"""Interval recalibration from each model's own track record.

The problem
-----------
A model's 95% interval is a claim: "the outcome will land in here 19 times out
of 20". Models routinely break that promise. They assume tidy bell-shaped
noise, ignore the uncertainty in their own fitted parameters, and cannot see
structural breaks coming, so their intervals are usually too narrow.

The fix: conformalized quantile regression (Romano, Patterson and Candes, 2019)
------------------------------------------------------------------------------
Check the claim against the past and widen (or narrow) the intervals until it
would have been true. For one model, one horizon and one interval level L
(say 95%), working on the log scale z = log1p(y):

1.  Take every past forecast whose outcome is already known.
2.  For each, measure by how much the interval missed, in units of that
    series' own volatility v (the SD of its past period-to-period log changes):

        E = max(lower - z, z - upper) / v

    E is positive when the outcome fell outside and negative when it fell
    inside. Dividing by v makes a miss in a jumpy series (Rwanda's malaria
    estimates) comparable to a miss in a smooth one (Nigeria's).
3.  Let c be the L-th empirical quantile of those E values, using the
    finite-sample level ceil((n + 1) * L) / n.
4.  Replace every new interval [lower, upper] with
    [lower - c*v, upper + c*v], using the new series' own v.

By construction the adjusted intervals would have covered a fraction L of the
past outcomes, and volatile series are widened more than stable ones (a
"locally adaptive" conformal margin; Lei et al., 2018). Because the shift is
on the log scale it widens an interval by the ratio exp(c*v) on the natural
scale. The median is never moved: calibration changes the stated uncertainty,
not the point forecast.

The formal guarantee of conformal prediction assumes past and future errors
are exchangeable. Time series are not exactly so, which is why the backtest
reports the coverage actually achieved instead of assuming it.

No leakage
----------
A forecast made at origin O is calibrated using only forecasts whose target
period is at or before O, and v uses only data up to each forecast's own
origin. Early origins, with too little history, are left uncalibrated.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from .backtest import Q_COLS

N_MIN = 30                 # minimum number of resolved forecasts before a margin is trusted
MARGIN_RANGE = (-1.0, 1.5)  # log-scale bounds on any one margin (exp(1.5) is about 4.5x)

_QI = {q: i for i, q in enumerate(config.QUANTILES)}
_PAIRS = [(level, _QI[round((1 - level) / 2, 4)], _QI[round(1 - (1 - level) / 2, 4)])
          for level in config.INTERVALS]
_M = config.MEDIAN_INDEX


def conformity_scores(z: np.ndarray, zq: np.ndarray, vol: np.ndarray, lo_idx: int, hi_idx: int) -> np.ndarray:
    """E = max(lower - z, z - upper) / v: how far outside (+) or inside (-) each outcome fell."""
    return np.maximum(zq[:, lo_idx] - z, z - zq[:, hi_idx]) / vol


def conformal_margin(scores: np.ndarray, level: float) -> float:
    """The margin (per unit of volatility) that would have given ``level`` coverage on ``scores``."""
    n = len(scores)
    if n < N_MIN:
        return 0.0
    k = min(math.ceil((n + 1) * level), n)
    return float(np.sort(scores)[k - 1])


def _margins(z: np.ndarray, zq: np.ndarray, vol: np.ndarray, horizon: np.ndarray, h: int) -> dict[float, float]:
    """Margins for horizon ``h``; falls back to all horizons pooled if ``h`` has too little history."""
    use = horizon == h
    if use.sum() < N_MIN:
        use = np.ones(len(z), dtype=bool)
    return {level: conformal_margin(conformity_scores(z[use], zq[use], vol[use], lo, hi), level)
            for level, lo, hi in _PAIRS}


def _apply(zq: np.ndarray, vol: np.ndarray, margins: dict[float, float]) -> np.ndarray:
    out = zq.copy()
    med = zq[:, _M]
    for level, lo, hi in _PAIRS:
        c = margins[level]
        if c == 0.0:
            continue
        shift = np.clip(c * vol, *MARGIN_RANGE)
        out[:, lo] = np.minimum(zq[:, lo] - shift, med)   # an interval never crosses its own median
        out[:, hi] = np.maximum(zq[:, hi] + shift, med)
    return np.sort(np.clip(out, 0.0, None), axis=1)  # z = log1p(y) cannot be negative


def calibrate_backtest(raw: pd.DataFrame) -> pd.DataFrame:
    """Recalibrate every backtest forecast using only outcomes known at its origin."""
    out = raw.copy()
    values = out[Q_COLS].to_numpy(float).copy()
    for (_, _), idx in raw.groupby(["target", "model"], sort=False).indices.items():
        grp = raw.iloc[idx]
        zq = np.log1p(grp[Q_COLS].to_numpy(float))
        z = np.log1p(grp["y"].to_numpy(float))
        origin = grp["origin"].astype(str).to_numpy()
        period = grp["period"].astype(str).to_numpy()
        horizon = grp["horizon"].to_numpy()
        vol = grp["vol"].to_numpy(float)
        known = np.isfinite(z)
        new = zq.copy()
        for o in np.unique(origin):
            cal = known & (period <= o)  # outcome already observed at this origin
            rows_o = origin == o
            for h in np.unique(horizon[rows_o]):
                sel = rows_o & (horizon == h)
                f = _margins(z[cal], zq[cal], vol[cal], horizon[cal], h) if cal.sum() >= N_MIN else None
                if f is not None:
                    new[sel] = _apply(zq[sel], vol[sel], f)
        values[idx] = np.expm1(new)
    out[Q_COLS] = values
    return out


def final_margins(raw: pd.DataFrame) -> pd.DataFrame:
    """Margins learned from all resolved backtest forecasts, for use on future forecasts.

    One extra rule applies here: a margin may not be smaller than the margin at
    the previous horizon. With few resolved forecasts (a single live outbreak)
    the per-horizon margins are noisy, and without this rule a four-week
    interval could come out narrower than a one-week interval.
    """
    rows = []
    for (target, model), grp in raw.dropna(subset=["y"]).groupby(["target", "model"], sort=False):
        zq = np.log1p(grp[Q_COLS].to_numpy(float))
        z = np.log1p(grp["y"].to_numpy(float))
        horizon = grp["horizon"].to_numpy()
        vol = grp["vol"].to_numpy(float)
        typical = float(np.median(vol))
        previous: dict[float, float] | None = None
        for h in np.sort(np.unique(horizon)):
            f = _margins(z, zq, vol, horizon, h)
            if previous is not None:  # published uncertainty never shrinks as the horizon grows
                f = {level: max(f[level], previous[level]) for level in f}
            previous = f
            n = int((horizon == h).sum())
            rows += [{"target": target, "model": model, "horizon": int(h), "level": level,
                      "margin_per_vol": f[level],
                      "typical_width_ratio": float(np.exp(np.clip(f[level] * typical, *MARGIN_RANGE))),
                      "n": n} for level in f]
    return pd.DataFrame(rows)


def calibrate_final(final: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """Apply stored margins to forecasts that have no outcome yet."""
    out = final.copy()
    values = out[Q_COLS].to_numpy(float).copy()
    vol = out["vol"].to_numpy(float)
    lookup = {(t, m, h): g.set_index("level")["margin_per_vol"].to_dict()
              for (t, m, h), g in margins.groupby(["target", "model", "horizon"])}
    for (t, m, h), idx in final.groupby(["target", "model", "horizon"], sort=False).indices.items():
        f = lookup.get((t, m, int(h)))
        if f is None:
            continue
        values[idx] = np.expm1(_apply(np.log1p(values[idx]), vol[idx], f))
    out[Q_COLS] = values
    return out
