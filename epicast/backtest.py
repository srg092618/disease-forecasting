"""Rolling-origin backtesting: replaying history to measure forecast skill honestly.

The procedure ("time-series cross-validation", Tashman 2000):

    for each past date O (the "origin"):
        hide everything after O
        fit every model on what is left
        forecast 1..H steps ahead
        compare with what really happened

Every forecast is therefore scored on data the model had never seen, and the
training window only ever grows forward in time. A random train/test split,
the default in most machine-learning tutorials, would let the model see the
future and is not valid for time series.

Two caveats that no backtest on these data can remove are written up in the
documentation: the annual estimates are revised retrospectively by WHO and
UNAIDS (we only have today's vintage), and the Ebola counts are the final
cleaned totals rather than what was known in real time.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config, metrics
from .models import BASELINE, ENSEMBLE, build_models, ensemble_members
from .models.ensemble import quantile_median

log = logging.getLogger(__name__)

Q_COLS = [f"q{q:g}" for q in config.QUANTILES]


def log_volatility(train: np.ndarray) -> float:
    """SD of period-to-period changes in log1p(y): how jumpy this series has been so far."""
    d = np.diff(np.log1p(np.asarray(train, dtype=float)))
    return float(max(d.std(), 0.01)) if len(d) else 0.01


def _rows(target_key: str, model: str, location: str, origin, horizons: Iterable[int],
          periods: list, actuals: list[float], fc: np.ndarray, scale: float, vol: float) -> list[dict]:
    rows = []
    for i, h in enumerate(horizons):
        row = {"target": target_key, "model": model, "location": location, "origin": origin,
               "horizon": h, "period": periods[i], "y": actuals[i], "mase_scale": scale, "vol": vol}
        row.update({c: fc[i, j] for j, c in enumerate(Q_COLS)})
        rows.append(row)
    return rows


# ------------------------------------------------------------------------ annual
def annual_origin_forecasts(target: config.Target, series: dict[str, pd.Series], origin: int) -> list[dict]:
    """All models' forecasts made at the end of year ``origin`` (no later data used)."""
    horizons = list(target.horizons)
    panel = {loc: s[s.index <= origin] for loc, s in series.items()}
    panel = {loc: s for loc, s in panel.items() if len(s) >= 5 and s.index[-1] == origin}
    active = {loc: s for loc, s in panel.items()
              if len(s) >= target.min_train and s.iloc[-1] >= target.min_level}
    if not active:
        return []

    models = build_models(target)
    arrays = {loc: s.to_numpy(float) for loc, s in panel.items()}
    for m in models.values():
        if m.is_global:
            m.fit_panel(arrays, horizons)  # one fit per origin, on data up to the origin only

    rows: list[dict] = []
    members = ensemble_members(target)
    for loc, s in active.items():
        y_train = s.to_numpy(float)
        full = series[loc]
        periods = [origin + h for h in horizons]
        actuals = [float(full.get(p, np.nan)) for p in periods]
        scale = metrics.mase_scale(y_train)
        vol = log_volatility(y_train)
        by_model = {name: m.forecast(y_train, horizons) for name, m in models.items()}
        by_model[ENSEMBLE] = quantile_median([by_model[n] for n in members])
        for name, fc in by_model.items():
            rows += _rows(target.key, name, loc, origin, horizons, periods, actuals, fc, scale, vol)
    return rows


def backtest_annual(target: config.Target, series: dict[str, pd.Series], n_jobs: int = 1) -> pd.DataFrame:
    last = max(s.index[-1] for s in series.values())
    origins = list(range(target.first_origin, last))
    log.info("%s: %d origins (%d-%d)", target.key, len(origins), origins[0], origins[-1])
    chunks = Parallel(n_jobs=n_jobs)(
        delayed(annual_origin_forecasts)(target, series, o) for o in origins)
    return pd.DataFrame([r for chunk in chunks for r in chunk])


def final_annual_forecasts(target: config.Target, series: dict[str, pd.Series]) -> pd.DataFrame:
    """Forecasts beyond the end of the data, from the latest year available."""
    last = max(s.index[-1] for s in series.values())
    return pd.DataFrame(annual_origin_forecasts(target, series, last))


# ---------------------------------------------------------------------- outbreak
def outbreak_origin_forecasts(target: config.Target, location: str, s: pd.Series, i: int) -> list[dict]:
    """All models' forecasts using weeks 0..i only."""
    horizons = list(target.horizons)
    y_train = s.to_numpy(float)[: i + 1]
    models = build_models(target)
    by_model = {name: m.forecast(y_train, horizons) for name, m in models.items()}
    by_model[ENSEMBLE] = quantile_median([by_model[n] for n in ensemble_members(target)])
    periods, actuals = [], []
    for h in horizons:
        j = i + h
        periods.append(s.index[j].strftime("%Y-%m-%d") if j < len(s)
                       else (s.index[-1] + pd.Timedelta(weeks=j - len(s) + 1)).strftime("%Y-%m-%d"))
        actuals.append(float(s.iloc[j]) if j < len(s) else np.nan)
    scale = metrics.mase_scale(y_train)
    vol = log_volatility(y_train)
    origin = s.index[i].strftime("%Y-%m-%d")
    rows: list[dict] = []
    for name, fc in by_model.items():
        rows += _rows(target.key, name, location, origin, horizons, periods, actuals, fc, scale, vol)
    return rows


def outbreak_origins(target: config.Target, s: pd.Series) -> list[int]:
    """Week indices used as origins: enough history, and transmission still under way."""
    y = s.to_numpy(float)
    return [i for i in range(target.min_train - 1, len(y) - 1)
            if y[max(0, i - 3): i + 1].sum() >= target.min_level]


def backtest_outbreak(target: config.Target, series: dict[str, pd.Series], n_jobs: int = 1) -> pd.DataFrame:
    jobs = [(loc, s, i) for loc, s in series.items() for i in outbreak_origins(target, s)]
    log.info("%s: %d location-origins", target.key, len(jobs))
    chunks = Parallel(n_jobs=n_jobs)(
        delayed(outbreak_origin_forecasts)(target, loc, s, i) for loc, s, i in jobs)
    return pd.DataFrame([r for chunk in chunks for r in chunk])


def final_outbreak_forecasts(target: config.Target, series: dict[str, pd.Series]) -> pd.DataFrame:
    """Forecasts beyond the newest week, for an outbreak that is still in progress."""
    rows: list[dict] = []
    for loc, s in series.items():
        rows += outbreak_origin_forecasts(target, loc, s, len(s) - 1)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- scoring
def score(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Attach every metric to each forecast row that has a known outcome."""
    df = forecasts.dropna(subset=["y"]).copy()
    q = df[Q_COLS].to_numpy(float)
    y = df["y"].to_numpy(float)
    med = q[:, config.MEDIAN_INDEX]
    df["wis"] = metrics.wis(y, q)
    df["wis_log"] = metrics.wis(np.log1p(y), np.log1p(q))
    df["ae"] = np.abs(y - med)
    df["ae_log"] = np.abs(np.log1p(y) - np.log1p(med))
    df["mase"] = df["ae"] / df["mase_scale"]
    for level in config.INTERVALS:
        df[f"cov{int(level * 100)}"] = metrics.covered(y, q, level)
    return df


def summarise(scores: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Average the scores and express WIS relative to the naive baseline.

    rel_wis_log < 1 means the model beat the random-walk baseline on the
    log-scale weighted interval score. All models are scored on exactly the
    same forecasts, so the ratio of means is a like-for-like comparison.
    """
    agg = (scores.groupby(by, observed=True)
           .agg(n=("y", "size"), wis_log=("wis_log", "mean"), wis=("wis", "mean"),
                ae_log=("ae_log", "mean"), mase=("mase", "mean"),
                cov50=("cov50", "mean"), cov80=("cov80", "mean"),
                cov90=("cov90", "mean"), cov95=("cov95", "mean"))
           .reset_index())
    base_keys = [c for c in by if c != "model"]
    base = agg[agg["model"] == BASELINE][base_keys + ["wis_log", "wis", "ae_log"]].rename(
        columns={"wis_log": "_b_wis_log", "wis": "_b_wis", "ae_log": "_b_ae_log"})
    agg = agg.merge(base, on=base_keys, how="left") if base_keys else agg.assign(
        _b_wis_log=base["_b_wis_log"].iloc[0], _b_wis=base["_b_wis"].iloc[0],
        _b_ae_log=base["_b_ae_log"].iloc[0])
    agg["rel_wis_log"] = agg["wis_log"] / agg["_b_wis_log"]
    agg["rel_wis"] = agg["wis"] / agg["_b_wis"]
    agg["rel_ae_log"] = agg["ae_log"] / agg["_b_ae_log"]
    return agg.drop(columns=["_b_wis_log", "_b_wis", "_b_ae_log"])


def bootstrap_rel_wis(scores: pd.DataFrame, n_boot: int = 2000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """95% interval for each model's relative log-WIS, resampling whole locations.

    Forecasts for the same country are strongly correlated (overlapping
    windows, the same shocks), so rows are not independent. Resampling
    countries rather than rows (a cluster bootstrap) respects that. With only
    three locations, as for Ebola, this interval is not meaningful and is not
    reported.
    """
    rng = np.random.default_rng(seed)
    per_loc = scores.groupby(["target", "model", "location"], observed=True)["wis_log"].agg(["sum", "size"])
    out = []
    for target, grp in per_loc.groupby(level="target"):
        sums = grp["sum"].unstack("location").droplevel("target")
        sizes = grp["size"].unstack("location").droplevel("target")
        locs = sums.columns.to_numpy()
        if len(locs) < 10 or BASELINE not in sums.index:
            continue
        draws = rng.integers(0, len(locs), size=(n_boot, len(locs)))
        s, n = sums.to_numpy(), sizes.to_numpy()
        means = s[:, draws].sum(axis=2) / n[:, draws].sum(axis=2)  # (models, n_boot)
        b = list(sums.index).index(BASELINE)
        ratio = means / means[b]
        for i, model in enumerate(sums.index):
            lo, hi = np.quantile(ratio[i], [0.025, 0.975])
            out.append({"target": target, "model": model, "rel_wis_log_lo": lo, "rel_wis_log_hi": hi})
    return pd.DataFrame(out)
