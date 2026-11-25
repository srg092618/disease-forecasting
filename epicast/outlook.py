"""Disease outlook: short, plain summaries derived from the forecasts.

Three kinds of outlook, matched to three kinds of evidence.

1. Annual burden (malaria, HIV). From the calibrated five-year forecast of the
   model that scored best in the backtest: is the 80% interval entirely below
   today's value ("likely to fall"), entirely above ("likely to rise"), or
   does it straddle it ("no clear direction")?

2. A live outbreak (Bundibugyo 2026). The current reproduction number with its
   uncertainty, and the calibrated four-week forecast.

3. Recurrence (new outbreaks of Ebola virus in DRC, Sudan virus in Uganda,
   Bundibugyo virus anywhere). How often have outbreaks started, and what does
   that rate imply for the next 12 months?

Recurrence model
----------------
Treat outbreak starts as a Poisson process with an unknown rate lambda per
year. Observing N outbreaks in T years, with the Jeffreys prior, gives the
posterior

    lambda | data ~ Gamma(shape = N + 1/2, rate = T)

Averaging the Poisson probability of "no outbreak next year" over that
posterior gives a closed form:

    P(at least one new outbreak in the next 12 months) = 1 - (T / (T + 1)) ** (N + 1/2)

This is a base rate. It assumes outbreaks arrive independently at a constant
average rate, and it knows nothing about surveillance, ecology, conflict or an
outbreak already under way. It answers "how often has this happened?", which
is the honest starting point for "how likely is it again?".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .backtest import Q_COLS
from .models.outbreak import Renewal

Q = list(config.QUANTILES)


def recurrence(n_outbreaks: int, years_observed: float) -> dict:
    """Rate of new outbreaks per year and the chance of at least one in the next 12 months."""
    shape = n_outbreaks + 0.5
    lo, hi = stats.gamma.ppf([0.05, 0.95], a=shape, scale=1.0 / years_observed)
    return {
        "n": int(n_outbreaks),
        "years": round(float(years_observed), 2),
        "rate_per_year": round(shape / years_observed, 3),
        "rate_lo": round(float(lo), 3),
        "rate_hi": round(float(hi), 3),
        "p_next_12_months": round(float(1.0 - (years_observed / (years_observed + 1.0)) ** shape), 3),
    }


def recurrence_table(outbreaks: pd.DataFrame, as_of: pd.Timestamp) -> list[dict]:
    """Recurrence statistics for the three outbreak groups the project tracks."""
    start = pd.Timestamp(year=config.REGISTRY_START_YEAR, month=1, day=1)
    years = (as_of - start).days / 365.25
    groups = [
        ("Ebola virus (Zaire) in DR Congo", (outbreaks["virus"] == "EBOV") & (outbreaks["country"] == "COD")),
        ("Sudan virus in Uganda", (outbreaks["virus"] == "SUDV") & (outbreaks["country"] == "UGA")),
        ("Bundibugyo virus, any country", outbreaks["virus"] == "BDBV"),
    ]
    rows = []
    for label, mask in groups:
        sub = outbreaks[mask & (outbreaks["start_year"] >= config.REGISTRY_START_YEAR)]
        sized = sub["cases"].dropna()
        row = {"group": label, **recurrence(len(sub), years),
               "median_cases": (float(sized.median()) if len(sized) else None),
               "largest_cases": (float(sized.max()) if len(sized) else None),
               "last_start": str(sub["start"].max()) if len(sub) else None,
               "ongoing": int((sub["status"] == "ongoing").sum())}
        rows.append(row)
    return rows


def best_model(metrics_overall: pd.DataFrame, target: str) -> str:
    """The model with the lowest calibrated relative error for a target."""
    m = metrics_overall[(metrics_overall["target"] == target) & (metrics_overall["stage"] == "calibrated")]
    return str(m.sort_values("rel_wis_log").iloc[0]["model"])


def annual_outlook(forecasts: pd.DataFrame, latest: dict[str, float], target: str, model: str) -> dict[str, dict]:
    """Per-country direction of travel from the final-horizon forecast of ``model``."""
    f = forecasts[(forecasts["target"] == target) & (forecasts["model"] == model)]
    h_max = f["horizon"].max()
    out = {}
    for loc, row in f[f["horizon"] == h_max].set_index("location").iterrows():
        now = latest.get(loc)
        if now is None:
            continue
        lo80, med, hi80 = row["q0.1"], row["q0.5"], row["q0.9"]
        direction = "likely to fall" if hi80 < now else "likely to rise" if lo80 > now else "no clear direction"
        out[loc] = {"latest": float(now), "year": int(row["period"]), "median": float(med),
                    "lo95": float(row["q0.025"]), "hi95": float(row["q0.975"]),
                    "lo80": float(lo80), "hi80": float(hi80), "direction": direction, "model": model}
    return out


def outbreak_outlook(series: pd.Series, final_fc: pd.DataFrame, location: str, model: str = "ensemble") -> dict:
    """Current R (posterior mean and 95% interval) and the calibrated four-week forecast."""
    y = series.to_numpy(float)
    renewal = Renewal()
    shape, rate = renewal.r_posterior(y, len(y) - 1)
    r_lo, r_hi = stats.gamma.ppf([0.025, 0.975], a=shape, scale=1.0 / rate)
    r_mean = shape / rate
    trend = "growing" if r_lo > 1 else "shrinking" if r_hi < 1 else "roughly steady"
    f = final_fc[(final_fc["location"] == location) & (final_fc["model"] == model)].sort_values("horizon")
    weeks = [{"week_ending": str(r["period"]), "median": float(r["q0.5"]),
              "lo95": float(r["q0.025"]), "hi95": float(r["q0.975"]),
              "lo50": float(r["q0.25"]), "hi50": float(r["q0.75"])} for _, r in f.iterrows()]
    return {"location": location, "as_of": series.index[-1].strftime("%Y-%m-%d"),
            "latest_week": float(y[-1]), "last4_total": float(y[-4:].sum()),
            "r_mean": round(float(r_mean), 2), "r_lo": round(float(r_lo), 2), "r_hi": round(float(r_hi), 2),
            "trend": trend, "model": model, "weeks": weeks}
