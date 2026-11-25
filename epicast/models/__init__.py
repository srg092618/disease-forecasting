"""Model registry: which models run for which kind of target.

The ensemble members are declared here, in advance, and are never changed in
response to backtest results.
"""
from __future__ import annotations

from ..config import Target
from .base import Forecaster
from .outbreak import Renewal, Richards
from .pooled import GlobalGBM
from .statistical import ARIMA, ETS, Drift, Naive

BASELINE = "naive"
ENSEMBLE = "ensemble"

ANNUAL_ENSEMBLE_MEMBERS = ("drift", "ets", "arima", "gbm")
OUTBREAK_ENSEMBLE_MEMBERS = ("ets", "renewal", "richards")

MODEL_LABELS = {
    "naive": "Naive (random walk)",
    "drift": "Random walk with drift",
    "ets": "Exponential smoothing (trend by AICc)",
    "arima": "ARIMA with drift (AICc)",
    "gbm": "Global gradient boosting",
    "renewal": "Renewal equation (R_t)",
    "richards": "Richards growth curve",
    "ensemble": "Ensemble (quantile median)",
}


def build_models(target: Target) -> dict[str, Forecaster]:
    """Fresh model instances for a target, keyed by name."""
    if target.freq == "A":
        return {
            "naive": Naive(),
            "drift": Drift(),
            "ets": ETS(),
            "arima": ARIMA(),
            "gbm": GlobalGBM(min_level=target.min_level),
        }
    return {
        "naive": Naive(),
        "ets": ETS(),
        "renewal": Renewal(),
        "richards": Richards(),
    }


def ensemble_members(target: Target) -> tuple[str, ...]:
    return ANNUAL_ENSEMBLE_MEMBERS if target.freq == "A" else OUTBREAK_ENSEMBLE_MEMBERS
