"""A global ("pooled") machine-learning model: one model trained on all countries.

Why pool?
---------
A single country gives 25 to 35 annual points. That is far too few to train a
flexible model. But 100+ countries give thousands of (history -> what happened
next) examples, and the dynamics are similar enough across countries to learn
from each other. This "cross-learning" is how machine-learning methods became
competitive in forecasting (Montero-Manso and Hyndman, 2021).

How it is framed as supervised learning
---------------------------------------
For every country, every past year t and every horizon h we build one row:

    features   the last three yearly changes in z = log1p(y)
               the mean and SD of the last five yearly changes
               the current level z_t
               the horizon h
    label      z_{t+h} - z_t          (the h-year change that actually followed)

All features are differences or levels on the log scale, so a country with
300 cases per 1,000 and one with 3 per 1,000 are directly comparable.

Why gradient boosting with quantile loss
----------------------------------------
Gradient boosting builds many small decision trees, each one correcting the
errors of the sum so far (Friedman, 2001). Training it with the pinball
(quantile) loss makes it predict a chosen quantile rather than the mean
(Koenker and Bassett, 1978). We train one model per quantile, which gives a
full predictive distribution whose width can depend on the features: wider
after volatile years, narrower after stable ones.

Leakage guard: ``fit_panel`` only ever receives data up to the forecast origin,
and a row is only built when its label year is inside that data.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import QUANTILES, RANDOM_SEED
from .base import Forecaster, enforce_monotone, from_z, to_z

N_LAGS = 3
WINDOW = 5
FEATURE_NAMES = ["change_1", "change_2", "change_3", "mean_change_5", "sd_change_5", "level", "horizon"]


def features_at(z: np.ndarray, t: int, h: int) -> list[float]:
    """Feature vector describing series ``z`` as seen at index ``t``, for horizon ``h``."""
    d = np.diff(z[: t + 1])
    recent = d[-WINDOW:]
    return [d[-1], d[-2], d[-3], float(recent.mean()), float(recent.std()), z[t], float(h)]


def build_training_rows(panel: dict[str, np.ndarray], horizons: Sequence[int],
                        min_level: float) -> tuple[np.ndarray, np.ndarray]:
    """Turn a panel of series into (X, y) rows. Uses nothing beyond each series' end."""
    X, y = [], []
    for series in panel.values():
        z = to_z(series)
        T = len(z)
        for t in range(N_LAGS, T - 1):
            if series[t] < min_level:
                continue  # train only on states where the disease is still present
            for h in horizons:
                if t + h < T:
                    X.append(features_at(z, t, h))
                    y.append(z[t + h] - z[t])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


class GlobalGBM(Forecaster):
    """Histogram gradient boosting, one model per quantile, shared by all countries."""

    name = "gbm"
    is_global = True

    def __init__(self, min_level: float = 0.0, max_iter: int = 150, learning_rate: float = 0.06,
                 max_leaf_nodes: int = 8, min_samples_leaf: int = 20, l2: float = 1.0):
        self.min_level = min_level
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate,
                           max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                           l2_regularization=l2, early_stopping=False, random_state=RANDOM_SEED)
        self.models_: list | None = None
        self.n_rows_: int = 0

    def fit_panel(self, panel, horizons):
        from sklearn.ensemble import HistGradientBoostingRegressor

        X, y = build_training_rows(panel, horizons, self.min_level)
        self.n_rows_ = len(y)
        if len(y) < 200:  # not enough examples to learn from; forecast() falls back
            self.models_ = None
            return
        self.models_ = [
            HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.params).fit(X, y)
            for q in QUANTILES
        ]

    def forecast(self, y, horizons):
        from .statistical import Drift

        z = to_z(y)
        if self.models_ is None or len(z) < N_LAGS + 1:
            return Drift().forecast(y, horizons)
        t = len(z) - 1
        X = np.asarray([features_at(z, t, h) for h in horizons], dtype=float)
        change = np.column_stack([m.predict(X) for m in self.models_])  # (H, Q)
        return from_z(enforce_monotone(z[t] + change))
