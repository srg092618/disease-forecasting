"""Shared model interface.

Every model, simple or complex, answers the same question in the same format:

    forecast(y, horizons) -> array of shape (len(horizons), len(QUANTILES))

``y`` is the observed history on its natural scale (cases, rates). The result
is a set of predictive quantiles for each horizon, also on the natural scale.
A common interface is what lets one backtest harness and one scoring module
serve every model and every disease.

The log1p scale
---------------
Disease counts and rates are positive, right-skewed, and change
multiplicatively (a country goes from 200 to 180, another from 20 to 18:
both fell 10%). Models therefore work on z = log(1 + y):

* differences of z are approximately growth rates,
* errors are closer to symmetric and constant in size,
* back-transformed forecasts can never be negative.

Because log1p is strictly increasing, the q-th quantile of z maps exactly to
the q-th quantile of y. (The same is not true of the mean, which is why the
point forecast we report is the median.)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import QUANTILES

Q = np.asarray(QUANTILES)
SIGMA_FLOOR = 0.005  # numerical floor on log-scale innovation SD (0.5%)


def to_z(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(y, dtype=float))


def from_z(z: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(z), 0.0, None)


def enforce_monotone(quantiles: np.ndarray) -> np.ndarray:
    """Sort each row so quantiles never cross (rearrangement).

    Sorting is the standard repair (Chernozhukov, Fernandez-Val and Galichon,
    2010). Swapping two crossed quantiles q_lo > q_hi at levels tau_lo < tau_hi
    lowers the summed pinball loss by exactly (tau_hi - tau_lo) * (q_lo - q_hi),
    whatever the outcome, so sorting can never make the forecast's score worse.
    """
    return np.sort(np.asarray(quantiles, dtype=float), axis=-1)


class Forecaster:
    """Base class. Subclasses implement ``forecast``; global models also ``fit_panel``."""

    name: str = "base"
    is_global: bool = False

    def fit_panel(self, panel: dict[str, np.ndarray], horizons: Sequence[int]) -> None:
        """Learn from many series at once. Local models ignore this."""

    def forecast(self, y: np.ndarray, horizons: Sequence[int]) -> np.ndarray:
        raise NotImplementedError


class LogScaleForecaster(Forecaster):
    """A model that works on z = log1p(y) and returns natural-scale quantiles."""

    def forecast(self, y: np.ndarray, horizons: Sequence[int]) -> np.ndarray:
        z = to_z(y)
        zq = self.forecast_z(z, list(horizons))
        return from_z(enforce_monotone(zq))

    def forecast_z(self, z: np.ndarray, horizons: list[int]) -> np.ndarray:
        raise NotImplementedError
