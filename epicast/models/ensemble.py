"""Combining models.

The ensemble takes, for every horizon and every quantile level, the median of
the member models' values:

    q_ensemble(tau, h) = median over models m of q_m(tau, h)

Why it works: different models make different mistakes, and the median ignores
whichever one is furthest off at that moment. In the large collaborative
epidemic-forecasting hubs, a simple equal-weight quantile ensemble of this
kind has been as accurate as, or more accurate than, nearly every individual
model, and more stable than weighted schemes that try to learn who is best
(Cramer et al., 2022; Ray et al., 2023).

Membership is fixed in models/__init__.py before any results are seen. Picking
members after looking at the backtest would be tuning on the test set.
"""
from __future__ import annotations

import numpy as np

from .base import enforce_monotone


def quantile_median(member_forecasts: list[np.ndarray]) -> np.ndarray:
    """Median across members of arrays shaped (..., n_quantiles)."""
    stacked = np.stack(member_forecasts, axis=0)
    return enforce_monotone(np.median(stacked, axis=0))
