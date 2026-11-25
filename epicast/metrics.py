"""Forecast scoring.

A forecast here is a set of quantiles, so it is scored as a probability
statement, not just as a single number.

pinball loss      the building block: scores one quantile against the outcome
WIS               weighted interval score: scores the whole set of quantiles
coverage          how often the X% interval actually contained the outcome
MASE              point-forecast error relative to a naive in-sample forecast

Lower is better for every loss. All losses are "proper": a forecaster cannot
improve their expected score by reporting anything other than what they
really believe, which is the property that makes the comparison fair
(Gneiting and Raftery, 2007).
"""
from __future__ import annotations

import numpy as np

from .config import INTERVALS, MEDIAN_INDEX, QUANTILES

Q = np.asarray(QUANTILES)


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, tau: np.ndarray | float) -> np.ndarray:
    """Pinball (quantile) loss.

        L_tau(y, q) = tau * (y - q)        if y >= q   (forecast too low)
                      (1 - tau) * (q - y)  if y <  q   (forecast too high)

    For tau = 0.9, being too low costs nine times more than being too high, so
    the loss is minimised on average by the true 90th percentile.
    """
    y = np.asarray(y, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    diff = y - q_pred
    return np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    """Interval score of a central (1 - alpha) prediction interval.

        IS_alpha = (u - l) + (2/alpha)*(l - y)*[y < l] + (2/alpha)*(y - u)*[y > u]

    Width is penalised (sharpness), and so is missing the outcome, in
    proportion to how far outside it fell.
    """
    y = np.asarray(y, dtype=float)
    return ((upper - lower)
            + (2.0 / alpha) * np.clip(lower - y, 0.0, None)
            + (2.0 / alpha) * np.clip(y - upper, 0.0, None))


def wis(y: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Weighted interval score (Bracher et al., 2021).

    With a median m and K central intervals at levels alpha_1..alpha_K:

        WIS = ( 0.5*|y - m| + sum_k (alpha_k/2) * IS_alpha_k ) / (K + 0.5)

    It generalises absolute error (a forecast with zero-width intervals gets
    WIS = |y - m|), is measured in the units of y, and approximates the
    continuous ranked probability score.

    ``y`` has shape (...,) and ``quantiles`` shape (..., n_quantiles) on the
    grid in config.QUANTILES.
    """
    y = np.asarray(y, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    total = 0.5 * np.abs(y - quantiles[..., MEDIAN_INDEX])
    for level in INTERVALS:
        alpha = 1.0 - level
        lo = quantiles[..., QUANTILES.index(round(alpha / 2, 4))]
        hi = quantiles[..., QUANTILES.index(round(1 - alpha / 2, 4))]
        total = total + (alpha / 2.0) * interval_score(y, lo, hi, alpha)
    return total / (len(INTERVALS) + 0.5)


def wis_from_pinball(y: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """The same score computed a second way: twice the mean pinball loss.

    (alpha/2) * IS_alpha equals the pinball losses of the interval's two end
    points added together, and 0.5*|y - m| is the pinball loss of the median.
    So WIS = 2 * (average pinball loss over all quantile levels). The test
    suite checks that both routes agree.
    """
    y = np.asarray(y, dtype=float)[..., None]
    return 2.0 * pinball_loss(y, np.asarray(quantiles, dtype=float), Q).mean(axis=-1)


def covered(y: np.ndarray, quantiles: np.ndarray, level: float) -> np.ndarray:
    """1 where the central ``level`` interval contains y, else 0."""
    alpha = 1.0 - level
    lo = quantiles[..., QUANTILES.index(round(alpha / 2, 4))]
    hi = quantiles[..., QUANTILES.index(round(1 - alpha / 2, 4))]
    y = np.asarray(y, dtype=float)
    return ((y >= lo) & (y <= hi)).astype(float)


def mase_scale(train: np.ndarray) -> float:
    """In-sample mean absolute one-step change: the naive forecast's typical error.

    MASE = |forecast error| / this scale (Hyndman and Koehler, 2006). A value
    below 1 means the forecast beat the average in-sample naive one-step
    error. Scaling by the series' own variability is what allows errors to be
    averaged across countries of very different size.
    """
    train = np.asarray(train, dtype=float)
    if len(train) < 2:
        return np.nan
    scale = np.mean(np.abs(np.diff(train)))
    return float(scale) if scale > 0 else np.nan
