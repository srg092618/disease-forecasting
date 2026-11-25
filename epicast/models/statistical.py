"""Classical time-series models. Each is fitted to one series at a time.

All four work on z_t = log1p(y_t) and produce Gaussian or Student-t predictive
distributions on that scale.

naive   z_{T+h} = z_T + noise                  random walk: "next year looks like this year"
drift   z_{T+h} = z_T + h*d + noise            random walk with drift: "the average trend continues"
ets     exponential smoothing, trend by AICc   level and (if supported) trend, weighted to recent years
arima   ARIMA(p,1,q) with drift, chosen by AICc  trend plus short-term momentum in the yearly changes

naive and drift are benchmarks. A model is only worth keeping if it beats them.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from .base import Q, SIGMA_FLOOR, LogScaleForecaster


class Naive(LogScaleForecaster):
    """Random walk without drift.

    Model:        z_t = z_{t-1} + e_t,   e_t ~ N(0, sigma^2)
    Forecast:     z_{T+h} | data  has mean z_T and variance h * sigma^2
    Estimate:     sigma^2 = mean of squared one-step changes (n = T-1 of them)

    Because sigma is estimated from only n changes, the predictive distribution
    is Student-t with n degrees of freedom rather than normal. With 20 points
    the difference at the 95% level is about 6%, so it matters.
    """

    name = "naive"

    def forecast_z(self, z, horizons):
        diffs = np.diff(z)
        n = len(diffs)
        sigma = max(np.sqrt(np.mean(diffs ** 2)), SIGMA_FLOOR)
        tq = stats.t.ppf(Q, df=max(n, 1))
        h = np.asarray(horizons, dtype=float)[:, None]
        return z[-1] + np.sqrt(h) * sigma * tq[None, :]


class Drift(LogScaleForecaster):
    """Random walk with drift.

    Model:        z_t = z_{t-1} + d + e_t
    Estimate:     d = mean one-step change = (z_T - z_1) / (T - 1)
    Forecast:     mean z_T + h*d, variance sigma^2 * h * (1 + h/(T-1))

    The extra factor (1 + h/(T-1)) is the uncertainty in the estimated drift
    itself: an error in d is multiplied by h, so it grows with the horizon.
    """

    name = "drift"

    def forecast_z(self, z, horizons):
        diffs = np.diff(z)
        n = len(diffs)
        d = diffs.mean()
        dof = max(n - 1, 1)
        sigma = max(np.sqrt(np.sum((diffs - d) ** 2) / dof), SIGMA_FLOOR)
        tq = stats.t.ppf(Q, df=dof)
        h = np.asarray(horizons, dtype=float)[:, None]
        mean = z[-1] + h * d
        sd = sigma * np.sqrt(h * (1.0 + h / n))
        return mean + sd * tq[None, :]


class ETS(LogScaleForecaster):
    """Exponential smoothing with additive errors; trend type chosen by AICc.

    Three candidates, from simplest to richest:

        ETS(A,N,N)   level only (simple exponential smoothing)
        ETS(A,A,N)   level + linear trend (Holt)
        ETS(A,Ad,N)  level + damped trend

    State equations of the damped model (level l, trend b, damping phi):

        z_t = l_{t-1} + phi*b_{t-1} + e_t
        l_t = l_{t-1} + phi*b_{t-1} + alpha*e_t
        b_t = phi*b_{t-1} + beta*e_t

    phi = 1 gives Holt's linear trend; dropping b gives the level-only model.
    Forecast mean: l_T + (phi + phi^2 + ... + phi^h) * b_T, so a damped trend
    flattens out instead of running to zero or infinity.

    Forecast variance. An error made j steps before the target period still
    moves the target by c_j = alpha + beta*(phi + ... + phi^j), so

        Var(z_{T+h}) = sigma^2 * (1 + c_1^2 + ... + c_{h-1}^2)

    Small samples. With 12 to 35 points, a five-parameter model can fit noise.
    Two protections: (1) AICc picks the simplest candidate the data support;
    (2) sigma^2 is estimated as SSE/(n - k), not SSE/n, and quantiles use a
    Student-t with n - k degrees of freedom (k = number of fitted parameters).
    Parameters are estimated by maximum likelihood (statsmodels ETSModel).
    """

    name = "ets"
    candidates = (  # (trend, damped, number of estimated parameters incl. initial states)
        (None, False, 2),
        ("add", False, 4),
        ("add", True, 5),
    )

    def forecast_z(self, z, horizons):
        import pandas as pd
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        H = max(horizons)
        n = len(z)
        best = None
        for trend, damped, k in self.candidates:
            if n - k < 3:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ETSModel(pd.Series(z), error="add", trend=trend, damped_trend=damped).fit(
                        disp=False, maxiter=200)
                if np.isfinite(res.aicc) and (best is None or res.aicc < best[0]):
                    best = (res.aicc, res, k)
            except Exception:
                continue
        if best is None:
            return Drift().forecast_z(z, horizons)
        _, res, k = best
        params = dict(zip(res.param_names, res.params))
        alpha = params["smoothing_level"]
        beta = params.get("smoothing_trend", 0.0)
        phi = params.get("damping_trend", 1.0)
        mean = np.asarray(res.forecast(H), dtype=float)
        dof = n - k
        sigma2 = float(res.sse) / dof
        sd = np.empty(H)
        for h in range(1, H + 1):
            c = alpha + beta * np.cumsum(phi ** np.arange(1, h))
            sd[h - 1] = np.sqrt(sigma2 * (1.0 + np.sum(c ** 2)))
        if not (np.isfinite(mean).all() and np.isfinite(sd).all()):
            return Drift().forecast_z(z, horizons)
        sd = np.maximum(sd, SIGMA_FLOOR)
        idx = np.asarray(horizons) - 1
        tq = stats.t.ppf(Q, df=dof)
        return mean[idx, None] + sd[idx, None] * tq[None, :]


class ARIMA(LogScaleForecaster):
    """ARIMA(p,1,q) with drift; (p,q) chosen from a small set by AICc.

    Writing w_t = z_t - z_{t-1} for the yearly change, the candidates are

        (0,1,0)  w_t = d + e_t                       (identical to ``drift``)
        (1,1,0)  w_t = d + phi*(w_{t-1} - d) + e_t   (changes have momentum)
        (0,1,1)  w_t = d + e_t + theta*e_{t-1}       (changes are partly noise that reverses)

    AICc = AIC + small-sample correction. It rewards fit and penalises extra
    parameters, which matters with 15 to 35 annual points. All candidates share
    d = 1, so their likelihoods are comparable (comparing AIC across different
    orders of differencing is not valid, so we do not).
    """

    name = "arima"
    candidates = ((0, 1, 0), (1, 1, 0), (0, 1, 1))

    def forecast_z(self, z, horizons):
        from statsmodels.tsa.arima.model import ARIMA as SMARIMA

        H = max(horizons)
        best = None
        for order in self.candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SMARIMA(z, order=order, trend="t").fit()
                    aicc = res.aicc
                if np.isfinite(aicc) and (best is None or aicc < best[0]):
                    best = (aicc, res, order)
            except Exception:
                continue
        if best is None:
            return Drift().forecast_z(z, horizons)
        _, res, order = best
        fc = res.get_forecast(steps=H)
        n = len(z) - 1                       # number of yearly changes
        k = 1 + order[0] + order[2]          # drift + AR + MA coefficients
        dof = max(n - k, 1)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        # statsmodels reports the maximum-likelihood SD (divides by n); rescale to
        # the unbiased n - k version and use Student-t quantiles, as for the other models.
        sd = np.maximum(np.asarray(fc.se_mean, dtype=float) * np.sqrt(n / dof), SIGMA_FLOOR)
        if not (np.isfinite(mean).all() and np.isfinite(sd).all()):
            return Drift().forecast_z(z, horizons)
        idx = np.asarray(horizons) - 1
        tq = stats.t.ppf(Q, df=dof)
        return mean[idx, None] + sd[idx, None] * tq[None, :]
