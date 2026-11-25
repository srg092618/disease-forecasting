"""Outbreak models for weekly case counts (used for Ebola).

An outbreak is a different statistical object from an annual burden estimate:
it starts from nothing, grows roughly exponentially, bends, peaks and dies
out, all within months. Two model families built for that shape:

renewal    semi-mechanistic. Estimates the reproduction number R_t from the
           last three weeks and simulates transmission forward.
richards   phenomenological. Fits an S-shaped growth curve to cumulative cases
           and extrapolates it.

Both produce their uncertainty by simulation, so they return quantiles of
simulated future counts.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import optimize, stats

from .. import config
from .base import Q, Forecaster, enforce_monotone


# ----------------------------------------------------------------------- helpers
def discretised_serial_interval(mean_days: float, sd_days: float, max_days: int = 60) -> np.ndarray:
    """Probability that the serial interval is s days, for s = 1..max_days.

    The serial interval is gamma distributed with the given mean and SD:
    shape = (mean/sd)^2, scale = sd^2/mean. Day s gets the probability mass
    between s-0.5 and s+0.5 (day 1 also takes the mass below 0.5), and the
    weights are renormalised to sum to 1.
    """
    shape = (mean_days / sd_days) ** 2
    scale = sd_days ** 2 / mean_days
    edges = np.concatenate([[0.0], np.arange(1.5, max_days + 1.0)])
    w = np.diff(stats.gamma.cdf(edges, a=shape, scale=scale))
    return w / w.sum()


def nb_dispersion(observed: np.ndarray, expected: np.ndarray) -> float:
    """Method-of-moments estimate of the negative binomial dispersion 1/k.

    A negative binomial count with mean mu has variance mu + mu^2/k. Setting
    the observed squared errors equal to that variance and solving gives

        1/k = sum((y - mu)^2 - mu) / sum(mu^2)

    1/k = 0 is the Poisson case; larger values mean noisier counts.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    denom = np.sum(expected ** 2)
    if denom <= 0:
        return 0.05
    inv_k = np.sum((observed - expected) ** 2 - expected) / denom
    return float(np.clip(inv_k, 1e-3, 2.0))


def draw_negbin(rng: np.random.Generator, mean: np.ndarray, inv_k: float) -> np.ndarray:
    """Draw negative binomial counts as a gamma-mixed Poisson."""
    mean = np.maximum(np.asarray(mean, dtype=float), 0.0)
    k = 1.0 / inv_k
    lam = rng.gamma(shape=k, scale=np.maximum(mean, 1e-12) / k)
    return rng.poisson(lam).astype(float)


# ----------------------------------------------------------------------- renewal
class Renewal(Forecaster):
    """Renewal-equation forecast with a time-varying reproduction number.

    The renewal equation says today's expected new cases are produced by past
    cases, weighted by how infectious a case is s days after its own onset:

        E[I_t] = R_t * Lambda_t,     Lambda_t = sum_s w_s * I_{t-s}

    w is the serial-interval distribution and R_t is the average number of
    people each case infects. R_t > 1 means growth, R_t < 1 means decline.

    Estimating R_t (Cori et al., 2013). With a Gamma(a, b) prior on R and
    Poisson-distributed cases, the posterior over a window of days is again a
    gamma distribution:

        R | data ~ Gamma(shape = a + sum I_t, rate = 1/b + sum Lambda_t)

    We use the last 21 days and the usual weak prior (mean 5, SD 5).

    Forecasting. For each of n simulated futures: draw R from that posterior,
    let log R take a random-walk step each week (transmission keeps changing as
    behaviour and control measures change), run the renewal equation forward
    day by day, and add negative binomial noise to each weekly total. The step
    size and the noise level are both estimated from how the outbreak has
    behaved so far, not chosen by hand.

    Guard rail. Simulated R is capped at ``r_max`` = 3. Published estimates of
    the basic reproduction number for this epidemic were 1.7 to 2.0 (WHO Ebola
    Response Team, 2014), so sustained transmission above 3 is not plausible.
    Without the cap, a few early weeks in which cases appear faster than past
    local cases can explain (imported or late-reported cases) produce R
    estimates near 10 and forecasts in the millions.

    Weekly data are spread evenly over the seven days of each week to run the
    daily recursion. That is an approximation; daily line lists would be better.
    """

    name = "renewal"

    def __init__(self, window_weeks: int = 3, n_sims: int = 2000,
                 prior_mean: float = 5.0, prior_sd: float = 5.0, r_max: float = 3.0):
        self.window_weeks = window_weeks
        self.n_sims = n_sims
        self.r_max = r_max
        self.prior_shape = (prior_mean / prior_sd) ** 2
        self.prior_scale = prior_sd ** 2 / prior_mean
        self.w = discretised_serial_interval(config.EBOLA_SERIAL_INTERVAL_MEAN_DAYS,
                                             config.EBOLA_SERIAL_INTERVAL_SD_DAYS)

    # -- pieces, exposed for the dashboard and for teaching
    def infection_pressure(self, daily: np.ndarray) -> np.ndarray:
        """Lambda_t = sum_s w_s * I_{t-s} for every day t."""
        kernel = np.concatenate([[0.0], self.w])
        return np.convolve(daily, kernel)[: len(daily)]

    def r_posterior(self, weekly: np.ndarray, end_week: int) -> tuple[float, float]:
        """Posterior (shape, rate) of R using the window that ends at ``end_week`` (inclusive)."""
        daily = np.repeat(np.asarray(weekly, dtype=float) / 7.0, 7)
        lam = self.infection_pressure(daily)
        lo = max(0, (end_week + 1 - self.window_weeks) * 7)
        hi = (end_week + 1) * 7
        shape = self.prior_shape + daily[lo:hi].sum()
        rate = 1.0 / self.prior_scale + lam[lo:hi].sum()
        return shape, rate

    def r_history(self, weekly: np.ndarray) -> np.ndarray:
        """Posterior mean of R at the end of every week (NaN for the first few)."""
        out = np.full(len(weekly), np.nan)
        for j in range(self.window_weeks, len(weekly)):
            shape, rate = self.r_posterior(weekly, j)
            out[j] = shape / rate
        return out

    def _one_week_ahead_means(self, weekly: np.ndarray, r_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """In-sample one-week-ahead expected counts, used to calibrate the noise."""
        S = len(self.w)
        w_rev = self.w[::-1]
        obs, exp = [], []
        daily = np.repeat(np.asarray(weekly, dtype=float) / 7.0, 7)
        for j in range(self.window_weeks + 1, len(weekly)):
            r = r_hist[j - 1]
            if not np.isfinite(r):
                continue
            r = min(r, self.r_max)
            buf = np.concatenate([np.zeros(S), daily[: j * 7]])
            total = 0.0
            for _ in range(7):
                m = r * float(buf[-S:] @ w_rev)
                buf = np.append(buf, m)
                total += m
            obs.append(weekly[j])
            exp.append(total)
        return np.asarray(obs), np.asarray(exp)

    def forecast(self, y: np.ndarray, horizons: Sequence[int]) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        H = max(horizons)
        T = len(y)
        rng = np.random.default_rng([config.RANDOM_SEED, T, int(y.sum()) % 100000])
        S = len(self.w)
        w_rev = self.w[::-1]

        r_hist = self.r_history(y)
        obs, exp = self._one_week_ahead_means(y, r_hist)
        inv_k = nb_dispersion(obs[-26:], exp[-26:]) if len(obs) >= 4 else 0.2

        log_r = np.log(np.clip(r_hist[np.isfinite(r_hist)], 1e-3, None))
        step = np.diff(log_r[-13:])
        rw_sd = float(np.clip(np.std(step), 0.03, 0.40)) if len(step) >= 3 else 0.15

        shape, rate = self.r_posterior(y, T - 1)
        r = rng.gamma(shape=shape, scale=1.0 / rate, size=self.n_sims)

        daily = np.repeat(y / 7.0, 7)
        hist = np.concatenate([np.zeros(S), daily])[-S:]
        D = np.empty((self.n_sims, S + 7 * H))
        D[:, :S] = hist
        weekly_draws = np.empty((self.n_sims, H))
        for h in range(H):
            r = np.minimum(r * np.exp(rng.normal(0.0, rw_sd, size=self.n_sims)), self.r_max)
            start = S + 7 * h
            for day in range(start, start + 7):
                D[:, day] = r * (D[:, day - S:day] @ w_rev)
            mu = D[:, start:start + 7].sum(axis=1)
            draws = draw_negbin(rng, mu, inv_k)
            scale = np.divide(draws, mu, out=np.ones_like(mu), where=mu > 1e-9)
            D[:, start:start + 7] *= scale[:, None]  # realised cases drive later transmission
            weekly_draws[:, h] = draws
        idx = np.asarray(horizons) - 1
        return enforce_monotone(np.quantile(weekly_draws[:, idx], Q, axis=0).T)


# ---------------------------------------------------------------------- Richards
def richards_cumulative(t: np.ndarray, K: float, r: float, a: float, ti: float) -> np.ndarray:
    """Richards growth curve for cumulative cases.

    It solves the differential equation

        dC/dt = r * C * (1 - (C/K)^a)

    C grows exponentially at rate r while small, then slows as it approaches
    the final size K. The exponent a sets how sharply growth switches off
    (a = 1 is the ordinary logistic curve); ti is the time of fastest growth.

        C(t) = K * (1 + a * exp(-a * r * (t - ti))) ** (-1/a)
    """
    expo = np.clip(-a * r * (np.asarray(t, dtype=float) - ti), -700.0, 700.0)
    return K * (1.0 + a * np.exp(expo)) ** (-1.0 / a)


class Richards(Forecaster):
    """Richards-curve forecast with parametric-bootstrap uncertainty.

    Fit. Weekly incidence implied by the curve is mu_t = C(t) - C(t-1). The
    four parameters are chosen to minimise sum (sqrt(y_t) - sqrt(mu_t))^2.
    The square root roughly equalises the variance of count data, so the peak
    weeks do not dominate the fit.

    Uncertainty (Chowell, 2017). (1) Estimate the count noise from the fit's
    residuals. (2) Simulate many synthetic outbreaks from the fitted curve with
    that noise. (3) Refit the curve to each one. (4) Project every refitted
    curve forward and add count noise again. The spread of those projections
    reflects both parameter uncertainty and observation noise.

    Known limitation, visible in the backtest: a single S-curve assumes one
    wave that ends. Before the peak the final size K is barely identified, and
    in a long, multi-wave tail the curve declares the outbreak over too early.
    """

    name = "richards"

    def __init__(self, n_boot: int = 80, draws_per_boot: int = 25):
        self.n_boot = n_boot
        self.draws_per_boot = draws_per_boot

    @staticmethod
    def _mu(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
        K, r, a, ti = np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2]), theta[3]
        return np.maximum(richards_cumulative(t, K, r, a, ti) - richards_cumulative(t - 1.0, K, r, a, ti), 1e-9)

    def _fit(self, y: np.ndarray, starts: list[np.ndarray]) -> np.ndarray:
        T = len(y)
        t = np.arange(1, T + 1, dtype=float)
        total = max(y.sum(), 1.0)
        lower = np.array([np.log(total * 1.001), np.log(0.01), np.log(0.02), -10.0])
        upper = np.array([np.log(total * 100.0), np.log(3.0), np.log(10.0), T + 60.0])
        root_y = np.sqrt(y)

        def resid(theta):
            return np.sqrt(self._mu(theta, t)) - root_y

        best, best_cost = None, np.inf
        for s in starts:
            s = np.clip(s, lower + 1e-6, upper - 1e-6)
            try:
                sol = optimize.least_squares(resid, s, bounds=(lower, upper), method="trf", max_nfev=200)
            except Exception:
                continue
            if sol.cost < best_cost:
                best, best_cost = sol.x, sol.cost
        if best is None:
            best = np.clip(starts[0], lower + 1e-6, upper - 1e-6)
        return best

    def _starts(self, y: np.ndarray) -> list[np.ndarray]:
        total = max(y.sum(), 1.0)
        peak = float(np.argmax(y) + 1)
        T = float(len(y))
        return [np.array([np.log(total * m), np.log(r), 0.0, ti])
                for m, r, ti in ((1.2, 0.3, peak), (2.0, 0.3, T), (5.0, 0.2, T + 5), (20.0, 0.15, T + 15),
                                 (1.5, 0.6, peak))]

    def forecast(self, y: np.ndarray, horizons: Sequence[int]) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        T = len(y)
        rng = np.random.default_rng([config.RANDOM_SEED, T, int(y.sum()) % 100000, 7])
        t_hist = np.arange(1, T + 1, dtype=float)
        t_fut = T + np.asarray(horizons, dtype=float)

        theta = self._fit(y, self._starts(y))
        mu_hist = self._mu(theta, t_hist)
        inv_k = nb_dispersion(y, mu_hist)

        sims = []
        for _ in range(self.n_boot):
            y_star = draw_negbin(rng, mu_hist, inv_k)
            theta_b = self._fit(y_star, [theta])
            mu_fut = self._mu(theta_b, t_fut)
            sims.append(draw_negbin(rng, np.tile(mu_fut, (self.draws_per_boot, 1)), inv_k))
        sims = np.concatenate(sims, axis=0)  # (n_boot * draws, H)
        return enforce_monotone(np.quantile(sims, Q, axis=0).T)
