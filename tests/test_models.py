import numpy as np
import pytest
from scipy import stats

from epicast import config
from epicast.config import MEDIAN_INDEX, QUANTILES
from epicast.models import build_models
from epicast.models.base import enforce_monotone, from_z, to_z
from epicast.models.outbreak import (Renewal, Richards, discretised_serial_interval, nb_dispersion,
                                     richards_cumulative)
from epicast.models.pooled import N_LAGS, GlobalGBM, build_training_rows
from epicast.models.statistical import ARIMA, ETS, Drift, Naive

H5 = [1, 2, 3, 4, 5]


def _well_formed(fc, n_h):
    assert fc.shape == (n_h, len(QUANTILES))
    assert np.isfinite(fc).all() and (fc >= 0).all()
    assert (np.diff(fc, axis=1) >= -1e-9).all(), "quantiles must not cross"


def test_log_transform_round_trip():
    y = np.array([0.0, 0.5, 10.0, 1e5])
    np.testing.assert_allclose(from_z(to_z(y)), y, rtol=1e-12, atol=1e-12)


def test_enforce_monotone_sorts_each_row():
    np.testing.assert_array_equal(enforce_monotone(np.array([[3.0, 1.0, 2.0]])), [[1.0, 2.0, 3.0]])


def test_naive_median_is_last_value_and_spread_grows_with_root_h():
    rng = np.random.default_rng(0)
    y = np.exp(np.cumsum(rng.normal(0, 0.1, 30)) + 5)
    fc = Naive().forecast(y, H5)
    _well_formed(fc, 5)
    np.testing.assert_allclose(fc[:, MEDIAN_INDEX], y[-1], rtol=1e-9)
    z = to_z(fc)
    width = z[:, -1] - z[:, 0]
    np.testing.assert_allclose(width / width[0], np.sqrt(H5), rtol=1e-6)


def test_drift_continues_an_exact_log_linear_trend():
    z = 6.0 - 0.05 * np.arange(20)                 # falls by 0.05 log units a year, no noise
    fc = Drift().forecast(from_z(z), H5)
    np.testing.assert_allclose(to_z(fc[:, MEDIAN_INDEX]), z[-1] - 0.05 * np.array(H5), atol=1e-6)


def test_drift_interval_matches_textbook_formula():
    rng = np.random.default_rng(3)
    z = np.cumsum(rng.normal(0.02, 0.1, 25)) + 4
    d = np.diff(z)
    n = len(d)
    sigma = np.sqrt(np.sum((d - d.mean()) ** 2) / (n - 1))
    h = 3
    expected_hi = z[-1] + h * d.mean() + stats.t.ppf(0.975, n - 1) * sigma * np.sqrt(h * (1 + h / n))
    fc = Drift().forecast(from_z(z), [h])
    assert to_z(fc[0, -1]) == pytest.approx(expected_hi, rel=1e-9)


@pytest.mark.parametrize("model_cls", [Naive, Drift, ETS, ARIMA])
def test_statistical_models_return_well_formed_quantiles(model_cls):
    rng = np.random.default_rng(1)
    y = np.exp(np.cumsum(rng.normal(-0.03, 0.08, 24)) + 5)
    _well_formed(model_cls().forecast(y, H5), 5)


@pytest.mark.parametrize("model_cls", [Naive, Drift, ETS, ARIMA])
def test_statistical_models_survive_a_constant_series(model_cls):
    _well_formed(model_cls().forecast(np.full(15, 100.0), H5), 5)


def test_training_rows_never_look_past_the_end_of_a_series():
    panel = {"A": np.arange(1.0, 13.0)}            # 12 points
    X, y = build_training_rows(panel, [1, 2, 3], min_level=0.0)
    T = 12
    expected = sum(1 for t in range(N_LAGS, T - 1) for h in (1, 2, 3) if t + h < T)
    assert len(y) == expected and X.shape == (expected, 7)
    assert X[:, -1].max() == 3                      # horizon is the last feature


def test_gbm_learns_a_shared_trend_across_series():
    rng = np.random.default_rng(2)
    panel = {f"S{i}": np.exp(rng.uniform(3, 8) - 0.06 * np.arange(25) + rng.normal(0, 0.01, 25))
             for i in range(60)}
    model = GlobalGBM(min_level=0.0)
    model.fit_panel(panel, H5)
    y = panel["S0"]
    fc = model.forecast(y, H5)
    _well_formed(fc, 5)
    implied = (to_z(fc[:, MEDIAN_INDEX]) - to_z(y[-1])) / np.array(H5)
    np.testing.assert_allclose(implied, -0.06, atol=0.02)   # about 6% decline per year


def test_gbm_falls_back_when_there_is_too_little_data():
    model = GlobalGBM()
    model.fit_panel({"A": np.arange(1.0, 9.0)}, H5)
    assert model.models_ is None
    _well_formed(model.forecast(np.arange(1.0, 9.0), H5), 5)


def test_serial_interval_is_a_distribution_with_the_published_mean():
    w = discretised_serial_interval(15.3, 9.3)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    assert np.sum(w * np.arange(1, len(w) + 1)) == pytest.approx(15.3, abs=0.4)


def test_richards_curve_solves_its_differential_equation():
    K, r, a, ti = 5000.0, 0.4, 0.7, 12.0
    t = np.linspace(2, 30, 50)
    eps = 1e-5
    dC = (richards_cumulative(t + eps, K, r, a, ti) - richards_cumulative(t - eps, K, r, a, ti)) / (2 * eps)
    C = richards_cumulative(t, K, r, a, ti)
    np.testing.assert_allclose(dC, r * C * (1 - (C / K) ** a), rtol=1e-5)
    assert richards_cumulative(np.array([1e6]), K, r, a, ti)[0] == pytest.approx(K)


def test_nb_dispersion_recovers_a_known_value():
    rng = np.random.default_rng(4)
    mu = np.full(4000, 80.0)
    inv_k = 0.25
    y = rng.poisson(rng.gamma(shape=1 / inv_k, scale=mu * inv_k))
    assert nb_dispersion(y, mu) == pytest.approx(inv_k, abs=0.04)


def test_renewal_estimates_r_near_one_for_a_flat_epidemic():
    y = np.full(30, 140.0)
    model = Renewal(n_sims=800)
    assert model.r_history(y)[-1] == pytest.approx(1.0, abs=0.05)
    fc = model.forecast(y, [1, 2, 3, 4])
    _well_formed(fc, 4)
    assert fc[0, MEDIAN_INDEX] == pytest.approx(140.0, rel=0.15)


def test_renewal_sees_growth_and_decline():
    model = Renewal()
    growing = 20 * 1.3 ** np.arange(14)
    shrinking = growing[::-1]
    assert model.r_history(growing)[-1] > 1.3
    assert model.r_history(shrinking)[-1] < 0.8


def test_richards_recovers_the_final_size_after_the_peak():
    t = np.arange(0, 31, dtype=float)
    cum = richards_cumulative(t, 4000.0, 0.45, 1.0, 12.0)
    y = np.diff(cum)                                # 30 noise-free weeks, peak near week 12
    model = Richards(n_boot=10, draws_per_boot=10)
    theta = model._fit(y, model._starts(y))
    assert np.exp(theta[0]) == pytest.approx(4000.0, rel=0.05)
    _well_formed(model.forecast(y, [1, 2, 3, 4]), 4)


@pytest.mark.parametrize("key", list(config.TARGETS))
def test_registry_builds_models_for_every_target(key):
    models = build_models(config.TARGETS[key])
    assert "naive" in models and len(models) >= 4
