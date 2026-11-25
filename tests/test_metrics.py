"""The scoring rules are the referee, so they are tested hardest."""
import numpy as np
import pytest

from epicast import metrics
from epicast.config import QUANTILES


def test_pinball_at_median_is_half_absolute_error():
    assert metrics.pinball_loss(10.0, 7.0, 0.5) == pytest.approx(1.5)
    assert metrics.pinball_loss(7.0, 10.0, 0.5) == pytest.approx(1.5)


def test_pinball_is_asymmetric_for_upper_quantiles():
    too_low = metrics.pinball_loss(10.0, 9.0, 0.9)   # outcome above the 90th percentile forecast
    too_high = metrics.pinball_loss(10.0, 11.0, 0.9)
    assert too_low == pytest.approx(0.9)
    assert too_high == pytest.approx(0.1)


def test_wis_matches_twice_mean_pinball_loss():
    rng = np.random.default_rng(0)
    q = np.sort(rng.normal(50, 10, size=(500, len(QUANTILES))), axis=1)
    y = rng.normal(50, 15, size=500)
    np.testing.assert_allclose(metrics.wis(y, q), metrics.wis_from_pinball(y, q), rtol=1e-12)


def test_wis_of_a_point_forecast_is_absolute_error():
    q = np.full((1, len(QUANTILES)), 20.0)
    assert metrics.wis(np.array([26.0]), q)[0] == pytest.approx(6.0)


def test_wis_rewards_sharpness_when_correct():
    y = np.array([100.0])
    narrow = np.linspace(95, 105, len(QUANTILES))[None, :]
    wide = np.linspace(50, 150, len(QUANTILES))[None, :]
    assert metrics.wis(y, narrow)[0] < metrics.wis(y, wide)[0]


def test_interval_score_penalises_misses_in_proportion_to_distance():
    inside = metrics.interval_score(np.array([5.0]), np.array([0.0]), np.array([10.0]), 0.1)[0]
    outside = metrics.interval_score(np.array([12.0]), np.array([0.0]), np.array([10.0]), 0.1)[0]
    assert inside == pytest.approx(10.0)
    assert outside == pytest.approx(10.0 + (2 / 0.1) * 2.0)


def test_coverage_indicator():
    q = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=float)  # 50% interval is [4, 6]
    assert metrics.covered(np.array([5.0]), q, 0.50)[0] == 1.0
    assert metrics.covered(np.array([6.5]), q, 0.50)[0] == 0.0
    assert metrics.covered(np.array([6.5]), q, 0.95)[0] == 1.0


def test_gaussian_quantiles_are_calibrated_and_proper():
    """A forecaster who reports the true distribution should cover at the nominal rate,
    and should score better than one who reports intervals half as wide."""
    from scipy import stats

    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, size=20000)
    z = stats.norm.ppf(QUANTILES)
    honest = np.tile(z, (len(y), 1))
    overconfident = honest * 0.5
    assert metrics.covered(y, honest, 0.95).mean() == pytest.approx(0.95, abs=0.01)
    assert metrics.covered(y, honest, 0.50).mean() == pytest.approx(0.50, abs=0.015)
    assert metrics.wis(y, honest).mean() < metrics.wis(y, overconfident).mean()


def test_mase_scale_is_mean_absolute_change():
    assert metrics.mase_scale(np.array([1.0, 3.0, 2.0, 6.0])) == pytest.approx((2 + 1 + 4) / 3)
    assert np.isnan(metrics.mase_scale(np.array([5.0, 5.0, 5.0])))
