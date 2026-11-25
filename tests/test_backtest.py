"""Tests for the two properties a backtest must have: no look-ahead, and honest scoring."""
import numpy as np
import pandas as pd
import pytest

from epicast import backtest as bt
from epicast import calibrate, config
from epicast.config import QUANTILES

MALARIA = config.TARGETS["malaria_incidence"]
EBOLA = config.TARGETS["ebola_weekly_cases"]


def _panel(seed=0, n_series=25, years=range(2000, 2025)):
    rng = np.random.default_rng(seed)
    years = list(years)
    return {f"S{i:02d}": pd.Series(
        np.exp(rng.uniform(3, 6) + np.cumsum(rng.normal(-0.02, 0.08, len(years)))), index=years)
        for i in range(n_series)}


def test_annual_forecasts_do_not_depend_on_the_future():
    """Scrambling every value after the origin must leave the origin's forecasts unchanged."""
    series = _panel()
    origin = 2016
    before = pd.DataFrame(bt.annual_origin_forecasts(MALARIA, series, origin))
    tampered = {k: s.copy() for k, s in series.items()}
    for s in tampered.values():
        s[s.index > origin] = s[s.index > origin] * 37.0 + 5.0
    after = pd.DataFrame(bt.annual_origin_forecasts(MALARIA, tampered, origin))
    assert len(before) > 0
    np.testing.assert_allclose(before[bt.Q_COLS].to_numpy(), after[bt.Q_COLS].to_numpy(), rtol=1e-12)
    assert not np.allclose(before["y"].to_numpy(), after["y"].to_numpy())  # only the answers changed


def test_annual_forecasts_cover_every_model_and_horizon():
    rows = pd.DataFrame(bt.annual_origin_forecasts(MALARIA, _panel(), 2018))
    assert set(rows["model"]) == {"naive", "drift", "ets", "arima", "gbm", "ensemble"}
    assert set(rows["horizon"]) == set(MALARIA.horizons)
    assert (rows["period"] == rows["origin"] + rows["horizon"]).all()


def test_inactive_series_are_not_forecast():
    series = _panel(n_series=6)
    first = next(iter(series))
    series[first] = series[first] * 0.0  # disease eliminated: below min_level
    rows = pd.DataFrame(bt.annual_origin_forecasts(MALARIA, series, 2018))
    assert first not in set(rows["location"])


def test_outbreak_origins_require_history_and_activity():
    y = np.concatenate([np.zeros(12), [1, 2, 5, 9, 20, 40], np.zeros(8)])
    s = pd.Series(y, index=pd.date_range("2014-01-05", periods=len(y), freq="W-SUN"))
    origins = bt.outbreak_origins(EBOLA, s)
    assert min(origins) >= EBOLA.min_train - 1
    assert all(y[max(0, i - 3): i + 1].sum() >= EBOLA.min_level for i in origins)
    assert len(y) - 1 not in origins  # the last week has no outcome to score against


def _forecast_table(n=400, spread=1.0, seed=0):
    """Synthetic backtest table: truth ~ N(0, 1) on the log scale around a known median."""
    rng = np.random.default_rng(seed)
    from scipy import stats
    zq = 5.0 + spread * stats.norm.ppf(QUANTILES)[None, :] * np.ones((n, 1))
    z = 5.0 + rng.normal(0, 1.0, n)
    df = pd.DataFrame(np.expm1(zq), columns=bt.Q_COLS)
    df.insert(0, "y", np.expm1(z))
    df.insert(0, "mase_scale", 1.0)
    df.insert(0, "vol", 1.0)
    df.insert(0, "period", [str(2000 + i // 20 + 1) for i in range(n)])
    df.insert(0, "horizon", 1)
    df.insert(0, "origin", [str(2000 + i // 20) for i in range(n)])
    df.insert(0, "location", [f"L{i % 20}" for i in range(n)])
    df.insert(0, "model", "demo")
    df.insert(0, "target", "t")
    return df


def test_scores_have_expected_columns_and_drop_unknown_outcomes():
    df = _forecast_table(40)
    df.loc[0, "y"] = np.nan
    scored = bt.score(df)
    assert len(scored) == 39
    assert {"wis", "wis_log", "ae", "mase", "cov50", "cov95"} <= set(scored.columns)


def test_calibration_repairs_overconfident_intervals():
    raw = _forecast_table(n=1200, spread=0.5)        # intervals half as wide as they should be
    cal = calibrate.calibrate_backtest(raw)
    late = raw["origin"].astype(int) >= 2010          # after enough history has built up
    cov_raw = bt.score(raw[late])["cov95"].mean()
    cov_cal = bt.score(cal[late])["cov95"].mean()
    assert cov_raw < 0.75
    assert cov_cal == pytest.approx(0.95, abs=0.04)
    np.testing.assert_allclose(raw["q0.5"], cal["q0.5"])  # the median is never moved


def test_calibration_uses_only_outcomes_known_at_the_origin():
    raw = _forecast_table(n=600, spread=0.5)
    cal = calibrate.calibrate_backtest(raw)
    tampered = raw.copy()
    future = tampered["period"].astype(int) > 2015
    tampered.loc[future, "y"] = tampered.loc[future, "y"] * 1000.0
    cal2 = calibrate.calibrate_backtest(tampered)
    at_origin = (raw["origin"].astype(int) == 2015).to_numpy()
    np.testing.assert_allclose(cal.loc[at_origin, bt.Q_COLS], cal2.loc[at_origin, bt.Q_COLS])


def test_first_origins_are_left_uncalibrated():
    raw = _forecast_table(n=600, spread=0.5)
    cal = calibrate.calibrate_backtest(raw)
    first = (raw["origin"] == "2000").to_numpy()
    np.testing.assert_allclose(raw.loc[first, bt.Q_COLS], cal.loc[first, bt.Q_COLS])


def test_published_margins_never_shrink_with_horizon():
    """Even when a later horizon happens to have smaller past misses, the published margin holds."""
    a = _forecast_table(n=400, spread=0.5, seed=1)                  # horizon 1: intervals far too narrow
    b = _forecast_table(n=400, spread=2.0, seed=2).assign(horizon=2)  # horizon 2: intervals far too wide
    margins = calibrate.final_margins(pd.concat([a, b], ignore_index=True))
    m = margins[margins["level"] == 0.95].set_index("horizon")["margin_per_vol"]
    assert m.loc[1] > 0 and m.loc[2] >= m.loc[1]


def test_ongoing_outbreaks_get_forecasts_beyond_the_data():
    rng = np.random.default_rng(5)
    y = np.round(200 + 40 * rng.standard_normal(20)).clip(1)
    s = pd.Series(y, index=pd.date_range("2026-06-07", periods=20, freq="W-SUN"))
    target = config.TARGETS["bvd_weekly_cases"]
    fc = bt.final_outbreak_forecasts(target, {"COD": s})
    assert set(fc["horizon"]) == {1, 2, 3, 4} and fc["y"].isna().all()
    assert sorted(fc["period"].unique())[0] == "2026-10-25"          # the week after the last one
    assert {"naive", "renewal", "richards", "ets", "ensemble"} == set(fc["model"])


def test_conformal_margin_is_zero_without_enough_history_and_signed_otherwise():
    assert calibrate.conformal_margin(np.array([0.3] * 5), 0.95) == 0.0
    inside = np.full(200, -0.2)     # outcomes always well inside: intervals can shrink
    outside = np.full(200, 0.3)     # outcomes always outside by 0.3: intervals must grow
    assert calibrate.conformal_margin(inside, 0.95) == pytest.approx(-0.2)
    assert calibrate.conformal_margin(outside, 0.95) == pytest.approx(0.3)


def test_volatile_series_are_widened_more_than_stable_ones():
    zq = np.tile(np.linspace(4.0, 6.0, len(QUANTILES)), (2, 1))
    out = calibrate._apply(zq, np.array([0.05, 0.50]), {level: 1.0 for level in config.INTERVALS})
    widen = out[:, -1] - zq[:, -1]
    assert widen[0] == pytest.approx(0.05) and widen[1] == pytest.approx(0.50)


def test_log_volatility_is_the_sd_of_log_changes():
    y = np.expm1(np.array([1.0, 1.2, 1.1, 1.5]))
    assert bt.log_volatility(y) == pytest.approx(np.std([0.2, -0.1, 0.4]))


def test_summarise_reports_skill_relative_to_naive():
    a = _forecast_table(200, spread=1.0, seed=1).assign(model="naive")
    b = _forecast_table(200, spread=1.0, seed=1).assign(model="better")
    b[bt.Q_COLS] = np.expm1(np.log1p(b[bt.Q_COLS].to_numpy()) * 0 + np.log1p(b[["y"]].to_numpy()))  # perfect
    out = bt.summarise(bt.score(pd.concat([a, b])), ["target", "model"]).set_index("model")
    assert out.loc["naive", "rel_wis_log"] == pytest.approx(1.0)
    assert out.loc["better", "rel_wis_log"] == pytest.approx(0.0, abs=1e-9)
