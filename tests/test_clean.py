import json

import numpy as np
import pandas as pd
import pytest

from epicast import clean, config, schema

MALARIA = config.TARGETS["malaria_incidence"]
HIV = config.TARGETS["hiv_new_infections"]


def test_make_monotone_revises_earlier_values_down():
    np.testing.assert_array_equal(clean.make_monotone([1, 3, 2, 5]), [1, 2, 2, 5])
    np.testing.assert_array_equal(clean.make_monotone([4, 4, 9, 7, 7]), [4, 4, 7, 7, 7])


def test_cumulative_to_weekly_recovers_a_constant_rate():
    dates = pd.date_range("2014-01-01", periods=60, freq="D")  # 10 new cases per day
    weekly = clean.cumulative_to_weekly(dates, np.arange(60) * 10.0)
    assert (weekly.index.dayofweek == 6).all()            # week-ending Sundays
    np.testing.assert_allclose(weekly.to_numpy(), 70.0)
    assert (weekly >= 0).all()


def test_cumulative_to_weekly_never_goes_negative_after_a_reclassification():
    dates = pd.to_datetime(["2014-01-05", "2014-01-12", "2014-01-19", "2014-01-26"])
    weekly = clean.cumulative_to_weekly(dates, np.array([10.0, 50.0, 40.0, 60.0]))
    assert (weekly >= 0).all()
    assert weekly.sum() == pytest.approx(50.0)  # final total minus first total is preserved


def test_clean_wdi_mirror_keeps_countries_only():
    csv = b"geo,time,sh_mlr_incd_p3\nnga,2023,299.0\nnga,2024,294.2\nnot_a_country,2024,10\nken,2024,\n"
    out = clean.clean_wdi_mirror(csv, MALARIA)
    assert list(out["location"]) == ["NGA", "NGA"]
    assert out["location_name"].iloc[0] == "Nigeria"
    assert set(out["period"]) == {2023, 2024}
    assert out["value"].iloc[-1] == pytest.approx(294.2)


def test_clean_who_gho_parses_country_rows_and_bounds():
    payload = {"value": [
        {"SpatialDimType": "COUNTRY", "SpatialDim": "ZAF", "TimeDim": 2024, "Dim1": None,
         "NumericValue": 140000, "Low": 96000, "High": 190000},
        {"SpatialDimType": "REGION", "SpatialDim": "AFR", "TimeDim": 2024, "Dim1": None,
         "NumericValue": 9e5, "Low": None, "High": None},
        {"SpatialDimType": "COUNTRY", "SpatialDim": "KEN", "TimeDim": 2024, "Dim1": None,
         "NumericValue": None, "Low": None, "High": None},
    ]}
    out = clean.clean_who_gho(json.dumps(payload).encode(), HIV, retrieved="test")
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["location"], row["period"], row["value"]) == ("ZAF", 2024, 140000)
    assert (row["lower"], row["upper"]) == (96000, 190000)


def _valid_frame():
    return pd.DataFrame({
        "disease": "malaria", "metric": "m", "location": ["NGA", "NGA"], "location_name": "Nigeria",
        "period": [2023, 2024], "freq": "A", "value": [1.0, 2.0], "lower": np.nan, "upper": np.nan,
        "source": "s", "source_version": "v"})


def test_validate_accepts_a_good_table():
    assert len(schema.validate(_valid_frame())) == 2


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.assign(value=[1.0, -2.0]), "negative"),
    (lambda d: d.assign(period=[2024, 2024]), "duplicate"),
    (lambda d: d.assign(location=["nga", "nga"]), "ISO3"),
    (lambda d: d.drop(columns="value"), "missing columns"),
    (lambda d: d.assign(lower=[5.0, 5.0], upper=[1.0, 1.0]), "lower bound"),
])
def test_validate_rejects_bad_tables(mutate, message):
    with pytest.raises(schema.SchemaError, match=message):
        schema.validate(mutate(_valid_frame()))


def test_longest_recent_run_drops_everything_before_a_gap():
    mask = schema.longest_recent_run(np.array([2000, 2001, 2005, 2006, 2007]))
    assert mask.tolist() == [False, False, True, True, True]


def test_series_for_cuts_annual_series_at_the_last_gap():
    df = pd.DataFrame({
        "disease": "malaria", "metric": MALARIA.metric, "location": "NGA", "location_name": "Nigeria",
        "period": ["2000", "2001", "2005", "2006"], "freq": "A", "value": [4.0, 3.0, 2.0, 1.0],
        "lower": np.nan, "upper": np.nan, "source": "s", "source_version": "v"})
    s = clean.series_for(df, MALARIA)["NGA"]
    assert s.index.tolist() == [2005, 2006]
