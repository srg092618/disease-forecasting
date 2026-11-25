"""Tests for the outbreak registry, the map geometry helpers and the outlook maths."""
import io
import json

import numpy as np
import pandas as pd
import pytest

from epicast import clean, config, geo, outlook, registry

BVD = config.TARGETS["bvd_weekly_cases"]


# ------------------------------------------------------------------ registry
def test_shipped_registry_is_valid_and_fully_sourced():
    sources = registry.load_sources()
    outbreaks = registry.load_outbreaks(sources=sources)
    assert len(outbreaks) >= 20
    assert set(outbreaks["virus"]) == {"EBOV", "SUDV", "BDBV"}
    for ids in outbreaks["source_ids"]:
        assert all(i in set(sources["source_id"]) for i in ids.split(";"))
    assert (outbreaks["start_year"] >= config.REGISTRY_START_YEAR).all()


def test_registry_holds_the_outbreaks_the_project_tracks():
    o = registry.load_outbreaks().set_index("outbreak_id")
    assert (o.loc["uga-2022-mubende", "cases"], o.loc["uga-2022-mubende", "deaths"]) == (164, 77)
    assert (o.loc["uga-2025-kampala", "cases"], o.loc["uga-2025-kampala", "deaths"]) == (14, 4)
    assert (o.loc["cod-2025-kasai", "cases"], o.loc["cod-2025-kasai", "deaths"]) == (64, 45)
    assert o.loc["cod-2026-ituri", "status"] == "ongoing" and o.loc["cod-2026-ituri", "virus"] == "BDBV"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


SOURCES_OK = "source_id,title,publisher,url,accessed,notes\ns1,A report,WHO,https://example.org/r,2026-11-17,\n"
HEADER = "outbreak_id,virus,country,admin1,place,place_precision,lat,lon,start,end,status,cases,deaths,case_note,source_ids,notes\n"


def test_registry_rejects_an_unknown_source(tmp_path):
    src = registry.load_sources(_write(tmp_path, "s.csv", SOURCES_OK))
    bad = _write(tmp_path, "o.csv", HEADER + "x,EBOV,COD,,Town,town,0,20,2020,2020,ended,10,5,,nope,\n")
    with pytest.raises(registry.RegistryError, match="unknown source"):
        registry.load_outbreaks(bad, sources=src)


def test_registry_rejects_more_deaths_than_cases(tmp_path):
    src = registry.load_sources(_write(tmp_path, "s.csv", SOURCES_OK))
    bad = _write(tmp_path, "o.csv", HEADER + "x,EBOV,COD,,Town,town,0,20,2020,2020,ended,5,10,,s1,\n")
    with pytest.raises(registry.RegistryError, match="deaths exceed"):
        registry.load_outbreaks(bad, sources=src)


def test_registry_rejects_a_source_without_a_link(tmp_path):
    bad = _write(tmp_path, "s.csv", "source_id,title,publisher,url,accessed,notes\ns1,A report,WHO,not-a-link,2026-11-17,\n")
    with pytest.raises(registry.RegistryError, match="http"):
        registry.load_sources(bad)


def test_source_links_follow_the_order_of_ids():
    sources = registry.load_sources()
    links = registry.source_links("who_don433;cdc_outbreak_history", sources)
    assert [l["publisher"] for l in links] == ["World Health Organization", "US Centers for Disease Control and Prevention"]
    assert all(l["url"].startswith("https://") for l in links)


# ----------------------------------------------------------------------- geo
def test_haversine_matches_known_distances():
    assert geo.haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.1)        # one degree at the equator
    assert geo.haversine_km(0.316, 32.582, -1.283, 36.817) == pytest.approx(503, abs=6)  # Kampala to Nairobi
    assert geo.haversine_km(5, 5, 5, 5) == 0


def test_distance_to_country_is_zero_inside_and_positive_outside():
    from shapely.geometry import box

    square = box(10, -5, 20, 5)                       # a 10 x 10 degree "country"
    assert geo.distance_to_country_km(0, 15, square) == 0.0
    assert geo.distance_to_country_km(0, 21, square) == pytest.approx(111.19, abs=0.2)


def test_rings_are_rounded_outer_boundaries():
    from shapely.geometry import box

    feature = geo._feature(box(1.23456, 2.34567, 3.0, 4.0), tolerance=0.0)
    assert feature["bbox"] == [1.23, 2.35, 3.0, 4.0]
    assert all(len(pt) == 2 for pt in feature["rings"][0])
    assert 1.23 <= feature["centroid"][0] <= 3.0


# ------------------------------------------------------------------- outlook
def test_recurrence_formula():
    r = outlook.recurrence(n_outbreaks=12, years_observed=20.0)
    assert r["rate_per_year"] == pytest.approx(12.5 / 20)
    assert r["p_next_12_months"] == pytest.approx(1 - (20 / 21) ** 12.5, abs=1e-3)
    assert r["rate_lo"] < r["rate_per_year"] < r["rate_hi"]


def test_recurrence_probability_rises_with_more_outbreaks_and_is_never_certain():
    few, many = outlook.recurrence(1, 20.0), outlook.recurrence(40, 20.0)
    assert 0 < few["p_next_12_months"] < many["p_next_12_months"] < 1


def test_recurrence_matches_a_simulation():
    """Draw rates from the posterior, then outbreaks from a Poisson: the closed form should agree."""
    rng = np.random.default_rng(0)
    lam = rng.gamma(shape=5.5, scale=1 / 19.9, size=400_000)
    simulated = np.mean(rng.poisson(lam) >= 1)
    assert outlook.recurrence(5, 19.9)["p_next_12_months"] == pytest.approx(simulated, abs=0.003)


def test_annual_outlook_direction_uses_the_80_percent_interval():
    cols = ["target", "model", "location", "horizon", "period"] + [f"q{q:g}" for q in config.QUANTILES]
    rows = [["t", "m", "AAA", 5, 2029, 10, 20, 30, 40, 50, 60, 70, 80, 90],     # 80% interval 30 to 70
            ["t", "m", "BBB", 5, 2029, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            ["t", "m", "CCC", 5, 2029, 10, 20, 30, 40, 50, 60, 70, 80, 90]]
    fc = pd.DataFrame(rows, columns=cols)
    out = outlook.annual_outlook(fc, {"AAA": 100.0, "BBB": 20.0, "CCC": 50.0}, "t", "m")
    assert out["AAA"]["direction"] == "likely to fall"
    assert out["BBB"]["direction"] == "likely to rise"
    assert out["CCC"]["direction"] == "no clear direction"


def test_best_model_is_the_lowest_calibrated_relative_error():
    m = pd.DataFrame({"target": ["t"] * 4, "stage": ["raw", "calibrated", "calibrated", "calibrated"],
                      "model": ["x", "naive", "gbm", "ets"], "rel_wis_log": [0.1, 1.0, 0.6, 0.9]})
    assert outlook.best_model(m, "t") == "gbm"


# ------------------------------------------------------- Bundibugyo line list
def _linelist(rows):
    cols = ["Case_status", "Date_confirmation", "Location_Admin0", "Location Admin1", "Health zone", "Outcome"]
    buf = io.StringIO()
    pd.DataFrame(rows, columns=cols).to_csv(buf, index=False)
    return buf.getvalue().encode()


DRC = "Democratic Republic of the Congo"
ROWS = (
    [["confirmed", "2026-06-01", DRC, "Ituri", "Bunia", "Death"]] * 3 +          # Mon, week ending 7 Jun
    [["confirmed", "2026-06-10", DRC, "Nord-Kivu", "Beni", ""]] * 2 +            # week ending 14 Jun
    [["confirmed", "2026-06-24", DRC, "Ituri", "Bunia", ""]] * 4 +               # week ending 28 Jun
    [["suspected", "2026-06-24", DRC, "Ituri", "Bunia", ""]] * 9 +               # not confirmed: ignored
    [["confirmed", "2026-06-30", DRC, "Ituri", "Nizi", ""]] * 5 +                # Tue: partial final week
    [["confirmed", "2026-06-12", "Uganda", "Kampala", "", ""]]                   # outside DRC
)


def test_bvd_weekly_series_counts_by_confirmation_week_and_drops_the_partial_week():
    out = clean.clean_bvd_linelist(_linelist(ROWS), BVD)
    national = out[out["location"] == "COD"].set_index("period")["value"]
    assert national.index.tolist() == ["2026-06-07", "2026-06-14", "2026-06-21", "2026-06-28"]
    assert national.tolist() == [3.0, 2.0, 0.0, 4.0]          # the empty week is kept as zero
    ituri = out[out["location"] == "COD-IT"].set_index("period")["value"]
    assert ituri.tolist() == [3.0, 0.0, 0.0, 4.0]
    assert set(out["location"]) == set(config.BVD_SERIES)


def test_bvd_summary_totals():
    s = clean.bvd_summary(_linelist(ROWS))
    assert s["confirmed"] == 15 and s["deaths_among_confirmed"] == 3
    assert s["as_of"] == "2026-06-30"
    assert {c["name"]: c["confirmed"] for c in s["countries"]} == {DRC: 14, "Uganda": 1}
    assert s["provinces"][0] == {"name": "Ituri", "confirmed": 12, "deaths": 3}
    assert json.dumps(s)  # serialisable


def test_subnational_location_codes_pass_the_schema():
    out = clean.clean_bvd_linelist(_linelist(ROWS), BVD)
    assert out["location"].str.fullmatch(r"[A-Z]{3}(-[A-Z]{2})?").all()
