"""Turn raw downloads into the canonical tidy table (see schema.py).

Each ``clean_*`` function is pure: bytes in, DataFrame out, no file access.
That makes them easy to unit-test with a few hand-written rows.
"""
from __future__ import annotations

import io
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, ingest, schema

log = logging.getLogger(__name__)

EBOLA_TOTAL_INDICATOR = "Cumulative number of confirmed, probable and suspected Ebola cases"
EBOLA_LAST_WEEK = "2015-12-27"  # reports are dense until here; one stray point follows in Mar 2016


@lru_cache(maxsize=1)
def countries() -> pd.DataFrame:
    """Reference table: ISO3 code, mirror id, English name, region, income group."""
    return pd.read_csv(config.REFERENCE_DIR / "countries.csv", keep_default_na=False)


# -------------------------------------------------------------- annual: mirror
def clean_wdi_mirror(content: bytes, target: config.Target) -> pd.DataFrame:
    """Parse one indicator file from the World Bank mirror.

    The file has three columns: ``geo`` (lower-case id), ``time`` (year) and one
    value column named after the indicator. Rows whose ``geo`` is not a country
    with an ISO3 code (regional aggregates and the like) are dropped.
    """
    raw = pd.read_csv(io.BytesIO(content))
    value_col = [c for c in raw.columns if c not in ("geo", "time")]
    if {"geo", "time"} - set(raw.columns) or len(value_col) != 1:
        raise schema.SchemaError(f"unexpected mirror columns: {list(raw.columns)}")
    ref = countries().set_index("geo")
    raw = raw[raw["geo"].isin(ref.index)].dropna(subset=value_col)
    out = pd.DataFrame({
        "disease": target.disease,
        "metric": target.metric,
        "location": ref.loc[raw["geo"], "iso3"].to_numpy(),
        "location_name": ref.loc[raw["geo"], "name"].to_numpy(),
        "period": raw["time"].astype(int).to_numpy(),
        "freq": "A",
        "value": raw[value_col[0]].astype(float).to_numpy(),
        "lower": np.nan,
        "upper": np.nan,
        "source": "World Bank WDI (open-numbers mirror)",
        "source_version": config.WDI_MIRROR_COMMIT[:10],
    })
    return schema.validate(out)


# ------------------------------------------------------------- annual: WHO GHO
def clean_who_gho(content: bytes, target: config.Target, retrieved: str = "") -> pd.DataFrame:
    """Parse one indicator from the WHO GHO OData API.

    Keeps country-level records only, and only the all-population total (records
    where the sex/age dimension ``Dim1`` is empty or a "both sexes"/"total" code).
    """
    records = json.loads(content)["value"]
    raw = pd.DataFrame.from_records(records)
    needed = {"SpatialDimType", "SpatialDim", "TimeDim", "NumericValue"}
    if needed - set(raw.columns):
        raise schema.SchemaError(f"WHO GHO response is missing {sorted(needed - set(raw.columns))}")
    raw = raw[(raw["SpatialDimType"] == "COUNTRY") & raw["NumericValue"].notna()]
    if "Dim1" in raw.columns:
        total = raw["Dim1"].isna() | raw["Dim1"].isin(["SEX_BTSX", "BTSX", "TOTAL", "AGEGROUP_ALLAGES"])
        raw = raw[total]
    ref = countries().set_index("iso3")
    raw = raw[raw["SpatialDim"].isin(ref.index)]
    out = pd.DataFrame({
        "disease": target.disease,
        "metric": target.metric,
        "location": raw["SpatialDim"].to_numpy(),
        "location_name": ref.loc[raw["SpatialDim"], "name"].to_numpy(),
        "period": raw["TimeDim"].astype(int).to_numpy(),
        "freq": "A",
        "value": raw["NumericValue"].astype(float).to_numpy(),
        "lower": pd.to_numeric(raw.get("Low"), errors="coerce").to_numpy(),
        "upper": pd.to_numeric(raw.get("High"), errors="coerce").to_numpy(),
        "source": "WHO Global Health Observatory",
        "source_version": retrieved,
    })
    return schema.validate(out)


# ------------------------------------------------------------------- Ebola
def make_monotone(cumulative: np.ndarray) -> np.ndarray:
    """Force a cumulative count to be non-decreasing by revising *earlier* values down.

    Reported cumulative totals sometimes fall, because suspected cases are
    later reclassified as not Ebola. Walking backwards and taking the running
    minimum trusts the later, corrected totals and keeps the final count intact.
    """
    return np.minimum.accumulate(np.asarray(cumulative, dtype=float)[::-1])[::-1]


def cumulative_to_weekly(dates: pd.DatetimeIndex, cumulative: np.ndarray,
                         last_week: str | None = None) -> pd.Series:
    """Convert irregularly dated cumulative counts to weekly new cases.

    Steps: (1) make the curve monotone, (2) linearly interpolate it onto a grid
    of week-ending Sundays, (3) difference. The first grid point has no
    previous week, so the series starts at the second Sunday.
    """
    order = np.argsort(dates.values)
    dates = dates[order]
    cum = make_monotone(np.asarray(cumulative, dtype=float)[order])
    first_sunday = dates[0] + pd.offsets.Week(weekday=6, n=0 if dates[0].weekday() == 6 else 1)
    end = min(dates[-1], pd.Timestamp(last_week)) if last_week else dates[-1]
    grid = pd.date_range(first_sunday, end, freq="W-SUN")
    x = (dates - dates[0]).days.to_numpy(dtype=float)
    gx = (grid - dates[0]).days.to_numpy(dtype=float)
    weekly_cum = np.interp(gx, x, cum)
    weekly_new = np.diff(weekly_cum)
    return pd.Series(np.round(weekly_new, 1), index=grid[1:], name="weekly_cases")


def clean_ebola(early: bytes, who: bytes, target: config.Target) -> pd.DataFrame:
    """Splice the two transcriptions and return weekly new cases per country.

    Before 29 Aug 2014 only the early transcription exists; from that date the
    WHO situation-report series is used. On the dates where both exist they
    agree to within a few percent, which is what justifies the splice.
    """
    e = pd.read_csv(io.BytesIO(early))
    e["Date"] = pd.to_datetime(e["Date"], format="%m/%d/%Y")
    w = pd.read_csv(io.BytesIO(who))
    w = w[w["Indicator"] == EBOLA_TOTAL_INDICATOR].copy()
    w["Date"] = pd.to_datetime(w["Date"])
    splice = pd.Timestamp(config.EBOLA_SPLICE_DATE)

    frames = []
    for iso3, name in config.EBOLA_COUNTRIES.items():
        early_col = "Cases_" + name.replace(" ", "")
        a = e.loc[e["Date"] < splice, ["Date", early_col]].dropna().rename(columns={early_col: "cum"})
        b = (w.loc[(w["Country"] == name) & (w["Date"] >= splice)]
             .groupby("Date", as_index=False)["value"].max().rename(columns={"value": "cum"}))
        both = pd.concat([a, b], ignore_index=True).sort_values("Date")
        weekly = cumulative_to_weekly(pd.DatetimeIndex(both["Date"]), both["cum"].to_numpy(), EBOLA_LAST_WEEK)
        frames.append(pd.DataFrame({
            "disease": target.disease,
            "metric": target.metric,
            "location": iso3,
            "location_name": name,
            "period": weekly.index.strftime("%Y-%m-%d"),
            "freq": "W",
            "value": weekly.to_numpy(),
            "lower": np.nan,
            "upper": np.nan,
            "source": "WHO / Ministry of Health situation reports (Rivers et al.; HDX)",
            "source_version": "cmrivers@ddcbf44, hdx-mirror@c23b14a",
        }))
    return schema.validate(pd.concat(frames, ignore_index=True))


# ------------------------------------------------------- Bundibugyo 2026 line list
def _bvd_confirmed(content: bytes) -> pd.DataFrame:
    """Confirmed cases with a confirmation date, one row per case."""
    raw = pd.read_csv(io.BytesIO(content), low_memory=False)
    needed = {"Case_status", "Date_confirmation", "Location_Admin0", "Location Admin1", "Health zone", "Outcome"}
    if needed - set(raw.columns):
        raise schema.SchemaError(f"line list is missing columns: {sorted(needed - set(raw.columns))}")
    df = raw[raw["Case_status"].astype(str).str.strip().str.lower() == "confirmed"].copy()
    df["date"] = pd.to_datetime(df["Date_confirmation"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        raise schema.SchemaError("line list has no confirmed cases with a confirmation date")
    df["week"] = df["date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    return df


def clean_bvd_linelist(content: bytes, target: config.Target, version: str = "") -> pd.DataFrame:
    """Weekly confirmed Bundibugyo cases for the DRC total and its main provinces.

    A case is counted in the week (ending Sunday) of its laboratory confirmation,
    because onset dates are almost never published. The newest week is dropped
    unless the data run to its Sunday: a partial week would look like a fall in
    cases. Weeks with no confirmed case are kept as zeros.
    """
    df = _bvd_confirmed(content)
    as_of = df["date"].max()
    drc = df[df["Location_Admin0"] == "Democratic Republic of the Congo"]
    last_full = as_of.to_period("W-SUN").end_time.normalize()
    if last_full > as_of.normalize():
        last_full -= pd.Timedelta(weeks=1)
    weeks = pd.date_range(drc["week"].min(), last_full, freq="W-SUN")
    frames = []
    for code, (label, admin1) in config.BVD_SERIES.items():
        sub = drc if admin1 is None else drc[drc["Location Admin1"] == admin1]
        counts = sub.groupby("week").size().reindex(weeks, fill_value=0)
        frames.append(pd.DataFrame({
            "disease": target.disease, "metric": target.metric, "location": code,
            "location_name": label, "period": weeks.strftime("%Y-%m-%d"), "freq": "W",
            "value": counts.to_numpy(float), "lower": np.nan, "upper": np.nan,
            "source": "Global.health Ebola BVD 2026 line list",
            "source_version": version or f"data to {as_of:%Y-%m-%d}",
        }))
    return schema.validate(pd.concat(frames, ignore_index=True))


def bvd_summary(content: bytes) -> dict:
    """Totals from the line list by country, province and health zone, for the map and panel."""
    df = _bvd_confirmed(content)
    died = df["Outcome"].astype(str).str.strip().str.lower() == "death"

    def table(col: str, frame: pd.DataFrame, top: int | None = None) -> list[dict]:
        g = (frame.assign(died=died.loc[frame.index]).groupby(col)
             .agg(confirmed=("died", "size"), deaths=("died", "sum")).sort_values("confirmed", ascending=False))
        if top:
            g = g.head(top)
        return [{"name": str(k), "confirmed": int(r.confirmed), "deaths": int(r.deaths)} for k, r in g.iterrows()]

    drc = df[df["Location_Admin0"] == "Democratic Republic of the Congo"]
    zones = drc[~drc["Health zone"].astype(str).str.lower().isin(["other", "nan", "unknown"])]
    return {
        "as_of": f"{df['date'].max():%Y-%m-%d}",
        "first_confirmation": f"{df['date'].min():%Y-%m-%d}",
        "confirmed": int(len(df)),
        "deaths_among_confirmed": int(died.sum()),
        "countries": table("Location_Admin0", df),
        "provinces": table("Location Admin1", drc),
        "health_zones": table("Health zone", zones, top=15),
        "n_health_zones": int(zones["Health zone"].nunique()),
    }


# ------------------------------------------------------------------ orchestration
def build_observations(source: str = "mirror", raw_dir: Path | None = None) -> pd.DataFrame:
    """Read the newest raw files and return one validated table for all targets."""
    frames = []
    for key in config.ANNUAL_TARGETS:
        target = config.TARGETS[key]
        if source == "mirror":
            path = ingest.latest_raw(f"mirror_{key}", raw_dir)
            frames.append(clean_wdi_mirror(path.read_bytes(), target))
        elif source == "who":
            path = ingest.latest_raw(f"who_{key}", raw_dir)
            stamp = path.stem.split("__")[-1][:8]
            frames.append(clean_who_gho(path.read_bytes(), target, retrieved=f"retrieved {stamp}"))
        else:
            raise ValueError("source must be 'mirror' or 'who'")
    frames.append(clean_ebola(
        ingest.latest_raw("ebola_early", raw_dir).read_bytes(),
        ingest.latest_raw("ebola_who", raw_dir).read_bytes(),
        config.TARGETS["ebola_weekly_cases"],
    ))
    bvd_path = ingest.latest_raw("bvd_linelist", raw_dir)
    frames.append(clean_bvd_linelist(bvd_path.read_bytes(), config.TARGETS["bvd_weekly_cases"]))
    obs = schema.validate(pd.concat(frames, ignore_index=True))
    log.info("observations: %d rows, %d series", len(obs),
             obs.groupby(["disease", "metric", "location"]).ngroups)
    return obs


def save_observations(obs: pd.DataFrame, processed_dir: Path | None = None) -> Path:
    processed_dir = processed_dir or config.PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "observations.csv"
    obs.to_csv(path, index=False)
    return path


def load_observations(processed_dir: Path | None = None) -> pd.DataFrame:
    path = (processed_dir or config.PROCESSED_DIR) / "observations.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m epicast clean` first.")
    return pd.read_csv(path, dtype={"period": str, "source_version": str}, keep_default_na=True)


def series_for(obs: pd.DataFrame, target: config.Target) -> dict[str, pd.Series]:
    """Return ``{ISO3: values indexed by period}`` for one target.

    Annual series are cut to their most recent unbroken run of years, because
    the models assume evenly spaced data. The index is int years for annual
    targets and week-ending Timestamps for weekly ones.
    """
    sub = obs[(obs["disease"] == target.disease) & (obs["metric"] == target.metric)]
    out: dict[str, pd.Series] = {}
    for loc, grp in sub.groupby("location"):
        if target.freq == "A":
            s = pd.Series(grp["value"].to_numpy(float), index=grp["period"].astype(int).to_numpy())
            s = s.sort_index()
            s = s[schema.longest_recent_run(s.index.to_numpy())]
        else:
            s = pd.Series(grp["value"].to_numpy(float), index=pd.to_datetime(grp["period"]))
            s = s.sort_index()
        out[loc] = s
    return out
