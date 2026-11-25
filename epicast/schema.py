"""The canonical tidy table every source is converted into, and its validation.

One row = one observation of one metric in one place at one time.

    disease        malaria | hiv | ebola
    metric         e.g. incidence_per_1000_at_risk, new_infections, weekly_cases
    location       ISO 3166-1 alpha-3 code (NGA), optionally with a two-letter
                   subdivision suffix for a province (COD-IT = Ituri)
    location_name  English country name
    period         integer year for annual data; week-ending date (ISO string) for weekly
    freq           A | W
    value          the estimate or count, non-negative
    lower, upper   the source's own uncertainty bounds, when published (else NaN)
    source         where the number came from
    source_version commit hash or retrieval date that pins the exact vintage

Keeping one schema means cleaning, modelling, scoring and the dashboard never
need to know which source a number came from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = [
    "disease", "metric", "location", "location_name", "period", "freq",
    "value", "lower", "upper", "source", "source_version",
]
KEY = ["disease", "metric", "location", "period"]


class SchemaError(ValueError):
    """Raised when a table breaks the canonical contract."""


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the canonical contract and return the frame sorted by key.

    Failing loudly here is deliberate: a silent bad value upstream becomes a
    confident wrong forecast downstream.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if df.empty:
        raise SchemaError("table has no rows")
    df = df[COLUMNS].copy()

    if df["value"].isna().any():
        raise SchemaError("null values present; drop or impute before validating")
    if not np.isfinite(df["value"].to_numpy(dtype=float)).all():
        raise SchemaError("non-finite values present")
    if (df["value"] < 0).any():
        raise SchemaError("negative values present; counts and rates cannot be negative")
    if not df["freq"].isin(["A", "W"]).all():
        raise SchemaError("freq must be 'A' or 'W'")
    bad_loc = ~df["location"].astype(str).str.fullmatch(r"[A-Z]{3}(-[A-Z]{2})?")
    if bad_loc.any():
        raise SchemaError(f"locations must be ISO3 upper case: {df.loc[bad_loc, 'location'].unique()[:5]}")
    dup = df.duplicated(KEY)
    if dup.any():
        raise SchemaError(f"{int(dup.sum())} duplicate keys, e.g. {df.loc[dup, KEY].head(3).to_dict('records')}")
    both = df["lower"].notna() & df["upper"].notna()
    if (df.loc[both, "lower"] > df.loc[both, "upper"]).any():
        raise SchemaError("lower bound above upper bound")
    return df.sort_values(KEY).reset_index(drop=True)


def longest_recent_run(periods: np.ndarray, step: int = 1) -> np.ndarray:
    """Boolean mask of the unbroken run of periods that ends at the last one.

    Time-series models assume evenly spaced observations. If a country has a
    gap, we keep the most recent contiguous block rather than interpolate
    numbers the source never published.
    """
    periods = np.asarray(periods)
    keep = np.zeros(len(periods), dtype=bool)
    if len(periods) == 0:
        return keep
    keep[-1] = True
    for i in range(len(periods) - 2, -1, -1):
        if periods[i + 1] - periods[i] == step:
            keep[i] = True
        else:
            break
    return keep
