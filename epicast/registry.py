"""The outbreak registry and the sources registry.

Two small hand-maintained CSV files in ``epicast/reference/``:

``outbreaks.csv``  one row per Ebola-disease outbreak (Ebola virus in DRC, Sudan
                   virus in Uganda, Bundibugyo virus anywhere) since 2007:
                   where, when, how many cases and deaths, and which sources
                   say so.
``sources.csv``    one row per source: a stable id, title, publisher, URL and
                   the date it was read.

Every outbreak row names at least one ``source_id``. The dashboard turns those
ids into links, so adding a link to the map is a matter of adding a row here.
``docs/ADDING_SOURCES.md`` walks through it.

Validation runs every time the files are loaded, so a typo in an id, a missing
URL or a death count above the case count stops the pipeline instead of
reaching the map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

VIRUSES = {
    "EBOV": "Ebola virus (Zaire)",
    "SUDV": "Sudan virus",
    "BDBV": "Bundibugyo virus",
}
OUTBREAK_COLUMNS = ["outbreak_id", "virus", "country", "admin1", "place", "place_precision", "lat", "lon",
                    "start", "end", "status", "cases", "deaths", "case_note", "source_ids", "notes"]
SOURCE_COLUMNS = ["source_id", "title", "publisher", "url", "accessed", "notes"]


class RegistryError(ValueError):
    """Raised when a registry file breaks its rules."""


def load_sources(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or config.REFERENCE_DIR / "sources.csv", dtype=str, keep_default_na=False)
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise RegistryError(f"sources.csv is missing columns: {missing}")
    if df["source_id"].duplicated().any():
        raise RegistryError(f"duplicate source_id: {df.loc[df['source_id'].duplicated(), 'source_id'].tolist()}")
    bad = df[~df["url"].str.match(r"https?://")]
    if len(bad):
        raise RegistryError(f"sources without an http(s) URL: {bad['source_id'].tolist()}")
    if (df["title"].str.strip() == "").any():
        raise RegistryError("every source needs a title")
    return df


def _year(value: str) -> int:
    return int(str(value)[:4])


def load_outbreaks(path: Path | None = None, sources: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load and validate the outbreak registry. Adds ``start_year`` and ``end_year``."""
    df = pd.read_csv(path or config.REFERENCE_DIR / "outbreaks.csv", dtype=str, keep_default_na=False)
    missing = [c for c in OUTBREAK_COLUMNS if c not in df.columns]
    if missing:
        raise RegistryError(f"outbreaks.csv is missing columns: {missing}")
    sources = load_sources() if sources is None else sources
    known = set(sources["source_id"])

    if df["outbreak_id"].duplicated().any():
        raise RegistryError("duplicate outbreak_id")
    if not df["virus"].isin(VIRUSES).all():
        raise RegistryError(f"virus must be one of {sorted(VIRUSES)}")
    if not df["country"].str.fullmatch(r"[A-Z]{3}").all():
        raise RegistryError("country must be an ISO3 code")
    if not df["status"].isin(["ended", "ongoing"]).all():
        raise RegistryError("status must be 'ended' or 'ongoing'")
    for _, row in df.iterrows():
        ids = [i for i in row["source_ids"].split(";") if i]
        if not ids:
            raise RegistryError(f"{row['outbreak_id']}: every outbreak needs at least one source")
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise RegistryError(f"{row['outbreak_id']}: unknown source ids {unknown}")
        if not row["start"]:
            raise RegistryError(f"{row['outbreak_id']}: start is required")
        if row["status"] == "ended" and not row["end"]:
            raise RegistryError(f"{row['outbreak_id']}: an ended outbreak needs an end")

    for col in ("lat", "lon", "cases", "deaths"):
        df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="raise")
    both = df["cases"].notna() & df["deaths"].notna()
    if (df.loc[both, "deaths"] > df.loc[both, "cases"]).any():
        raise RegistryError("deaths exceed cases in at least one row")
    located = df["lat"].notna()
    if ((df.loc[located, "lat"].abs() > 90) | (df.loc[located, "lon"].abs() > 180)).any():
        raise RegistryError("latitude or longitude out of range")
    df["start_year"] = df["start"].map(_year)
    df["end_year"] = [(_year(e) if e else np.nan) for e in df["end"]]
    if (df["end_year"] < df["start_year"]).any():
        raise RegistryError("an outbreak ends before it starts")
    return df


def source_links(source_ids: str, sources: pd.DataFrame) -> list[dict]:
    """``[{title, publisher, url}]`` for a semicolon-separated list of source ids."""
    idx = sources.set_index("source_id")
    return [{"title": idx.loc[i, "title"], "publisher": idx.loc[i, "publisher"], "url": idx.loc[i, "url"]}
            for i in source_ids.split(";") if i]
