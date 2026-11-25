"""Download raw data and store it untouched.

Rules this module enforces:

1. Raw bytes are written exactly as received. Nothing is parsed or "fixed" here.
2. Every download gets a UTC timestamp in its file name, so an old file is
   never overwritten and any past run can be reproduced.
3. Every download is logged in ``manifest.jsonl`` with its URL and SHA-256
   hash, which is the audit trail for "where did this number come from?".

Usage
-----
    python -m epicast ingest                 # pinned mirror + Ebola (default)
    python -m epicast ingest --source who    # live WHO GHO API + Ebola
    python -m epicast ingest --live-outbreak # newest Bundibugyo line list instead of the pinned one
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- primitives
def fetch(url: str, params: dict | None = None, retries: int = 3) -> bytes:
    """GET a URL and return the response body, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as err:  # network hiccup, 5xx, timeout
            last_error = err
            log.warning("fetch failed (attempt %d/%d): %s", attempt, retries, err)
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not fetch {url}") from last_error


def save_raw(content: bytes, name: str, ext: str, url: str, raw_dir: Path | None = None) -> Path:
    """Write bytes to ``raw_dir/<name>__<UTC stamp>.<ext>`` and log the download."""
    raw_dir = raw_dir or config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = raw_dir / f"{name}__{stamp}.{ext}"
    path.write_bytes(content)
    entry = {
        "name": name,
        "file": path.name,
        "url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with (raw_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    log.info("saved %s (%d bytes)", path.name, len(content))
    return path


def latest_raw(name: str, raw_dir: Path | None = None) -> Path:
    """Return the most recent raw file for ``name`` (stamps sort chronologically)."""
    raw_dir = raw_dir or config.RAW_DIR
    matches = sorted(raw_dir.glob(f"{name}__*"))
    if not matches:
        raise FileNotFoundError(
            f"no raw file for '{name}' in {raw_dir}. Run `python -m epicast ingest` first."
        )
    return matches[-1]


# ---------------------------------------------------------------------- sources
def wdi_mirror_url(wdi_code: str) -> str:
    """URL of one indicator file in the pinned World Bank mirror.

    ``SH.MLR.INCD.P3`` is stored as ``sh_mlr_incd_p3``.
    """
    concept = wdi_code.lower().replace(".", "_")
    return f"{config.WDI_MIRROR_BASE}/datapoints/ddf--datapoints--{concept}--by--geo--time.csv"


def ingest_wdi_mirror(raw_dir: Path | None = None) -> list[Path]:
    """Download malaria and HIV indicators from the pinned mirror."""
    paths = []
    for key in config.ANNUAL_TARGETS:
        target = config.TARGETS[key]
        url = wdi_mirror_url(target.wdi_code)
        paths.append(save_raw(fetch(url), f"mirror_{key}", "csv", url, raw_dir))
    return paths


def ingest_who_gho(raw_dir: Path | None = None) -> list[Path]:
    """Download malaria and HIV indicators live from the WHO GHO OData API.

    The API returns ``{"value": [ {record}, ... ]}``. Each record carries
    ``SpatialDimType`` (COUNTRY, REGION, ...), ``SpatialDim`` (ISO3 code),
    ``TimeDim`` (year), ``NumericValue`` and the uncertainty bounds ``Low`` and
    ``High``. Parsing happens in clean.py; here we only store the JSON.
    """
    paths = []
    for key in config.ANNUAL_TARGETS:
        target = config.TARGETS[key]
        url = f"{config.WHO_GHO_BASE}/{target.gho_code}"
        paths.append(save_raw(fetch(url), f"who_{key}", "json", url, raw_dir))
    return paths


def ingest_ebola(raw_dir: Path | None = None) -> list[Path]:
    """Download the two West Africa 2014-2016 case-count transcriptions."""
    return [
        save_raw(fetch(config.EBOLA_EARLY_URL), "ebola_early", "csv", config.EBOLA_EARLY_URL, raw_dir),
        save_raw(fetch(config.EBOLA_WHO_URL), "ebola_who", "csv", config.EBOLA_WHO_URL, raw_dir),
    ]


def ingest_bvd(live: bool = False, raw_dir: Path | None = None) -> list[Path]:
    """Download the 2026 Bundibugyo line list (pinned commit, or the newest if ``live``)."""
    url = config.BVD_LINELIST_URL.format(ref="main" if live else config.BVD_LINELIST_COMMIT)
    return [save_raw(fetch(url), "bvd_linelist", "csv", url, raw_dir)]


def ingest_boundaries(raw_dir: Path | None = None) -> list[Path]:
    """Download the map boundaries: countries, admin-1 units, current DRC provinces."""
    jobs = [
        ("ne_countries", f"{config.NATURAL_EARTH_BASE}/ne_50m_admin_0_countries.geojson"),
        ("ne_admin1", f"{config.NATURAL_EARTH_BASE}/ne_10m_admin_1_states_provinces.geojson"),
        ("cod_provinces", config.GEOBOUNDARIES_COD_ADM1_URL),
    ]
    return [save_raw(fetch(url), name, "geojson", url, raw_dir) for name, url in jobs]


def ingest(source: str = "mirror", raw_dir: Path | None = None, live_outbreak: bool = False) -> list[Path]:
    """Download everything the pipeline needs. ``source`` is 'mirror' or 'who'."""
    if source == "mirror":
        paths = ingest_wdi_mirror(raw_dir)
    elif source == "who":
        paths = ingest_who_gho(raw_dir)
    else:
        raise ValueError("source must be 'mirror' or 'who'")
    return paths + ingest_ebola(raw_dir) + ingest_bvd(live_outbreak, raw_dir) + ingest_boundaries(raw_dir)
