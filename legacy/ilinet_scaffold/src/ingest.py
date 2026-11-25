"""Download raw data for a configured source and save it, untouched, with a UTC timestamp.

Usage:
    python -m src.ingest            # default source: ilinet
    python -m src.ingest ilinet
"""
import argparse
import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src import config

log = logging.getLogger(__name__)


def download(source: str) -> bytes:
    """Fetch the raw bytes for a configured source."""
    spec = config.SOURCES[source]
    resp = requests.get(spec["url"], params=spec["params"], timeout=config.REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.content


def validate(content: bytes, expected_columns: list[str]) -> pd.DataFrame:
    """Parse CSV bytes and check the expected columns are present and data is non-empty."""
    df = pd.read_csv(io.BytesIO(content))
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Downloaded data has no rows")
    return df


def save_raw(content: bytes, source: str, raw_dir: Path = config.RAW_DIR) -> Path:
    """Write bytes exactly as received to raw_dir/<source>_<YYYYmmddTHHMMSSZ>.csv."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = raw_dir / f"{source}_{stamp}.csv"
    path.write_bytes(content)
    return path


def ingest(source: str = "ilinet", raw_dir: Path = config.RAW_DIR) -> Path:
    content = download(source)
    df = validate(content, config.SOURCES[source]["expected_columns"])
    path = save_raw(content, source, raw_dir)
    log.info("Saved %d rows to %s", len(df), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", default="ilinet", choices=sorted(config.SOURCES))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest(args.source)


if __name__ == "__main__":
    main()
