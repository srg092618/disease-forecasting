"""Central configuration: data sources, paths, and expected schemas."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# CDC ILINet (influenza-like illness), national level, via the Delphi Epidata API.
# Public, no API key required. Swap in another source by adding an entry here.
SOURCES = {
    "ilinet": {
        "url": "https://api.delphi.cmu.edu/epidata/fluview/",
        "params": {
            "regions": "nat",
            "epiweeks": "201001-203052",
            "format": "csv",
        },
        "expected_columns": ["region", "epiweek", "wili", "ili", "num_ili", "num_patients"],
    },
}

REQUEST_TIMEOUT_SECONDS = 60
