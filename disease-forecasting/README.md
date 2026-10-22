# Disease Forecasting

Forecasting infectious disease activity, starting with CDC ILINet (influenza-like illness) data.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\Activate.ps1 in PowerShell
pip install -r requirements.txt
```

## Ingest data

```bash
python -m src.ingest            # downloads ILINet national data to data/raw/ilinet_<UTC timestamp>.csv
```

Raw files are never modified; cleaned outputs go to `data/processed/`. Sources, URLs and
expected columns live in `src/config.py`.

## Tests

```bash
pytest
```

## Layout

| Path | Purpose |
|---|---|
| `src/config.py` | URLs, paths, expected columns |
| `src/ingest.py` | Download + validate + save raw data |
| `tests/` | Unit tests (no network needed) |
| `data/raw/` | Untouched, timestamped downloads (git-ignored) |
| `data/processed/` | Cleaned data (git-ignored) |
| `notebooks/` | Exploration |
| `docker/` | Container setup |
| `.github/workflows/` | CI |
