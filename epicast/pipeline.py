"""End-to-end pipeline: ingest -> clean -> backtest -> forecast -> report.

Each step reads the previous step's files and writes its own, so any step can
be re-run alone and every intermediate result can be inspected.
"""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import calibrate, clean, config, geo, ingest, registry

log = logging.getLogger(__name__)


def _round_sig(df: pd.DataFrame, cols: list[str], sig: int = 5) -> pd.DataFrame:
    """Round numeric columns to ``sig`` significant figures (keeps files small)."""
    out = df.copy()
    for c in cols:
        v = out[c].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = np.where(v != 0, np.floor(np.log10(np.abs(v))), 0.0)
        mag = np.where(np.isfinite(mag), mag, 0.0)
        factor = 10.0 ** (sig - 1 - mag)
        out[c] = np.where(np.isfinite(v), np.round(v * factor) / factor, v)
    return out


def step_ingest(source: str = "mirror", live_outbreak: bool = False) -> None:
    ingest.ingest(source, live_outbreak=live_outbreak)


def step_clean(source: str = "mirror") -> pd.DataFrame:
    """Raw files -> tidy observations, map geometry and the Bundibugyo summary."""
    registry.load_outbreaks()  # fail early if the hand-maintained registry has an error
    obs = clean.build_observations(source)
    clean.save_observations(obs)
    geo.save_geo(geo.build_geo())
    summary = clean.bvd_summary(ingest.latest_raw("bvd_linelist").read_bytes())
    (config.PROCESSED_DIR / "bvd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return obs


def step_backtest(n_jobs: int = 1, targets: list[str] | None = None) -> pd.DataFrame:
    """Rolling-origin backtest, then recalibration, then scoring of both stages."""
    obs = clean.load_observations()
    frames = []
    for key in targets or list(config.TARGETS):
        target = config.TARGETS[key]
        series = clean.series_for(obs, target)
        fn = bt.backtest_annual if target.freq == "A" else bt.backtest_outbreak
        frames.append(fn(target, series, n_jobs=n_jobs))
    raw = pd.concat(frames, ignore_index=True)
    calibrated = calibrate.calibrate_backtest(raw)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = bt.Q_COLS + ["mase_scale", "vol"]
    _round_sig(raw, cols).to_csv(config.OUTPUT_DIR / "backtest_forecasts_raw.csv.gz", index=False)
    _round_sig(calibrated, cols).to_csv(config.OUTPUT_DIR / "backtest_forecasts.csv.gz", index=False)
    calibrate.final_margins(raw).round(4).to_csv(config.OUTPUT_DIR / "calibration_margins.csv", index=False)

    overall, by_h = [], []
    for stage, table in (("raw", raw), ("calibrated", calibrated)):
        scores = bt.score(table)
        o = bt.summarise(scores, ["target", "model"])
        ci = bt.bootstrap_rel_wis(scores)
        if not ci.empty:
            o = o.merge(ci, on=["target", "model"], how="left")
        overall.append(o.assign(stage=stage))
        by_h.append(bt.summarise(scores, ["target", "model", "horizon"]).assign(stage=stage))
    overall = pd.concat(overall, ignore_index=True)
    overall.round(4).to_csv(config.OUTPUT_DIR / "metrics_overall.csv", index=False)
    pd.concat(by_h, ignore_index=True).round(4).to_csv(config.OUTPUT_DIR / "metrics_by_horizon.csv", index=False)
    log.info("\n%s", overall[["stage", "target", "model", "n", "rel_wis_log", "mase", "cov50", "cov95"]]
             .round(3).to_string(index=False))
    return overall


def step_forecast() -> pd.DataFrame:
    """Forecasts beyond the data, recalibrated with the margins learned in the backtest."""
    obs = clean.load_observations()
    frames = []
    for key, target in config.TARGETS.items():
        series = clean.series_for(obs, target)
        if target.freq == "A":
            frames.append(bt.final_annual_forecasts(target, series))
        elif target.ongoing:
            frames.append(bt.final_outbreak_forecasts(target, series))
    fc = pd.concat(frames, ignore_index=True).drop(columns=["y", "mase_scale"])
    margins_path = config.OUTPUT_DIR / "calibration_margins.csv"
    if not margins_path.exists():
        raise FileNotFoundError("calibration_margins.csv not found. Run `python -m epicast backtest` first.")
    fc = calibrate.calibrate_final(fc, pd.read_csv(margins_path)).drop(columns=["vol"])
    _round_sig(fc, bt.Q_COLS).to_csv(config.OUTPUT_DIR / "forecasts.csv", index=False)
    return fc


def step_report(source: str = "mirror") -> Path:
    from . import report

    path = report.build_dashboard()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "wdi_mirror_commit": config.WDI_MIRROR_COMMIT,
        "bvd_linelist_commit": config.BVD_LINELIST_COMMIT,
        "python": platform.python_version(),
        "packages": _versions(),
        "random_seed": config.RANDOM_SEED,
        "quantiles": list(config.QUANTILES),
    }
    (config.OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def _versions() -> dict[str, str]:
    import importlib

    out = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "sklearn", "requests", "shapely"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:  # pragma: no cover
            out[name] = "not installed"
    return out


def run_all(source: str = "mirror", n_jobs: int = 1, skip_ingest: bool = False,
            live_outbreak: bool = False) -> Path:
    if not skip_ingest:
        step_ingest(source, live_outbreak)
    step_clean(source)
    step_backtest(n_jobs=n_jobs)
    step_forecast()
    return step_report(source)
