"""Command line interface.

    python -m epicast run                # everything, start to finish
    python -m epicast ingest             # download raw data
    python -m epicast clean              # raw -> tidy observations.csv
    python -m epicast backtest           # rolling-origin evaluation
    python -m epicast forecast           # forecasts beyond the data
    python -m epicast report             # build outputs/dashboard.html

Options: --source mirror|who   --jobs N   --live-outbreak
"""
from __future__ import annotations

import argparse
import logging

from . import pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="epicast", description="Malaria, HIV and Ebola forecasting pipeline.")
    parser.add_argument("step", choices=["run", "ingest", "clean", "backtest", "forecast", "report"])
    parser.add_argument("--source", choices=["mirror", "who"], default="mirror",
                        help="mirror = pinned, reproducible snapshot (default); who = live WHO GHO API")
    parser.add_argument("--jobs", type=int, default=1, help="parallel workers for the backtest")
    parser.add_argument("--live-outbreak", action="store_true",
                        help="read the newest Bundibugyo line list instead of the pinned commit")
    parser.add_argument("--skip-ingest", action="store_true", help="with 'run': reuse the raw files already on disk")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.step == "run":
        path = pipeline.run_all(args.source, args.jobs, args.skip_ingest, args.live_outbreak)
        print(f"Done. Open {path}")
    elif args.step == "ingest":
        pipeline.step_ingest(args.source, args.live_outbreak)
    elif args.step == "clean":
        pipeline.step_clean(args.source)
    elif args.step == "backtest":
        pipeline.step_backtest(args.jobs)
    elif args.step == "forecast":
        pipeline.step_forecast()
    elif args.step == "report":
        print(f"Wrote {pipeline.step_report(args.source)}")


if __name__ == "__main__":
    main()
