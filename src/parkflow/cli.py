"""Command-line entry point: ``parkflow <command>``.

Commands:
    run             execute the full pipeline and write artifacts
    info            print resolved configuration
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import pipeline
from .config import Config
from .logging_utils import configure, get_logger

log = get_logger("cli")


def _cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    result = pipeline.run(cfg, horizon=args.horizon, live=args.live)
    m = result.metrics
    log.info(
        "Done. MAE model=%.3f baseline=%.3f -> %s",
        m["model"]["mae"],
        m["baseline"]["mae"],
        "model wins" if m["model_beats_baseline"] else "baseline wins",
    )
    return 0


def _cmd_info(args: argparse.Namespace, cfg: Config) -> int:
    log.info("raw_data:      %s", cfg.paths.raw_data)
    log.info("artifacts_dir: %s", cfg.paths.artifacts_dir)
    log.info("bin_hours:     %d (%d bins/day)", cfg.temporal.bin_hours, cfg.temporal.bins_per_day)
    log.info("test_fraction: %.2f", cfg.model.test_fraction)
    log.info("objective:     %s", cfg.model.objective)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parkflow", description="ParkFlow-AI pipeline")
    p.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument(
        "--horizon", type=int, default=None,
        help="number of future bins to forecast recursively (default: config forecast_horizon_bins)",
    )
    r.add_argument(
        "--live", action="store_true",
        help="relabel the forecast timeline to start now() ('as if run against live feeds'); "
             "does not alter which historical data feeds the model",
    )
    r.set_defaults(func=_cmd_run)

    i = sub.add_parser("info", help="print resolved configuration")
    i.set_defaults(func=_cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    configure(cfg.log_level)
    return args.func(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
