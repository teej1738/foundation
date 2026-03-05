"""Foundation CLI entry point (AD-8).

Every pipeline step is a CLI command that reads TOML config, produces
JSON output, and returns an exit code. CC orchestrates by calling CLI
commands in sequence.

Usage:
    foundation validate <config.toml> [--type experiment|instrument|environment]
    foundation status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a config file against its Pydantic schema."""
    from foundation.config.loader import (
        load_environment,
        load_experiment,
        load_instrument,
    )

    path = Path(args.config)
    if not path.exists():
        print(json.dumps({"status": "error", "message": f"File not found: {path}"}))
        return 1

    loaders = {
        "experiment": load_experiment,
        "instrument": load_instrument,
        "environment": load_environment,
    }
    loader = loaders.get(args.type)
    if loader is None:
        print(json.dumps({"status": "error", "message": f"Unknown type: {args.type}"}))
        return 1

    try:
        loader(path)
        print(json.dumps({"status": "ok", "type": args.type, "file": str(path)}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


def cmd_download(args: argparse.Namespace) -> int:
    """Download raw data from Binance."""
    from foundation.data.downloaders.candles import CandleDownloader
    from foundation.data.downloaders.funding import FundingRateDownloader
    from foundation.data.downloaders.oi import OIMetricsDownloader

    dataset = args.dataset
    output_dir = Path(args.output) if args.output else Path("data/raw/btcusdt")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if dataset in ("candles-1m", "candles-5m"):
            interval = dataset.split("-")[1]
            dl = CandleDownloader(output_dir, interval=interval)
            sy, sm = map(int, args.start.split("-"))
            ey, em = map(int, args.end.split("-"))
            paths = dl.run(sy, sm, ey, em)
            result = {"status": "ok", "dataset": dataset, "files_saved": len(paths)}
        elif dataset == "oi":
            dl_oi = OIMetricsDownloader(output_dir)
            sy, sm = map(int, args.start.split("-"))
            ey, em = map(int, args.end.split("-"))
            paths = dl_oi.run(sy, sm, ey, em)
            result = {"status": "ok", "dataset": dataset, "files_saved": len(paths)}
        elif dataset == "funding":
            dl_fund = FundingRateDownloader(output_dir)
            start = f"{args.start}-01"
            end = f"{args.end}-28"
            path = dl_fund.run(start, end)
            result = {"status": "ok", "dataset": dataset, "files_saved": 1 if path.exists() else 0}
        else:
            result = {"status": "error", "message": f"Unknown dataset: {dataset}"}
            print(json.dumps(result))
            return 1

        print(json.dumps(result))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Run pipeline diagnostics (AD-22)."""
    from foundation.diagnostics.models import PlantedSignalConfig
    from foundation.diagnostics.signal_recovery import test_signal_recovery

    if args.diagnostic == "planted-signal":
        config = PlantedSignalConfig(
            strength=args.strength,
            seed=args.seed,
        )
        result = test_signal_recovery(config=config, n_rows=args.n_rows)
        output = {
            "status": "ok",
            "diagnostic": "planted-signal",
            "planted_auc": result.planted_auc,
            "baseline_auc": result.baseline_auc,
            "recovery_ratio": result.recovery_ratio,
            "passed": result.passed,
            "threshold": result.threshold,
            "strength": result.strength,
            "n_folds": result.n_folds,
        }
        print(json.dumps(output, indent=2))
        return 0 if result.passed else 1
    else:
        print(json.dumps({"status": "error", "message": f"Unknown diagnostic: {args.diagnostic}"}))
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Print Foundation status as JSON."""
    import foundation

    status = {
        "version": foundation.__version__,
        "phase": "0",
        "status": "scaffold complete",
    }
    print(json.dumps(status, indent=2))
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="foundation",
        description="Foundation ML trading system CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # validate
    val_parser = sub.add_parser("validate", help="Validate a config file")
    val_parser.add_argument("config", help="Path to TOML config file")
    val_parser.add_argument(
        "--type",
        "-t",
        choices=["experiment", "instrument", "environment"],
        default="experiment",
        help="Config type (default: experiment)",
    )

    # download
    dl_parser = sub.add_parser("download", help="Download raw data")
    dl_parser.add_argument(
        "dataset",
        choices=["candles-1m", "candles-5m", "oi", "funding"],
        help="Dataset to download",
    )
    dl_parser.add_argument("--start", required=True, help="Start month (YYYY-MM)")
    dl_parser.add_argument("--end", required=True, help="End month (YYYY-MM)")
    dl_parser.add_argument("--output", "-o", help="Output directory")

    # diagnose
    diag_parser = sub.add_parser("diagnose", help="Run pipeline diagnostics (AD-22)")
    diag_parser.add_argument(
        "diagnostic",
        choices=["planted-signal"],
        help="Diagnostic to run",
    )
    diag_parser.add_argument(
        "--strength", type=float, default=0.7, help="Signal strength (default: 0.7)"
    )
    diag_parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed (default: 42)"
    )
    diag_parser.add_argument(
        "--n-rows", type=int, default=10000, help="Synthetic data rows (default: 10000)"
    )

    # status
    sub.add_parser("status", help="Print Foundation status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "validate": cmd_validate,
        "download": cmd_download,
        "diagnose": cmd_diagnose,
        "status": cmd_status,
    }
    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 0

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
