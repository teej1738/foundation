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

    # status
    sub.add_parser("status", help="Print Foundation status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "validate": cmd_validate,
        "status": cmd_status,
    }
    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 0

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
