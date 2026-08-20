"""Command-line entry point for hardware-free Phase 1 workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from redsun_aht.configurations import PROFILES, run_simulation

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aht")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("profiles", help="list declared composition profiles")
    simulate = subparsers.add_parser(
        "simulate", help="run and replay a hardware-free lifecycle smoke test"
    )
    simulate.add_argument(
        "--journal",
        type=Path,
        default=Path("run-events.jsonl"),
        help="append-only event-journal path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AHT command line."""
    args = _build_parser().parse_args(argv)
    if args.command == "profiles":
        for profile in PROFILES.values():
            print(f"{profile.name}: {profile.description}")
        return 0
    if args.command == "simulate":
        result = run_simulation(args.journal)
        print(f"run {result.run_id} replayed as {result.state.value}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
