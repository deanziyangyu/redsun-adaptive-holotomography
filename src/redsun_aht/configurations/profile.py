"""Build a headless instrument session from a packaged profile and overlays."""

import argparse
import threading
from collections.abc import Sequence
from pathlib import Path

from redsun import Session


def run_profile(argv: Sequence[str]) -> int:
    """Select an instrument profile and keep its headless session running."""
    parser = argparse.ArgumentParser(prog="aht session profile")
    parser.add_argument("instrument", choices=("arh", "holospim"))
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--check", action="store_true", help="Build and shut down in mock mode")
    options = parser.parse_args(argv)
    profiles = Path(__file__).resolve().parents[1] / "profiles"
    filename = {"arh": "arh_system.yaml", "holospim": "holospim.yaml"}[options.instrument]
    layers = [profiles / filename, *options.overlay]
    if options.check:
        layers.append({"mock": True, "strict": True})
    session = Session.from_config(layers)
    try:
        session.build()
        if not options.check:
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        session.shutdown()
    return 0
