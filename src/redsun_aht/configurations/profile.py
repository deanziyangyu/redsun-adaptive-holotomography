"""Build a headless instrument session from a packaged profile and overlays."""

import argparse
import threading
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import yaml
from redsun import Session

from .arh_pva import arh_direct_pva

ARH_WORKSPACE_PROFILE = Path("02_docs_offline/instruments/arh/setup/arh-due.yaml")
ARH_SCRATCH_ROOT = Path("tmp/arh-due")


def read_instrument_profile(path: Path) -> dict:
    """Load either instrument profile and resolve relative MM configuration paths."""
    path = path.expanduser().resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Expected a profile mapping in {path}")
    services = config.get("services", {})
    if not isinstance(services, dict):
        raise ValueError(f"Expected a services mapping in {path}")
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        args = service.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"Expected an args mapping for {name!r} in {path}")
        for key in ("mm-config", "mmcore-config"):
            value = args.get(key)
            if isinstance(value, str) and value.startswith(("./", ".\\")):
                args[key] = str((path.parent / value).resolve())
    pattern_presenter = config.get("presenters", {}).get("led_pattern_ctrl", {})
    if "sources" in pattern_presenter:
        sources = pattern_presenter["sources"]
        if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
            raise ValueError("led_pattern_ctrl.sources must be a list of filenames")
        pattern_presenter["sources"] = [
            str((path.parent / Path(filename).expanduser()).resolve()) for filename in sources
        ]
    return config


def read_arh_profile(path: Path) -> dict:
    """Load an ARH profile, retaining the existing desktop entry point."""
    return read_instrument_profile(path)


def without_arh_due(config: dict) -> dict:
    """Omit Due services and their client devices from an ARH profile."""
    services = config.get("services", {})
    omitted = {
        name for name, service in services.items()
        if isinstance(service, dict) and service.get("plugin_id") == "unified-emmsh"
    }
    return {
        **config,
        "services": {name: service for name, service in services.items() if name not in omitted},
        "devices": {
            name: device for name, device in config.get("devices", {}).items()
            if device.get("service") not in omitted and device.get("plugin_id") != "unified-emmsh"
        },
    }


def run_profile(argv: Sequence[str]) -> int:
    """Select an instrument profile and keep its headless session running."""
    parser = argparse.ArgumentParser(prog="aht session profile")
    parser.add_argument("instrument", choices=("arh", "holospim"))
    parser.add_argument("--profile", type=Path, help="Use a complete instrument profile")
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--check", action="store_true", help="Build and shut down in mock mode")
    options = parser.parse_args(argv)
    profiles = Path(__file__).resolve().parents[1] / "profiles"
    filename = {"arh": "arh_system.yaml", "holospim": "holospim.yaml"}[options.instrument]
    profile = options.profile
    if profile is None and options.instrument == "arh":
        if ARH_WORKSPACE_PROFILE.is_file():
            profile = ARH_WORKSPACE_PROFILE
    base = profile or profiles / filename
    try:
        layers = [read_instrument_profile(path) for path in (base, *options.overlay)]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if options.check:
        check = {"mock": True, "strict": True}
        if options.instrument == "arh":
            check["storage"] = {"base_dir": str(ARH_SCRATCH_ROOT.resolve())}
        layers.append(check)
    with arh_direct_pva() if options.instrument == "arh" else nullcontext():
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
