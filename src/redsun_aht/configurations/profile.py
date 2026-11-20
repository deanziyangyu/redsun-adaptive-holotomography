"""Build a headless instrument session from a packaged profile and overlays."""

import argparse
import math
import threading
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path

import yaml
from redsun import Session
from redsun._config import load

from redsun_aht.services.mmcore_stage import parse_axis_limits

from .arh_pva import arh_direct_pva

ARH_WORKSPACE_PROFILE = Path("02_docs_offline/instruments/arh/setup/arh-due.yaml")
ARH_SCRATCH_ROOT = Path("tmp/arh-due")


def _placeholder_fields(value: object, field: str) -> Iterator[str]:
    if isinstance(value, str) and value.startswith("REPLACE_"):
        yield field
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _placeholder_fields(item, f"{field}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _placeholder_fields(item, f"{field}[{index}]")


def check_holospim_host(layers: Sequence[dict], *, force: bool = False) -> None:
    """Validate resolved host settings before any HoloSPIM service is launched."""
    config = load(layers)
    if config.get("mock") is True and not force:
        return
    problems: list[str] = []
    for name, service in config.get("services", {}).items():
        if not isinstance(service, dict):
            continue
        field = f"services.{name}.args"
        args = service.get("args", {})
        if args and not (
            service.get("module")
            or (service.get("plugin_name") and service.get("plugin_id"))
        ):
            problems.append(
                f"services.{name} supplies args without a service module or plugin"
            )
        unresolved = list(_placeholder_fields(args, field))
        problems.extend(f"{item} is still a placeholder" for item in unresolved)
        for key in ("mm-config", "mmcore-config"):
            path = args.get(key)
            if (
                isinstance(path, str)
                and not path.startswith("REPLACE_")
                and not Path(path).expanduser().is_file()
            ):
                problems.append(f"{field}.{key} does not identify an existing file")
        plugin_id = service.get("plugin_id")
        if plugin_id in ("esp300-stage", "unified-mcu"):
            if not isinstance(service.get("prefix"), str) or not service["prefix"]:
                problems.append(
                    f"services.{name}.prefix must identify the PVA endpoint"
                )
            elif args.get("prefix") != service["prefix"]:
                problems.append(f"{field}.prefix must match the service PVA prefix")
        if plugin_id != "esp300-stage":
            continue
        if not isinstance(args.get("port"), str) or not args["port"].strip():
            problems.append(f"{field}.port must identify the ESP300 serial endpoint")
        identities = args.get("axis-identities")
        if (
            not isinstance(identities, (list, tuple))
            or len(identities) != 3
            or any(not isinstance(item, str) or not item.strip() for item in identities)
        ):
            problems.append(f"{field}.axis-identities must name X, Y and Z in order")
        bounds = args.get("limits")
        if not isinstance(bounds, str):
            problems.append(f"{field}.limits must supply all three travel bounds")
        elif not bounds.startswith("REPLACE_"):
            try:
                limits = parse_axis_limits(bounds, "xyz")
                if set(limits) != set("xyz"):
                    raise ValueError("limits must be supplied for x, y and z")
            except ValueError as error:
                problems.append(f"{field}.limits: {error}")
    services = config.get("services", {})
    for name, device in config.get("devices", {}).items():
        if not isinstance(device, dict):
            continue
        field = f"devices.{name}"
        service_name = device.get("service")
        service = (
            services.get(service_name, {}) if isinstance(service_name, str) else {}
        )
        args = service.get("args", {}) if isinstance(service, dict) else {}
        if device.get("enable_controller_jog") is True:
            roll_bounds = (
                args.get("roll-bounds-steps") if isinstance(args, dict) else None
            )
            if (
                not isinstance(args, dict)
                or args.get("require-holospim-panel") is not True
                or not isinstance(roll_bounds, (list, tuple)) or len(roll_bounds) != 2
                or any(type(value) is not int for value in roll_bounds)
                or not -(2**31) <= roll_bounds[0] < roll_bounds[1] < 2**31
            ):
                problems.append(
                    f"{field} controller jog requires commissioned HoloSPIM roll bounds"
                )
        scale = device.get("x28_um_per_turn")
        bounds = device.get("x28_jog_bounds_tenths")
        if scale is None and bounds is None:
            continue
        if (
            type(scale) not in (int, float) or not math.isfinite(scale) or scale == 0
            or not isinstance(bounds, (tuple, list)) or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or not -3600 <= bounds[0] < bounds[1] <= 3600
        ):
            problems.append(f"{field} requires a signed X28 scale and angular bounds")
            continue
        if (
            not isinstance(args, dict) or args.get("require-holospim-panel") is not True
            or args.get("x28-jog-bounds-tenths") not in (tuple(bounds), list(bounds))
        ):
            problems.append(f"{field} X28 bounds must match its HoloSPIM MCU service")
        lower, upper = bounds
        largest_angle = max(abs(lower), abs(upper), min(upper - lower, 3600))
        if largest_angle * abs(scale) / 3600 > 1000:
            problems.append(
                f"{field} X28 bounds exceed 1000 µm at the configured scale"
            )
    if problems:
        raise ValueError("HoloSPIM host configuration:\n  " + "\n  ".join(problems))


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
        if not isinstance(sources, list) or any(
            not isinstance(item, str) for item in sources
        ):
            raise ValueError("led_pattern_ctrl.sources must be a list of filenames")
        pattern_presenter["sources"] = [
            str((path.parent / Path(filename).expanduser()).resolve())
            for filename in sources
        ]
    return config


def read_arh_profile(path: Path) -> dict:
    """Load an ARH profile, retaining the existing desktop entry point."""
    return read_instrument_profile(path)


def read_holospim_profile(path: Path) -> dict:
    """Load a full HoloSPIM profile with the packaged laser as its default."""
    config = read_instrument_profile(path)
    devices = config.setdefault("devices", {})
    if "laser" not in devices:
        packaged = Path(__file__).resolve().parents[1] / "profiles" / "holospim.yaml"
        default = read_instrument_profile(packaged)
        services = config.setdefault("services", {})
        services.setdefault("transport", "pv-access")
        services.setdefault("laser_service", default["services"]["laser_service"])
        devices["laser"] = default["devices"]["laser"]
    return config


def without_arh_due(config: dict) -> dict:
    """Omit Due services and their client devices from an ARH profile."""
    services = config.get("services", {})
    omitted = {
        name
        for name, service in services.items()
        if isinstance(service, dict) and service.get("plugin_id") == "unified-emmsh"
    }
    return {
        **config,
        "services": {
            name: service for name, service in services.items() if name not in omitted
        },
        "devices": {
            name: device
            for name, device in config.get("devices", {}).items()
            if device.get("service") not in omitted
            and device.get("plugin_id") != "unified-emmsh"
        },
    }


def run_profile(argv: Sequence[str]) -> int:
    """Select an instrument profile and keep its headless session running."""
    parser = argparse.ArgumentParser(prog="aht session profile")
    parser.add_argument("instrument", choices=("arh", "holospim"))
    parser.add_argument(
        "--profile", type=Path, help="Use a complete instrument profile"
    )
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--check",
        "--check-profile",
        action="store_true",
        help="Build and shut down in mock mode",
    )
    checks.add_argument(
        "--check-host",
        action="store_true",
        help="Validate HoloSPIM host settings without starting services",
    )
    options = parser.parse_args(argv)
    profiles = Path(__file__).resolve().parents[1] / "profiles"
    filename = {"arh": "arh_system.yaml", "holospim": "holospim.yaml"}[
        options.instrument
    ]
    profile = options.profile
    if (
        profile is None
        and options.instrument == "arh"
        and ARH_WORKSPACE_PROFILE.is_file()
    ):
        profile = ARH_WORKSPACE_PROFILE
    base = profile or profiles / filename
    try:
        read_base = (
            read_holospim_profile if options.instrument == "holospim"
            else read_arh_profile
        )
        layers = [
            read_base(base),
            *(read_instrument_profile(path) for path in options.overlay),
        ]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if options.check:
        check = {"mock": True, "strict": True}
        if options.instrument == "arh":
            check["storage"] = {"base_dir": str(ARH_SCRATCH_ROOT.resolve())}
        layers.append(check)
    if options.check_host and options.instrument != "holospim":
        parser.error("--check-host is available for HoloSPIM")
    if options.instrument == "holospim":
        try:
            check_holospim_host(layers, force=options.check_host)
        except ValueError as error:
            parser.error(str(error))
        if options.check_host:
            print("HoloSPIM host configuration passed static checks.")
            return 0
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
