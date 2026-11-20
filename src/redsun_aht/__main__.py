"""Command-line entry point for the adaptive holotomography application.

Adapted from the IX83 imaging bundle's command line and its lazy session
package (``redsun-searching-is-imaging``, revision ``3a3bf06``, Apache-2.0).
The dispatch stays lazy: importing this module must not import napari, Qt or
``pymmcore_plus``. The session and service registry is read only when a
command needs it, through :func:`load_registry`.

The application is PVA-only. ``scripts/run_session.ps1`` is the supported host
entry point because it relocates the caches and prepares a short writable
temporary root before this module is imported. ``aht --check`` reports what the
current environment would use, so it can be run before and after that setup.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Final

#: Where this host keeps acquisition data; replaced only when it refuses writes.
DEFAULT_DATA_ROOT: Final = Path("R:/redsun_aht_data")

#: Longest temporary root ``p4p`` reliably builds its event base on.
MAX_PVA_TEMP: Final = 90

#: Environment variable an out-of-tree launcher uses to name the checkout.
REPOSITORY_ENV: Final = "REDSUN_AHT_REPO"

_PROBE_PREFIX: Final = ".aht-write-probe-"

Entry = Callable[[Sequence[str]], int | None]
"""One registry entry: runs a session or service with the remaining arguments."""


@dataclass(frozen=True)
class Registry:
    """Sessions and services that this checkout can dispatch."""

    sessions: Mapping[str, Entry] = field(default_factory=dict)
    services: Mapping[str, Entry] = field(default_factory=dict)


def writable(path: Path) -> bool:
    """Return whether *path* accepts a probe file, creating the directory if needed.

    Parameters
    ----------
    path : Path
        Directory to create and probe.

    Returns
    -------
    bool
        ``True`` when a probe file could be written and removed, else ``False``.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f"{_PROBE_PREFIX}{os.getpid()}"
        probe.write_text("probe\n", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def resolve_repository() -> Path:
    """Return the checkout that holds the running package.

    Returns
    -------
    Path
        ``REDSUN_AHT_REPO`` when the launcher set it, else the checkout whose
        ``pyproject.toml`` contains this module, else the current directory.
    """
    override = os.environ.get(REPOSITORY_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


def resolve_data_root(repository: Path) -> tuple[Path, str | None]:
    """Return the writable data root and a note when the default was replaced.

    ``REDSUN_DATA_ROOT`` is treated as an explicit request and is never
    silently replaced. The default ``R:/redsun_aht_data`` falls back to
    ``<repository>/.redsun_data`` when it refuses writes, and the returned note
    says so.

    Parameters
    ----------
    repository : Path
        Checkout used for the repository-relative fallback.

    Returns
    -------
    tuple[Path, str | None]
        The resolved root and an optional fallback note.
    """
    configured = os.environ.get("REDSUN_DATA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser(), None
    if writable(DEFAULT_DATA_ROOT):
        return DEFAULT_DATA_ROOT, None
    fallback = repository / ".redsun_data"
    note = f"{DEFAULT_DATA_ROOT} is not writable; using repository fallback"
    return fallback, note


def resolve_temp_root(repository: Path) -> tuple[Path | None, str | None]:
    """Return a short, writable temporary root for the native PVA client.

    ``p4p`` builds its event base in a local socket under ``TEMP``/``TMP``.
    Both a short path and write access are required, so each candidate is
    length-checked and probed before it is accepted.

    Parameters
    ----------
    repository : Path
        Checkout used for the repository-relative candidates.

    Returns
    -------
    tuple[Path | None, str | None]
        The accepted root, or ``None`` with a diagnostic when none qualified.
    """
    candidates = (
        Path(os.environ.get("TMP") or os.environ.get("TEMP") or "."),
        repository.parent.parent / ".pva-tmp",
        repository / ".pva-tmp",
        Path((os.environ.get("SYSTEMDRIVE") or "C:") + "/pvatmp"),
        Path(os.environ.get("USERPROFILE", "C:/")) / "pva-tmp",
    )
    for candidate in candidates:
        if len(str(candidate)) > MAX_PVA_TEMP:
            continue
        if writable(candidate):
            return candidate, None
    diagnostic = (
        "no short writable temporary root found; p4p cannot build its event base"
    )
    return None, diagnostic


def resolve_napari_cache(repository: Path) -> Path:
    """Return the directory napari should use for its per-environment cache.

    Parameters
    ----------
    repository : Path
        Checkout used for the default cache location.

    Returns
    -------
    Path
        ``REDSUN_NAPARI_CACHE`` when set, else ``<repository>/.napari_cache``.
    """
    configured = os.environ.get("REDSUN_NAPARI_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return repository / ".napari_cache"


def resolve_redsun_data_root(data_root: Path) -> Path:
    """Return where the Redsun framework would write its user data.

    The launcher patches ``redsun.log.user_data_dir`` at startup, so a launcher
    run reports the relocated root and a bare ``aht`` run reports the framework
    default (``%LOCALAPPDATA%/redsun`` on Windows), which is the diagnostic
    ``--check`` needs.

    Parameters
    ----------
    data_root : Path
        Root to report when Redsun cannot be imported.

    Returns
    -------
    Path
        Redsun's resolved user data directory.
    """
    try:
        import redsun.log as log_module
    except ImportError:
        return data_root
    resolver: Any = getattr(log_module, "user_data_dir", None)
    if not callable(resolver):
        return data_root
    try:
        return Path(resolver("redsun", appauthor=False))
    except Exception:
        return data_root


def check_environment() -> int:
    """Report every launcher path and whether it accepts a write.

    Returns
    -------
    int
        ``0`` when every path is writable, ``1`` otherwise.
    """
    repository = resolve_repository()
    data_root, data_note = resolve_data_root(repository)
    temp_root, temp_note = resolve_temp_root(repository)
    napari_cache = resolve_napari_cache(repository)
    redsun_root = resolve_redsun_data_root(data_root)

    rows: list[tuple[str, Path | None, str | None]] = [
        ("repository", repository, None),
        ("data root", data_root, data_note),
        ("temp root", temp_root, temp_note),
        ("napari cache", napari_cache, None),
        ("redsun data", redsun_root, None),
    ]
    healthy = True
    for label, path, note in rows:
        if path is None:
            print(f"{label:<12}  {note or 'unavailable'}")
            healthy = False
            continue
        accepts = writable(path)
        if label == "temp root":
            detail = f"  ({len(str(path))} chars, short and writable)"
        elif note is not None:
            detail = f"  ({note})"
        else:
            detail = ""
        state = "writable" if accepts else "NOT WRITABLE"
        print(f"{label:<12}  {path}  {state}{detail}")
        healthy = healthy and accepts

    if not healthy:
        print(
            "hint: run scripts/run_session.ps1 -Check so the wrapper prepares "
            "these paths",
            file=sys.stderr,
        )
        return 1
    return 0


def load_registry() -> Registry:
    """Return the configured sessions and services, tolerating an absent registry.

    The coordinator owns ``redsun_aht.configurations`` and may not have added
    ``registry()`` yet. A missing function, a missing module or a mapping that
    cannot be read yields an empty :class:`Registry` instead of raising, so the
    launcher is usable before the registry lands.

    Returns
    -------
    Registry
        The session and service mappings, empty when none are configured.

    Notes
    -----
    The coordinator implements the registry in ``redsun_aht.configurations``
    as ``registry() -> tuple[Mapping[str, Entry], Mapping[str, Entry]]``,
    where each entry is ``Callable[[Sequence[str]], int | None]`` and receives
    the arguments after ``--``. The package must import Qt and napari lazily,
    because ``aht`` imports it for ``--list`` and for dispatch alike.
    """
    try:
        module = import_module("redsun_aht.configurations")
    except ImportError:
        return Registry()
    factory: Any = getattr(module, "registry", None)
    if not callable(factory):
        return Registry()
    try:
        sessions, services = factory()
        return Registry(sessions=dict(sessions), services=dict(services))
    except Exception as error:
        print(
            f"aht: warning: the session registry could not be loaded: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return Registry()


def _known(entries: Mapping[str, Entry]) -> str:
    """Return a readable list of registered names for an error message."""
    return ", ".join(sorted(entries)) if entries else "none registered yet"


def _list_entries(label: str, entries: Mapping[str, Entry]) -> None:
    """Print one registry section, one named entry per line."""
    print(f"{label}s")
    if not entries:
        print(f"  no {label}s registered yet")
        return
    for name in sorted(entries):
        summary = (entries[name].__doc__ or "").strip().splitlines()
        print(f"  {name:<24} {summary[0] if summary else 'no description'}")


def _dispatch(
    kind: str,
    name: str,
    arguments: Sequence[str],
    entries: Mapping[str, Entry],
) -> int:
    """Run one registered entry and return its exit status."""
    entry = entries.get(name)
    if entry is None:
        print(
            f"aht: error: unknown {kind} {name!r}; known {kind}s: {_known(entries)}",
            file=sys.stderr,
        )
        return 2
    try:
        result = entry(list(arguments))
    except KeyboardInterrupt:
        print(f"aht: {kind} {name!r} interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(
            f"aht: {kind} {name!r} failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    if result is None:
        return 0
    try:
        return int(result)
    except (TypeError, ValueError):
        print(
            f"aht: {kind} {name!r} returned {result!r}, not an exit status",
            file=sys.stderr,
        )
        return 1


def _build_parser() -> argparse.ArgumentParser:
    """Return the ``aht`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="aht",
        description=(
            "Adaptive holotomography acquisition: sessions, services and checks."
        ),
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--check",
        action="store_true",
        help="report the launcher paths and whether each accepts a write",
    )
    actions.add_argument(
        "--list",
        action="store_true",
        dest="list_entries",
        help="list the configured sessions and services",
    )
    subparsers = parser.add_subparsers(dest="command")
    for kind, summary in (
        ("session", "run one configured session (usually a Qt application)"),
        ("service", "run one configured PVA service"),
    ):
        command = subparsers.add_parser(kind, help=summary, description=summary)
        command.add_argument("name", nargs="?", help=f"registered {kind} name")
        command.add_argument(
            "arguments",
            nargs=argparse.REMAINDER,
            help=f"arguments passed through to the {kind}; separate them with '--'",
        )
    recon = subparsers.add_parser(
        "reconstruct-tiff", help="reconstruct DPCT or IDT from TIFF"
    )
    recon.add_argument("--solver", choices=("dpct", "idt"), default="dpct")
    recon.add_argument("--sample", type=Path, required=True)
    recon.add_argument("--reference", type=Path)
    recon.add_argument("--config", type=Path, required=True)
    recon.add_argument("--output", type=Path, required=True)
    recon.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    recon.add_argument("--gpu-id", type=int, default=0)
    recon.add_argument("--method", choices=("tikhonov", "tv", "linear"))
    recon.add_argument("--tile-size", metavar="HEIGHT,WIDTH")
    recon.add_argument("--output-dtype", choices=("uint16", "float32"))
    recon.add_argument(
        "--median-output", type=Path, help="save the matched IDT reference as CYX"
    )
    recon.add_argument("--roi", metavar="Y,X,HEIGHT,WIDTH")
    recon.add_argument("--zero-policy", choices=("error", "neutral"), default="error")
    migration = subparsers.add_parser(
        "migrate-recon-config", help="migrate legacy DPCT YAML"
    )
    migration.add_argument("--solver", choices=("dpct",), default="dpct")
    migration.add_argument("--input", type=Path, required=True)
    migration.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``aht`` command line and return its exit status.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments without the program name. ``None`` uses
        ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``2`` for usage errors and ``1`` for runtime
        failures. Every failure is reported as a message, never a traceback.
    """
    parser = _build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        options = parser.parse_args(arguments)
    except SystemExit as exit_code:
        return int(exit_code.code) if isinstance(exit_code.code, int) else 1

    if options.check:
        return check_environment()
    if options.list_entries:
        registry = load_registry()
        _list_entries("session", registry.sessions)
        _list_entries("service", registry.services)
        return 0
    if options.command is None:
        parser.print_usage(sys.stderr)
        print(
            "aht: error: a command is required",
            file=sys.stderr,
        )
        return 2

    if options.command in {"reconstruct-tiff", "migrate-recon-config"}:
        try:
            import yaml

            if options.command == "reconstruct-tiff":
                common = (
                    options.sample,
                    options.reference,
                    options.config,
                    options.output,
                    options.roi,
                    options.zero_policy,
                )
                if options.solver == "idt":
                    from .idt_tiff import reconstruct as reconstruct_idt

                    if (
                        options.method not in (None, "linear")
                        or options.tile_size
                        or options.output_dtype == "uint16"
                    ):
                        raise ValueError(
                            "IDT supports the linear method and float32 output"
                        )
                    reconstruct_idt(
                        *common,
                        backend=options.backend,
                        gpu_id=options.gpu_id,
                        median_output=options.median_output,
                    )
                else:
                    from .dpct_tiff import reconstruct as reconstruct_dpct

                    if options.median_output is not None:
                        raise ValueError("--median-output is available for IDT")
                    reconstruct_dpct(
                        *common,
                        backend=options.backend,
                        gpu_id=options.gpu_id,
                        method=options.method,
                        tile_size=options.tile_size,
                        output_dtype=options.output_dtype or "uint16",
                    )
            else:
                from .dpct_tiff import migrate_config

                config = migrate_config(options.input)
                with options.output.open("x", encoding="utf-8") as stream:
                    yaml.safe_dump(config, stream, sort_keys=False)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            MemoryError,
            ImportError,
            yaml.YAMLError,
        ) as error:
            print(f"aht: error: {error}", file=sys.stderr)
            return 1
        print(options.output)
        return 0

    registry = load_registry()
    entries = registry.sessions if options.command == "session" else registry.services
    if options.name is None:
        print(
            f"aht: error: {options.command!r} needs a name; "
            f"known {options.command}s: {_known(entries)}",
            file=sys.stderr,
        )
        return 2
    return _dispatch(options.command, options.name, options.arguments, entries)


if __name__ == "__main__":
    raise SystemExit(main())
