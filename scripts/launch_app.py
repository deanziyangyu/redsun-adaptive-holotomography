"""Prepare the AHT environment, then hand over to the ``aht`` command line.

Adapted from the IX83 imaging bundle's launcher (``redsun-searching-is-imaging``,
revision ``3a3bf06``, Apache-2.0). It relocates napari's theme cache, relocates
the Redsun data root (Redsun takes it from ``platformdirs.user_data_dir``, not
from an environment variable), and makes sure ``TEMP``/``TMP`` are a short,
writable root because the native PVA client builds its event base there.

The redirection is deliberately not a global patch of ``platformdirs``: the
Micro-Manager camera service needs the real local app data directory to find its
device-interface adapter bundle.

Run it through ``scripts/run_session.ps1``, or directly::

    python scripts/launch_app.py --check
    python scripts/launch_app.py session camera-gui --prefix AHT:CAM:PVCAM:
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: Where the redirected napari cache lives unless REDSUN_NAPARI_CACHE says otherwise.
DEFAULT_CACHE = REPO / ".napari_cache"

#: Where Redsun's data root lives unless REDSUN_DATA_ROOT says otherwise.
DEFAULT_DATA = Path("R:/redsun_aht_data")

#: Longest temporary root ``p4p`` reliably builds its event base on.
MAX_PVA_TEMP = 90

_PROBE_PREFIX = ".aht-write-probe-"


def writable(path: Path) -> bool:
    """Return whether a probe file can be created in *path*, creating it if needed.

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


def cache_directory() -> Path:
    """Return the cache directory napari should use.

    Returns
    -------
    Path
        ``REDSUN_NAPARI_CACHE`` when set, else ``.napari_cache`` in this checkout.
    """
    override = os.environ.get("REDSUN_NAPARI_CACHE", "").strip()
    return Path(override).expanduser() if override else DEFAULT_CACHE


def redirect_napari_cache() -> Path:
    """Point napari's cache at a writable directory before its themes are built.

    ``napari.utils.theme`` registers the built-in themes at import time and
    rebuilds their SVG icons into napari's per-environment cache. A cache that
    refuses writes kills the process before any window exists.

    Returns
    -------
    Path
        The directory napari now uses.
    """
    target = cache_directory()
    target.mkdir(parents=True, exist_ok=True)

    try:
        import napari.utils._platformdirs as napari_platformdirs
    except ImportError:
        # Headless installs without the pyqt/pyside extra still get the
        # environment prepared; there is simply no napari cache to redirect.
        return target

    napari_platformdirs.user_cache_dir = lambda *args, **kwargs: str(target)
    # A consumer imported before this patch keeps its own bound reference, so
    # refresh it too rather than trusting import order.
    icons = sys.modules.get("napari.resources._icons")
    if icons is not None:
        icons.user_cache_dir = napari_platformdirs.user_cache_dir
    return target


def redirect_redsun_data() -> tuple[Path, str | None]:
    """Move Redsun's logs, settings and session storage under one writable root.

    ``REDSUN_DATA_ROOT`` is an explicit request and is used as given. The
    default ``R:/redsun_aht_data`` falls back to ``<repository>/.redsun_data``
    when it refuses writes, and the returned note says so.

    Returns
    -------
    tuple[Path, str | None]
        The root in force and an optional fallback note.
    """
    configured = os.environ.get("REDSUN_DATA_ROOT", "").strip()
    note: str | None = None
    if configured:
        root = Path(configured).expanduser()
    else:
        root = DEFAULT_DATA
        if not writable(root):
            root = REPO / ".redsun_data"
            note = f"{DEFAULT_DATA} is not writable; using repository fallback"
    root.mkdir(parents=True, exist_ok=True)

    import redsun._settings as settings_module
    import redsun.log as log_module
    import redsun.path_provider as path_provider_module

    def relocated(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return str(root)

    log_module.user_data_dir = relocated
    path_provider_module.user_data_dir = relocated
    settings_module.user_config_dir = relocated
    return root, note


def ensure_pva_temp() -> Path | None:
    """Make ``TEMP``/``TMP`` a short *and* writable root for ``p4p``.

    ``p4p`` builds its event base on a local socket under the temp directory.
    If that directory cannot be written, or its path is long, the service dies
    before it is ready with ``RuntimeError: event_base_new() fails``.

    Returns
    -------
    Path | None
        The root now in force, or ``None`` when no candidate qualified.
    """
    candidates = (
        Path(os.environ.get("TMP") or os.environ.get("TEMP") or "."),
        REPO.parent.parent / ".pva-tmp",
        REPO / ".pva-tmp",
        Path((os.environ.get("SYSTEMDRIVE") or "C:") + "/pvatmp"),
        Path(os.environ.get("USERPROFILE", "C:/")) / "pva-tmp",
    )
    for candidate in candidates:
        if len(str(candidate)) > MAX_PVA_TEMP:
            continue
        if not writable(candidate):
            continue
        os.environ["TEMP"] = str(candidate)
        os.environ["TMP"] = str(candidate)
        return candidate
    return None


def check() -> int:
    """Report the paths this launcher uses and whether they accept a write.

    Returns
    -------
    int
        ``0`` when every path is writable and the plain PVA client opens,
        ``1`` otherwise.
    """
    try:
        redirect_napari_cache()
        redirect_redsun_data()
        ensure_pva_temp()
    except OSError as error:
        print(f"launcher       cannot prepare a writable root: {error}")
        return 1

    print(f"python        {sys.executable}")

    from redsun_aht.__main__ import check_environment

    status = check_environment()

    theme: Path | None = None
    svg_count = 0
    try:
        import napari.resources._icons as icons

        # Importing the theme module is what rebuilds the SVGs, so force it
        # here rather than reporting a directory that is still empty.
        import napari.utils.theme  # noqa: F401  (imported for its side effect)

        theme = icons._theme_path("dark")
        svg_count = len(list(theme.glob("*.svg")))
    except ImportError:
        print("napari theme  not installed; the GUI extras are absent")
    except Exception as error:
        print(f"napari theme  FAILS: {type(error).__name__}: {error}")
        status = 1
    else:
        print(f"napari theme  {theme}  ({svg_count} svg)")

    # Open a real handler: this is the exact call that fails when the data root
    # is unwritable, so a pass here is the check that matters.
    try:
        from redsun.log import SessionFileHandler

        handler = SessionFileHandler("launch-check")
    except Exception as error:
        print(f"session log   FAILS: {type(error).__name__}: {error}")
        status = 1
    else:
        try:
            print(f"session log   {handler.baseFilename}")
        finally:
            handler.close()

    # And the other call that fails in every service before it is ready.
    try:
        from p4p.client.raw import Context

        context = Context("pva", conf={"EPICS_PVA_ADDR_LIST": "127.0.0.1"})
    except Exception as error:
        print(f"pva           event base FAILS: {type(error).__name__}: {error}")
        return 1
    try:
        print("pva           event base ready")
    finally:
        context.close()
    return status


def main(argv: list[str] | None = None) -> int:
    """Prepare the environment and run the requested AHT command.

    Parameters
    ----------
    argv : list[str] | None
        Arguments without the program name. ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    int
        The application's exit status, or ``1`` when the environment could not
        be prepared. Failures are printed, not raised.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Only a leading `--check` is the launcher's own check; a later one belongs
    # to the command being forwarded (for example `session <name> -- --check`).
    if arguments[:1] == ["--check"]:
        return check()

    try:
        cache = redirect_napari_cache()
        data, note = redirect_redsun_data()
        temp = ensure_pva_temp()
    except OSError as error:
        print(f"aht launcher: cannot prepare a writable root: {error}", file=sys.stderr)
        return 1

    source = REPO / "src"
    if source.is_dir():
        sys.path.insert(0, str(source))
    os.environ["REDSUN_DATA_ROOT"] = str(data)
    os.environ.setdefault("REDSUN_AHT_REPO", str(REPO))

    if not writable(cache):
        print(f"aht launcher: napari cache {cache} is not writable", file=sys.stderr)
        return 1
    if not writable(data):
        print(f"aht launcher: data root {data} is not writable", file=sys.stderr)
        return 1
    if temp is None:
        print(
            "aht launcher: no short writable temporary root found; "
            "p4p cannot build its event base",
            file=sys.stderr,
        )
        return 1

    from redsun_aht.__main__ import main as aht_main

    print(f"napari cache -> {cache}", file=sys.stderr)
    print(f"redsun data  -> {data}", file=sys.stderr)
    print(f"temp         -> {temp}", file=sys.stderr)
    if note is not None:
        print(f"warning      -> {note}", file=sys.stderr)
    return int(aht_main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
