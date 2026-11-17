"""Sessions and services this bundle dispatches, loaded only when used.

``aht`` reads :func:`registry` for ``--list`` and for dispatch, on every run,
including one that only asks which commands exist. Importing this package must
therefore stay free of Qt, napari and ``pymmcore``: the registry entries and
the lazy attributes below import their module inside the call that needs it.
``tests/test_session_bundle.py`` pins that with a fresh interpreter.

The registry has two sessions, ``camera`` and ``dpct``, and the PVA
services the bundle serves. A service entry passes the arguments after ``--``
unchanged to that module's ``main``, so ``aht service mmcore-camera -- --prefix
AHT:CAM1:`` runs the same parser as running that module directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import Any

__all__ = [
    "CameraSession",
    "DetectorPresenter",
    "DetectorPreview",
    "DpctPresenter",
    "DpctSession",
    "DualCameraSession",
    "build_camera_session",
    "build_dpct_session",
    "registry",
    "run_camera",
    "run_dpct",
    "wire_detector",
    "wire_preview",
]

Entry = Callable[[Sequence[str]], int | None]
"""One dispatchable entry: a session or a service, given the arguments after it."""


def _run_camera(argv: Sequence[str]) -> int | None:
    """Run the editable PVA camera session, with one or two detectors."""
    from .camera import run_camera

    run_camera(argv)


def _run_dpct(argv: Sequence[str]) -> int | None:
    """Run the four-pattern DPCT session over the camera, stage and LED panel."""
    from .dpct import run_dpct

    run_dpct(argv)


def _run_arh(argv: Sequence[str]) -> int | None:
    """Run the ARH instrument desktop session."""
    from .arh import run_arh

    run_arh(argv)


def _run_holospim(argv: Sequence[str]) -> int | None:
    """Run the HoloSPIM instrument desktop."""
    from .holospim import run_holospim

    run_holospim(argv)


def _run_rotation(argv: Sequence[str]) -> int | None:
    """Run the rotation motor widget."""
    from .rotation import run_rotation

    run_rotation(argv)


def _serve_mmcore_camera(argv: Sequence[str]) -> int | None:
    """Serve one Micro-Manager camera over PVAccess."""
    from ..services.mmcore_camera import main

    return main(list(argv))


def _run_profile(argv: Sequence[str]) -> int:
    """Build a headless ARH or HoloSPIM profile."""
    from .profile import run_profile

    return run_profile(argv)


def _serve_lx4000(argv: Sequence[str]) -> int:
    """Serve the ASI LX4000 XYZ stage."""
    from ..robotic.services.asi_stage import main

    return main(list(argv))


def _serve_lumencor(argv: Sequence[str]) -> int:
    """Serve Lumencor Spectra illumination through MMCore."""
    from ..robotic.services.lumencor_spectra import main

    return main(list(argv))


def _serve_emmsh(argv: Sequence[str]) -> int:
    """Serve the Due EMMSH controller."""
    from ..robotic.services.unified_emmsh import main

    return main(list(argv))


def _serve_unified_mcu(argv: Sequence[str]) -> int:
    """Serve the HoloSPIM ESP32-S3 controller."""
    from ..services.unified_mcu import main

    return main(list(argv))


def _serve_spinnaker_camera(argv: Sequence[str]) -> int | None:
    """Serve one camera through the separately installed Spinnaker SDK."""
    from ..services.spinnaker_camera import main

    return main(list(argv))


def _serve_mmcore_stage(argv: Sequence[str]) -> int | None:
    """Serve one Micro-Manager stage over PVAccess."""
    from ..services.mmcore_stage import main

    return main(list(argv))


def _serve_arduino_led(argv: Sequence[str]) -> int | None:
    """Serve the MM-compatible Arduino LED service over PVAccess."""
    from ..services.arduino_led import main

    return main(list(argv))


def _serve_workcell_webcam(argv: Sequence[str]) -> int:
    """Serve the OpenCV workcell recorder over PVAccess."""
    from ..robotic.services.workcell_webcam import main

    return main(list(argv))


_SESSIONS: Mapping[str, Entry] = {
    "camera": _run_camera,
    "dpct": _run_dpct,
    "arh": _run_arh,
    "holospim": _run_holospim,
    "rotation": _run_rotation,
    "profile": _run_profile,
}

_SERVICES: Mapping[str, Entry] = {
    "workcell-webcam": _serve_workcell_webcam,
    "mmcore-camera": _serve_mmcore_camera,
    "spinnaker-camera": _serve_spinnaker_camera,
    "mmcore-stage": _serve_mmcore_stage,
    "arduino-led": _serve_arduino_led,
    "asi-lx4000": _serve_lx4000,
    "lumencor-spectra": _serve_lumencor,
    "unified-emmsh": _serve_emmsh,
    "unified-mcu": _serve_unified_mcu,
}

_LAZY: Mapping[str, tuple[str, str]] = {
    "CameraSession": (".camera", "CameraSession"),
    "DetectorPresenter": (".camera", "DetectorPresenter"),
    "DetectorPreview": (".wiring", "DetectorPreview"),
    "DpctPresenter": (".dpct", "DpctPresenter"),
    "DpctSession": (".dpct", "DpctSession"),
    "DualCameraSession": (".camera", "DualCameraSession"),
    "build_camera_session": (".camera", "build_camera_session"),
    "build_dpct_session": (".dpct", "build_dpct_session"),
    "run_camera": (".camera", "run_camera"),
    "run_dpct": (".dpct", "run_dpct"),
    "wire_detector": (".wiring", "wire_detector"),
    "wire_preview": (".wiring", "wire_preview"),
}


def registry() -> tuple[Mapping[str, Entry], Mapping[str, Entry]]:
    """Return the sessions and services this checkout dispatches.

    Returns
    -------
    tuple[Mapping[str, Entry], Mapping[str, Entry]]
        ``(sessions, services)``, each keyed by the name ``aht`` takes. Fresh
        mappings, so a caller cannot change what the next lookup sees.
    """
    return dict(_SESSIONS), dict(_SERVICES)


def __getattr__(name: str) -> Any:
    """Load a session, presenter or wiring helper only when it is used.

    Parameters
    ----------
    name : str
        Public attribute requested from this package.

    Returns
    -------
    Any
        The attribute from its defining module.

    Raises
    ------
    AttributeError
        If *name* is not one of the lazy names above.
    """
    try:
        module_name, attribute = _LAZY[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(f"{__name__}{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the registry and the lazily exported names for introspection."""
    return sorted({*globals(), *_LAZY})
