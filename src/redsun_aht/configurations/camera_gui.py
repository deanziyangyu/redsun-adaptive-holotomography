"""Composition for the optional camera-service Qt client."""

from __future__ import annotations

from typing import Any


def run_camera_acquisition_gui(prefix: str, *, run_event_loop: bool = True) -> Any:
    """Launch a GUI client for an already-running isolated camera service."""
    try:
        from redsun_aht.view.camera import launch_camera_acquisition_gui
    except ImportError as error:
        raise RuntimeError(
            "camera GUI is unavailable; install the pyqt or pyside extra"
        ) from error
    return launch_camera_acquisition_gui(prefix, run_event_loop=run_event_loop)


__all__ = ["run_camera_acquisition_gui"]
