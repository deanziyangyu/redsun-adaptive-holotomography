"""Hardware-free Qt/Napari processing application composition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from redsun_aht.presenter import OfflineProcessingPresenter

if TYPE_CHECKING:
    from pathlib import Path


def build_processing_gui_presenter() -> OfflineProcessingPresenter:
    """Build the processing application service without importing Qt."""
    return OfflineProcessingPresenter()


def run_processing_gui(
    *,
    input_root: Path | None = None,
    output_root: Path | None = None,
    run_event_loop: bool = True,
) -> tuple[Any, Any]:
    """Launch the optional Napari view without constructing hardware devices."""
    try:
        from redsun_aht.view.processing import launch_processing_gui
    except ImportError as error:
        raise RuntimeError(
            "processing GUI is unavailable; install the pyqt or pyside extra"
        ) from error
    return launch_processing_gui(
        build_processing_gui_presenter(),
        input_root=None if input_root is None else str(input_root.resolve()),
        output_root=None if output_root is None else str(output_root.resolve()),
        run_event_loop=run_event_loop,
    )


__all__ = ["build_processing_gui_presenter", "run_processing_gui"]
