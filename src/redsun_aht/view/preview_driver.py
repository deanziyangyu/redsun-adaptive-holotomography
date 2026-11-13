"""Timer that samples the live preview buffer into its layer controller.

Ported from ``redsun_searching_is_imaging/view/preview_driver.py`` at donor
revision ``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed;
copyright the Redsun contributors. The timer, its cadence and the rule that a
repeated sequence is skipped by the controller are the donor's.

Changed for :mod:`redsun_aht`:

- The source is any :class:`PreviewSource` with a ``snapshot`` method, which
  :class:`redsun_aht.view.preview.LatestPreview` satisfies. The searching
  bundle's ``acquisition.live.PreviewBuffer`` is not part of this package.
- The module imports ``QtCore`` at import time and constructs no timer until a
  caller constructs a driver. Importing it never creates an application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from qtpy import QtCore

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from .preview import PreviewFrame, PreviewLayerController

__all__ = ["PreviewDriver", "PreviewSource"]


class PreviewSource(Protocol):
    """The part of a preview buffer this driver samples."""

    def snapshot(self) -> PreviewFrame | None:
        """Return the newest frame, or ``None`` when nothing arrived yet."""
        ...


class PreviewDriver(QtCore.QObject):
    """Refresh a preview layer at a fixed cadence on the Qt thread.

    The driver owns one timer and samples a thread-safe source, so a worker can
    keep offering frames while the viewer draws at 2 to 5 Hz. A repeated
    snapshot sequence is skipped by the controller, not by the timer.
    """

    def __init__(
        self,
        buffer: PreviewSource,
        controller: PreviewLayerController,
        *,
        interval_ms: int = 200,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if interval_ms < 1:
            raise ValueError("interval_ms must be at least one")
        self._buffer = buffer
        self._controller = controller
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.refresh)

    @property
    def interval_ms(self) -> int:
        """Refresh interval in milliseconds."""
        return self._timer.interval()

    @property
    def active(self) -> bool:
        """Whether the timer is running."""
        return self._timer.isActive()

    def start(self) -> None:
        """Start refreshing."""
        self._timer.start()

    def stop(self) -> None:
        """Stop refreshing."""
        self._timer.stop()

    def refresh(self) -> bool:
        """Draw one snapshot now and return whether the layer changed."""
        return self._controller.refresh(self._buffer.snapshot())
