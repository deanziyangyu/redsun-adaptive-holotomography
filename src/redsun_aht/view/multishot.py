"""Qt controls for an explicitly bounded finite four-shot list scan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from qtpy import QtCore, QtWidgets


@dataclass(frozen=True, slots=True)
class MultishotScanRequest:
    """User-selected finite scan intent independent of hardware composition."""

    z_positions: int = 40
    z_step_um: float = 1.0
    pattern_ids: tuple[int, int, int, int] = (101, 102, 103, 104)

    def __post_init__(self) -> None:
        """Keep every scan endpoint inside the current 100 µm admission."""
        if self.z_positions <= 0 or self.z_step_um <= 0:
            raise ValueError("Z positions and Z step must be positive")
        if len(self.pattern_ids) != 4 or len(set(self.pattern_ids)) != 4:
            raise ValueError("a multishot plan requires exactly four patterns")
        if (self.z_positions - 1) * self.z_step_um > 100:
            raise ValueError("scan exceeds the admitted 100 um Z displacement")

    @property
    def shot_count(self) -> int:
        """Return the exact finite detector-frame count per camera."""
        return self.z_positions * len(self.pattern_ids)


@dataclass(frozen=True, slots=True)
class MultishotScanSummary:
    """Human-readable output of a completed composed hardware scan."""

    run_id: str
    frame_count: int
    fixed_median_checksum: str
    frame_rate_hz: float
    configured_exposure_ms: str


class MultishotScanClient(Protocol):
    """Application-composed scan authority consumed by the Qt surface."""

    def run(self, request: MultishotScanRequest) -> MultishotScanSummary:
        """Execute the explicit request and return only completed-run evidence."""


class _ScanSignals(QtCore.QObject):
    completed = QtCore.Signal(object)  # type: ignore[attr-defined]
    failed = QtCore.Signal(str)  # type: ignore[attr-defined]


class _ScanTask(QtCore.QRunnable):
    """Keep a potentially long physical acquisition off the Qt event loop."""

    def __init__(self, client: MultishotScanClient, request: MultishotScanRequest):
        super().__init__()
        self.client = client
        self.request = request
        self.signals = _ScanSignals()

    def run(self) -> None:
        """Execute the injected authority and marshal its result back to Qt."""
        try:
            self.signals.completed.emit(self.client.run(self.request))
        except Exception as error:
            self.signals.failed.emit(f"{type(error).__name__}: {error}")


class MultishotScanWidget(QtWidgets.QWidget):
    """User-operable finite-scan panel with no implicit hardware construction."""

    def __init__(
        self,
        client: MultishotScanClient,
        *,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.client = client
        self._task: _ScanTask | None = None
        thread_pool = QtCore.QThreadPool.globalInstance()
        if thread_pool is None:  # pragma: no cover - Qt runtime invariant
            raise RuntimeError("Qt global thread pool is unavailable")
        self._thread_pool: QtCore.QThreadPool = thread_pool
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("AHT four-shot list scan")
        layout = QtWidgets.QFormLayout(self)
        self.patterns_label = QtWidgets.QLabel("101, 102, 103, 104 (red panel91)")
        self.z_positions = QtWidgets.QSpinBox()
        self.z_positions.setRange(1, 101)
        self.z_positions.setValue(40)
        self.z_step_um = QtWidgets.QDoubleSpinBox()
        self.z_step_um.setRange(0.001, 100.0)
        self.z_step_um.setDecimals(3)
        self.z_step_um.setValue(1.0)
        self.run_button = QtWidgets.QPushButton("Run four-shot scan")
        self.run_button.clicked.connect(self.run_scan)
        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setWordWrap(True)
        layout.addRow("Pattern IDs", self.patterns_label)
        layout.addRow("Z positions", self.z_positions)
        layout.addRow("Z step (um)", self.z_step_um)
        layout.addRow(self.run_button)
        layout.addRow("Status", self.status_label)

    def run_scan(self) -> None:
        """Validate the bounded request, then run it on the Qt worker pool."""
        if self._task is not None:
            return
        try:
            request = MultishotScanRequest(
                z_positions=self.z_positions.value(),
                z_step_um=self.z_step_um.value(),
            )
        except ValueError as error:
            self.status_label.setText(f"Invalid: {error}")
            return
        task = _ScanTask(self.client, request)
        task.signals.completed.connect(self._completed)
        task.signals.failed.connect(self._failed)
        self._task = task
        self.run_button.setEnabled(False)
        self.status_label.setText(
            f"Running {request.shot_count} frames; Z extent "
            f"{(request.z_positions - 1) * request.z_step_um:g} um"
        )
        self._thread_pool.start(task)

    def _completed(self, summary: object) -> None:
        if not isinstance(summary, MultishotScanSummary):
            self._failed("runner returned an invalid multishot summary")
            return
        self.status_label.setText(
            f"Completed {summary.run_id}: {summary.frame_count} frames; "
            f"{summary.frame_rate_hz:.3g} Hz; exposure "
            f"{summary.configured_exposure_ms} ms; median "
            f"{summary.fixed_median_checksum}"
        )
        self._finish()

    def _failed(self, message: str) -> None:
        self.status_label.setText(f"Failed: {message}")
        self._finish()

    def _finish(self) -> None:
        self._task = None
        self.run_button.setEnabled(True)


__all__ = [
    "MultishotScanClient",
    "MultishotScanRequest",
    "MultishotScanSummary",
    "MultishotScanWidget",
]
