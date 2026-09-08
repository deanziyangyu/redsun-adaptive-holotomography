"""Optional Qt view for one acknowledged frame from a camera service."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from redsun.view import ViewPosition
from redsun.view.qt import QtView

from redsun_aht.presenter.camera import (
    AcquiredCameraFrame,
    CameraGuiClient,
    EpicsCameraGuiClient,
    EpicsLiveCameraGuiClient,
    LiveCameraGuiClient,
)


class CameraAcquisitionWidget(QtView):
    """Explicit, single-frame GUI client for an already-running camera IOC."""

    def __init__(
        self,
        name: str,
        /,
        *,
        prefix: str | None = None,
        client: CameraGuiClient | None = None,
        live_client: LiveCameraGuiClient | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if prefix is None:
            prefix = name
            name = "camera-acquisition"
        super().__init__(name)
        if parent is not None:
            self.setParent(parent)
        self.prefix = prefix
        self.client = client or EpicsCameraGuiClient(prefix)
        self.live_client = live_client or EpicsLiveCameraGuiClient(prefix)
        self.last_capture: AcquiredCameraFrame | None = None
        self.last_frame: np.ndarray[Any, Any] | None = None
        self._live_timer = QtCore.QTimer(self)
        self._live_timer.setInterval(1000 // 60)
        self._live_timer.timeout.connect(self._poll_live)
        self._build_ui()

    @property
    def view_position(self) -> ViewPosition:
        """Place the camera control and preview in the central workspace."""
        return ViewPosition.CENTER

    def _build_ui(self) -> None:
        self.setWindowTitle("AHT camera acquisition")
        layout = QtWidgets.QVBoxLayout(self)
        self.prefix_label = QtWidgets.QLabel(f"Service: {self.prefix}")
        layout.addWidget(self.prefix_label)
        self.capture_button = QtWidgets.QPushButton("Capture one frame")
        self.capture_button.clicked.connect(self.capture_once)
        layout.addWidget(self.capture_button)
        self.start_live_button = QtWidgets.QPushButton("Start live view")
        self.start_live_button.clicked.connect(self.start_live)
        layout.addWidget(self.start_live_button)
        self.stop_live_button = QtWidgets.QPushButton("Stop live view")
        self.stop_live_button.clicked.connect(self.stop_live)
        self.stop_live_button.setEnabled(False)
        layout.addWidget(self.stop_live_button)
        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.preview_label = QtWidgets.QLabel("No frame captured")
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_label)

    def capture_once(self) -> None:
        """Capture one explicit snap, then present its detached verified copy."""
        self.capture_button.setEnabled(False)
        self.status_label.setText("Acquiring one frameâ€¦")
        try:
            frame = self.client.acquire_one()
            self.last_capture = frame
            self.last_frame = frame.array
            self._present(frame.array)
            self.status_label.setText(
                f"Captured {frame.service_id} sequence {frame.sequence}; "
                f"SHA-256 {frame.checksum}"
            )
        except Exception as error:
            self.status_label.setText(f"Failed: {type(error).__name__}: {error}")
        finally:
            self.capture_button.setEnabled(True)

    def start_live(self) -> None:
        """Start one sequence-backed, latest-wins service live view."""
        self.start_live_button.setEnabled(False)
        self.status_label.setText("Starting sequence-backed live view…")
        try:
            self.live_client.start()
            self.capture_button.setEnabled(False)
            self.stop_live_button.setEnabled(True)
            self._live_timer.start()
            self.status_label.setText(
                "Live view running (EPICS latest frame, max 60 Hz)"
            )
        except Exception as error:
            self.status_label.setText(
                f"Live view failed: {type(error).__name__}: {error}"
            )
            self.start_live_button.setEnabled(True)

    def stop_live(self) -> None:
        """Stop the live sequence and re-enable explicit single-frame capture."""
        self._live_timer.stop()
        try:
            self.live_client.stop()
            self.status_label.setText("Live view stopped")
        except Exception as error:
            self.status_label.setText(
                f"Live stop failed: {type(error).__name__}: {error}"
            )
        finally:
            self.capture_button.setEnabled(True)
            self.start_live_button.setEnabled(True)
            self.stop_live_button.setEnabled(False)

    def _poll_live(self) -> None:
        """Render at most one latest frame per GUI timer tick."""
        try:
            frame = self.live_client.read_latest()
            if frame is None:
                return
            self.last_frame = frame.array
            self._present(frame.array)
            self.status_label.setText(f"Live sequence {frame.sequence}")
        except Exception as error:
            self._live_timer.stop()
            self.capture_button.setEnabled(True)
            self.start_live_button.setEnabled(True)
            self.stop_live_button.setEnabled(False)
            self.status_label.setText(
                f"Live view failed: {type(error).__name__}: {error}"
            )

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        """Release the service-owned sequence when the optional viewer closes."""
        if self._live_timer.isActive():
            self.stop_live()
        super().closeEvent(event)

    def _present(self, array: np.ndarray[Any, Any]) -> None:
        image = np.asarray(array)
        if image.ndim != 2:
            raise ValueError(f"camera GUI accepts one 2D frame, got {image.shape!r}")
        minimum = float(np.min(image))
        maximum = float(np.max(image))
        if maximum > minimum:
            preview = np.asarray((image - minimum) * (255.0 / (maximum - minimum)))
        else:
            preview = np.zeros(image.shape, dtype=np.uint8)
        preview = np.ascontiguousarray(preview, dtype=np.uint8)
        qimage = QtGui.QImage(
            preview.tobytes(),
            preview.shape[1],
            preview.shape[0],
            preview.strides[0],
            QtGui.QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )


def launch_camera_acquisition_gui(
    prefix: str, *, run_event_loop: bool = True
) -> CameraAcquisitionWidget:
    """Show a RedSun-composed client without creating SDK/MMCore objects."""
    from typing import cast

    from redsun.containers import declare_view
    from redsun.qt import QtAppContainer

    from redsun_aht.configurations.profiles import profile_path

    class AHTCameraContainer(QtAppContainer, config=profile_path("camera-gui")):
        camera_view = declare_view(CameraAcquisitionWidget, prefix=prefix)

    container = AHTCameraContainer(
        session="AHT camera acquisition", frontend="pyqt"
    ).build()
    widget = cast("CameraAcquisitionWidget", container.views["camera_view"])
    widget._redsun_container = container  # type: ignore[attr-defined]
    widget.show()
    if run_event_loop:
        application = QtWidgets.QApplication.instance()
        if not isinstance(application, QtWidgets.QApplication):
            raise RuntimeError("RedSun Qt container did not create QApplication")
        try:
            application.exec()
        finally:
            container.shutdown()
    return widget


__all__ = ["CameraAcquisitionWidget", "launch_camera_acquisition_gui"]
