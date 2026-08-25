"""Optional Qt view for one acknowledged frame from a camera service."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from redsun_aht.presenter.camera import (
    AcquiredCameraFrame,
    CameraGuiClient,
    EpicsCameraGuiClient,
)


class CameraAcquisitionWidget(QtWidgets.QWidget):
    """Explicit, single-frame GUI client for an already-running camera IOC."""

    def __init__(
        self,
        prefix: str,
        *,
        client: CameraGuiClient | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.prefix = prefix
        self.client = client or EpicsCameraGuiClient(prefix)
        self.last_capture: AcquiredCameraFrame | None = None
        self.last_frame: np.ndarray[Any, Any] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("AHT camera acquisition")
        layout = QtWidgets.QVBoxLayout(self)
        self.prefix_label = QtWidgets.QLabel(f"Service: {self.prefix}")
        layout.addWidget(self.prefix_label)
        self.capture_button = QtWidgets.QPushButton("Capture one frame")
        self.capture_button.clicked.connect(self.capture_once)
        layout.addWidget(self.capture_button)
        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.preview_label = QtWidgets.QLabel("No frame captured")
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_label)

    def capture_once(self) -> None:
        """Capture one frame, present its detached copy, and report integrity."""
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
    """Show the GUI client without creating camera SDK or MMCore objects."""
    application = QtWidgets.QApplication.instance()
    if not isinstance(application, QtWidgets.QApplication):
        application = QtWidgets.QApplication([])
    widget = CameraAcquisitionWidget(prefix)
    widget.show()
    if run_event_loop:
        application.exec()
    return widget


__all__ = ["CameraAcquisitionWidget", "launch_camera_acquisition_gui"]
