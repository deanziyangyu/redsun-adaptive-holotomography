"""HoloSPIM operator desktop over the devices declared in its profile."""

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, NoReturn

from qtpy import QtCore, QtWidgets
from redsun import AsHook, AsPresenter, AsView, Link, Serves, links_between, slot
from redsun.presenter import PositionerPresenter
from redsun.qt import QtHook, QtSession
from redsun.view.qt.builtins import PositionerView

from ..device.esp300_stage import Esp300Stage
from ..device.pva_rotation import PvaRotationMotor
from ..hooks import NapariApplication
from ..operator_input.live_jog import LiveJog
from ..operator_input.xinput import XInputPoller
from ..presenter.acquisition import AcquisitionPresenter
from ..presenter.detector import DetectorPresenter
from ..presenter.holospim import HolospimPresenter
from ..presenter.led_patterns import LedPatternPresenter
from ..view.acquisition import AcquisitionView
from ..view.detector import DetectorView
from ..view.image import ImageView
from ..view.led_patterns import LedPatternView
from .profile import check_holospim_host, read_instrument_profile
from .wiring import (
    DetectorPreview,
    DetectorPreviewSource,
    wire_acquisition,
    wire_detector,
    wire_led_patterns,
    wire_locks,
    wire_preview,
)


class HolospimPositionerView(PositionerView):
    """Route foreground Xbox jogs to the configured Positioner axes."""

    def configure_jog(self, targets: dict[str, tuple[str, str]]) -> None:
        """Start polling for the discovered jog axes."""
        try:
            poller = XInputPoller()
        except OSError:
            return
        self._jog_targets = targets
        self._jog_devices = {device for device, _ in targets.values()}
        self._jog_pending: str | None = None
        self._jog_moving: set[str] = set()
        self._jog = LiveJog(
            poller.read, self._move_jog, self._stop_jog, halt_mcu=self._stop_jog,
            zdt_is_focus=True, supported_axes=targets,
        )
        self._jog_timer = QtCore.QTimer(self)
        self._jog_timer.setInterval(150)
        self._jog_timer.timeout.connect(self._poll_survey_jog)
        QtWidgets.QApplication.instance().applicationStateChanged.connect(
            self._survey_application_state
        )
        self._jog_timer.start()

    def _survey_application_state(self, state: QtCore.Qt.ApplicationState) -> None:
        if state != QtCore.Qt.ApplicationState.ApplicationActive:
            self._jog.stop()

    def _poll_survey_jog(self) -> None:
        if (self._jog_devices & self._held_boxes or not self.isVisible()
                or QtWidgets.QApplication.applicationState()
                != QtCore.Qt.ApplicationState.ApplicationActive):
            self._jog.stop()
            return
        self._jog.poll()

    def _move_jog(self, axis: str, delta: float) -> None:
        if (
            self._jog_moving or self._jog_pending
            or self._jog_devices & self._held_boxes
        ):
            return
        device, row = self._jog_targets[axis]
        self._jog_pending = device
        self.sig_move.emit(device, row, delta)

    def _stop_jog(self) -> None:
        if self._jog_pending is not None:
            device, self._jog_pending = self._jog_pending, None
            self.sig_stop_device.emit(device)

    @slot(signal="sig_moving")
    def set_moving(self, device: str, moving: bool) -> None:
        """Track whether an admitted jog still needs a stop on release."""
        super().set_moving(device, moving)
        if device not in getattr(self, "_jog_devices", ()):
            return
        if moving:
            self._jog_moving.add(device)
        else:
            self._jog_moving.discard(device)
            if self._jog_pending == device:
                self._jog_pending = None

    @slot(signal="sig_failed")
    def set_failed(self, device: str, message: str) -> None:
        """Disarm the controller after a rejected or failed move."""
        super().set_failed(device, message)
        if device == getattr(self, "_jog_pending", None):
            self._jog_pending = None
            self._jog.stop()

    @slot(signal="sig_locks_changed")
    def set_locked(self, names: frozenset[str]) -> None:
        """Disarm immediately when an acquisition takes a jog device."""
        super().set_locked(names)
        if getattr(self, "_jog_devices", set()) & self._held_boxes:
            self._jog.stop()


class HolospimSession(QtSession):
    """Camera, acquisition, positioner and 91-LED preview for HoloSPIM."""

    config: ClassVar[dict[str, object]] = {
        "presenters": {
            "det_ctrl": {"timeout": 2.0},
            "acq_ctrl": {},
            "holospim_ctrl": {},
            "positioner_ctrl": {},
            "led_pattern_ctrl": {"panel": "holospim"},
        },
        "views": {
            "det_widget": {},
            "acq_widget": {},
            "img_widget": {},
            "positioner_widget": {},
            "led_pattern_widget": {},
        },
    }

    napari: Annotated[AsHook[NapariApplication], Serves(QtHook.CREATE_APPLICATION)]
    det_ctrl: AsPresenter[DetectorPresenter]
    acq_ctrl: AsPresenter[AcquisitionPresenter]
    holospim_ctrl: AsPresenter[HolospimPresenter]
    positioner_ctrl: AsPresenter[PositionerPresenter]
    led_pattern_ctrl: AsPresenter[LedPatternPresenter]
    det_widget: AsView[DetectorView]
    acq_widget: AsView[AcquisitionView]
    img_widget: AsView[ImageView]
    positioner_widget: AsView[HolospimPositionerView]
    led_pattern_widget: AsView[LedPatternView]
    preview: DetectorPreview

    def wire(self) -> Iterator[Link]:
        """Route the acquisition, detector, and positioner controls."""
        self.preview = DetectorPreview()
        yield from wire_detector(self.det_ctrl, self.det_widget, self.img_widget)
        yield from wire_preview(self.det_ctrl, self.preview)
        yield from wire_acquisition(self.acq_ctrl, self.acq_widget)
        yield from wire_locks(
            self.acq_ctrl,
            self.det_ctrl,
            self.det_widget,
            self.positioner_ctrl,
            self.positioner_widget,
        )
        yield self.acq_ctrl.sig_locks_changed, self.img_widget.set_roi_locked
        yield from links_between(self.positioner_widget, self.positioner_ctrl)
        targets = {}
        survey = self.devices.get("survey")
        axes = self.positioner_ctrl.axes
        if isinstance(survey, Esp300Stage):
            box = survey.axis.name
            targets.update({axis: (box, axis) for axis in axes.get(box, {})})
        if "x28" in axes.get("mcu", {}):
            targets["zdt"] = ("mcu", "x28")
        rotation = self.devices.get("rotation")
        if (
            isinstance(rotation, PvaRotationMotor)
            and (rotation.rotation.simulated or rotation.enable_controller_jog)
            and "rotation" in axes.get("rotation", {})
        ):
            targets["roll"] = ("rotation", "rotation")
        if targets:
            self.positioner_widget.configure_jog(targets)
        yield from wire_led_patterns(self.led_pattern_ctrl, self.led_pattern_widget)
        if isinstance(self.img_widget, ImageView):
            for name in self.det_ctrl.detectors:
                self.img_widget.attach_preview(
                    name, DetectorPreviewSource(self.preview, name)
                )


def run_holospim(argv: Sequence[str]) -> NoReturn:
    """Launch the desktop with the selected profile and host overlays."""
    parser = argparse.ArgumentParser(prog="aht session holospim")
    parser.add_argument(
        "--profile", type=Path, help="Complete HoloSPIM session profile"
    )
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument(
        "--mock", action="store_true", help="Override the profile to use mock devices"
    )
    args = parser.parse_args(list(argv))
    packaged = Path(__file__).resolve().parents[1] / "profiles" / "holospim.yaml"
    try:
        config = [
            read_instrument_profile(path)
            for path in (args.profile or packaged, *args.overlay)
        ]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.mock:
        config.append({"mock": True})
    try:
        check_holospim_host(config)
    except ValueError as error:
        parser.error(str(error))
    HolospimSession(config).run()
