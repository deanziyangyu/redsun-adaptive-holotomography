"""ARH desktop session over an explicitly selected instrument profile."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, NoReturn

from redsun import AsHook, AsPresenter, AsView, Link, Serves, links_between, slot
from redsun.aio import run_coro
from redsun.presenter import PositionerPresenter
from redsun.qt import Dock, QtHook, QtSession
from redsun.qt.styles.napari import NapariStyle
from redsun.view.qt.builtins import PositionerView
from qtpy import QtCore, QtWidgets

from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.hooks import NapariApplication
from redsun_aht.operator_input.xinput import XInputPoller
from redsun_aht.operator_input.live_jog import JOG_STEPS, ROTARY_JOG_STEPS, LiveJog
from redsun_aht.presenter.acquisition import AcquisitionPresenter
from redsun_aht.presenter.arh import ArhPresenter
from redsun_aht.presenter.led_patterns import LedPatternPresenter
from redsun_aht.presenter.detector import DetectorPresenter, LIVE_STREAM
from redsun_aht.protocols import LiveFlyer
from redsun_aht.view.acquisition import AcquisitionView
from redsun_aht.robotic.device.auxiliary import LumencorSpectra, UnifiedEmmsh
from redsun_aht.robotic.device.asi._service import AsiFilterWheel, AsiSerialStage
from redsun_aht.robotic.device.workcell_webcam import WorkcellWebcam
from redsun_aht.view.arh import ArhControls
from redsun_aht.view.led_patterns import LedPatternView
from redsun_aht.view.detector import DetectorView
from redsun_aht.view.image import ImageView

from .arh_pva import arh_direct_pva
from .profile import ARH_SCRATCH_ROOT, ARH_WORKSPACE_PROFILE, read_arh_profile, without_arh_due
from .wiring import (
    DetectorPreview,
    DetectorPreviewSource,
    wire_acquisition,
    wire_detector,
    wire_led_patterns,
    wire_preview,
)


class ArhPositionerView(PositionerView):
    """Display the LX4000 focus axis as F while retaining its Z device key."""

    placement = Dock("right", group="arh-controls")

    def configure_live_jog(self, device: str | None, zdt_device: str | None) -> None:
        try:
            poller = XInputPoller()
        except OSError:
            return
        self._live_jog = LiveJog(
            poller.read,
            lambda axis, delta: self._move_live_jog(device, zdt_device, axis, delta),
            lambda: self.sig_stop_device.emit(device) if device is not None else None,
            self._show_live_jog_step,
            lambda: self.sig_stop_device.emit(zdt_device) if zdt_device is not None else None,
        )
        self._jog_timer = QtCore.QTimer(self)
        self._jog_timer.setInterval(150)
        self._jog_timer.timeout.connect(self._poll_live_jog)
        app = QtWidgets.QApplication.instance()
        app.applicationStateChanged.connect(self._live_jog_application_state)
        self._show_live_jog_step(JOG_STEPS[0])
        self._jog_timer.start()

    def _move_live_jog(self, stage: str | None, mcu: str | None, axis: str, delta: float) -> None:
        device = mcu if axis in ("zdt", "roll", "pitch", "yaw") else stage
        if device is not None:
            self.sig_move.emit(device, axis, delta)

    def _show_live_jog_step(self, step: float) -> None:
        self.setToolTip(
            f"Xbox step: {step:g} µm / {ROTARY_JOG_STEPS[JOG_STEPS.index(step)]:g}°. LB/RB change step. "
            "Hold A: left stick XY, D-pad focus/ZDT, right stick roll/pitch, triggers yaw. "
            "Releasing A or B stops active rotary jogging and latches Due stop."
        )

    @QtCore.Slot(QtCore.Qt.ApplicationState)
    def _live_jog_application_state(self, state) -> None:
        if state != QtCore.Qt.ApplicationState.ApplicationActive:
            self._live_jog.stop()

    def _poll_live_jog(self) -> None:
        if QtWidgets.QApplication.applicationState() == QtCore.Qt.ApplicationState.ApplicationActive:
            self._live_jog.poll()
        else:
            self._live_jog.stop()

    def _build_groups(self, positioner) -> None:
        super()._build_groups(positioner)
        for group in self._groups.values():
            if "zdt" in group._rows:
                rows = group._rows
                group.layout().insertWidget(
                    group.layout().indexOf(next(iter(rows.values()))), rows["zdt"]
                )
            for label in group.findChildren(QtWidgets.QLabel):
                if label.text() == "z":
                    label.setText("F (focus)")
                elif label.text() == "zdt":
                    label.setText("Z (ZDT)")


class ArhSession(QtSession):
    """Add operator panels to the devices declared in the ARH host profile."""

    config: ClassVar[dict[str, object]] = {
        "color_scheme": "dark",
        "presenters": {
            "det_ctrl": {"timeout": 2.0},
            "arh_ctrl": {},
            "led_pattern_ctrl": {"panel": "arh"},
            "positioner_ctrl": {},
            "acq_ctrl": {},
        },
        "views": {
            "det_widget": {},
            "acq_widget": {},
            "img_widget": {},
            "hardware_widget": {},
            "led_pattern_widget": {},
            "positioner_widget": {"steps": list(JOG_STEPS)},
        },
    }

    napari: Annotated[AsHook[NapariApplication], Serves(QtHook.CREATE_APPLICATION)]
    napari_style: Annotated[AsHook[NapariStyle], Serves(QtHook.CONFIGURE_APPLICATION)]
    det_ctrl: AsPresenter[DetectorPresenter]
    arh_ctrl: AsPresenter[ArhPresenter]
    led_pattern_ctrl: AsPresenter[LedPatternPresenter]
    positioner_ctrl: AsPresenter[PositionerPresenter]
    acq_ctrl: AsPresenter[AcquisitionPresenter]
    det_widget: AsView[DetectorView]
    acq_widget: AsView[AcquisitionView]
    img_widget: AsView[ImageView]
    positioner_widget: AsView[ArhPositionerView]
    hardware_widget: AsView[ArhControls]
    led_pattern_widget: AsView[LedPatternView]
    preview: DetectorPreview

    def wire(self) -> Iterator[Link]:
        """Bind cameras, acquisition and native XYZ jog controls."""
        self.preview = DetectorPreview()
        self.arh_ctrl.preview_frame = self.preview.update_layers
        yield from wire_detector(self.det_ctrl, self.det_widget, self.img_widget)
        yield from wire_preview(self.det_ctrl, self.preview)
        yield from wire_acquisition(self.acq_ctrl, self.acq_widget)
        yield from wire_led_patterns(self.led_pattern_ctrl, self.led_pattern_widget)
        yield self.hardware_widget.sig_led_map_uploaded, self.led_pattern_ctrl.show_uploaded
        if isinstance(self.img_widget, ImageView):
            for detector in self.det_ctrl.detectors:
                source = DetectorPreviewSource(self.preview, detector)
                self.img_widget.attach_preview(detector, source)
            if "due_z_stack" in self.arh_ctrl.plan_map() and "dcam" not in self.det_ctrl.detectors:
                source = DetectorPreviewSource(self.preview, "dcam")
                self.img_widget.attach_preview("dcam", source, standalone=True)
            for device in self.devices.values():
                if isinstance(device, WorkcellWebcam):
                    source = DetectorPreviewSource(self.preview, device.name)
                    self.img_widget.attach_preview(device.name, source, standalone=True)
        yield from links_between(self.positioner_widget, self.positioner_ctrl)
        self._positioner_locks = frozenset()
        self._positioner_plan = None
        self._plan_cameras = frozenset()
        self._plan_stages = frozenset()
        yield self.acq_ctrl.sig_locks_changed, self._set_positioner_locks
        yield self.acq_ctrl.sig_plan_started, self._hold_positioner
        yield self.acq_ctrl.sig_plan_done, self._release_positioner
        yield self.acq_ctrl.sig_plan_stopped, self._release_positioner
        yield self.acq_ctrl.sig_plan_failed, self._release_positioner
        self._hardware_names = {}
        for role, device_type in (
            ("wheel", AsiFilterWheel), ("illumination", LumencorSpectra),
            ("stage", AsiSerialStage), ("mcu", UnifiedEmmsh),
        ):
            matches = [name for name, device in self.devices.items() if isinstance(device, device_type)]
            if role in matches:
                self._hardware_names[role] = role
            elif len(matches) == 1:
                self._hardware_names[role] = matches[0]
        self.hardware_widget.set_available_devices(set(self._hardware_names))
        wheel = self.devices.get(self._hardware_names.get("wheel", ""))
        if isinstance(wheel, AsiFilterWheel):
            try:
                busy = run_coro(wheel.busy.get_value(cached=False), timeout=2)
            except (RuntimeError, TimeoutError) as error:
                self.logger.warning(f"FW1000 startup state could not be read: {error}")
            else:
                self.hardware_widget.set_wheel_busy(busy)
            yield wheel.busy, self.hardware_widget.on_wheel_busy
            yield wheel.last_slot, self.hardware_widget.on_wheel_slot
        light = self.devices.get(self._hardware_names.get("illumination", ""))
        if isinstance(light, LumencorSpectra):
            try:
                channel, shutter_open = run_coro(_read_spectra_state(light))
            except (RuntimeError, TimeoutError) as error:
                self.logger.warning(f"Spectra startup state could not be read: {error}")
            else:
                self.hardware_widget.set_initial_illumination(channel, shutter_open)
            yield light.intensity, self.hardware_widget.on_spectra_intensity
            yield light.channel, self.hardware_widget.on_spectra_channel
            yield light.shutter_open, self.hardware_widget.on_spectra_shutter
        mcu = self.devices.get(self._hardware_names.get("mcu", ""))
        if isinstance(mcu, UnifiedEmmsh):
            async def led_capability() -> tuple[int, bool]:
                pixels = await mcu.led_pixels.get_value(cached=False)
                supports_map = await mcu.led_map_supported.get_value(cached=False)
                return pixels, supports_map

            try:
                self.hardware_widget.set_led_array_capability(*run_coro(led_capability(), timeout=3))
            except (RuntimeError, TimeoutError) as error:
                self.logger.warning(f"Due LED Array capability could not be read: {error}")
        self.hardware_widget.set_dynamic_dpct_cameras([
            name for name, device in self.devices.items()
            if isinstance(device, MMCamera) and "arh_dynamic_dpct" in self.acq_ctrl.plans
        ])
        stage = self.devices.get(self._hardware_names.get("stage", ""))
        if isinstance(stage, AsiSerialStage):
            try:
                bounds = run_coro(_read_stage_limits(stage))
            except (RuntimeError, TimeoutError, ValueError) as error:
                self.logger.warning(f"LX4000 travel limits could not be read: {error}")
            else:
                self.hardware_widget.set_stage_limits(bounds)
        stage_name = stage.xy.axis.name if isinstance(stage, AsiSerialStage) else None
        mcu_name = mcu.name if isinstance(mcu, UnifiedEmmsh) else None
        if stage_name is not None or mcu_name is not None:
            self.positioner_widget.configure_live_jog(stage_name, mcu_name)
        yield self.hardware_widget.sig_launch, self._launch_hardware_plan
        yield self.acq_ctrl.sig_plan_started, self.hardware_widget.plan_started
        yield self.acq_ctrl.sig_plan_done, self.hardware_widget.plan_finished
        yield self.acq_ctrl.sig_plan_stopped, self.hardware_widget.plan_stopped
        yield self.acq_ctrl.sig_plan_failed, self.hardware_widget.plan_failed

    @slot(thread="main")
    def _set_positioner_locks(self, names: frozenset[str]) -> None:
        self._positioner_locks = names
        names = names | self._plan_stages
        self.positioner_ctrl.set_locked(names)
        self.positioner_widget.set_locked(names)
        detector_locks = names | self._plan_cameras
        self.det_ctrl.set_locked(detector_locks)
        self.det_widget.set_locked(detector_locks)
        self.img_widget.set_roi_locked(detector_locks)

    @slot(thread="main")
    def _hold_positioner(self, plan: str, values: dict) -> None:
        if plan == LIVE_STREAM:
            self._positioner_plan = plan
            detectors = values.get("detectors")
            if not detectors and not values.get("webcam"):
                detectors = [device.name for device in self.det_ctrl.detectors.values()
                             if isinstance(device, LiveFlyer)]
            self._plan_cameras = frozenset(detectors or ())
            self._set_positioner_locks(self._positioner_locks)
        elif plan in {"lx4000_z_stack", "arh_dpct", "arh_dynamic_dpct", "due_z_stack", "due_software_z_stack", "zdt_relative_stack", "arh_hand_eye_3d", "due_zdt_jog"}:
            self._positioner_plan = plan
            self._plan_stages = frozenset(
                device.name for device in self.devices.values() if isinstance(device, (AsiSerialStage, UnifiedEmmsh))
            )
            camera = values.get("camera")
            self._plan_cameras = frozenset({camera}) if isinstance(camera, str) else frozenset()
            light = values.get("light")
            if plan == "zdt_relative_stack":
                self._plan_cameras |= frozenset(values.get("additional_cameras") or ())
                if isinstance(light, str):
                    self._plan_stages |= {light}
            self._set_positioner_locks(self._positioner_locks)

    @slot(thread="main")
    def _release_positioner(self, plan: str, reason: str = "") -> None:
        if self._positioner_plan == plan:
            self._positioner_plan = None
            self._plan_stages = frozenset()
            self._plan_cameras = frozenset()
            self._set_positioner_locks(self._positioner_locks)

    @slot(thread="main")
    def _launch_hardware_plan(self, plan: str, values: dict) -> None:
        if plan not in self.acq_ctrl.plans:
            self.hardware_widget.plan_failed(plan, f"{plan} is unavailable in this profile")
            return
        if not self.acq_ctrl.accepting_plan:
            self.hardware_widget.plan_failed(plan, "An acquisition is already active")
            return
        values = dict(values)
        for field, role in (("light", "illumination"), ("wheel", "wheel"), ("mcu", "mcu"), ("stage", "stage")):
            if values.get(field) == role:
                if role not in self._hardware_names:
                    self.hardware_widget.plan_failed(plan, f"No unique {role} device in this profile")
                    return
                values[field] = self._hardware_names[role]
        self.acq_ctrl.launch(plan, values)


async def _read_spectra_state(light: LumencorSpectra) -> tuple[str, bool]:
    async with asyncio.timeout(2):
        channel = await light.channel.get_value(cached=False)
        shutter_open = await light.shutter_open.get_value(cached=False)
    return channel, shutter_open


async def _read_stage_limits(stage: AsiSerialStage) -> dict[str, list[float]]:
    async with asyncio.timeout(2):
        xy = json.loads(await stage.motion.xy_limits.get_value(cached=False))
        z = json.loads(await stage.motion.z_limits.get_value(cached=False))
    if not isinstance(xy, dict) or set(xy) != {"x", "y"}:
        raise ValueError("LX4000 service did not publish X and Y travel intervals")
    for axis, interval in (*xy.items(), ("z", z)):
        if (
            not isinstance(interval, list) or len(interval) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in interval)
            or interval[0] >= interval[1]
        ):
            raise ValueError(f"Invalid LX4000 {axis.upper()} travel interval")
    return {**xy, "z": z}


async def _read_rotary_configuration(
    mcu: UnifiedEmmsh,
) -> tuple[dict[str, tuple[int, int]], dict[str, int], bool]:
    async with asyncio.timeout(3):
        status = json.loads(await mcu.commanded_steps.get_value(cached=False))
        enabled = await mcu.rotary_enabled.get_value(cached=False)
    bounds = status.get("rotary_limits") if isinstance(status, dict) else None
    if bounds == {} and not enabled:
        return {}, {}, False
    if not isinstance(bounds, dict) or not bounds or set(bounds) - {"roll", "pitch", "yaw"}:
        raise ValueError("Due service published invalid rotary axes")
    result = {}
    for axis, limits in bounds.items():
        if (not isinstance(limits, list) or len(limits) != 2
                or any(type(value) is not int for value in limits)
                or not -2047 <= limits[0] < limits[1] < 2048):
            raise ValueError(f"Invalid Due {axis} range")
        result[axis] = (limits[0], limits[1])
    targets = status.get("steps")
    if (not isinstance(targets, dict) or not result.keys() <= targets.keys()
            or any(type(value) is not int or not -2047 <= value < 2048 for value in targets.values())):
        raise ValueError("Invalid Due commanded target readback")
    return result, targets, enabled


def run_arh(argv: Sequence[str]) -> NoReturn:
    """Run against the profile supplied by the instrument owner."""
    parser = argparse.ArgumentParser(prog="aht session arh")
    parser.add_argument("--profile", type=Path, help="Defaults to the instrument profile from the ARH root")
    parser.add_argument("--mock", action="store_true", help="Use mock devices and workspace scratch storage")
    parser.add_argument("--without-due", action="store_true", help="Omit the Due service and devices")
    parser.add_argument("--workcell-profile", type=Path, help="Add the optional workcell webcam")
    parser.add_argument("--external-dcam-config", type=Path)
    parser.add_argument("--dcam-polarity", choices=("POSITIVE", "NEGATIVE"))
    parser.add_argument("--dcam-mm-path", type=Path, help="Micro-Manager installation for external DCAM")
    options = parser.parse_args(list(argv))
    if options.profile is None and not ARH_WORKSPACE_PROFILE.is_file():
        parser.error("Supply --profile when launching outside the ARH workspace root")
    profile = (options.profile or ARH_WORKSPACE_PROFILE).expanduser()
    try:
        config = read_arh_profile(profile)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if options.without_due:
        if options.external_dcam_config:
            parser.error("External DCAM capture requires the Due")
        config = without_arh_due(config)
        presenters = config.setdefault("presenters", {})
        presenters.setdefault("arh_ctrl", {})["include_due_plans"] = False
    if options.mock:
        if options.external_dcam_config:
            parser.error("--mock cannot open an external DCAM checkout")
        scratch = ARH_SCRATCH_ROOT.resolve()
        config = [config, {"mock": True, "storage": {"base_dir": str(scratch)}}]
    if bool(options.external_dcam_config) != bool(options.dcam_polarity):
        parser.error("Specify both --external-dcam-config and --dcam-polarity")
    if options.dcam_mm_path and not options.external_dcam_config:
        parser.error("--dcam-mm-path requires --external-dcam-config")
    if options.external_dcam_config:
        from .arh_external_dcam import external_dcam_profile

        try:
            config = external_dcam_profile(
                profile, options.external_dcam_config.expanduser(), options.dcam_polarity,
                mm_path=options.dcam_mm_path.expanduser().resolve() if options.dcam_mm_path else None,
            )
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
    base = config[0] if isinstance(config, list) else config
    services = base.get("services", {})
    due_devices = [
        device for device in base.get("devices", {}).values()
        if device.get("plugin_id") == "unified-emmsh"
    ]
    due_args = [
        service.get("args", {}) for device in due_devices
        if (service := services.get(device.get("service"))) is not None
        if isinstance(service, dict) and service.get("plugin_id") == "unified-emmsh"
    ]
    base.setdefault("presenters", {}).setdefault("arh_ctrl", {}).update(
        include_workcell=any(
            device.get("plugin_id") == "workcell-webcam"
            and isinstance(service := services.get(device.get("service")), dict)
            and service.get("plugin_id") == "workcell-webcam"
            for device in base.get("devices", {}).values()
        ),
        include_due_plans=bool(due_args),
        include_due_stack_recovery=any(args.get("zdt-stack") is True for args in due_args),
        include_due_software_stack=any(
            args.get("zdt-stack") is True and args.get("zdt-software") is True for args in due_args
        ),
        include_due_dpct=any(args.get("software-dpct") is True for args in due_args),
    )
    if options.workcell_profile:
        # Both adapters own the same physical camera. A legacy recorder profile
        # explicitly selects that service instead of the live-stream adapter.
        opencv_services = {name for name, service in services.items()
                           if isinstance(service, dict) and service.get("plugin_id") == "opencv-camera"}
        for name in opencv_services:
            del services[name]
        for name, device in tuple(base.get("devices", {}).items()):
            if device.get("service") in opencv_services:
                del base["devices"][name]
        try:
            workcell = read_arh_profile(options.workcell_profile)
            service = workcell["services"]["webcam_service"]
            device = workcell["devices"]["webcam"]
            if (service["plugin_id"] != "workcell-webcam" or device["plugin_id"] != "workcell-webcam"
                    or device["service"] != "webcam_service"):
                raise ValueError("Expected the ARH workcell webcam service and device")
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        layers = config if isinstance(config, list) else [config]
        config = [*layers, {
            "services": {"webcam_service": service},
            "devices": {"webcam": device},
            "presenters": {"arh_ctrl": {"include_workcell": True}},
        }]
    with arh_direct_pva():
        ArhSession(config).run()
