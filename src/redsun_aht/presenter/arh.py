"""ARH operator actions and host-stepped LX4000 Z acquisition."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from secrets import randbelow
from typing import Any, Literal, get_args
from uuid import uuid4

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.protocols import DataKey, Reading
from bluesky.utils import MsgGenerator
from numpy.typing import NDArray
from redsun.registry import PlanEntry

# Redsun resolves these annotations to populate the GUI device fields.
from redsun_aht.contracts.frames import RunIdentity
from redsun_aht.device.arh_dpct_patterns import demo_phases
from redsun_aht.device.dcam_process import DcamProcess
from redsun_aht.robotic.device.asi._stage import quantize_tenths
from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.device.software_stepped_plan import _Plane
from redsun_aht.device.unimmcore_triggered_camera import UniMMCoreTriggeredCamera
from redsun_aht.plans.dpct import _bounded_call
from redsun_aht.plans.hand_eye import HAND_EYE_AXES, calibration_waypoints, signed_offsets
from redsun_aht.robotic.device.asi._service import AsiFilterWheel, AsiSerialStage
from redsun_aht.robotic.device.auxiliary import LumencorSpectra, UnifiedEmmsh
from redsun_aht.robotic.device.workcell_webcam import WorkcellWebcam
from redsun_aht.robotic.device.due_stack import (
    DUE_STACK_CLEANUP_BUDGET_S,
    DueStackStatus,
    capture_due_stack,
    validate_due_geometry,
    wait_due_capture_plane,
    wait_due_stack,
)
from redsun_aht.storage.writer import FrameStamp, RunFrameStore, local_store_path

Channel = Literal["off", "brightfield", "514_teal", "640_red", "470_cyan", "561_green",
                  "445_blue", "395_violet"]
RotaryAxis = Literal["roll", "pitch", "yaw"]
CHANNELS = get_args(Channel)
FILTER_SLOTS = ("610BP60", "555BP30", "676BP34", "open", "closed", "470BP24", "518BP32", "632BP60")


def z_targets(origin: float, step_um: float, planes: int) -> tuple[float, ...]:
    """Validate a relative, monotone stack before issuing any move."""
    if type(planes) is not int or not 1 <= planes <= 32:
        raise ValueError("planes must be 1..32")
    if not math.isfinite(origin) or not math.isfinite(step_um):
        raise ValueError("origin and step must be finite")
    if planes > 1 and (step_um == 0 or abs(step_um) * (planes - 1) > 1000):
        raise ValueError("stack must have a nonzero step and span at most 1000 um")
    targets = tuple(origin + index * step_um for index in range(planes))
    if not all(math.isfinite(target) for target in targets):
        raise ValueError("stack target is not finite")
    if len(set(targets)) != planes:
        raise ValueError("stack step is too small to produce distinct Z targets")
    return targets


class ArhPresenter:
    """Expose hardware actions through the acquisition page's RunEngine."""

    def __init__(
        self, name: str, *,
        due_capture_factory: Callable[[], DcamProcess | UniMMCoreTriggeredCamera] | None = None,
        due_store_uri: Callable[[str], str] | None = None,
        include_due_plans: bool = True,
        include_due_stack_recovery: bool = True,
        include_due_software_stack: bool = True,
        include_due_dpct: bool = True,
        include_workcell: bool = False,
    ) -> None:
        self.name = name
        self._due_capture_factory = due_capture_factory
        self._due_store_uri = due_store_uri
        self._include_due_plans = include_due_plans
        self._include_due_stack_recovery = include_due_stack_recovery
        self._include_due_software_stack = include_due_software_stack
        self._include_due_dpct = include_due_dpct
        self._include_workcell = include_workcell
        self.preview_frame: Callable[[Mapping[str, Reading[Any]]], None] | None = None

    def _publish_preview(self, key: str, pixels: NDArray[Any]) -> None:
        if self.preview_frame is not None:
            self.preview_frame({
                key: {"value": pixels, "timestamp": time.time()},
            })

    def plan_map(self) -> Mapping[str, PlanEntry]:
        plans: dict[str, PlanEntry] = {
            "lx4000_z_stack": {"plan": self.z_stack},
            "arh_hand_eye_3d": {"plan": self.hand_eye_3d},
            "spectra_channel": {"plan": self.channel},
            "spectra_intensity": {"plan": self.spectra_intensity},
            "spectra_shutter": {"plan": self.shutter},
            "filter_wheel_slot": {"plan": self.wheel_slot},
        }

        if self._include_workcell:
            plans["workcell_snap"] = {"plan": self.workcell_snap}
            plans["workcell_record"] = {"plan": self.workcell_record}
        if not self._include_due_plans:
            return plans
        plans.update({
            "emmsh_rotary_target": {"plan": self.rotary_target},
            "emmsh_rotary_jog": {"plan": self.rotary_jog},
            "emmsh_stop": {"plan": self.rotary_stop},
            "emmsh_enable": {"plan": self.enable_rotary},
            "translation_bounds": {"plan": self.translation_bounds},
            "emmsh_zero_rotary": {"plan": self.zero_rotary},
            "emmsh_blank": {"plan": self.blank},
            "emmsh_led_array": {"plan": self.led_array},
        })
        if self._include_workcell:
            plans["workcell_rotary_check"] = {"plan": self.workcell_rotary_check}
        if self._include_due_stack_recovery:
            plans["due_z_stack_abort"] = {"plan": self.due_z_stack_abort}
            plans["due_z_stack_status"] = {"plan": self.due_z_stack_status}
        if self._include_due_software_stack:
            plans["due_software_z_stack"] = {"plan": self.due_software_z_stack}
            plans["zdt_relative_stack"] = {"plan": self.zdt_relative_stack}
            plans["due_zdt_jog"] = {"plan": self.zdt_jog}
        if self._include_due_dpct:
            plans["arh_dpct"] = {"plan": self.dpct}
            plans["arh_dynamic_dpct"] = {"plan": self.dynamic_dpct}
        if self._due_capture_factory is not None and self._due_store_uri is not None:
            plans["due_z_stack"] = {"plan": self.due_z_stack}
        return plans

    def _workcell_image(self, webcam: WorkcellWebcam) -> MsgGenerator[str]:
        async def snap() -> str:
            previous = json.loads(await webcam.status.get_value(cached=False))
            await webcam.snap.set(True, timeout=10)
            status = json.loads(await webcam.status.get_value(cached=False))
            if status["state"] == "error":
                raise RuntimeError(status["error"])
            path = status.get("path")
            if (status["state"] != "complete" or status.get("frames") != 1
                    or not isinstance(path, str) or path == previous.get("path")
                    or Path(path).suffix != ".png"):
                raise RuntimeError("Workcell did not report a new completed still image")
            return path

        return (yield from _bounded_call(snap, timeout_s=12, what="capture workcell still"))

    def workcell_snap(self, webcam: WorkcellWebcam) -> MsgGenerator[None]:
        """Save a workcell PNG and publish its path in the run metadata."""
        path = yield from self._workcell_image(webcam)
        yield from bps.open_run(md={"mode": "workcell_snap", "workcell_image_path": path})
        yield from bps.close_run()

    def workcell_preview(self, webcam: WorkcellWebcam, frames: int = 10, *,
                         continuous: bool = False) -> MsgGenerator[None]:
        """Refresh the workcell preview from successive software snapshots."""
        if type(frames) is not int or not 1 <= frames <= 120:
            raise ValueError("Workcell preview requires 1 to 120 frames")
        import cv2

        def preview():
            yield from bps.clear_checkpoint()
            remaining = frames
            while continuous or remaining:
                path = yield from self._workcell_image(webcam)
                pixels = cv2.imread(path)
                if pixels is None:
                    raise RuntimeError(f"Could not decode workcell image {path}")
                self._publish_preview(f"{webcam.name}-buffer", cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                remaining -= 1

        yield from bpp.run_wrapper(preview(), md={"mode": "workcell_preview", "frames": frames})

    def workcell_record(self, webcam: WorkcellWebcam) -> MsgGenerator[None]:
        """Record an AVI for the duration configured on the workcell service."""
        def record():
            yield from bps.clear_checkpoint()
            previous = json.loads((yield from bps.rd(webcam.status)))
            if previous["state"] in {"recording", "snapping"}:
                raise RuntimeError("Workcell camera is busy")
            yield from bps.abs_set(webcam.start, True, wait=True, timeout=5)
            try:
                deadline = time.monotonic() + 610
                while time.monotonic() < deadline:
                    status = json.loads((yield from _bounded_call(
                        lambda: webcam.status.get_value(cached=False),
                        timeout_s=5, what="read workcell recording status",
                    )))
                    if status["state"] == "error":
                        raise RuntimeError(status["error"])
                    path = status.get("path")
                    if status["state"] == "complete" and path != previous.get("path"):
                        if not isinstance(path, str) or Path(path).suffix.lower() != ".avi":
                            raise RuntimeError("Workcell did not report an AVI recording")
                        return path
                    yield from bps.sleep(0.25)
                raise TimeoutError("Workcell recording did not complete")
            finally:
                yield from bps.abs_set(webcam.finish, True, wait=True, timeout=8)

        path = yield from record()
        yield from bps.open_run(md={"mode": "workcell_record", "workcell_video_path": path})
        yield from bps.close_run()

    def workcell_rotary_check(
        self, mcu: UnifiedEmmsh, webcam: WorkcellWebcam, axis: RotaryAxis = "pitch",
        confirm_clearance: bool = False, travel_deg: float = 0.0,
        counts_per_degree: float = 0.0, settle_s: float = 1.0,
    ) -> MsgGenerator[None]:
        """Capture a rotary round trip using one count or a calibrated angular move."""
        if confirm_clearance is not True or axis not in get_args(RotaryAxis):
            raise ValueError("Select a rotary axis and confirm clearance for the round trip")
        if not math.isfinite(travel_deg) or not 0 <= travel_deg <= 10:
            raise ValueError("Rotary observation travel must be between 0 and 10 degrees")
        if not math.isfinite(settle_s) or settle_s < 0:
            raise ValueError("Rotary observation settle time must be finite and nonnegative")
        increment = 1
        if travel_deg:
            count_span = travel_deg * counts_per_degree
            if not math.isfinite(count_span) or not 0.5 < count_span <= 2047:
                raise ValueError("Supply a calibrated Due target count scale for this axis")
            increment = round(count_span)
        if (yield from bps.rd(mcu.rotary_enabled)) is not True:
            raise RuntimeError("Rotary targets are disabled in the Due service")

        def read_status():
            return (yield from _bounded_call(
                lambda: mcu.commanded_steps.get_value(cached=False),
                timeout_s=5, what="read Due rotary targets",
            ))

        status = json.loads((yield from read_status()))
        origin = status["steps"][axis]
        if type(origin) is not int or not -2047 <= origin < 2048:
            raise ValueError(f"Invalid Due {axis} commanded target readback")
        target = origin + increment
        limits = status["rotary_limits"].get(axis)
        if limits is None or not limits[0] <= origin < target <= limits[1]:
            raise ValueError(f"Due {axis} round trip exceeds its configured limits")

        reading = _WorkcellRotaryReading()

        def capture(phase, expected):
            steps = json.loads((yield from read_status()))["steps"][axis]
            if steps != expected:
                raise RuntimeError(f"Due {axis} reported {steps} after requesting {expected}")
            path = yield from self._workcell_image(webcam)
            reading.values = {"phase": phase, "workcell_image_path": path, "commanded_steps": steps}
            yield from bps.create(name="workcell_rotary")
            yield from bps.read(reading)
            yield from bps.save()

        def move(steps):
            try:
                yield from self.rotary_target(mcu, axis, steps)
            except Exception as error:
                try:
                    yield from self.rotary_stop(mcu)
                except Exception as stop_error:
                    error.add_note(f"Due stop request failed: {stop_error}")
                raise

        def observe():
            yield from bps.clear_checkpoint()
            yield from capture("before", origin)
            yield from move(target)
            try:
                yield from bps.sleep(settle_s)
                yield from capture("target", target)
            finally:
                yield from move(origin)
            yield from bps.sleep(settle_s)
            yield from capture("after_return_command", origin)

        yield from bpp.run_wrapper(observe(), md={
            "mode": "workcell_rotary_check", "axis": axis,
            "origin_steps": origin, "target_steps": target,
            "requested_travel_deg": travel_deg or None,
            "counts_per_degree": counts_per_degree if travel_deg else None,
            "quantized_travel_deg": increment / counts_per_degree if travel_deg else None,
            "rotary_position_provenance": "commanded_target",
            "settle_s": settle_s,
        })

    @staticmethod
    def _require_due_stack_enabled(mcu: UnifiedEmmsh) -> MsgGenerator[None]:
        enabled = yield from _bounded_call(
            lambda: mcu.zdt_stack_enabled.get_value(cached=False),
            timeout_s=5, what="verify Due X-stack capability",
        )
        if not enabled:
            raise RuntimeError("The Due service has not enabled ZDT-X stacks")

    def due_z_stack_status(self, mcu: UnifiedEmmsh, run_id: int) -> MsgGenerator[None]:
        """Record the firmware state and edge count for an identified Due run."""
        if type(run_id) is not int or not 1 <= run_id <= 2147483647:
            raise ValueError("Invalid Due stack run ID")
        yield from self._require_due_stack_enabled(mcu)
        yield from bps.abs_set(mcu.zdt_status_request, run_id, wait=True, timeout=5)
        readback = yield from _bounded_call(
            lambda: mcu.zdt_status.get_value(cached=False),
            timeout_s=5, what="read Due Z-stack status",
        )
        status = DueStackStatus.parse(readback, run_id, 32)

        def report() -> MsgGenerator[None]:
            yield from bps.create(name="due_stack_status")
            yield from bps.read(_DueStackStatusReading(status))
            yield from bps.save()

        yield from bpp.run_wrapper(report(), md={"due_run_id": run_id})

    def due_z_stack_abort(
        self, mcu: UnifiedEmmsh, run_id: int,
    ) -> MsgGenerator[None]:
        """Request abort and panel blanking for an identified Due stack."""
        if type(run_id) is not int or not 1 <= run_id <= 2147483647:
            raise ValueError("Invalid Due stack run ID")
        yield from self._require_due_stack_enabled(mcu)
        try:
            yield from bps.abs_set(mcu.zdt_abort, run_id, wait=True, timeout=5)
        except Exception as error:
            try:
                yield from bps.abs_set(mcu.blank, True, wait=True, timeout=5)
            except Exception as blank_error:
                error.add_note(f"Due panel blanking could not be confirmed: {blank_error}")
            raise
        else:
            yield from bps.abs_set(mcu.blank, True, wait=True, timeout=5)

    def due_z_stack(
        self, mcu: UnifiedEmmsh, run_id: int, first_tenths_deg: int, step_tenths_deg: int, count: int,
        lower_tenths_deg: int, upper_tenths_deg: int, confirm_exclusive_camera: bool = False,
        confirm_illumination_ready: bool = False,
    ) -> MsgGenerator[None]:
        """Capture an external-trigger stack. Angular parameters are tenths of a degree."""
        if self._due_capture_factory is None or self._due_store_uri is None:
            raise RuntimeError("No commissioned external-trigger camera binding")
        if not (confirm_exclusive_camera is True and confirm_illumination_ready is True):
            raise ValueError("Confirm camera ownership and illumination readiness")
        definition = dict(run_id=run_id, first=first_tenths_deg, step=step_tenths_deg, count=count,
                          lower=lower_tenths_deg, upper=upper_tenths_deg)
        validate_due_geometry(definition)
        yield from self._require_due_stack_enabled(mcu)
        capture_id = uuid4().hex
        store_uri = self._due_store_uri(capture_id)
        run = RunIdentity(capture_id, "due_z_stack", "Due-triggered DCAM stack", run_id)
        capture_timeout_s = 60

        async def acquire():
            try:
                return await capture_due_stack(
                    mcu, self._due_capture_factory(), definition, timeout_s=capture_timeout_s,
                    exclusive_camera=True, illumination_ready=True,
                )
            except Exception as error:
                # Redsun displays str(error), which omits exception notes.
                notes = getattr(error, "__notes__", ())
                if isinstance(error, ExceptionGroup):
                    details = [str(cause) for cause in error.exceptions]
                elif notes:
                    details = [str(error)]
                else:
                    raise
                raise RuntimeError(" ".join((*details, *notes))) from error

        def capture() -> MsgGenerator[None]:
            status, frames = yield from _bounded_call(
                acquire, timeout_s=capture_timeout_s + DUE_STACK_CLEANUP_BUDGET_S + 1,
                what="Due Z stack",
            )
            store = RunFrameStore(
                store_uri, "due_external", frames[0].data,
                planned_frames=count, record_timing=True, run=run, detector_id="dcam",
            )
            try:
                for frame in frames:
                    store.append(
                        frame.data, stamp=FrameStamp(producer_sequence=frame.sequence),
                        arrival_ns=frame.receipt_ns,
                    )
            finally:
                store.close()
            self._publish_preview("dcam-buffer", frames[-1].data)
            yield from bps.create(name="due_stack")
            yield from bps.read(_DueStackResult(status.edges, len(frames)))
            yield from bps.save()

        yield from bpp.run_wrapper(capture(), md={
            "mode": "due_z_stack", "association": "host_order_inferred",
            "definition": definition, "store_uri": store_uri, "capture_id": capture_id,
        })

    def zdt_relative_stack(
        self, mcu: UnifiedEmmsh, camera: MMCamera, light: LumencorSpectra,
        step_um: float = 2.0, frames: int = 50, channel: Channel = "470_cyan",
        additional_cameras: Sequence[MMCamera] = (), centered: bool = False,
    ) -> MsgGenerator[None]:
        """Acquire a software-triggered stack from the current ZDT position."""
        if type(frames) is not int or not 1 <= frames <= 1000:
            raise ValueError("frames must be 1..1000")
        if not math.isfinite(step_um) or step_um == 0 or abs(step_um) * frames > 1000:
            raise ValueError("Enter a nonzero step with a scan window of at most 1000 µm")
        if channel not in CHANNELS or channel in ("off", "brightfield"):
            raise ValueError("Select a Spectra LED channel")
        cameras = (camera, *additional_cameras)
        if len({detector.name for detector in cameras}) != len(cameras):
            raise ValueError("Select each camera only once")
        for detector in cameras:
            yield from self._require_idle_camera(detector)
        origin = yield from _bounded_call(
            mcu.read_zdt_position, timeout_s=6, what="read ZDT stack origin",
        )
        first = origin - step_um * frames / 2 if centered else origin
        targets = tuple(round((first + i * step_um) * 3.6) / 3.6 for i in range(frames))
        endpoint = first + step_um * frames
        low, high = sorted((first, endpoint))
        window = (math.ceil(low * 3.6 - 1e-8), math.floor(high * 3.6 + 1e-8))
        if not -10000 <= low <= high <= 10000:
            raise ValueError("ZDT targets exceed the scan window or controller range")
        if centered:
            targets = tuple(min(max(round(target * 3.6), window[0]), window[1]) / 3.6 for target in targets)
        if len(set(targets)) != frames:
            raise ValueError("ZDT step is too small for distinct controller targets")
        capture_id = uuid4().hex
        uris = {detector.name: detector.dpct_store_uri(f"zdt-relative-{capture_id}") for detector in cameras}
        run = RunIdentity(capture_id, "zdt_relative_stack", "Relative ZDT stack", 1)

        def capture():
            stores = {}
            previous_counts = {}
            try:
                yield from self.channel(light, channel)
                if centered:
                    yield from _bounded_call(
                        lambda: mcu.move_zdt(targets[0], window=window),
                        timeout_s=25, what="move ZDT to first centered plane",
                    )
                yield from self.shutter(light, True)
                for index, target in enumerate(targets):
                    if index:
                        yield from _bounded_call(
                            lambda: mcu.move_zdt(target, window=window),
                            timeout_s=25, what=f"move ZDT to plane {index + 1}",
                        )
                    for detector in cameras:
                        name = detector.name
                        count, pixels = yield from self._snap(detector)
                        if name in previous_counts and count != previous_counts[name] + 1:
                            raise RuntimeError(f"{name} snap count is discontinuous")
                        previous_counts[name] = count
                        if name not in stores:
                            stores[name] = RunFrameStore(
                                uris[name], "software_snap", pixels, planned_frames=frames,
                                run=run, detector_id=name,
                            )
                        stores[name].append(pixels, stamp=FrameStamp(scan_index=index, producer_sequence=count))
                        self._publish_preview(detector.buffer.name, pixels)
            finally:
                for store in stores.values():
                    store.close()

        def cleanup():
            try:
                yield from self.shutter(light, False)
            finally:
                yield from self.channel(light, "off")

        yield from bpp.run_wrapper(bpp.finalize_wrapper(capture(), cleanup()), md={
            "mode": "zdt_relative_stack", "store_uri": uris[camera.name], "store_uris": uris, "origin_um": origin,
            "step_um": step_um, "targets_um": targets, "channel": channel,
        })

    def due_software_z_stack(
        self, mcu: UnifiedEmmsh, camera: MMCamera, run_id: int,
        first_tenths_deg: int, step_tenths_deg: int, count: int,
        lower_tenths_deg: int, upper_tenths_deg: int,
        confirm_exclusive_camera: bool = False,
    ) -> MsgGenerator[None]:
        """Save one software snap per settled Due ZDT plane.

        Enter absolute targets and bounds in tenths of a degree. The ARH
        drive advances nominally 1 mm per 3600 tenths of a degree.
        """
        if confirm_exclusive_camera is not True:
            raise ValueError("Confirm that the camera is reserved for software snaps")
        definition = dict(run_id=run_id, first=first_tenths_deg, step=step_tenths_deg,
                          count=count, lower=lower_tenths_deg, upper=upper_tenths_deg)
        validate_due_geometry(definition)
        if not (yield from bps.rd(mcu.zdt_software_enabled)):
            raise RuntimeError("Due firmware software capture is disabled")
        yield from self._require_idle_camera(camera)
        capture_id = uuid4().hex
        uri = camera.dpct_store_uri(f"due-software-z-{capture_id}")
        run = RunIdentity(capture_id, "due_software_z_stack", "Due software Z stack", run_id)

        def wait_plane(index):
            yield from _bounded_call(
                lambda: wait_due_capture_plane(mcu, run_id, count, index),
                timeout_s=11, what=f"wait for Due plane {index}",
            )

        def capture():
            store = None
            previous_count = None
            try:
                request = json.dumps({**definition, "camera_ready": True, "software": True})
                yield from bps.abs_set(mcu.zdt_start, request, wait=True, timeout=10)
                for index in range(count):
                    yield from wait_plane(index)
                    frame_count, pixels = yield from self._snap(camera)
                    if previous_count is not None and frame_count != previous_count + 1:
                        raise RuntimeError("Camera snap count is discontinuous")
                    previous_count = frame_count
                    if store is None:
                        store = RunFrameStore(uri, "software_snap", pixels, planned_frames=count,
                                              run=run, detector_id=camera.name)
                    store.append(pixels, stamp=FrameStamp(
                        scan_index=index, producer_sequence=frame_count,
                    ))
                    self._publish_preview(camera.buffer.name, pixels)
                    yield from bps.abs_set(mcu.zdt_captured, f"{run_id} {index}", wait=True, timeout=2)
                yield from wait_plane(count)
                yield from bps.create(name="due_stack")
                yield from bps.read(_DueStackResult(0, count, software_acknowledgements=count))
                yield from bps.save()
            except GeneratorExit:
                raise
            except BaseException as error:
                try:
                    yield from bps.abs_set(mcu.zdt_abort, run_id, wait=True, timeout=3)
                except Exception as abort_error:
                    raise RuntimeError(
                        f"{error}. Due stack abort could not be confirmed: {abort_error}. "
                        f"Check due_z_stack_status for run {run_id}."
                    ) from error
                raise
            finally:
                if store is not None:
                    store.close()

        yield from bpp.run_wrapper(capture(), md={
            "mode": "due_software_z_stack", "definition": definition,
            "store_uri": uri, "capture_id": capture_id, "detector": camera.name,
            "zdt_lead_mm_per_rev": 1.0,
            "nominal_z_targets_um": tuple(
                (first_tenths_deg + index * step_tenths_deg) * 1000 / 3600
                for index in range(count)
            ),
        })

    def channel(self, light: LumencorSpectra, channel: Channel = "off") -> MsgGenerator[None]:
        """Select the Spectra channel. The service closes the shutter on change."""
        yield from bps.abs_set(light.channel, channel, wait=True, timeout=10)

    def spectra_intensity(self, light: LumencorSpectra, intensity: int) -> MsgGenerator[None]:
        """Set the active Spectra channel intensity in percent."""
        yield from bps.abs_set(light.intensity, intensity, wait=True, timeout=10)

    def shutter(self, light: LumencorSpectra, opened: bool = False) -> MsgGenerator[None]:
        """Open or close the Spectra shutter."""
        yield from bps.abs_set(light.shutter_open, opened, wait=True, timeout=10)

    def wheel_slot(self, wheel: AsiFilterWheel, slot: int = 4) -> MsgGenerator[None]:
        """Request a filter slot. An idle response does not measure the slot."""
        if type(slot) is not int or not 0 <= slot <= 7:
            raise ValueError("slot must be 0..7")
        yield from bps.abs_set(wheel.slot, slot, wait=True, timeout=20)

    def rotary_target(
        self, mcu: UnifiedEmmsh, axis: RotaryAxis = "roll", steps: int = 0
    ) -> MsgGenerator[None]:
        """Set an absolute commanded step target within the service's bounds."""
        if type(steps) is not int or not -2047 <= steps <= 2047:
            raise ValueError("steps must be an integer within -2047 to 2047")
        enabled = yield from bps.rd(mcu.rotary_enabled)
        if enabled is not True:
            raise RuntimeError("Rotary targets are disabled in the Due service")
        target = {"roll": mcu.roll_target, "pitch": mcu.pitch_target, "yaw": mcu.yaw_target}
        yield from bps.abs_set(target[axis], steps, wait=True, timeout=10)

    def rotary_stop(self, mcu: UnifiedEmmsh) -> MsgGenerator[None]:
        """Stop the Due rotary motors and request abort of a reserved ZDT stack."""
        yield from bps.abs_set(mcu.stop_, True, wait=True, timeout=10)

    def translation_bounds(self, stage: AsiSerialStage, mcu: UnifiedEmmsh, enabled: bool) -> MsgGenerator[None]:
        """Set the LX4000 and ZDT manual travel bounds for this session."""
        if enabled:
            yield from bps.abs_set(stage.motion.travel_bounds, True, wait=True, timeout=5)
            yield from bps.abs_set(mcu.zdt_bounds, True, wait=True, timeout=5)
            return
        try:
            yield from bps.abs_set(stage.motion.travel_bounds, False, wait=True, timeout=5)
            yield from bps.abs_set(mcu.zdt_bounds, False, wait=True, timeout=5)
        except Exception:
            yield from bps.abs_set(stage.motion.travel_bounds, True, wait=True, timeout=5)
            yield from bps.abs_set(mcu.zdt_bounds, True, wait=True, timeout=5)
            raise

    def zero_rotary(self, mcu: UnifiedEmmsh) -> MsgGenerator[None]:
        """Set the current roll, pitch and yaw positions to zero without motion."""
        yield from bps.abs_set(mcu.rotary_zero, True, wait=True, timeout=5)

    def enable_rotary(self, mcu: UnifiedEmmsh) -> MsgGenerator[None]:
        """Enable and hold the Due rotary coils."""
        yield from bps.abs_set(mcu.enable_rotary, True, wait=True, timeout=5)

    def rotary_jog(self, mcu: UnifiedEmmsh, axis: RotaryAxis, delta: int) -> MsgGenerator[None]:
        """Jog relative to the latest controller target within the service limits."""
        if type(delta) is not int or delta == 0:
            raise ValueError("Enter a nonzero rotary jog in counts")
        status = json.loads((yield from _bounded_call(
            lambda: mcu.commanded_steps.get_value(cached=False), timeout_s=5,
            what="read current Due rotary target",
        )))
        target = status["steps"][axis] + delta
        yield from bps.abs_set(mcu.rotary_jog_request, f"{axis}={target}", wait=True, timeout=12)

    def zdt_jog(self, mcu: UnifiedEmmsh, delta_um: float) -> MsgGenerator[None]:
        """Jog the ZDT in micrometres using its nominal 1 mm/rev lead."""
        if not math.isfinite(delta_um) or not 0 < abs(delta_um) <= 100:
            raise ValueError("Enter a nonzero ZDT jog up to 100 µm")
        delta = round(delta_um * 3.6)
        if delta == 0:
            raise ValueError("ZDT jog is below one controller increment")
        run_id = 1 + randbelow(2147483647)
        request = json.dumps(dict(run_id=run_id, jog=delta, camera_ready=True, software=True))
        yield from bps.abs_set(mcu.zdt_start, request, wait=True, timeout=10)
        try:
            yield from _bounded_call(lambda: wait_due_stack(mcu, run_id, 1, timeout_s=10),
                                     timeout_s=11, what="verify ZDT jog completion")
        except Exception as error:
            try:
                yield from bps.abs_set(mcu.zdt_abort, run_id, wait=True, timeout=3)
            except Exception as abort_error:
                error.add_note(f"ZDT jog abort was not confirmed: {abort_error}")
            raise

    def led_array(self, mcu: UnifiedEmmsh, request: str) -> MsgGenerator[None]:
        """Apply an LED Array setting through the Due service."""
        if not (yield from bps.rd(mcu.led_pixels)):
            raise RuntimeError("Compact LED Array unavailable")
        yield from bps.abs_set(mcu.led_array_request, request, wait=True, timeout=10)

    def blank(self, mcu: UnifiedEmmsh) -> MsgGenerator[None]:
        """Blank the Due LED panel."""
        yield from bps.abs_set(mcu.blank, True, wait=True, timeout=10)

    def dynamic_dpct(
        self, camera: MMCamera, mcu: UnifiedEmmsh, confirm_exclusive_camera: bool = False,
    ) -> MsgGenerator[None]:
        """Capture the semicircle pair, graded groups 1–4 and solid groups 1–4."""
        if confirm_exclusive_camera is not True:
            raise ValueError("Confirm that the camera is reserved for software snaps")
        if (yield from bps.rd(mcu.led_pixels)) != 61:
            raise RuntimeError("Dynamic DPCT requires the 61-LED Due panel")
        yield from self._require_idle_camera(camera)
        phases = demo_phases()
        labels = (
            "Graded semicircle", "Solid semicircle",
            *(f"Graded DPCT group {n}" for n in range(1, 5)),
            *(f"Solid DPCT group {n}" for n in range(1, 5)),
        )
        capture_id = uuid4().hex
        uri = camera.dpct_store_uri(f"arh-dynamic-dpct-{capture_id}")
        run = RunIdentity(capture_id, "arh_dynamic_dpct", "ARH dynamic DPCT", scan_id=0)

        def command(action, value):
            request = json.dumps({"action": action, "value": value})
            yield from bps.abs_set(mcu.led_array_request, request, wait=True, timeout=15)

        def capture():
            store = None
            ordinal = 0
            previous_count = None
            source = _DpctFrame(0, 0, 0, -1)
            try:
                yield from self.blank(mcu)
                yield from command("color", "blue")
                for maps in phases:
                    receipts = []
                    recipe_id = 1 + randbelow(0xFFFFFFFF)
                    for slot, pixels in enumerate(maps, 1):
                        receipt = yield from _bounded_call(
                            lambda slot=slot, pixels=pixels: mcu.upload_dynamic_pattern(slot, pixels),
                            timeout_s=15, what=f"upload dynamic slot {slot}",
                        )
                        receipts.append(receipt)
                    yield from command("dynamic_arm", [
                        recipe_id,
                        *([r.slot, r.generation, r.crc32] for r in receipts),
                    ])
                    for index, receipt in enumerate(receipts):
                        yield from command("dynamic_next", [
                            recipe_id, index, receipt.slot, receipt.generation,
                        ])
                        yield from bps.sleep(3)
                        count, pixels = yield from self._snap(camera)
                        if previous_count is not None and count != previous_count + 1:
                            raise RuntimeError("Camera snap count is discontinuous")
                        previous_count = count
                        if store is None:
                            store = RunFrameStore(
                                uri, "software_snap", pixels, planned_frames=len(labels),
                                run=run, detector_id=camera.name,
                            )
                        frame_index = store.append(pixels, stamp=FrameStamp(
                            pattern_id=ordinal + 1, producer_sequence=count,
                        ))
                        yield from command("dynamic_ack", [recipe_id, index])
                        self._publish_preview(camera.buffer.name, pixels)
                        yield from bps.create(name="dpct")
                        source.values = {
                            **_DpctFrame(ordinal, ordinal + 1, count, frame_index).values,
                            "dynamic_run": recipe_id,
                            "dynamic_slot": receipt.slot,
                            "dynamic_generation": receipt.generation,
                            "dynamic_crc32": receipt.crc32,
                        }
                        yield from bps.read(source)
                        yield from bps.save()
                        ordinal += 1
                    yield from command("dynamic_finish", [recipe_id, len(receipts)])
            finally:
                try:
                    yield from self.blank(mcu)
                finally:
                    if store is not None:
                        store.close()

        yield from bpp.run_wrapper(capture(), md={
            "mode": "arh_dynamic_dpct",
            "detector": camera.name,
            "capture_id": capture_id,
            "store_uri": uri,
            "pattern_labels": labels,
            "map_sha256": tuple(hashlib.sha256(pixels).hexdigest() for maps in phases for pixels in maps),
        })

    def dpct(
        self,
        camera: MMCamera,
        mcu: UnifiedEmmsh,
        confirm_exclusive_camera: bool = False,
    ) -> MsgGenerator[None]:
        """Capture the four Due DPCT patterns using an idle MMCore camera."""
        if confirm_exclusive_camera is not True:
            raise ValueError("Confirm that the camera is reserved for software snaps")
        if not (yield from bps.rd(mcu.software_dpct_enabled)):
            raise RuntimeError("Software DPCT is disabled in the Due service")
        yield from self._require_idle_camera(camera)
        capture_id = uuid4().hex
        run = RunIdentity(capture_id, "arh_dpct", "ARH software DPCT", scan_id=0)
        store_uri = camera.dpct_store_uri(f"arh-dpct-{capture_id}")

        def capture() -> MsgGenerator[None]:
            store = None
            source = _DpctFrame(0, 0, 0, -1)
            previous_count = None
            capture_error = None
            try:
                yield from bps.abs_set(mcu.blank, True, wait=True, timeout=10)
                for ordinal, pattern in enumerate((11, 12, 13, 14)):
                    yield from bps.abs_set(mcu.dpct_pattern, pattern, wait=True, timeout=10)
                    yield from bps.sleep(0.02)
                    count, pixels = yield from self._snap(camera)
                    if previous_count is not None and count != previous_count + 1:
                        raise RuntimeError("Camera snap count is discontinuous")
                    previous_count = count
                    if store is None:
                        store = RunFrameStore(
                            store_uri, "software_snap", pixels, planned_frames=4,
                            run=run, detector_id=camera.name,
                        )
                    frame_index = store.append(pixels, stamp=FrameStamp(pattern_id=pattern))
                    self._publish_preview(camera.buffer.name, pixels)
                    yield from bps.create(name="dpct")
                    source.values = _DpctFrame(ordinal, pattern, count, frame_index).values
                    yield from bps.read(source)
                    yield from bps.save()
            except Exception as error:
                capture_error = error
                raise
            finally:
                try:
                    yield from bps.abs_set(mcu.blank, True, wait=True, timeout=10)
                except Exception as error:
                    detail = f"Due panel blanking could not be confirmed: {error}"
                    if capture_error is not None:
                        detail = f"DPCT capture failed: {capture_error}. {detail}"
                    raise RuntimeError(detail) from error
                finally:
                    if store is not None:
                        store.close()

        yield from bpp.run_wrapper(capture(), md={
            "mode": "arh_dpct_software_snap", "association": "host_order_inferred",
            "detector": camera.name, "pattern_commands": (11, 12, 13, 14),
            "store_uri": store_uri, "capture_id": capture_id,
        })

    @staticmethod
    def _require_idle_camera(camera: MMCamera) -> MsgGenerator[None]:
        for signal in (camera.acquire, camera.capture):
            active = yield from _bounded_call(
                lambda: signal.get_value(cached=False),
                timeout_s=5, what="check software-snap camera state",
            )
            if active:
                raise RuntimeError("Camera acquisition or capture is active")

    @staticmethod
    def _snap(camera: MMCamera) -> MsgGenerator[tuple[int, NDArray[Any]]]:
        def read_frame_count() -> MsgGenerator[int]:
            return (yield from _bounded_call(
                lambda: camera.frame_count.get_value(cached=False),
                timeout_s=5, what="read software-snap frame count",
            ))

        before = yield from read_frame_count()
        yield from bps.abs_set(camera.snap_request, True, wait=True, timeout=20)
        count = yield from read_frame_count()
        if count != before + 1:
            raise RuntimeError("Camera snap count is discontinuous")
        pixels = yield from _bounded_call(
            lambda: camera.buffer.get_value(cached=False),
            timeout_s=20,
            what="read software-snap pixels",
        )
        if pixels is None or pixels.size == 0:
            raise RuntimeError("Camera snap returned no pixels")
        return count, pixels

    def hand_eye_3d(
        self, stage: AsiSerialStage, camera: MMCamera,
        xy_step_um: float = 50.0, z_step_um: float = 2.0,
        confirm_exclusive_camera: bool = False,
        sample_x: bool = True, sample_y: bool = True, sample_z: bool = True,
        preview_only: bool = False,
        preview_path: str = "",
        simulator_review_path: str = "",
    ) -> MsgGenerator[None]:
        """Acquire an anchor and six signed XYZ offsets on a fixed 3D fiducial."""
        if not preview_only and confirm_exclusive_camera is not True:
            raise ValueError("Confirm the camera is reserved and the fiducial is in view")
        if preview_only and preview_path:
            raise ValueError("Choose either to export a preview or to capture using one")
        if simulator_review_path and (preview_only or not preview_path):
            raise ValueError("A simulator review requires its original preview file")
        if any(not math.isfinite(step) or not 1 <= step <= 500 for step in (xy_step_um, z_step_um)):
            raise ValueError("Calibration steps must be between 1 and 500 µm")
        axes = [stage.axis[key] for key in "xyz"]

        def read_position():
            return (yield from _bounded_call(
                lambda: _read_calibration_position(axes), timeout_s=5,
                what="read LX4000 calibration position",
            ))

        origin = yield from read_position()
        xy = json.loads((yield from bps.rd(stage.motion.xy_limits)))
        motion_profile = json.loads((yield from bps.rd(stage.motion.profile)))
        xy_backlash_um = {
            axis: 1000 * float(motion_profile[axis]["backlash_mm"]) for axis in "xy"
        }
        if any(not math.isfinite(value) or value < 0 for value in xy_backlash_um.values()):
            raise ValueError("Invalid LX4000 XY backlash readback")
        z_backlash_um = 1000 * float(motion_profile["z"]["backlash_mm"])
        if not math.isfinite(z_backlash_um) or z_backlash_um < 0:
            raise ValueError("Invalid LX4000 Z backlash readback")
        bounds = [xy["x"], xy["y"], json.loads((yield from bps.rd(stage.motion.z_limits)))]
        steps = {
            axis: step for axis, step, selected in zip(
                "xyz", (xy_step_um, xy_step_um, z_step_um),
                (sample_x, sample_y, sample_z), strict=True,
            ) if selected
        }
        poses = tuple(signed_offsets("arh", steps))
        backlash = {**xy_backlash_um, "z": z_backlash_um}
        for axis, value, (low, high) in zip("xyz", origin, bounds, strict=True):
            margin = steps[axis] + backlash[axis] if axis in steps else 0.0
            if (not all(map(math.isfinite, (value, low, high)))
                    or not low <= value - margin <= value + margin <= high):
                raise ValueError(f"Calibration path including backlash exceeds LX4000 {axis.upper()} travel limits")
        capture_id = uuid4().hex
        uri = camera.dpct_store_uri(f"arh-hand-eye-{capture_id}")
        run = RunIdentity(capture_id, "arh_hand_eye_3d", "XYZ fiducial calibration", scan_id=0)
        metadata = {
            "store_uri": uri, "capture_id": capture_id, "origin_um": origin,
            "steps_um": steps, "detector": camera.name,
            "instrument": "arh", "calibration_axes": list(steps),
            "waypoint_axes": HAND_EYE_AXES["arh"],
            "waypoint_units": ("um", "um", "um", "deg", "deg", "deg"),
            "relative_waypoints": calibration_waypoints("arh", steps),
            "capture_waypoint_indices": (0, *range(1, 2 * len(poses) - 1, 2)),
            "xy_backlash_um": xy_backlash_um,
            "z_backlash_um": z_backlash_um,
            "travel_limits_um": dict(zip("xyz", bounds, strict=True)),
            "association": "host_order_inferred",
        }
        path_fields = (
            "instrument", "detector", "origin_um", "calibration_axes",
            "waypoint_axes", "waypoint_units", "relative_waypoints", "capture_waypoint_indices",
            "xy_backlash_um", "z_backlash_um", "travel_limits_um",
        )
        path_data = json.loads(json.dumps({key: metadata[key] for key in path_fields}))
        if preview_only:
            path = local_store_path(uri).with_suffix(".hand_eye_path.json")
            partial = path.with_suffix(".json.tmp")
            preview_metadata = {key: value for key, value in metadata.items() if key != "store_uri"}
            try:
                partial.write_text(json.dumps(preview_metadata, indent=2) + "\n", encoding="utf-8")
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
            yield from bps.open_run(md={**preview_metadata, "hand_eye_path": str(path)})
            yield from bps.close_run()
            return

        if preview_path:
            with Path(preview_path).open("rb") as source:
                contents = source.read(16385)
            if len(contents) > 16384:
                raise ValueError("Hand-eye preview exceeds the expected file size")
            preview = json.loads(contents)
            if not isinstance(preview, dict):
                raise ValueError("Hand-eye preview must contain a JSON object")
            for key, expected in path_data.items():
                if preview.get(key) != expected:
                    raise ValueError(f"Hand-eye preview does not match current {key}")
            metadata["preview_capture_id"] = preview.get("capture_id")
            metadata["preview_path_sha256"] = hashlib.sha256(contents).hexdigest()
            metadata["preview_path_file"] = "hand_eye_preview.json"
        if simulator_review_path:
            with Path(simulator_review_path).open("rb") as source:
                review_bytes = source.read(16385)
            if len(review_bytes) > 16384:
                raise ValueError("Simulator review exceeds the expected file size")
            review = json.loads(review_bytes)
            if (not isinstance(review, dict)
                    or review.get("schema") != "redsun.hand_eye.sampled_collision.v1"
                    or review.get("instrument") != "arh"
                    or review.get("path_sha256") != metadata["preview_path_sha256"]):
                raise ValueError("Simulator review does not match the ARH preview")
            if (type(review.get("sample_count")) is not int or review["sample_count"] < 1
                    or "first_collision_sample" not in review or review["first_collision_sample"] is not None):
                raise ValueError("Simulator review must report sampled poses without a collision")
            checked_paths = review.get("checked_paths", ["nominal"])
            if checked_paths not in (["nominal"], ["nominal", "backlash_-1", "backlash_+1"]):
                raise ValueError("Simulator review has an incomplete or unknown path list")
            metadata["simulator_review_sha256"] = hashlib.sha256(review_bytes).hexdigest()
            metadata["simulator_checked_paths"] = checked_paths
            metadata["simulator_review_file"] = "simulator_review.json"

        yield from self._require_idle_camera(camera)

        def capture():
            store = None
            previous = None
            point = _CalibrationPoint()
            observations = []
            if preview_path and (yield from read_position()) != origin:
                raise RuntimeError("LX4000 origin changed after checking the hand-eye preview")
            try:
                for index, (axis_name, offset) in enumerate(poses):
                    axis = "xyz".index(axis_name) if axis_name is not None else None
                    target = list(origin)
                    if axis is not None:
                        target[axis] += offset
                        yield from bps.abs_set(axes[axis], target[axis], wait=True, timeout=60)
                    before = yield from read_position()
                    if any(not math.isfinite(a) or abs(a - b) > 0.5
                           for a, b in zip(before, target, strict=True)):
                        raise RuntimeError(f"LX4000 did not settle near calibration target {target}")
                    count, pixels = yield from self._snap(camera)
                    after = yield from read_position()
                    if (any(not math.isfinite(value) for value in (*before, *after))
                            or max(abs(a - b) for a, b in zip(before, after, strict=True)) > 0.25):
                        raise RuntimeError("LX4000 readback changed during the calibration snap")
                    if previous is not None and count != previous + 1:
                        raise RuntimeError("Calibration camera snap count is discontinuous")
                    previous = count
                    if store is None:
                        store = RunFrameStore(uri, "software_snap", pixels, planned_frames=len(poses),
                                              run=run, detector_id=camera.name)
                    frame = store.append(pixels, stamp=FrameStamp(scan_index=index))
                    self._publish_preview(camera.buffer.name, pixels)
                    yield from bps.create(name="hand_eye")
                    point.update(frame, 2 * index - 1 if index else 0, count, target, before, after)
                    observations.append(point.values)
                    yield from bps.read(point)
                    yield from bps.save()
                    if axis is not None:
                        yield from bps.abs_set(axes[axis], origin[axis], wait=True, timeout=60)
                        returned = yield from read_position()
                        if any(not math.isfinite(a) or abs(a - b) > 0.5
                               for a, b in zip(returned, origin, strict=True)):
                            raise RuntimeError(f"LX4000 did not return to calibration origin {origin}")
            except GeneratorExit:
                raise
            except BaseException as error:
                try:
                    yield from bps.abs_set(stage.motion.stop_, True, wait=True, timeout=10)
                except Exception as halt_error:
                    error.add_note(f"LX4000 halt could not be confirmed: {halt_error}")
                raise
            finally:
                if store is not None:
                    store.close()
            sidecar = local_store_path(uri) / "hand_eye_path.json"
            provenance_files = {}
            if preview_path:
                provenance_files[metadata["preview_path_file"]] = contents
            if simulator_review_path:
                provenance_files[metadata["simulator_review_file"]] = review_bytes
            for filename, data in provenance_files.items():
                copy = sidecar.with_name(filename)
                partial_copy = copy.with_suffix(".json.tmp")
                try:
                    partial_copy.write_bytes(data)
                    partial_copy.replace(copy)
                finally:
                    partial_copy.unlink(missing_ok=True)
            partial = sidecar.with_suffix(".json.tmp")
            try:
                partial.write_text(
                    json.dumps({**metadata, "observations": observations}, indent=2) + "\n",
                    encoding="utf-8",
                )
                partial.replace(sidecar)
            finally:
                partial.unlink(missing_ok=True)

        yield from bpp.run_wrapper(capture(), md=metadata)

    def z_stack(
        self,
        stage: AsiSerialStage,
        camera: MMCamera,
        step_um: float = 1.0,
        planes: int = 3,
        confirm_exclusive_camera: bool = False,
    ) -> MsgGenerator[None]:
        """Move LX4000 Z, verify settled readback, and software-snap each plane.

        The camera must be idle in deferred software-snap mode. The operator
        sets illumination separately. The service enforces its startup-relative
        travel limits on every move. The stack leaves Z at its last target.
        """
        if confirm_exclusive_camera is not True:
            raise ValueError("Confirm that the camera is reserved for software snaps")
        z = stage.axis["z"]
        def read_z() -> MsgGenerator[float]:
            return (yield from _bounded_call(
                lambda: z.position.get_value(cached=False), timeout_s=5,
                what="read LX4000 Z position",
            ))

        origin = yield from read_z()
        targets = z_targets(float(origin), step_um, planes)
        bounds = json.loads((yield from bps.rd(stage.motion.z_limits)))
        if (
            not isinstance(bounds, list) or len(bounds) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds)
            or bounds[0] > bounds[1]
        ):
            raise RuntimeError("LX4000 Z travel limits are unavailable")
        if min(targets) < bounds[0] or max(targets) > bounds[1]:
            raise ValueError("Stack targets exceed the LX4000 Z travel interval")
        counts_per_um = yield from bps.rd(stage.motion.z_counts_per_um)
        if not math.isfinite(counts_per_um) or counts_per_um <= 0:
            raise RuntimeError("LX4000 Z controller scale is unavailable")
        encoded_targets = targets
        if planes > 1:
            counts = tuple(quantize_tenths(target, counts_per_um) for target in targets)
            if len(set(counts)) != planes:
                raise ValueError("Stack step is too small to produce distinct controller targets")
            encoded_targets = tuple(count / counts_per_um for count in counts)
            if min(encoded_targets) < bounds[0] or max(encoded_targets) > bounds[1]:
                raise ValueError("Encoded stack targets exceed the LX4000 Z travel interval")
        yield from self._require_idle_camera(camera)
        capture_id = uuid4().hex
        run = RunIdentity(capture_id, "lx4000_z_stack", "LX4000 software Z stack", scan_id=0)
        store_uri = camera.dpct_store_uri(f"lx4000-z-{capture_id}")

        def capture() -> MsgGenerator[None]:
            store = None
            previous_count = None
            source = _Plane(0, 0.0, 0.0, 0, 0.0, -1)
            try:
                for index, (target, encoded_target) in enumerate(zip(targets, encoded_targets, strict=True)):
                    if planes > 1:
                        yield from bps.abs_set(z, target, wait=True, timeout=60)
                    actual = yield from read_z()
                    error_counts = abs((float(actual) - encoded_target) * counts_per_um)
                    if not math.isfinite(error_counts) or error_counts > 1 + 1e-9:
                        raise RuntimeError(f"Z readback {actual:g} disagrees with {target:g} um")
                    count, pixels = yield from self._snap(camera)
                    if previous_count is not None and count != previous_count + 1:
                        raise RuntimeError("Camera snap count is discontinuous")
                    previous_count = count
                    if store is None:
                        store = RunFrameStore(
                            store_uri, "software_snap", pixels, planned_frames=planes,
                            run=run, detector_id=camera.name,
                        )
                    frame_index = store.append(pixels, stamp=FrameStamp(scan_index=index))
                    self._publish_preview(camera.buffer.name, pixels)
                    yield from bps.create(name="software_step")
                    source.values = _Plane(index, target, float(actual), count,
                                           float(pixels.mean()), frame_index).values
                    yield from bps.read(source)
                    yield from bps.save()
            except GeneratorExit:
                raise
            except BaseException as error:
                if planes > 1:
                    try:
                        yield from bps.abs_set(stage.motion.stop_, True, wait=True, timeout=10)
                    except Exception as halt_error:
                        error.add_note(f"LX4000 halt could not be confirmed: {halt_error}")
                raise
            finally:
                if store is not None:
                    store.close()

        yield from bpp.run_wrapper(capture(), md={
            "mode": "lx4000_software_step", "association": "host_order_inferred",
            "detector": camera.name, "targets_um": targets, "store_uri": store_uri,
            "capture_id": capture_id,
            "z_counts_per_um": counts_per_um, "encoded_targets_um": encoded_targets,
        })


async def _read_calibration_position(axes):
    return tuple([float(await axis.position.get_value(cached=False)) for axis in axes])


class _CalibrationPoint:
    name = "hand_eye_point"

    def update(self, index, waypoint, count, target, before, after):
        self.values = {"stored_index": index, "waypoint_index": waypoint, "camera_frame_count": count}
        for phase, position in (("target", target), ("before", before), ("after", after)):
            self.values.update({f"{axis}_{phase}_um": value for axis, value in zip("xyz", position, strict=True)})

    def read(self):
        now = time.time()
        return {key: {"value": value, "timestamp": now} for key, value in self.values.items()}

    def describe(self):
        return {
            key: {"source": "ARH:hand-eye", "dtype": "integer" if isinstance(value, int) else "number", "shape": []}
            for key, value in self.values.items()
        }


class _DpctFrame:
    name = "dpct_frame"

    def __init__(self, ordinal: int, pattern: int, count: int, index: int) -> None:
        self.values = {
            "pattern_ordinal": ordinal,
            "pattern_command": pattern,
            "camera_frame_count": count,
            "stored_index": index,
        }

    def read(self) -> dict[str, Reading[int]]:
        now = time.time()
        return {
            key: {"value": value, "timestamp": now}
            for key, value in self.values.items()
        }

    def describe(self) -> dict[str, DataKey]:
        return {
            key: {"source": "ARH:software-dpct", "dtype": "integer", "shape": []}
            for key in self.values
        }


class _DueStackResult:
    name = "due_stack_result"

    def __init__(self, edges: int, frames: int, *, software_acknowledgements: int | None = None) -> None:
        self.values = {"trigger_edges": edges, "camera_frames": frames}
        if software_acknowledgements is not None:
            self.values["software_acknowledgements"] = software_acknowledgements

    def read(self) -> dict[str, Reading[int]]:
        now = time.time()
        return {key: {"value": value, "timestamp": now} for key, value in self.values.items()}

    def describe(self) -> dict[str, DataKey]:
        return {key: {"source": "ARH:due-z-stack", "dtype": "integer", "shape": []}
                for key in self.values}


class _WorkcellRotaryReading:
    name = "workcell_rotary"

    def __init__(self) -> None:
        self.values: dict[str, str | int] = {}

    def read(self) -> dict[str, Reading]:
        now = time.time()
        return {key: {"value": value, "timestamp": now} for key, value in self.values.items()}

    def describe(self) -> dict[str, DataKey]:
        return {key: {"source": "ARH:workcell-rotary", "dtype": dtype, "shape": []}
                for key, dtype in (("phase", "string"), ("workcell_image_path", "string"),
                                   ("commanded_steps", "integer"))}


class _DueStackStatusReading:
    name = "due_stack_status"

    def __init__(self, status: DueStackStatus) -> None:
        self.status = status

    def read(self) -> dict[str, Reading]:
        now = time.time()
        return {
            "firmware_state": {"value": self.status.state, "timestamp": now},
            "trigger_edges": {"value": self.status.edges, "timestamp": now},
        }

    def describe(self) -> dict[str, DataKey]:
        return {
            "firmware_state": {"source": "ARH:due-z-stack", "dtype": "string", "shape": []},
            "trigger_edges": {"source": "ARH:due-z-stack", "dtype": "integer", "shape": []},
        }
