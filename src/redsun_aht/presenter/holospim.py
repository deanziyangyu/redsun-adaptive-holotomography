"""HoloSPIM operator actions on the desktop RunEngine."""

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from uuid import uuid4

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.utils import MsgGenerator
from redsun.engine.plan_stubs import lock_wrapper
from redsun.registry import PlanEntry

from redsun_aht.device.aaaotf import AttachedAAAOTF
from redsun_aht.device.dynamic_patterns import DynamicPattern
from redsun_aht.device.esp300_stage import Esp300Stage
from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.device.pva_automatic_stack import acquire_pva_automatic_stack
from redsun_aht.device.pva_capture_evidence import (
    PvaCaptureEvidence,
    require_idle_camera,
)
from redsun_aht.device.pva_pattern_shots import PatternPlane, acquire_pva_pattern_shots
from redsun_aht.device.pva_rotation import PvaRotationMotor
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.spinnaker_camera import SpinnakerCamera
from redsun_aht.device.unified_mcu import AngularStack
from redsun_aht.plans.dpct import _bounded_call
from redsun_aht.plans.hand_eye import HAND_EYE_AXES, calibration_waypoints
from redsun_aht.plans.holospim_hand_eye import capture_reviewed_path
from redsun_aht.plans.survey_stack import survey_z_stack
from redsun_aht.plans.timelapse import _PointRecord
from redsun_aht.presenter.led_patterns import PANELS, pattern_weights

logger = logging.getLogger(__name__)
LedColor = Literal["blue", "green", "red"]
LaserPreset = Literal["All Off", "488 nm", "561 nm"]
_COLOR_COMMAND = {"blue": "51", "green": "52", "red": "53"}
# Both swept settings were verified on slot 1 in separate runs.
_LASER_RF_PRESETS = {"488 nm": (112.28, 1750), "561 nm": (92.63, 1900)}


class HolospimPresenter:
    """Route operator actions to the HoloSPIM profile's MCU service."""

    def __init__(
        self,
        name: str,
        *,
        x28_bounds_tenths: Sequence[int] | None = None,
        enable_dynamic_dpct: bool = False,
        enable_x28_triggered_stack: bool = False,
        hand_eye_roll_limits_deg: Sequence[float] | None = None,
    ) -> None:
        self.name = name
        self._enable_dynamic_dpct = enable_dynamic_dpct
        self._enable_x28_triggered_stack = enable_x28_triggered_stack
        self._roll_limits = None
        if hand_eye_roll_limits_deg is not None:
            low, high = hand_eye_roll_limits_deg
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("hand-eye roll limits must be finite and increasing")
            self._roll_limits = (low, high)
        self._bounds: tuple[int, int] | None = None
        if x28_bounds_tenths is not None:
            lower, upper = x28_bounds_tenths
            AngularStack(1, lower, 0, 1, lower, upper)
            if lower == upper:
                raise ValueError("X28 limits must define an increasing interval")
            self._bounds = (lower, upper)

    def plan_map(self) -> Mapping[str, PlanEntry]:
        """Expose the stack only when the host supplies travel bounds."""
        plans: dict[str, PlanEntry] = {"holospim_led_blank": {"plan": self.blank}}
        plans["holospim_laser_profile"] = {"plan": self.laser_profile}
        plans["holospim_laser_settings"] = {"plan": self.laser_settings}
        plans["holospim_hand_eye_preview"] = {"plan": self.hand_eye_preview}
        plans["holospim_hand_eye_capture"] = {"plan": self.hand_eye_capture}
        plans["holospim_survey_z_stack"] = {"plan": self.survey_stack}
        plans["holospim_x28_readback"] = {"plan": self.x28_readback}
        if self._bounds is not None:
            plans["holospim_x28_software_stack"] = {"plan": self.x28_software_stack}
            if self._enable_x28_triggered_stack:
                plans["holospim_x28_triggered_stack"] = {
                    "plan": self.x28_triggered_stack,
                }
        if self._enable_dynamic_dpct:
            plans["holospim_dynamic_dpct"] = {"plan": self.dynamic_dpct}
        return plans

    def blank(self, mcu: PvaUnifiedMcu) -> MsgGenerator[None]:
        """Blank the HoloSPIM LED panel through its serial owner."""
        yield from bps.abs_set(mcu.illumination, "00", wait=True)

    def laser_settings(
        self, laser: AttachedAAAOTF, channel: int, frequency_mhz: float,
        maximum_intensity: int, power_percent: float,
    ) -> MsgGenerator[None]:
        """Close all slots and edit RF settings through the leased service."""
        if (
            type(channel) is not int or not 1 <= channel <= 8
            or type(maximum_intensity) is not int or not 0 <= maximum_intensity <= 2200
            or not math.isfinite(frequency_mhz) or not 50 <= frequency_mhz <= 200
            or not math.isfinite(power_percent) or not 0 <= power_percent <= 100
        ):
            raise ValueError("AOTF settings exceed the adapter ranges")

        def apply() -> MsgGenerator[None]:
            yield from bps.abs_set(laser, "all_off", wait=True)
            yield from bps.abs_set(laser, {
                "Channel": channel, "Frequency (MHz)": frequency_mhz,
                "Maximum intensity (dB)": maximum_intensity,
                "Power (% of max)": power_percent,
            }, wait=True)

        yield from lock_wrapper(apply(), laser)

    def laser_profile(
        self, laser: AttachedAAAOTF, profile: LaserPreset = "All Off"
    ) -> MsgGenerator[None]:
        """Apply a HoloSPIM laser preset through the open AAAOTF service."""
        preset = _LASER_RF_PRESETS.get(profile)
        if preset is None:
            command = "all_off" if profile == "All Off" else profile
            yield from lock_wrapper(bps.abs_set(laser, command, wait=True), laser)
            return

        frequency, intensity = preset

        def switch_wavelength() -> MsgGenerator[None]:
            yield from bps.abs_set(laser, "all_off", wait=True)
            yield from bps.abs_set(laser, {
                "Channel": 1,
                "Frequency (MHz)": frequency,
                "Maximum intensity (dB)": intensity,
                "Power (% of max)": 100,
                "State": 1,
            }, wait=True)

        yield from lock_wrapper(switch_wavelength(), laser)

    def x28_readback(self, mcu: PvaUnifiedMcu) -> MsgGenerator[None]:
        """Record controller-reported X28 position, speed and drive flags."""
        state = yield from _bounded_call(
            mcu.read_state, timeout_s=5, what="read X28 controller state",
        )
        logger.info(
            "X28 controller position %.1f°, target %.1f°, speed %.1f RPM, "
            "home flags 0x%02x, motor flags 0x%02x",
            state.position_tenths / 10, state.target_tenths / 10,
            state.speed_tenths_rpm / 10, state.home_flags, state.motor_flags,
        )
        yield from bps.open_run(md={"plan_name": "holospim_x28_readback"})
        point = _PointRecord()
        point.values = asdict(state)
        yield from bps.create("x28_controller")
        yield from bps.read(point)
        yield from bps.save()
        yield from bps.close_run()

    def survey_stack(
        self,
        camera: MMCamera,
        survey: Esp300Stage,
        step_um: float,
        planes: int,
        frames_per_plane: int = 1,
        confirm_setup: bool = False,
    ) -> MsgGenerator[None]:
        """Collect internal snaps along ESP300 Z with the configured optical setup."""
        if confirm_setup is not True:
            raise ValueError(
                "confirm exclusive camera, illumination and survey clearance"
            )
        yield from survey_z_stack(camera, survey, step_um, planes, frames_per_plane)

    def hand_eye_preview(
        self,
        camera: MMCamera,
        survey: Esp300Stage,
        rotation: PvaRotationMotor,
        parked_x28_mm: float,
        x_step_um: float = 0.0,
        y_step_um: float = 0.0,
        z_step_um: float = 0.0,
        roll_step_deg: float = 0.0,
    ) -> MsgGenerator[None]:
        """Export a signed XYZ/roll path without motion or camera capture."""
        offsets = (x_step_um, y_step_um, z_step_um, roll_step_deg)
        if not math.isfinite(parked_x28_mm):
            raise ValueError("parked X28 simulator coordinate must be finite")
        if any(not math.isfinite(step) or step < 0 for step in offsets):
            raise ValueError("steps must be finite and nonnegative; zero omits an axis")
        if any(step > 1000 for step in offsets[:3]):
            raise ValueError("linear hand-eye offsets must be at most 1000 um")
        if roll_step_deg and self._roll_limits is None:
            raise ValueError(
                "configure hand_eye_roll_limits_deg before previewing roll"
            )
        axis = rotation.rotation
        if roll_step_deg:
            roll_step_deg = abs(axis.to_degrees(axis.to_steps(roll_step_deg)))
            if roll_step_deg == 0:
                raise ValueError("roll step is below one controller step")
        axes = HAND_EYE_AXES["holospim"]
        steps = {
            name: step
            for name, step in zip(axes, (*offsets[:3], roll_step_deg), strict=True)
            if step
        }
        waypoints = calibration_waypoints("holospim", steps)

        def preview() -> MsgGenerator[None]:
            reading = yield from _bounded_call(
                survey.read_position, timeout_s=5, what="read survey anchor"
            )
            xyz = {row.name: row.value for row in reading.axes}
            if set(xyz) != set("xyz") or set(survey.axes) != set("xyz"):
                raise RuntimeError("survey XYZ readback and bounds are required")
            roll = yield from _bounded_call(
                axis.read_position, timeout_s=5, what="read roll anchor"
            )
            if not all(math.isfinite(value) for value in (*xyz.values(), roll)):
                raise RuntimeError("nonfinite hand-eye anchor")
            limits = {
                name: (spec.lower, spec.upper) for name, spec in survey.axes.items()
            }
            origin = tuple(xyz[name] for name in "xyz")
            for name in "xyz":
                low, high = limits[name]
                extent = steps.get(name, 0)
                if not low <= xyz[name] - extent <= xyz[name] + extent <= high:
                    raise ValueError(
                        f"hand-eye path exceeds survey {name.upper()} limits"
                    )
            if roll_step_deg:
                assert self._roll_limits is not None
                low, high = self._roll_limits
                if not low <= roll - roll_step_deg <= roll + roll_step_deg <= high:
                    raise ValueError("hand-eye path exceeds roll limits")
            metadata = {
                "schema": "redsun.hand_eye.path.v1",
                "instrument": "holospim",
                "detector": camera.name,
                "calibration_axes": list(steps),
                "waypoint_axes": axes,
                "waypoint_units": ("um", "um", "um", "deg"),
                "relative_waypoints": waypoints,
                "capture_waypoint_indices": (0, *range(1, len(waypoints), 2)),
                "origin_um": origin,
                "origin_roll_deg": roll,
                "travel_limits_um": limits,
                "roll_limits_deg": self._roll_limits,
                "parked_x28_mm": parked_x28_mm,
                "parked_x28_source": "operator_supplied",
                "survey_anchor_quality": reading.quality,
                "roll_anchor_quality": "simulated" if axis.simulated else "commanded",
            }
            info = camera.path_provider(camera.name)
            path = Path(info.directory_path) / f"{info.filename}.hand_eye_path.json"
            partial = path.with_suffix(f".{uuid4().hex}.tmp")
            try:
                partial.write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
            yield from bps.open_run(md={**metadata, "hand_eye_path": str(path)})
            yield from bps.close_run()

        yield from lock_wrapper(preview(), survey, rotation)

    def hand_eye_capture(
        self,
        camera: MMCamera,
        survey: Esp300Stage,
        rotation: PvaRotationMotor,
        preview_path: str,
        simulator_review_path: str,
        confirm_reviewed_setup: bool = False,
    ) -> MsgGenerator[None]:
        """Capture the reviewed path with X28 parked and the camera in internal trigger.

        Confirm the installed scene matches the review, the camera is exclusive,
        the fiducial is visible and the physical interlocks are ready.
        """
        if confirm_reviewed_setup is not True:
            raise ValueError("confirm the camera, parked X28, scene and interlocks")
        yield from capture_reviewed_path(
            camera,
            survey,
            rotation,
            preview_path,
            simulator_review_path,
            self._roll_limits,
        )

    def dynamic_dpct(
        self,
        mcu: PvaUnifiedMcu,
        camera: MMCamera,
        survey: Esp300Stage,
        rotation: PvaRotationMotor,
        pattern_file: str,
        color: LedColor,
        repeats: int = 1,
        ready_for_internal_snaps: bool = False,
    ) -> MsgGenerator[None]:
        """Capture an ordered recipe of one to five JSON arrays of panel weights."""
        if not self._enable_dynamic_dpct:
            raise RuntimeError("dynamic DPCT is disabled in this profile")
        if ready_for_internal_snaps is not True:
            raise ValueError("confirm camera and illumination readiness")
        if type(repeats) is not int or not 1 <= repeats <= 32:
            raise ValueError("repeats must be 1..32")
        with Path(pattern_file).open("rb") as stream:
            raw = stream.read(8193)
        if len(raw) > 8192:
            raise ValueError("dynamic pattern recipe exceeds 8192 bytes")
        recipe = json.loads(raw)
        if not isinstance(recipe, list) or not 1 <= len(recipe) <= 5:
            raise ValueError("expected an array of one to five 91-pixel maps")
        maps = tuple(
            bytes(pattern_weights(PANELS["holospim"], entry)) for entry in recipe
        )
        color_command = _COLOR_COMMAND[color]
        info = camera.path_provider(camera.name)
        uri = str(Path(info.directory_path) / f"{info.filename}.zarr")
        count = len(maps) * repeats

        async def acquire(receipts: list[DynamicPattern]) -> list[int]:
            capture = PvaCaptureEvidence(camera)
            frame_counts: list[int] = []

            async def snap(_ordinal: int, _pattern: DynamicPattern) -> None:
                frame_counts.append(await capture.snap())

            try:
                await capture.prepare(
                    uri,
                    count,
                    internal_trigger_configured=True,
                    software_snap=True,
                )
                await mcu.dynamic_shots(receipts, repeats, snap, ready=True)
                evidence = await capture.collect()
                if evidence.stored_frames != count:
                    raise RuntimeError("dynamic DPCT camera store is incomplete")
            finally:
                await capture.close()
            return frame_counts

        @bpp.run_decorator(
            md={
                "plan_name": "holospim_dynamic_dpct",
                "camera": camera.name,
                "store_uri": uri,
                "led_color": color,
                "repeats": repeats,
                "association": "host_order_inferred",
                "pattern_sha256": [
                    hashlib.sha256(pixels).hexdigest() for pixels in maps
                ],
            }
        )
        def run() -> MsgGenerator[None]:
            yield from bps.clear_checkpoint()
            yield from _bounded_call(
                lambda: require_idle_camera(camera, internal_trigger=True),
                timeout_s=5,
                what="verify camera ready for internal snaps",
            )
            yield from bps.abs_set(mcu.auto_pulses, False, wait=True)
            yield from _bounded_call(mcu.reserve, timeout_s=5, what="reserve MCU")
            failed = False
            try:
                yield from _bounded_call(
                    lambda: mcu.set_reserved_illumination(color_command),
                    timeout_s=5,
                    what="set LED color",
                )
                receipts: list[DynamicPattern] = []
                for slot, pixels in enumerate(maps, 1):
                    receipt = yield from _bounded_call(
                        lambda slot=slot, pixels=pixels: mcu.upload_dynamic_pattern(
                            slot, pixels
                        ),
                        timeout_s=15,
                        what="upload dynamic map",
                    )
                    receipts.append(receipt)
                frames = yield from _bounded_call(
                    lambda: acquire(receipts),
                    timeout_s=15 * (count + 2),
                    what="capture dynamic DPCT recipe",
                )
                record = _PointRecord()
                for ordinal, camera_count in enumerate(frames):
                    receipt = receipts[ordinal % len(receipts)]
                    record.values = {
                        "frame": ordinal,
                        "frame_index": ordinal,
                        "camera_count": camera_count,
                        "slot": receipt.slot,
                        "generation": receipt.generation,
                        "crc32": receipt.crc32,
                    }
                    yield from bps.create("dynamic_dpct_frame")
                    yield from bps.read(record)
                    yield from bps.save()
            except BaseException:
                failed = True
                raise
            finally:
                yield from self._release_mcu(mcu, failed)

        yield from lock_wrapper(run(), mcu, camera, survey, rotation)

    @staticmethod
    def _release_mcu(mcu: PvaUnifiedMcu, failed: bool) -> MsgGenerator[None]:
        if mcu.reservation_held:
            try:
                yield from _bounded_call(
                    lambda: mcu.set_reserved_illumination("00"),
                    timeout_s=5,
                    what="blank LED",
                )
                yield from _bounded_call(
                    mcu.release_reservation,
                    timeout_s=5,
                    what="release MCU",
                )
            except Exception:
                if not failed:
                    raise
                logger.exception("MCU cleanup failed after acquisition error")
        elif not failed:
            raise RuntimeError("MCU reservation uncertain; blanking unconfirmed")
        else:
            logger.warning("MCU reservation uncertain; blanking unconfirmed")

    def x28_triggered_stack(
        self,
        mcu: PvaUnifiedMcu,
        camera: SpinnakerCamera,
        survey: Esp300Stage,
        rotation: PvaRotationMotor,
        first_tenths: int,
        step_tenths: int,
        planes: int,
        color: LedColor,
        pattern: int,
        frames_per_plane: int = 1,
        interval_ms: int = 100,
        timeout_s: float = 60,
        confirm_setup: bool = False,
    ) -> MsgGenerator[None]:
        """Acquire a bounded X28 stack using GPIO13 and FLIR Line0."""
        if not self._enable_x28_triggered_stack or self._bounds is None:
            raise RuntimeError("X28 triggered stack requires profile opt-in and bounds")
        lower, upper = self._bounds
        AngularStack(1, first_tenths, step_tenths, planes, lower, upper)
        if planes > 1 and step_tenths == 0:
            raise ValueError("a multi-plane X28 stack requires a nonzero step")
        if type(pattern) is not int or not 1 <= pattern <= 5:
            raise ValueError("LED pattern must be 1..5")
        if type(frames_per_plane) is not int or not 1 <= frames_per_plane <= 32:
            raise ValueError("frames per plane must be 1..32")
        if type(interval_ms) is not int or not 1 <= interval_ms < 30000:
            raise ValueError("interval must be 1..29999 milliseconds")
        if not 0 < timeout_s <= 300 or confirm_setup is not True:
            raise ValueError(
                "confirm trigger, camera, illumination and travel readiness"
            )
        color_command = _COLOR_COMMAND[color]
        info = camera.path_provider(camera.name)
        uri = str(Path(info.directory_path) / f"{info.filename}.zarr")

        async def verify_camera() -> None:
            await require_idle_camera(camera)
            expected = {
                "TriggerMode": "On", "TriggerSelector": "FrameStart",
                "TriggerSource": "Line0", "TriggerActivation": "FallingEdge",
            }
            for name, value in expected.items():
                signal = camera.properties.get(name)
                if signal is None or await signal.get_value(cached=False) != value:
                    raise RuntimeError(f"FLIR must report {name}={value}")

        @bpp.run_decorator(md={
            "plan_name": "holospim_x28_triggered_stack",
            "camera": camera.name,
            "store_uri": uri,
            "first_tenths": first_tenths,
            "step_tenths": step_tenths,
            "planes": planes,
            "led_color": color,
            "led_pattern": pattern,
            "frames_per_plane": frames_per_plane,
            "interval_ms": interval_ms,
            "association_quality": "count_matched_host_order_inferred",
        })
        def run() -> MsgGenerator[None]:
            yield from bps.clear_checkpoint()
            yield from _bounded_call(
                verify_camera, timeout_s=10, what="verify FLIR Line0",
            )
            failed = False
            reserved = False
            try:
                yield from bps.abs_set(mcu.auto_pulses, True, wait=True)
                yield from _bounded_call(mcu.reserve, timeout_s=5, what="reserve MCU")
                reserved = True
                (run_id,) = yield from _bounded_call(
                    lambda: mcu.allocate_run_ids(1), timeout_s=5,
                    what="allocate run ID",
                )
                yield from _bounded_call(
                    lambda: mcu.set_reserved_illumination(color_command),
                    timeout_s=5, what="set LED color",
                )
                definition = AngularStack(
                    run_id, first_tenths, step_tenths, planes, lower, upper,
                )
                result = yield from _bounded_call(
                    lambda: acquire_pva_automatic_stack(
                        mcu, camera, definition, frames_per_plane, 1 << (pattern - 1),
                        interval_ms, uri,
                        resources_exclusive=True, external_trigger_configured=True,
                        continuous_capture_configured=True, illumination_ready=True,
                        timeout_s=timeout_s,
                    ),
                    timeout_s=timeout_s + 15, what="X28 triggered stack",
                )
                record = _PointRecord()
                record.values = {
                    "attempted_pulses": result.attempted_pulses,
                    "stored_frames": result.evidence.stored_frames,
                    "camera_frame_count_delta": result.camera_count_delta,
                }
                yield from bps.create("x28_triggered_stack")
                yield from bps.read(record)
                yield from bps.save()
            except BaseException:
                failed = True
                raise
            finally:
                try:
                    if reserved:
                        yield from self._release_mcu(mcu, failed)
                finally:
                    try:
                        yield from bps.abs_set(mcu.auto_pulses, False, wait=True)
                    except Exception:
                        if not failed:
                            raise
                        logger.exception("Could not disable pulses after stack error")

        yield from lock_wrapper(run(), mcu, camera, survey, rotation)

    def x28_software_stack(
        self,
        mcu: PvaUnifiedMcu,
        camera: MMCamera,
        survey: Esp300Stage,
        rotation: PvaRotationMotor,
        first_tenths: int,
        step_tenths: int,
        planes: int,
        color: LedColor,
        pattern: int,
        frames_per_plane: int = 1,
        ready_for_internal_snaps: bool = False,
    ) -> MsgGenerator[None]:
        """Acquire a bounded X28 stack with host snaps and physical TTL disabled.

        Confirm the camera is exclusive and in internal trigger mode, the LED
        path is ready, and the travel path is clear. X28 stays at the last plane.
        """
        if self._bounds is None:
            raise RuntimeError("X28 travel bounds are missing from the profile")
        lower, upper = self._bounds
        AngularStack(1, first_tenths, step_tenths, planes, lower, upper)
        if planes > 1 and step_tenths == 0:
            raise ValueError("a multi-plane X28 stack requires a nonzero step")
        if type(pattern) is not int or not 1 <= pattern <= 5:
            raise ValueError("LED pattern must be 1..5")
        if type(frames_per_plane) is not int or not 1 <= frames_per_plane <= 32:
            raise ValueError("frames per plane must be 1..32")
        if ready_for_internal_snaps is not True:
            raise ValueError("confirm camera, illumination and travel readiness")
        color_command = _COLOR_COMMAND[color]
        stores = []
        for _ in range(planes):
            info = camera.path_provider(camera.name)
            stores.append(str(Path(info.directory_path) / f"{info.filename}.zarr"))

        @bpp.run_decorator(
            md={
                "plan_name": "holospim_x28_software_stack",
                "camera": camera.name,
                "first_tenths": first_tenths,
                "step_tenths": step_tenths,
                "planes": planes,
                "led_color": color,
                "led_pattern": pattern,
                "plane_stores": stores,
            }
        )
        def run() -> MsgGenerator[None]:
            yield from bps.clear_checkpoint()
            yield from _bounded_call(
                lambda: require_idle_camera(camera, internal_trigger=True),
                timeout_s=5,
                what="verify camera ready for internal snaps",
            )
            yield from bps.abs_set(mcu.auto_pulses, False, wait=True)
            yield from _bounded_call(mcu.reserve, timeout_s=5, what="reserve MCU")
            failed = False
            try:
                ids = yield from _bounded_call(
                    lambda: mcu.allocate_run_ids(1),
                    timeout_s=5,
                    what="allocate X28 run ID",
                )
                yield from _bounded_call(
                    lambda: mcu.set_reserved_illumination(color_command),
                    timeout_s=5,
                    what="set LED color",
                )
                definition = AngularStack(
                    ids[0], first_tenths, step_tenths, planes, lower, upper
                )
                complete: list[PatternPlane] = []

                async def ignore_shot(_shot: object) -> None:
                    pass

                async def plane_complete(plane: PatternPlane) -> None:
                    complete.append(plane)

                yield from _bounded_call(
                    lambda: acquire_pva_pattern_shots(
                        mcu,
                        camera,
                        definition,
                        (pattern,) * frames_per_plane,
                        stores.__getitem__,
                        ignore_shot,
                        plane_complete,
                        resources_exclusive=True,
                        internal_trigger_configured=True,
                        illumination_ready=True,
                    ),
                    timeout_s=15 * (planes * frames_per_plane + 2),
                    what="X28 software stack",
                )
                record = _PointRecord()
                for plane in complete:
                    record.values = {
                        "plane": plane.plane,
                        "target_tenths": plane.target_tenths,
                        "stored_frames": plane.evidence.stored_frames,
                        "store_uri": stores[plane.plane],
                    }
                    yield from bps.create("x28_plane")
                    yield from bps.read(record)
                    yield from bps.save()
            except BaseException:
                failed = True
                raise
            finally:
                yield from self._release_mcu(mcu, failed)

        yield from lock_wrapper(run(), mcu, camera, survey, rotation)
