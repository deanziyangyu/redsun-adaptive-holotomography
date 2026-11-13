"""Experimental Bluesky time-lapse over service-owned camera and MCU."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Literal

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp

from redsun_aht.device.pva_internal_plane_capture import (
    PlaneCapture,
    PlaneTiming,
    acquire_pva_internal_planes,
)
from redsun_aht.device.pva_pattern_shots import acquire_pva_pattern_shots
from redsun_aht.device.unified_mcu import AngularStack
from redsun_aht.plans.dpct import _bounded_call
from redsun_aht.plans.timelapse import _PointRecord
from redsun_aht.storage.timelapse_journal import TimeLapseJournal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Sequence

logger = logging.getLogger(__name__)


def pva_timelapse_plan(
    owner: Any,
    camera: Any,
    positions: Sequence[Any],
    timepoints: int,
    channels: Sequence[Any],
    channel_patterns: Sequence[int],
    channel_exposures_ms: Sequence[float],
    frames_per_plane: int,
    *,
    stack_for_block: Callable[[int], AngularStack],
    store_for_plane: Callable[[int, int], str],
    move_position: Callable[[Any], Generator],
    configure_channel: Callable[[Any], Generator],
    open_gate: Callable[[], Generator],
    close_gate: Callable[[], Generator],
    resources_exclusive: bool,
    internal_trigger_configured: bool,
    illumination_ready: bool,
    software_trigger_configured: bool = False,
    bind_stage: bool = True,
    before_plane: Callable[[int, int], Awaitable[None]] | None = None,
    on_plane_timing: Callable[[PlaneTiming], None] | None = None,
    hold_illumination: bool = False,
    end_illumination: Literal["blank", "center"] = "blank",
    axis_order: Literal["position-time", "time-position"] = "position-time",
    timepoint_period_s: float = 0,
    missed_timepoint_policy: Literal["continue", "abort"] = "continue",
    position_ids: Sequence[str] | None = None,
    channel_labels: Sequence[str] | None = None,
    confirmation_id: str | None = None,
    channel_frame_patterns: Sequence[Sequence[int]] | None = None,
    block_timeout_s: float = 120,
    journal_path: str | None = None,
    auto_shutter: bool = False,
    open_shot_gate: Callable[[], Awaitable[None]] | None = None,
    close_shot_gate: Callable[[], Awaitable[None]] | None = None,
) -> Generator:
    """Acquire bounded service stores, publishing one host-associated event per plane.

    The caller owns the camera and a cooperative MCU reservation, verifies camera
    trigger/exposure configuration, and supplies gate and channel bindings. Store
    URIs are precomputed before motion and included in run metadata. No camera
    exposure timestamps or hardware-trigger associations are implied.
    On success the LED ends blank or at center. The legacy center command can
    pulse the camera output when automatic pulses are enabled.
    """
    positions, channels, channel_patterns, channel_exposures_ms = (
        tuple(positions),
        tuple(channels),
        tuple(channel_patterns),
        tuple(channel_exposures_ms),
    )
    if not positions or not channels or type(timepoints) is not int or timepoints < 1:
        raise ValueError(
            "nonempty positions and channels and positive timepoints required"
        )
    if len(channel_patterns) != len(channels) or any(
        type(pattern) is not int or not 1 <= pattern <= 5
        for pattern in channel_patterns
    ):
        raise ValueError("one valid pattern per channel required")
    if len(channel_exposures_ms) != len(channels) or any(
        isinstance(exposure, bool)
        or not isinstance(exposure, (int, float))
        or not math.isfinite(exposure)
        or exposure <= 0
        for exposure in channel_exposures_ms
    ):
        raise ValueError("one positive exposure per channel required")
    if type(frames_per_plane) is not int or not 1 <= frames_per_plane <= 32:
        raise ValueError("frames per plane must be 1..32")
    recipes = (
        tuple(tuple(recipe) for recipe in channel_frame_patterns)
        if channel_frame_patterns is not None
        else tuple((pattern,) * frames_per_plane for pattern in channel_patterns)
    )
    if len(recipes) != len(channels) or any(
        len(recipe) != frames_per_plane
        or any(type(pattern) is not int or not 1 <= pattern <= 5 for pattern in recipe)
        for recipe in recipes
    ):
        raise ValueError("one valid frame pattern per channel and frame required")
    if (
        any(
            value is not True
            for value in (
                resources_exclusive,
                illumination_ready,
            )
        )
        or not owner.reservation_held
        or type(internal_trigger_configured) is not bool
        or type(software_trigger_configured) is not bool
        or internal_trigger_configured == software_trigger_configured
    ):
        raise ValueError("reserved MCU and exclusive configured resources required")
    if software_trigger_configured and not hasattr(camera, "software_trigger"):
        raise ValueError("camera service does not expose software triggering")
    if type(bind_stage) is not bool or type(hold_illumination) is not bool:
        raise ValueError("stage binding and hold must be boolean")
    if type(auto_shutter) is not bool:
        raise ValueError("auto_shutter must be boolean")
    if auto_shutter:
        if not callable(open_shot_gate) or not callable(close_shot_gate):
            raise ValueError("auto shutter requires per-shot gate callbacks")
    elif open_shot_gate is not None or close_shot_gate is not None:
        raise ValueError("per-shot gate callbacks require auto shutter")
    if end_illumination not in ("blank", "center"):
        raise ValueError("end illumination must be blank or center")
    if before_plane is not None and bind_stage:
        raise ValueError("external plane hook requires unbound stage")
    if axis_order not in ("position-time", "time-position"):
        raise ValueError("invalid axis order")
    if missed_timepoint_policy not in ("continue", "abort"):
        raise ValueError("invalid missed timepoint policy")
    labels = tuple(position_ids) if position_ids is not None else None
    channel_names = tuple(channel_labels) if channel_labels is not None else None
    if channel_names is not None and (
        len(channel_names) != len(channels)
        or any(
            not isinstance(label, str) or not label.strip() for label in channel_names
        )
    ):
        raise ValueError("channel labels must be nonempty and match channels")
    if confirmation_id is not None and (
        not isinstance(confirmation_id, str) or not confirmation_id.strip()
    ):
        raise ValueError("confirmation ID must be a nonempty string")
    if labels is not None and (
        len(labels) != len(positions)
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("position IDs must be nonempty, unique and match positions")
    for value, name, allow_zero in (
        (timepoint_period_s, "timepoint period", True),
        (block_timeout_s, "block timeout", False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            raise ValueError(f"invalid {name}")
    count = len(positions) * timepoints * len(channels)
    if count > 10000:
        raise ValueError("too many blocks")
    definitions = tuple(stack_for_block(i) for i in range(count))
    if any(not isinstance(item, AngularStack) for item in definitions):
        raise ValueError("stack factory must return AngularStack")
    if sum(item.count * frames_per_plane for item in definitions) > 10000:
        raise ValueError("too many frames")
    uris = tuple(
        tuple(store_for_plane(i, p) for p in range(item.count))
        for i, item in enumerate(definitions)
    )
    flat_uris = tuple(uri for group in uris for uri in group)
    if any(not isinstance(uri, str) or not uri.strip() for uri in flat_uris) or len(
        set(flat_uris)
    ) != len(flat_uris):
        raise ValueError("unique plane stores required across the run")
    record = _PointRecord()

    @bpp.run_decorator(
        md={
            "plan_name": "pva_timelapse",
            "plane_stores": uris,
            "frames_per_plane": frames_per_plane,
            "channel_exposures_ms": channel_exposures_ms,
            "channel_frame_patterns": recipes,
            "axis_order": axis_order,
            "timepoint_period_s": timepoint_period_s,
            "missed_timepoint_policy": missed_timepoint_policy,
            "position_ids": labels,
            "channel_labels": channel_names,
            "stage_binding": "mcu" if bind_stage else "external",
            "requested_stacks": [
                {
                    "first_tenths": item.first,
                    "step_tenths": item.step,
                    "planes": item.count,
                    "lower_tenths": item.lower,
                    "upper_tenths": item.upper,
                }
                for item in definitions
            ],
            "confirmation_id": confirmation_id,
            "mcu_run_ids": [definition.run_id for definition in definitions],
            "association": "host_order_inferred",
            "end_illumination": end_illumination,
            "auto_shutter": auto_shutter,
            "camera_trigger_mode": (
                "software" if software_trigger_configured else "internal"
            ),
        }
    )
    def run() -> Generator:
        def body() -> Generator:
            block = 0
            last_ns = None
            gate_channel = None
            outer_count = (
                len(positions) if axis_order == "position-time" else timepoints
            )
            inner_count = (
                timepoints if axis_order == "position-time" else len(positions)
            )
            for outer in range(outer_count):
                if axis_order == "position-time":
                    last_ns = None
                for inner in range(inner_count):
                    p, t = (
                        (outer, inner)
                        if axis_order == "position-time"
                        else (inner, outer)
                    )
                    if axis_order == "position-time" and t == 0:
                        yield from move_position(positions[p])
                    if axis_order == "position-time" or p == 0:
                        deadline = None
                        if last_ns is not None and timepoint_period_s:
                            deadline = last_ns + round(timepoint_period_s * 1e9)
                            delay = (deadline - time.perf_counter_ns()) / 1e9
                            if delay < 0 and missed_timepoint_policy == "abort":
                                raise TimeoutError("timepoint deadline missed")
                            while delay > 0:
                                yield from bps.sleep(delay)
                                delay = (deadline - time.perf_counter_ns()) / 1e9
                        last_ns = time.perf_counter_ns()
                        lateness_ns = (
                            max(0, last_ns - deadline) if deadline is not None else 0
                        )
                    if axis_order == "time-position":
                        yield from move_position(positions[p])
                    for c, channel in enumerate(channels):
                        if gate_channel is not None and gate_channel != c:
                            yield from close_gate()
                            gate_channel = None
                        yield from configure_channel(channel)
                        exposure = yield from _bounded_call(
                            lambda: camera.exposure.get_value(cached=False),
                            timeout_s=5,
                            what="camera exposure readback",
                        )
                        if (
                            not math.isfinite(exposure)
                            or abs(exposure - channel_exposures_ms[c]) > 0.01
                        ):
                            raise ValueError("camera exposure readback mismatch")
                        if not auto_shutter and gate_channel is None:
                            yield from open_gate()
                            gate_channel = c
                        captured = []
                        frames = []

                        async def on_plane(result: Any, captured=captured) -> None:
                            captured.append(result)

                        async def on_pattern_shot(result: Any, frames=frames) -> None:
                            frames.append(result)

                        async def on_pattern_plane(
                            result: Any, captured=captured
                        ) -> None:
                            captured.append(
                                PlaneCapture(
                                    result.plane,
                                    result.target_tenths,
                                    0,
                                    result.evidence,
                                )
                            )

                        async def acquire(
                            block=block,
                            c=c,
                            on_plane=on_plane,
                            on_pattern_shot=on_pattern_shot,
                            on_pattern_plane=on_pattern_plane,
                        ) -> None:
                            hold = (hold_illumination and block < count - 1) or (
                                end_illumination == "center" and block == count - 1
                            )
                            if (
                                software_trigger_configured
                                or auto_shutter
                                or len(set(recipes[c])) > 1
                            ):
                                await acquire_pva_pattern_shots(
                                    owner,
                                    camera,
                                    definitions[block],
                                    recipes[c],
                                    lambda index: uris[block][index],
                                    on_pattern_shot,
                                    on_pattern_plane,
                                    resources_exclusive=True,
                                    internal_trigger_configured=internal_trigger_configured,
                                    software_trigger_configured=software_trigger_configured,
                                    illumination_ready=True,
                                    bind_stage=bind_stage,
                                    before_plane=before_plane,
                                    hold_illumination=hold,
                                    open_shot_gate=open_shot_gate,
                                    close_shot_gate=close_shot_gate,
                                )
                                return
                            await acquire_pva_internal_planes(
                                owner,
                                camera,
                                definitions[block],
                                frames_per_plane,
                                [recipes[c][0]] * definitions[block].count,
                                lambda index: uris[block][index],
                                on_plane,
                                resources_exclusive=True,
                                internal_trigger_configured=True,
                                illumination_ready=True,
                                bind_stage=bind_stage,
                                before_plane=before_plane,
                                on_timing=on_plane_timing,
                                hold_illumination=hold,
                            )

                        try:
                            yield from _bounded_call(
                                acquire,
                                timeout_s=block_timeout_s,
                                what="PVA time-lapse block",
                            )
                        finally:
                            for frame in frames:
                                record.values = {
                                    "timelapse_position": p,
                                    "timelapse_timepoint": t,
                                    "timelapse_channel": c,
                                    "timelapse_z": frame.plane,
                                    "timelapse_block": block,
                                    "timelapse_frame": frame.index,
                                    "timelapse_pattern": frame.pattern,
                                    "timelapse_camera_count": frame.camera_count,
                                    "timelapse_host_receipt_ns": frame.host_receipt_ns,
                                }
                                yield from bps.create(name="timelapse_frame")
                                yield from bps.read(record)
                                yield from bps.save()
                            for plane in captured:
                                record.values = {
                                    "timelapse_position": p,
                                    "timelapse_timepoint": t,
                                    "timelapse_channel": c,
                                    "timelapse_z": plane.plane,
                                    "timelapse_block": block,
                                    "timelapse_pattern": plane.pattern,
                                    "timelapse_target_tenths": plane.target_tenths,
                                    "timelapse_frames": plane.evidence.stored_frames,
                                    "timelapse_timepoint_start_ns": last_ns,
                                    "timelapse_timepoint_lateness_ns": lateness_ns,
                                }
                                yield from bps.create(name="timelapse_plane")
                                yield from bps.read(record)
                                yield from bps.save()
                        block += 1

        failed = False
        try:
            yield from body()
        except BaseException:
            failed = True
            raise
        finally:
            try:
                yield from close_gate()
            except Exception:
                if not failed:
                    failed = True
                    raise
                logger.exception(
                    "Failed to close acquisition gate after acquisition error"
                )
            finally:
                if owner.reservation_held:
                    try:
                        commands = (
                            ("00",)
                            if failed or end_illumination == "blank"
                            else ("01", "15")
                        )
                        for command in commands:
                            yield from _bounded_call(
                                lambda command=command: owner.set_reserved_illumination(
                                    command
                                ),
                                timeout_s=5,
                                what="set final illumination",
                            )
                    except Exception:
                        if not failed:
                            raise
                        logger.exception(
                            "Failed to set final illumination after acquisition error"
                        )
                else:
                    if not failed:
                        raise RuntimeError(
                            "MCU reservation uncertain; blanking not confirmed"
                        )
                    logger.warning("MCU reservation uncertain; blanking not confirmed")

    if journal_path is None:
        yield from run()
    else:
        journal = TimeLapseJournal(journal_path)
        failed = False
        try:
            yield from bpp.subs_wrapper(run(), journal)
        except BaseException:
            failed = True
            raise
        finally:
            try:
                journal.close()
            except Exception:
                if not failed:
                    raise
                logger.exception("Journal failed after acquisition error")
