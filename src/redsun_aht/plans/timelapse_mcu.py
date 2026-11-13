"""Experimental Bluesky time-lapse adapter for MCU-managed Z blocks."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Literal

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp

from redsun_aht.device.software_shot_core import run_core_software_shots
from redsun_aht.device.unified_mcu import AngularStack
from redsun_aht.plans.dpct import _bounded_call
from redsun_aht.plans.timelapse import TimeLapsePoint, _PointRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


def timelapse_mcu_plan(
    owner: Any,
    core: Any,
    sink: Any,
    positions: Sequence[Any],
    timepoints: int,
    channels: Sequence[Any],
    patterns: Sequence[int],
    *,
    stack_for_block: Callable[[int], Any],
    move_position: Callable[[Any], Generator],
    configure_channel: Callable[[Any], Generator],
    open_shutter: Callable[[], Generator],
    close_shutter: Callable[[], Generator],
    shutter_gate: Literal["per-snap", "acquisition"],
    camera_exclusive: bool,
    illumination_ready: bool,
    bind_stage: bool = True,
    before_plane: Callable[[int, int], None] | None = None,
    block_timeout_s: float = 120.0,
    frame_period_s: float = 0.0,
    expected_exposure_ms: float | None = None,
    maximum_duration_s: float = 3600.0,
    plane_sequence: bool = False,
    camera_internal_triggered: bool = False,
    channel_patterns: Sequence[Sequence[int]] | None = None,
    channel_exposures_ms: Sequence[float] | None = None,
    frame_gate: Any | None = None,
    channel_gates: Sequence[Any] | None = None,
    timepoint_period_s: float = 0.0,
    axis_order: Literal["position-time", "time-position"] = "position-time",
    missed_timepoint_policy: Literal["continue", "abort"] = "continue",
    position_ids: Sequence[str] | None = None,
) -> Generator:
    """Acquire position/time/channel blocks; the MCU coordinates each Z stack.

    Caller owns the camera, illumination, stage, sink lifecycle and unique stack
    IDs. The sink must finish persisting each image before advancing. The
    frame period is a host frame-start floor across blocks. The timepoint
    period is a host start-to-start floor per position or global timepoint,
    depending on axis order. Neither is a
    sensor-exposure timestamp. Event docs are emitted after each
    complete block, while the sink retains partial data.
    """
    positions, channels, patterns = tuple(positions), tuple(channels), tuple(patterns)
    if not positions or not channels or not patterns:
        raise ValueError("positions, channels and patterns must be nonempty")
    labels = tuple(position_ids) if position_ids is not None else None
    if labels is not None and (
        len(labels) != len(positions)
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("position IDs must be nonempty, unique and match positions")
    if type(timepoints) is not int or timepoints < 1:
        raise ValueError("timepoints must be a positive integer")
    if axis_order not in ("position-time", "time-position"):
        raise ValueError("invalid time/position axis order")
    if missed_timepoint_policy not in ("continue", "abort"):
        raise ValueError("invalid missed timepoint policy")
    if shutter_gate not in ("per-snap", "acquisition"):
        raise ValueError("invalid shutter gate")
    if type(plane_sequence) is not bool or type(camera_internal_triggered) is not bool:
        raise ValueError("sequence policy must be boolean")
    if plane_sequence and (
        shutter_gate != "acquisition"
        or not camera_internal_triggered
        or frame_period_s != 0
    ):
        raise ValueError(
            "plane sequence requires internal trigger, open shutter "
            "and zero frame period"
        )
    if camera_exclusive is not True or illumination_ready is not True:
        raise ValueError("exclusive camera and illumination readiness required")
    if (
        isinstance(block_timeout_s, bool)
        or not math.isfinite(block_timeout_s)
        or block_timeout_s <= 0
    ):
        raise ValueError("invalid block timeout")
    for value, name, positive in (
        (frame_period_s, "frame period", False),
        (timepoint_period_s, "timepoint period", False),
        (maximum_duration_s, "maximum duration", True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)
        ):
            raise ValueError(f"invalid {name}")
    exposure = float(core.getExposure())
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("invalid exposure readback")
    if expected_exposure_ms is not None and (
        isinstance(expected_exposure_ms, bool)
        or not isinstance(expected_exposure_ms, (int, float))
        or not math.isfinite(expected_exposure_ms)
        or expected_exposure_ms <= 0
        or abs(expected_exposure_ms - exposure) > 0.01
    ):
        raise ValueError("exposure readback mismatch")
    recipes = (
        tuple(tuple(p) for p in channel_patterns)
        if channel_patterns is not None
        else tuple(patterns for _ in channels)
    )
    exposures = (
        tuple(channel_exposures_ms)
        if channel_exposures_ms is not None
        else tuple(exposure for _ in channels)
    )
    if len(recipes) != len(channels) or len(exposures) != len(channels):
        raise ValueError("one pattern recipe and exposure per channel required")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in exposures
    ):
        raise ValueError("invalid channel exposure")
    if frame_period_s and frame_period_s < max(exposures) / 1000:
        raise ValueError("frame period is shorter than exposure")
    if type(bind_stage) is not bool or (bind_stage and before_plane is not None):
        raise ValueError("external plane hook requires unbound stage")
    if any(
        not recipe
        or len(recipe) > 64
        or any(type(value) is not int or not 1 <= value <= 5 for value in recipe)
        for recipe in recipes
    ):
        raise ValueError("invalid illumination patterns")
    if shutter_gate == "acquisition" and any(
        len(set(recipe)) != 1 for recipe in recipes
    ):
        raise ValueError("open illumination requires one pattern per plane")
    if any(
        (len(recipe) if shutter_gate == "acquisition" else 1)
        * max(frame_period_s, channel_exposure / 1000)
        >= 15
        for recipe, channel_exposure in zip(recipes, exposures, strict=True)
    ):
        raise ValueError("frame schedule exceeds MCU callback timeout")
    if shutter_gate == "acquisition" and core.getAutoShutter():
        raise ValueError("core auto-shutter must be disabled")
    if channel_gates is not None and frame_gate is not None:
        raise ValueError("select a common gate or per-channel gates")
    gates = (
        tuple(channel_gates)
        if channel_gates is not None
        else tuple(frame_gate for _ in channels)
    )
    if len(gates) != len(channels):
        raise ValueError("one gate per channel required")
    if any(gate is not None for gate in gates) and (
        shutter_gate != "per-snap"
        or any(
            gate is None
            or not callable(getattr(gate, "open", None))
            or not callable(getattr(gate, "close", None))
            for gate in gates
        )
        or core.getAutoShutter()
    ):
        raise ValueError(
            "native frame gate requires per-snap policy and core auto-shutter off"
        )
    if (
        shutter_gate == "per-snap"
        and all(gate is None for gate in gates)
        and (not core.getAutoShutter() or not core.getShutterDevice())
    ):
        raise ValueError("enabled core auto-shutter and a shutter device are required")
    definitions = tuple(
        stack_for_block(i) for i in range(len(positions) * timepoints * len(channels))
    )
    if any(not isinstance(definition, AngularStack) for definition in definitions):
        raise ValueError("stack factory must return AngularStack")
    frame_count = sum(
        definition.count * len(recipes[i % len(channels)])
        for i, definition in enumerate(definitions)
    )
    if frame_count > 10_000 or sink.expected != frame_count:
        raise ValueError("sink frame count must match bounded acquisition")
    duration_estimate = sum(
        definition.count
        * len(recipes[i % len(channels)])
        * max(frame_period_s, exposures[i % len(channels)] / 1000)
        for i, definition in enumerate(definitions)
    )
    duration_estimate = max(
        duration_estimate,
        (len(positions) if axis_order == "position-time" else 1)
        * (timepoints - 1)
        * timepoint_period_s,
    )
    if duration_estimate > maximum_duration_s:
        raise ValueError("estimated duration exceeds maximum")
    record = _PointRecord()

    @bpp.run_decorator(
        md={
            "plan_name": "timelapse_mcu",
            "shutter_gate": shutter_gate,
            "channel_patterns": [list(recipe) for recipe in recipes],
            "channel_exposures_ms": list(exposures),
            "requested_frames": frame_count,
            "frame_period_s": frame_period_s,
            "timepoint_period_s": timepoint_period_s,
            "axis_order": axis_order,
            "position_ids": list(labels) if labels is not None else None,
            "missed_timepoint_policy": missed_timepoint_policy,
            "exposure_ms": exposure,
            "plane_sequence": plane_sequence,
        }
    )
    def run() -> Generator:
        sequence = 0
        block = 0
        last_started_ns: list[int] = []

        def body() -> Generator:
            nonlocal sequence, block
            if shutter_gate == "acquisition":
                yield from open_shutter()
            last_timepoint_ns: int | None = None
            outer_count = (
                len(positions) if axis_order == "position-time" else timepoints
            )
            inner_count = (
                timepoints if axis_order == "position-time" else len(positions)
            )
            for outer in range(outer_count):
                if axis_order == "position-time":
                    last_timepoint_ns = None
                for inner in range(inner_count):
                    p, t = (
                        (outer, inner)
                        if axis_order == "position-time"
                        else (inner, outer)
                    )
                    if axis_order == "position-time" and t == 0:
                        yield from move_position(positions[p])
                    if axis_order == "position-time" or p == 0:
                        deadline: int | None = None
                        if last_timepoint_ns is not None and timepoint_period_s > 0:
                            deadline = last_timepoint_ns + round(
                                timepoint_period_s * 1e9
                            )
                            if (
                                missed_timepoint_policy == "abort"
                                and time.perf_counter_ns() > deadline
                            ):
                                raise TimeoutError("timepoint deadline missed")
                            while (
                                delay := (deadline - time.perf_counter_ns()) / 1e9
                            ) > 0:
                                yield from bps.sleep(delay)
                        timepoint_start_ns = time.perf_counter_ns()
                        timepoint_lateness_ns = (
                            max(0, timepoint_start_ns - deadline)
                            if deadline is not None
                            else 0
                        )
                        last_timepoint_ns = timepoint_start_ns
                    if axis_order == "time-position":
                        yield from move_position(positions[p])
                    for c, channel in enumerate(channels):
                        yield from configure_channel(channel)
                        current_exposure = float(core.getExposure())
                        if (
                            not math.isfinite(current_exposure)
                            or abs(current_exposure - exposures[c]) > 0.01
                        ):
                            raise ValueError(
                                "exposure changed after channel configuration"
                            )
                        definition = definitions[block]
                        patterns = recipes[c]
                        selected_gate = gates[c]
                        starts: dict[int, int] = {}
                        receipts: dict[int, int] = {}
                        completed: list[TimeLapsePoint] = []

                        def started(shot: int, ns: int, starts=starts) -> None:
                            if ns == -1:
                                starts[shot] = -1
                                return
                            del ns
                            if last_started_ns:
                                deadline = last_started_ns[0] + round(
                                    frame_period_s * 1e9
                                )
                                while (
                                    delay := (deadline - time.perf_counter_ns()) / 1e9
                                ) > 0:
                                    time.sleep(delay)
                            actual = time.perf_counter_ns()
                            starts[shot] = actual
                            last_started_ns[:] = [actual]

                        def received(shot: int, ns: int, receipts=receipts) -> None:
                            receipts[shot] = ns

                        def receive(
                            shot: int,
                            plane: int,
                            pattern: int,
                            target: int,
                            image: Any,
                            p=p,
                            t=t,
                            c=c,
                            sequence=sequence,
                            starts=starts,
                            completed=completed,
                            receipts=receipts,
                            patterns=patterns,
                            timepoint_start_ns=timepoint_start_ns,
                            timepoint_lateness_ns=timepoint_lateness_ns,
                        ) -> None:
                            del target
                            point = TimeLapsePoint(
                                p,
                                t,
                                plane,
                                c,
                                shot % len(patterns),
                                sequence + shot,
                                starts[shot],
                                pattern,
                                timepoint_start_ns,
                                timepoint_lateness_ns,
                                labels[p] if labels is not None else None,
                            )
                            if plane_sequence:
                                sink.append(
                                    point, image, host_receipt_ns=receipts[shot]
                                )
                            else:
                                sink.append(point, image)
                            completed.append(point)

                        async def acquire(
                            definition=definition,
                            receive=receive,
                            started=started,
                            received=received,
                            block=block,
                            patterns=patterns,
                            selected_gate=selected_gate,
                        ) -> None:
                            await run_core_software_shots(
                                owner,
                                core,
                                definition,
                                len(patterns),
                                patterns * definition.count,
                                receive,
                                camera_exclusive=True,
                                illumination_ready=True,
                                bind_stage=bind_stage,
                                before_plane=before_plane,
                                auto_shutter=shutter_gate == "per-snap",
                                hold_illumination=shutter_gate == "acquisition"
                                and block < len(definitions) - 1,
                                frame_started=started,
                                frame_received=received,
                                plane_sequence=plane_sequence,
                                camera_internal_triggered=camera_internal_triggered,
                                frame_gate=selected_gate,
                            )

                        yield from _bounded_call(
                            acquire,
                            timeout_s=block_timeout_s,
                            what="MCU time-lapse block",
                        )
                        for point in completed:
                            record.values = {
                                "timelapse_position": point.position,
                                "timelapse_timepoint": point.timepoint,
                                "timelapse_z": point.z,
                                "timelapse_channel": point.channel,
                                "timelapse_pattern": point.pattern,
                                "timelapse_sequence": point.sequence,
                                "timelapse_host_start_ns": point.host_start_ns,
                                "timelapse_timepoint_start_ns": (
                                    point.timepoint_start_ns
                                ),
                                "timelapse_timepoint_lateness_ns": (
                                    point.timepoint_lateness_ns
                                ),
                            }
                            yield from bps.create(name="timelapse")
                            yield from bps.read(record)
                            yield from bps.save()
                        sequence += len(completed)
                        block += 1

        if shutter_gate == "acquisition":
            yield from bpp.finalize_wrapper(body(), close_shutter())
        else:
            yield from body()

    yield from run()
