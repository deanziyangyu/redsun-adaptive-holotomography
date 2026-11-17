"""Experimental multidimensional extension of the IX83 time-lapse ordering.

Device-specific motion, camera acquisition and durable storage are supplied by
the caller.  This plan emits host-order coordinates after each capture returns;
they are not measured stage positions or exposure timestamps.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


@dataclass(frozen=True)
class TimeLapsePoint:
    """One requested frame in the position/time/Z/channel/pattern order."""

    position: int
    timepoint: int
    z: int
    channel: int
    pattern: int
    sequence: int
    host_start_ns: int = 0
    pattern_id: int | None = None
    timepoint_start_ns: int = -1
    timepoint_lateness_ns: int = -1
    position_id: str | None = None


class _PointRecord:
    name = "timelapse_frame"

    def __init__(self) -> None:
        self.values: dict[str, int | float | str] = {}

    def read(self) -> dict[str, dict[str, Any]]:
        timestamp = time.time()
        return {
            key: {"value": val, "timestamp": timestamp}
            for key, val in self.values.items()
        }

    def describe(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "source": "PLAN:host_clock"
                if key.endswith("_ns")
                else "PLAN:requested",
                "dtype": (
                    "string"
                    if isinstance(value, str)
                    else "number"
                    if isinstance(value, float)
                    else "integer"
                ),
                "shape": [],
            }
            for key, value in self.values.items()
        }


def timelapse_plan(
    positions: Sequence[Any],
    frames_per_position: int,
    *,
    z_values: Sequence[Any] = (None,),
    channels: Sequence[Any] = (None,),
    patterns: Sequence[Any] = (None,),
    frame_period_s: float = 0.0,
    exposure_ms: float,
    maximum_duration_s: float = 3600.0,
    shutter_gate: Literal["per-snap", "acquisition"] = "per-snap",
    move_position: Callable[[Any], Generator],
    capture: Callable[[TimeLapsePoint, Any, Any, Any], Generator],
    open_shutter: Callable[[], Generator],
    close_shutter: Callable[[], Generator],
) -> Generator:
    """Execute a finite, caller-owned time-lapse; callback completion is required.

    The exposure is an already verified readback supplied by the caller.  Frame
    storage and hardware cleanup beyond the shutter are the caller's responsibility.
    The duration estimate is a preflight lower bound, not a runtime deadline.
    """
    positions, z_values, channels, patterns = map(
        tuple, (positions, z_values, channels, patterns)
    )
    if not all((positions, z_values, channels, patterns)):
        raise ValueError("all dimensions must be non-empty")
    if type(frames_per_position) is not int or frames_per_position < 1:
        raise ValueError("frames_per_position must be a positive integer")
    for value, name, positive in (
        (frame_period_s, "frame_period_s", False),
        (exposure_ms, "exposure_ms", True),
        (maximum_duration_s, "maximum_duration_s", True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)
        ):
            raise ValueError(f"invalid {name}")
    if frame_period_s and frame_period_s < exposure_ms / 1000:
        raise ValueError("frame period is shorter than exposure")
    if shutter_gate not in ("per-snap", "acquisition"):
        raise ValueError("invalid shutter gate")
    count = (
        len(positions)
        * frames_per_position
        * len(z_values)
        * len(channels)
        * len(patterns)
    )
    if count > 10_000:
        raise ValueError("time-lapse exceeds 10000 frames")
    estimate = count * max(frame_period_s, exposure_ms / 1000)
    if estimate > maximum_duration_s:
        raise ValueError("estimated duration exceeds maximum")
    record = _PointRecord()

    @bpp.run_decorator(
        md={
            "plan_name": "timelapse",
            "requested_frames": count,
            "frame_period_s": frame_period_s,
            "exposure_ms": exposure_ms,
            "shutter_gate": shutter_gate,
        }
    )
    def run() -> Generator:
        last_start: float | None = None
        started_ns = 0
        sequence = 0

        def body() -> Generator:
            nonlocal last_start, sequence
            if shutter_gate == "acquisition":
                yield from open_shutter()
            for p, position in enumerate(positions):
                yield from move_position(position)
                for t in range(frames_per_position):
                    for z, z_value in enumerate(z_values):
                        for c, channel in enumerate(channels):
                            for s, pattern in enumerate(patterns):
                                if last_start is not None:
                                    delay = frame_period_s - (
                                        time.perf_counter() - last_start
                                    )
                                    while delay > 0:
                                        yield from bps.sleep(delay)
                                        delay = frame_period_s - (
                                            time.perf_counter() - last_start
                                        )
                                point = TimeLapsePoint(p, t, z, c, s, sequence)

                                def take(
                                    point=point,
                                    z_value=z_value,
                                    channel=channel,
                                    pattern=pattern,
                                ) -> Generator:
                                    nonlocal last_start, started_ns
                                    if shutter_gate == "per-snap":
                                        yield from open_shutter()
                                    started_ns = time.perf_counter_ns()
                                    last_start = started_ns / 1_000_000_000
                                    yield from capture(
                                        replace(point, host_start_ns=started_ns),
                                        z_value,
                                        channel,
                                        pattern,
                                    )

                                if shutter_gate == "per-snap":
                                    yield from bpp.finalize_wrapper(
                                        take(), close_shutter()
                                    )
                                else:
                                    yield from take()
                                record.values = {
                                    "timelapse_position": p,
                                    "timelapse_timepoint": t,
                                    "timelapse_z": z,
                                    "timelapse_channel": c,
                                    "timelapse_pattern": s,
                                    "timelapse_sequence": sequence,
                                    "timelapse_host_start_ns": started_ns,
                                }
                                yield from bps.create(name="timelapse")
                                yield from bps.read(record)
                                yield from bps.save()
                                sequence += 1

        if shutter_gate == "acquisition":
            yield from bpp.finalize_wrapper(body(), close_shutter())
        else:
            yield from body()

    yield from run()
