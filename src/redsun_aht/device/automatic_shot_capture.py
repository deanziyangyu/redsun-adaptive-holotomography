"""Finite automatic MCU stacks with concurrent, independently counted camera frames.

Experimental integration: the caller must establish physical camera readiness,
exclusive capture and electrical compatibility. Counts do not prove exposure order.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from redsun_aht.device.unified_mcu import (
        AngularStack,
        AutomaticShotSummary,
        UnifiedMcu,
    )


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """A camera sequence and host receipt timestamp, with caller-owned data."""

    sequence: int
    receipt_ns: int
    data: Any


class AutomaticCamera(Protocol):
    """One exclusively owned finite triggered-camera acquisition."""

    async def arm(self, frame_count: int) -> int:
        """Arm external triggering and return the preceding camera counter."""
        ...

    async def collect(self, frame_count: int) -> tuple[CameraFrame, ...]:
        """Drain independently counted frames while the MCU runs."""
        ...

    async def close(self) -> None:
        """Stop collection and release camera resources."""
        ...


class AutomaticCaptureTimeout(TimeoutError):
    """Timeout with an unassociated copied-frame snapshot and final count, if known."""

    def __init__(self, frames: tuple[CameraFrame, ...], attempted_pulses: int | None):
        super().__init__("automatic capture timed out")
        self.frames = frames
        self.attempted_pulses = attempted_pulses


@dataclass(frozen=True, slots=True)
class AutomaticFrameAssociation:
    """Host-order inferred association of one frame and one requested shot."""

    shot: int
    plane: int
    target_tenths: int
    pattern: int
    frame: CameraFrame


@dataclass(frozen=True, slots=True)
class AutomaticCapture:
    """Reconciled or unassociated finite automatic capture result."""

    status: str
    attempted_pulses: int
    frames: tuple[CameraFrame, ...]
    associations: tuple[AutomaticFrameAssociation, ...]
    quality: str


async def acquire_automatic_shots(
    owner: UnifiedMcu,
    camera: AutomaticCamera,
    definition: AngularStack,
    shots_per_plane: int,
    pattern_mask: int,
    interval_ms: int,
    *,
    capture_exclusive: bool,
    illumination_ready: bool,
    timeout_s: float,
) -> AutomaticCapture:
    """Arm camera, drain concurrently, and associate only matching finite counts.

    The owner's cancellation path requests MCU abort and quarantines the owner;
    this does not verify physical stopping. A failed capture is not replayed.
    """
    if capture_exclusive is not True or illumination_ready is not True:
        raise ValueError("exclusive camera and illumination readiness required")
    if type(shots_per_plane) is not int or not 1 <= shots_per_plane <= 64:
        raise ValueError("invalid shots per plane")
    if type(pattern_mask) is not int or not 1 <= pattern_mask <= 31:
        raise ValueError("invalid pattern mask")
    if type(interval_ms) is not int or not 1 <= interval_ms < 30000:
        raise ValueError("invalid interval")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("positive finite timeout required")
    count = definition.count * shots_per_plane
    tasks: list[asyncio.Task[Any]] = []
    collector: asyncio.Task[Any] | None = None
    try:
        baseline = await asyncio.wait_for(camera.arm(count), timeout_s)
        if type(baseline) is not int or baseline < 0:
            raise ValueError("invalid camera baseline")
        collector = asyncio.create_task(camera.collect(count))
        execution = asyncio.create_task(
            owner.automatic_shots(
                definition,
                shots_per_plane,
                pattern_mask,
                interval_ms,
                camera_armed=True,
                illumination_ready=True,
                before_start=getattr(camera, "confirm_ready", None),
            )
        )
        tasks = [collector, execution]
        done, pending = await asyncio.wait(
            tasks, timeout=timeout_s, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            task.result()
        if pending:
            frames = getattr(camera, "partial_frames", ())
            if not isinstance(frames, tuple) or not all(
                isinstance(f, CameraFrame) for f in frames
            ):
                frames = ()
            attempts = (
                execution.result().attempted_pulses
                if execution.done()
                and not execution.cancelled()
                and execution.exception() is None
                else None
            )
            raise AutomaticCaptureTimeout(frames, attempts)
        finish = getattr(camera, "finish", None)
        if callable(finish):
            await asyncio.wait_for(finish(), timeout_s)
        summary: AutomaticShotSummary = execution.result()
        frames = collector.result()
        if not isinstance(frames, tuple) or any(
            not isinstance(frame, CameraFrame)
            or type(frame.sequence) is not int
            or frame.sequence < 0
            or type(frame.receipt_ns) is not int
            or frame.receipt_ns < 0
            for frame in frames
        ):
            raise ValueError("invalid camera frame records")
        if any(b.receipt_ns < a.receipt_ns for a, b in itertools.pairwise(frames)):
            raise ValueError("camera receipt timestamps out of order")
        if summary.attempted_pulses != count or tuple(
            f.sequence for f in frames
        ) != tuple(range(baseline + 1, baseline + count + 1)):
            return AutomaticCapture(
                "count_or_sequence_mismatch",
                summary.attempted_pulses,
                frames,
                (),
                "unassociated",
            )
        patterns = tuple(i + 1 for i in range(5) if pattern_mask & (1 << i))
        associations = tuple(
            AutomaticFrameAssociation(
                i,
                i // shots_per_plane,
                definition.first + definition.step * (i // shots_per_plane),
                patterns[i % len(patterns)],
                frame,
            )
            for i, frame in enumerate(frames)
        )
        return AutomaticCapture(
            "complete_order_inferred", count, frames, associations, "order_inferred"
        )
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
    finally:
        pending_error = sys.exc_info()[1]
        pending: set[asyncio.Task[Any]] = set()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=2)
            for task in done:
                if not task.cancelled():
                    task.exception()
            for task in pending:
                task.add_done_callback(
                    lambda task: None if task.cancelled() else task.exception()
                )
        if pending:
            logger.error("Acquisition tasks remain active; resource state is uncertain")
        if collector is not None and collector in pending:
            if pending_error is None:
                raise RuntimeError("camera collector still active; release unverified")
        else:
            try:
                await asyncio.shield(asyncio.wait_for(camera.close(), 2))
            except BaseException:
                if pending_error is None:
                    raise
