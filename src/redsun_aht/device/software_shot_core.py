"""Host software snaps coordinated by the exclusive MCU owner, without MDAEngine."""

from __future__ import annotations

import asyncio
import time
from threading import Event
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


async def run_core_software_shots(
    owner: Any,
    core: Any,
    definition: Any,
    shots_per_plane: int,
    patterns: Sequence[int],
    sink: Callable,
    *,
    camera_exclusive: bool,
    illumination_ready: bool,
    bind_stage: bool = True,
    before_plane: Callable[[int, int], None] | None = None,
    auto_shutter: bool,
    frame_started: Callable[[int, int], None] | None = None,
    timeout_s: float = 15.0,
    hold_illumination: bool = False,
    plane_sequence: bool = False,
    camera_internal_triggered: bool = False,
    frame_received: Callable[[int, int], None] | None = None,
    frame_gate: Any | None = None,
) -> None:
    """Take ordered software snaps while the MCU owns Z and illumination.

    A blocked camera worker is joined on cancellation before returning camera
    ownership. Host readiness and frame association are not exposure evidence.
    """
    if camera_exclusive is not True or illumination_ready is not True:
        raise ValueError("exclusive camera and illumination readiness required")
    if type(bind_stage) is not bool or (bind_stage and before_plane is not None):
        raise ValueError("external plane hook requires unbound stage")
    if type(shots_per_plane) is not int or not 1 <= shots_per_plane <= 64:
        raise ValueError("invalid shots per plane")
    if type(auto_shutter) is not bool:
        raise ValueError("auto-shutter policy must be boolean")
    if frame_gate is not None and (
        not auto_shutter
        or not callable(getattr(frame_gate, "open", None))
        or not callable(getattr(frame_gate, "close", None))
    ):
        raise ValueError("frame gate requires per-frame policy and open/close")
    if type(hold_illumination) is not bool or (auto_shutter and hold_illumination):
        raise ValueError("illumination hold requires auto-shutter off")
    if type(plane_sequence) is not bool or type(camera_internal_triggered) is not bool:
        raise ValueError("sequence policy must be boolean")
    if plane_sequence and (auto_shutter or not camera_internal_triggered):
        raise ValueError("plane sequence requires internal trigger and open shutter")
    patterns = tuple(patterns)
    if len(patterns) != definition.count * shots_per_plane or any(
        type(pattern) is not int or not 1 <= pattern <= 5 for pattern in patterns
    ):
        raise ValueError("one valid pattern per frame required")
    if not 0 < timeout_s < 20:
        raise ValueError("invalid snap timeout")
    if auto_shutter:
        if frame_gate is None and (
            not core.getAutoShutter() or not core.getShutterDevice()
        ):
            raise ValueError(
                "enabled core auto-shutter and a shutter device are required"
            )
        if frame_gate is not None and core.getAutoShutter():
            raise ValueError("core auto-shutter must be disabled for a native gate")
    else:
        if core.getAutoShutter():
            raise ValueError("core auto-shutter must be disabled")
        if any(
            len(set(patterns[i : i + shots_per_plane])) != 1
            for i in range(0, len(patterns), shots_per_plane)
        ):
            raise ValueError("open illumination requires one pattern per plane")
    stop = Event()

    def snap_one(shot: int, plane: int, pattern: int, target: int) -> None:
        if stop.is_set():
            raise RuntimeError("capture cancelled")
        if before_plane is not None and shot % shots_per_plane == 0:
            before_plane(plane, target)
        if frame_started is not None:
            frame_started(shot, time.perf_counter_ns())
        if stop.is_set():
            raise RuntimeError("capture cancelled before snap")
        core.snapImage()
        image = np.asarray(core.getImage()).copy()
        if frame_received is not None:
            frame_received(shot, time.perf_counter_ns())
        if stop.is_set():
            raise RuntimeError("capture cancelled before frame delivery")
        sink(shot, plane, pattern, target, image)

    def sequence_plane(plane: int, pattern: int, target: int) -> None:
        if stop.is_set():
            raise RuntimeError("capture cancelled")
        if before_plane is not None:
            before_plane(plane, target)
        if core.isSequenceRunning():
            raise RuntimeError("camera sequence already running")
        core.clearCircularBuffer()
        frames = []
        try:
            core.startSequenceAcquisition(shots_per_plane, 0, True)
            deadline = time.monotonic() + timeout_s
            while len(frames) < shots_per_plane:
                if stop.is_set():
                    raise RuntimeError("capture cancelled")
                if core.isBufferOverflowed():
                    raise RuntimeError("camera buffer overflow")
                if core.getRemainingImageCount():
                    image = np.asarray(core.popNextImage()).copy()
                    frames.append((image, time.perf_counter_ns()))
                elif not core.isSequenceRunning():
                    raise RuntimeError("camera sequence ended before all frames")
                elif time.monotonic() >= deadline:
                    raise TimeoutError("camera sequence timed out")
                else:
                    time.sleep(0.001)
            while core.isSequenceRunning():
                if stop.is_set():
                    raise RuntimeError("capture cancelled")
                if core.isBufferOverflowed():
                    raise RuntimeError("camera buffer overflow")
                if time.monotonic() >= deadline:
                    raise TimeoutError("camera sequence did not stop")
                time.sleep(0.001)
            if core.isBufferOverflowed() or core.getRemainingImageCount():
                raise RuntimeError("camera sequence overflow or extra frames")
        finally:
            if core.isSequenceRunning():
                core.stopSequenceAcquisition()
        for offset, (image, received_ns) in enumerate(frames):
            if stop.is_set():
                raise RuntimeError("capture cancelled before frame delivery")
            index = plane * shots_per_plane + offset
            if frame_started is not None:
                frame_started(index, -1)
            if frame_received is not None:
                frame_received(index, received_ns)
            sink(index, plane, pattern, target, image)

    async def run_worker(function: Callable, *args: Any) -> None:
        worker = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            await asyncio.shield(worker)
        except BaseException:
            stop.set()
            await asyncio.gather(worker, return_exceptions=True)
            raise

    async def capture(shot: int, plane: int, pattern: int, target: int) -> None:
        if frame_gate is None:
            await run_worker(snap_one, shot, plane, pattern, target)
            return
        try:
            await frame_gate.open()
            await run_worker(snap_one, shot, plane, pattern, target)
        finally:
            await asyncio.shield(frame_gate.close())

    async def capture_plane(shot: int, plane: int, pattern: int, target: int) -> None:
        if shot != plane:
            raise RuntimeError("MCU plane order mismatch")
        if plane_sequence:
            await run_worker(sequence_plane, plane, pattern, target)
            return
        for offset in range(shots_per_plane):
            await capture(plane * shots_per_plane + offset, plane, pattern, target)

    await owner.software_shots(
        definition,
        shots_per_plane if auto_shutter else 1,
        patterns if auto_shutter else patterns[::shots_per_plane],
        capture if auto_shutter else capture_plane,
        camera_exclusive=True,
        illumination_ready=True,
        snap_timeout_s=timeout_s,
        bind_stage=bind_stage,
        hold_illumination=hold_illumination,
    )
