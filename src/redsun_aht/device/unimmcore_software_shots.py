"""Experimental finite MCU software-shot acquisition through one MDA runner.

The MCU owns Z; MDA events carry indices but do not command a focus stage.
Frame associations are inferred from the exclusive, ordered host handshake.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event
from typing import TYPE_CHECKING, Any

from pymmcore_plus.mda import MDAEngine, MDARunner
from useq import MDAEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from redsun_aht.device.unified_mcu import AngularStack, UnifiedMcu


@dataclass(frozen=True)
class _Ready:
    shot: int
    plane: int
    pattern: int
    target: int
    consumed: Future[None]


class _ShotEngine(MDAEngine):
    def __init__(
        self,
        core: Any,
        ready: Queue,
        sink: Callable,
        timeout_s: float,
        stop: Event,
        shots_per_plane: int,
        before_plane: Callable[[int, int], None] | None,
        frame_started: Callable[[int, int], None] | None,
    ):
        super().__init__(
            core, use_hardware_sequencing=False, restore_initial_state=False
        )
        self.ready, self.sink, self.timeout_s, self.stop = ready, sink, timeout_s, stop
        self.shots_per_plane, self.before_plane = shots_per_plane, before_plane
        self.frame_started = frame_started
        self.completed = 0

    def exec_single_event(self, event: MDAEvent):
        try:
            entry = self.ready.get(timeout=self.timeout_s)
        except Empty as exc:
            raise TimeoutError("MCU readiness timed out") from exc
        if not isinstance(entry, _Ready):
            raise RuntimeError("MCU acquisition ended before readiness")
        try:
            if self.stop.is_set():
                raise RuntimeError("shot run cancelled")
            if entry.shot != self.completed or event.index != {
                "z": entry.plane,
                "s": entry.shot,
            }:
                raise RuntimeError("MCU and MDA shot order mismatch")
            if self.before_plane is not None and entry.shot % self.shots_per_plane == 0:
                self.before_plane(entry.plane, entry.target)
            if self.frame_started is not None:
                self.frame_started(entry.shot, time.monotonic_ns())
            if self.stop.is_set():
                raise RuntimeError("shot run cancelled before camera snap")
            payloads = list(super().exec_single_event(event))
            if len(payloads) != 1:
                raise RuntimeError("expected exactly one camera frame per shot")
            if self.stop.is_set():
                raise RuntimeError("shot run cancelled before frame delivery")
            image, _, _ = payloads[0]
            self.sink(entry.shot, entry.plane, entry.pattern, entry.target, image)
            yield payloads[0]
            self.completed += 1
            if not entry.consumed.done():
                entry.consumed.set_result(None)
        except BaseException as exc:
            if not entry.consumed.done():
                entry.consumed.set_exception(exc)
            raise


async def run_software_shot_mda(
    owner: UnifiedMcu,
    core: Any,
    definition: AngularStack,
    shots_per_plane: int,
    patterns: Sequence[int],
    sink: Callable,
    *,
    camera_exclusive: bool,
    illumination_ready: bool,
    timeout_s: float = 15.0,
    bind_stage: bool = True,
    before_plane: Callable[[int, int], None] | None = None,
    auto_shutter: bool | None = None,
    frame_started: Callable[[int, int], None] | None = None,
) -> None:
    """Run one ordered MDA sequence; sink executes synchronously in runner thread.

    The caller owns the core exclusively. A failed shot requests the existing
    owner's abort and quarantines it. This is not physical stop confirmation.
    With auto_shutter=False the caller keeps illumination open, patterns must
    be constant within each plane, and one MCU shot coordinates each plane.
    None preserves the existing per-frame handshake. With auto_shutter=True,
    the caller supplies a configured core shutter and enables core auto-shutter;
    the core handles shutter cycling for each per-frame software snap.
    The optional frame_started callback reports a host timestamp immediately
    before the core's snap path; it is not an exposure timestamp.
    """
    if camera_exclusive is not True or illumination_ready is not True:
        raise ValueError("exclusive camera and illumination readiness required")
    if type(bind_stage) is not bool or (bind_stage and before_plane is not None):
        raise ValueError("external plane hook requires unbound stage")
    if type(shots_per_plane) is not int or not 1 <= shots_per_plane <= 64:
        raise ValueError("invalid shots per plane")
    patterns = tuple(patterns)
    if len(patterns) != definition.count * shots_per_plane or any(
        type(p) is not int or not 1 <= p <= 5 for p in patterns
    ):
        raise ValueError("one valid pattern per shot required")
    if not 0 < timeout_s < 30:
        raise ValueError("timeout must be positive and below firmware timeout")
    if auto_shutter is not None and type(auto_shutter) is not bool:
        raise ValueError("invalid auto-shutter policy")
    if auto_shutter is True and (
        not core.getAutoShutter() or not core.getShutterDevice()
    ):
        raise ValueError("enabled core auto-shutter and a shutter device are required")
    if auto_shutter is False and any(
        len(set(patterns[i : i + shots_per_plane])) != 1
        for i in range(0, len(patterns), shots_per_plane)
    ):
        raise ValueError("open illumination requires one pattern per plane")
    if auto_shutter is False and core.getAutoShutter():
        raise ValueError("core auto-shutter must be disabled")
    ready: Queue = Queue(maxsize=1)
    stop = Event()
    engine = _ShotEngine(
        core, ready, sink, timeout_s, stop, shots_per_plane, before_plane, frame_started
    )
    engine.include_frame_position_metadata = False
    runner = MDARunner()
    runner.set_engine(engine)
    events = [
        MDAEvent(index={"z": i // shots_per_plane, "s": i})
        for i in range(len(patterns))
    ]

    async def capture(shot: int, plane: int, pattern: int, target: int) -> None:
        consumed: Future[None] = Future()
        ready.put_nowait(_Ready(shot, plane, pattern, target, consumed))
        await asyncio.wrap_future(consumed)

    def run_events() -> None:
        runner.run(events)
        if engine.completed != len(patterns):
            raise RuntimeError("MDA ended before all shots")

    async def run_owner() -> None:
        if auto_shutter is False:

            async def capture_plane(
                shot: int, plane: int, pattern: int, target: int
            ) -> None:
                if shot != plane:
                    raise RuntimeError("MCU plane order mismatch")
                for offset in range(shots_per_plane):
                    await capture(
                        plane * shots_per_plane + offset, plane, pattern, target
                    )

            await owner.software_shots(
                definition,
                1,
                patterns[::shots_per_plane],
                capture_plane,
                camera_exclusive=True,
                illumination_ready=True,
                snap_timeout_s=timeout_s,
                bind_stage=bind_stage,
            )
            return
        await owner.software_shots(
            definition,
            shots_per_plane,
            patterns,
            capture,
            camera_exclusive=True,
            illumination_ready=True,
            snap_timeout_s=timeout_s,
            bind_stage=bind_stage,
        )

    owner_task = asyncio.create_task(run_owner())
    runner_task = asyncio.create_task(asyncio.to_thread(run_events))
    try:
        await asyncio.gather(asyncio.shield(owner_task), asyncio.shield(runner_task))
        if engine.completed != len(patterns):
            raise RuntimeError("MDA ended before all shots")
    except BaseException:
        stop.set()
        # A runner failure may already have reached the owner callback. Give
        # its abort path time to finish before cancelling an active operation.
        if not owner_task.done():
            await asyncio.wait({owner_task}, timeout=2.0)
        if not owner_task.done():
            owner_task.cancel()
        with contextlib.suppress(Full):
            ready.put_nowait(None)
        await asyncio.gather(owner_task, return_exceptions=True)
        # Do not detach a worker that may still own the camera.
        await asyncio.gather(asyncio.shield(runner_task), return_exceptions=True)
        raise
