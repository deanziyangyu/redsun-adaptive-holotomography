"""Shared IX83 stage axis and motion I/O used by the ASI service."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from fastcs.attributes import (
    AttributeIO,
    AttributeIORef,
    AttrR,
    AttrRW,
    AttrW,
)
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, Float, String
from fastcs.logging import logger

from redsun_aht.robotic.device._motion import HALTED_OUTCOME, MoveHalted
from redsun_aht.robotic.device.asi._stage import AsiXYStage
from redsun_aht.robotic.device.ludl._commands import (
    decode_relative_move_command,
    decode_traverse_command,
)
from ._stage_telemetry import LIVE_TELEMETRY_PERIOD, TelemetryRef

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

STAGE_GROUP: Final = "stage"
UNITS: Final = "um"
DEFAULT_POLL_PERIOD_MS: Final = 200
MIN_POLL_PERIOD_MS: Final = 20
MAX_POLL_PERIOD_MS: Final = 5000


@dataclass
class AxisRef(AttributeIORef):
    """Names the axis served by a position attribute."""

    axis: str = ""


@dataclass
class MotionStateRef(AttributeIORef):
    """Selects one service-side motion-state readback."""

    key: str = ""


@dataclass
class StopRef(AttributeIORef):
    """Distinct reference type for the independent stop write path."""


@dataclass
class TraverseRef(AttributeIORef):
    """Distinct reference type for a bounded absolute XY traverse request."""


@dataclass
class RelativeMoveRef(AttributeIORef):
    """Distinct reference type for a bounded relative XY move request."""


class MotionProfileRef(AttributeIORef):
    """Select the read/write controller motion-profile record."""


class LudlSerialStageIO(AttributeIO[float, AxisRef]):
    """Adapt one bounded XY stage facade to FastCS axis position attributes."""

    def __init__(self, stage: AsiXYStage) -> None:
        super().__init__()
        self._stage = stage
        self._axis_lock = asyncio.Lock()
        self.busy = False
        self.outcome = "idle"

    async def _move_axis(self, axis: str, value: float) -> tuple[float, float]:
        current = await self._stage.read_xy()
        target = (value, current[1]) if axis == "x" else (current[0], value)
        return await self._stage.move_xy(target)

    async def send(self, attr: AttrW[float, AxisRef], value: float) -> None:
        """Apply a requested axis target through the serial client."""
        if attr.io_ref.axis not in ("x", "y"):
            raise ValueError(f"unsupported lateral axis {attr.io_ref.axis!r}")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("stage target must be finite")
        async with self._axis_lock:
            if self.busy:
                raise RuntimeError("stage motion is already active")
            self.busy = True
            self.outcome = "moving"
            try:
                actual = await self._move_axis(attr.io_ref.axis, float(value))
            except BaseException:
                # A failed wait is not evidence that the physical move stopped.
                self.outcome = "unknown"
                raise
            else:
                self.outcome = "completed"
            finally:
                self.busy = False
        if isinstance(attr, AttrR):
            await attr.update(actual[0] if attr.io_ref.axis == "x" else actual[1])

    async def update(self, attr: AttrR[float, AxisRef]) -> None:
        """Refresh one axis readback from the service."""
        if attr.io_ref.axis not in ("x", "y"):
            raise ValueError(f"unsupported lateral axis {attr.io_ref.axis!r}")
        async with self._axis_lock:
            actual = await self._stage.read_xy()
        await attr.update(actual[0] if attr.io_ref.axis == "x" else actual[1])


class LudlStopIO(AttributeIO[bool, StopRef]):
    """Send a service-level hard-stop request without waiting on axis I/O."""

    def __init__(self, stage: AsiXYStage, axis_io: LudlSerialStageIO) -> None:
        super().__init__()
        self._stage = stage
        self._axis_io = axis_io

    async def send(self, attr: AttrW[bool, StopRef], value: bool) -> None:
        """Apply or acknowledge a stop request."""
        del attr
        if value:
            if self._axis_io.busy:
                self._axis_io.outcome = "stop_requested"
            try:
                await self._stage.stop()
            except BaseException:
                if self._axis_io.busy:
                    self._axis_io.outcome = "unknown"
                raise


class LudlMotionStateIO(AttributeIO[Any, MotionStateRef]):
    """Publish service-side motion state without making a serial query."""

    def __init__(self, axis_io: LudlSerialStageIO) -> None:
        super().__init__()
        self._axis_io = axis_io

    async def update(self, attr: AttrR[bool | str, MotionStateRef]) -> None:
        """Publish the service's motion state."""
        if attr.io_ref.key == "busy":
            await attr.update(self._axis_io.busy)
        elif attr.io_ref.key == "outcome":
            await attr.update(self._axis_io.outcome)
        else:
            raise ValueError(f"unsupported motion state key {attr.io_ref.key!r}")


class LudlTraverseIO(AttributeIO[str, TraverseRef]):
    """Accept bounded XY targets and execute them in a service-owned task."""

    def __init__(self, stage: AsiXYStage, axis_io: LudlSerialStageIO) -> None:
        super().__init__()
        self._stage = stage
        self._axis_io = axis_io
        self._request_number = 0
        self._task: asyncio.Task[None] | None = None
        self._request_windows: list[tuple[str, int, int | None]] = []

    async def send(self, attr: AttrW[str, TraverseRef], value: str) -> None:
        """Start a bounded traverse request and record its identity."""
        del attr
        target = decode_traverse_command(value)
        if self._axis_io.busy:
            logger.warning(
                f"traverse rejected: axis busy={self._axis_io.busy} "
                f"task_done={None if self._task is None else self._task.done()}"
            )
            raise RuntimeError("stage motion is already active")
        if self._task is not None and not self._task.done():
            raise RuntimeError("a traverse task is already active")

        self._request_number += 1
        request_id = f"traverse-{self._request_number:06d}"
        accepted_ns = time.monotonic_ns()
        self._request_windows.append((request_id, accepted_ns, None))
        self._axis_io.busy = True
        self._axis_io.outcome = "accepted"
        self._task = asyncio.create_task(
            self._run(target, request_id), name=f"ludl-{request_id}"
        )

    def request_for_receive_bracket(
        self, started_ns: int, finished_ns: int
    ) -> str | None:
        """Return the unique traverse that contains a complete report bracket."""
        matches = [
            request_id
            for request_id, accepted_ns, completed_ns in self._request_windows
            if started_ns >= accepted_ns
            and (completed_ns is None or finished_ns <= completed_ns)
        ]
        return matches[0] if len(matches) == 1 else None

    def _close_request_window(self, request_id: str, completed_ns: int) -> None:
        """Close the matching request interval when the service resolves it."""
        for index in range(len(self._request_windows) - 1, -1, -1):
            existing_id, accepted_ns, existing_end_ns = self._request_windows[index]
            if existing_id == request_id and existing_end_ns is None:
                self._request_windows[index] = (
                    existing_id,
                    accepted_ns,
                    completed_ns,
                )
                return

    async def _run(self, target: tuple[float, float], request_id: str) -> None:
        result = "unknown"
        error: str | None = None
        actual_xy_um: tuple[float, float] | None = None
        try:
            async with self._axis_io._axis_lock:
                self._axis_io.outcome = "moving"
                actual_xy_um = await self._stage.move_xy(
                    (float(target[0]), float(target[1]))
                )
        except asyncio.CancelledError:
            self._axis_io.outcome = "unknown"
            error = "traverse task cancelled during service shutdown"
            raise
        except MoveHalted as exc:
            # A stop request ended the move: the facade reports the settled position and the
            # service publishes a known outcome instead of staying busy until its timeout.
            actual_xy_um = exc.position_xy_um
            result = HALTED_OUTCOME
            self._axis_io.outcome = result
            logger.info(f"Bounded traverse halted: {exc}")
        except Exception as exc:
            self._axis_io.outcome = "unknown"
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Native Ludl bounded traverse failed; outcome unknown")
        else:
            result = "completed"
            self._axis_io.outcome = "completed"
        finally:
            self._axis_io.busy = False
            completed_ns = time.monotonic_ns()
            self._close_request_window(request_id, completed_ns)

    async def cancel_and_wait(self) -> None:
        """Cancel the owned task during shutdown and wait for serial cleanup."""
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class LudlRelativeMoveIO(AttributeIO[str, RelativeMoveRef]):
    """Accept a bounded relative XY displacement in a service-owned task."""

    def __init__(
        self,
        move_relative: Callable[[tuple[float, float]], Awaitable[tuple[float, float]]],
        axis_io: LudlSerialStageIO,
    ) -> None:
        super().__init__()
        self._move_relative = move_relative
        self._axis_io = axis_io
        self._request_number = 0
        self._task: asyncio.Task[None] | None = None

    async def send(self, attr: AttrW[str, RelativeMoveRef], value: str) -> None:
        """Apply a requested relative XY move."""
        del attr
        delta = decode_relative_move_command(value)
        if self._axis_io.busy:
            raise RuntimeError("stage motion is already active")
        if self._task is not None and not self._task.done():
            raise RuntimeError("a relative move task is already active")
        self._request_number += 1
        request_id = f"relative-{self._request_number:06d}"
        self._axis_io.busy = True
        self._axis_io.outcome = "accepted"
        self._task = asyncio.create_task(
            self._run(delta, request_id), name=f"ludl-{request_id}"
        )

    async def _run(self, delta: tuple[float, float], request_id: str) -> None:
        result = "unknown"
        error: str | None = None
        actual_xy_um: tuple[float, float] | None = None
        try:
            async with self._axis_io._axis_lock:
                self._axis_io.outcome = "moving"
                actual_xy_um = await self._move_relative(delta)
        except asyncio.CancelledError:
            self._axis_io.outcome = "unknown"
            error = "relative move cancelled during service shutdown"
            raise
        except Exception as exc:
            self._axis_io.outcome = "unknown"
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Ludl relative move failed; outcome unknown")
        else:
            result = "completed"
            self._axis_io.outcome = "completed"
        finally:
            self._axis_io.busy = False

    async def cancel_and_wait(self) -> None:
        """Cancel the owned task during shutdown and wait for serial cleanup."""
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class LudlAxisController(Controller):
    """One named stage axis exposed as a read/write position."""

    def __init__(self, axis: str, io: LudlSerialStageIO, poll_period_s: float) -> None:
        super().__init__(ios=[io])
        self.position = AttrRW(
            Float(units=UNITS),
            io_ref=AxisRef(axis=axis, update_period=poll_period_s),
        )
