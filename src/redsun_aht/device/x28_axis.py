"""Micrometre Positioner axis backed by the unified MCU point-move service."""

from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING

from ophyd_async.core import (
    MovableLogic,
    StandardMovable,
    StandardReadable,
    soft_signal_rw,
)

if TYPE_CHECKING:
    from ophyd_async.core import TimeoutCalculator

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu


class X28Logic(MovableLogic[float]):
    """Translate Positioner commands and cancel through the active MCU request."""

    def __init__(self, axis: X28Axis) -> None:
        super().__init__(setpoint=axis.setpoint, readback=axis.position)
        self.axis = axis

    async def check_move(self, new_position: float) -> None:
        """Check the commissioned travel bounds before submitting a request."""
        self.axis.to_tenths(new_position)

    async def calculate_timeout(
        self, old_position: float, new_position: float
    ) -> float:
        """Allow time for the service to preflight and settle."""
        return 20

    async def move(self, new_position: float, timeout: TimeoutCalculator) -> None:
        """Submit one finite move and update the settled readback."""
        await self.setpoint.set(new_position, timeout=timeout())

    async def stop(self) -> None:
        """Cancel the matching move at the service."""
        if not self.axis._simulated:
            await self.axis._cancel()


class X28Axis(StandardReadable, StandardMovable[float]):
    """Expose fast Z with an explicit signed linear scale and angular bounds."""

    def __init__(
        self, owner: PvaUnifiedMcu, um_per_turn: float, bounds: tuple[int, int]
    ) -> None:
        low, high = bounds
        if (
            not math.isfinite(um_per_turn) or um_per_turn == 0
            or type(low) is not int or type(high) is not int
            or not -3600 <= low < high <= 3600
        ):
            raise ValueError("X28 requires a signed scale and valid angular limits")
        self._um_per_turn = um_per_turn
        self._bounds = bounds
        self._read = owner.read_state
        self._move = owner.move_x_absolute
        self._cancel = owner.stop_x_move
        self._simulated = False
        self._mock_position = 0
        self.position = soft_signal_rw(float, 0.0, units="µm", getter=self._position)
        self.setpoint = soft_signal_rw(
            float, 0.0, units="µm", setter=self._set_position
        )
        super().__init__()
        self.add_readables([self.position])

    def to_tenths(self, position: float) -> int:
        """Convert micrometres to the nearest controller step inside the limits."""
        if not math.isfinite(position):
            raise ValueError("X28 position must be finite")
        low, high = sorted(self._to_um(value) for value in self._bounds)
        if not low <= position <= high:
            raise ValueError("X28 position outside the configured travel limits")
        return round(position / self._um_per_turn * 3600)

    def _to_um(self, angle: int) -> float:
        return angle / 3600 * self._um_per_turn

    async def _position(self) -> float:
        if self._simulated:
            return self._to_um(self._mock_position)
        return self._to_um((await self._read()).position_tenths)

    async def _set_position(self, position: float | None) -> float:
        if position is None:
            raise ValueError("X28 position required")
        target = self.to_tenths(position)
        if self._simulated:
            self._mock_position = target
            actual = target
        else:
            actual = (await self._move(target)).position_tenths
        settled = self._to_um(actual)
        await self.position.set(settled)
        return settled

    @cached_property
    def movable_logic(self) -> X28Logic:
        """Expose the Redsun axis protocols."""
        return X28Logic(self)
