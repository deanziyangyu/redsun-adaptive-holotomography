"""Open-loop rotation motor served by the shared unified MCU owner."""

from __future__ import annotations

import asyncio
import math
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    Device,
    MovableLogic,
    SignalR,
    SignalW,
    StandardMovable,
    StandardReadable,
    StandardReadableFormat,
    soft_signal_rw,
)
from ophyd_async.fastcs.core import fastcs_connector

from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu

if TYPE_CHECKING:
    from ophyd_async.core import DeviceMock, TimeoutCalculator

    from redsun_aht.device.unified_mcu import RotationState


class RotationClient(Device):
    """Rotation-only PVA interface to the shared MCU service."""

    rotation_request: SignalW[str]
    rotation_result: SignalR[str]
    reservation: SignalW[str]
    reservation_status: SignalR[str]

    def __init__(self, prefix: str) -> None:
        super().__init__(connector=fastcs_connector(prefix, self))
        self._active = False
        self._uncertain = False
        self._reserving = False
        self._reservation_token: str | None = None
        self._last_allocated_id = 0
        self._reservation_generation = 0
        self._rotation_lock = asyncio.Lock()

    async def rotation(self, action: str, steps: int | None = None) -> RotationState:
        """Execute a correlated rotation transaction."""
        return await PvaUnifiedMcu.rotation(cast("PvaUnifiedMcu", self), action, steps)

    async def _rotation_transaction(
        self, action: str, steps: int | None
    ) -> RotationState:
        return await PvaUnifiedMcu._rotation_transaction(
            cast("PvaUnifiedMcu", self), action, steps
        )

    async def reserve(self) -> None:
        """Acquire the service's cooperative reservation."""
        await PvaUnifiedMcu.reserve(cast("PvaUnifiedMcu", self))

    async def release_reservation(self) -> None:
        """Release the service's cooperative reservation."""
        await PvaUnifiedMcu.release_reservation(cast("PvaUnifiedMcu", self))

    @property
    def reservation_held(self) -> bool:
        """Whether this client holds a certain reservation."""
        return (
            self._reservation_token is not None
            and not self._reserving
            and not self._uncertain
        )


class RotationLogic(MovableLogic[float]):
    """Wait for the firmware step counter to reach the requested grid point."""

    def __init__(self, axis: PvaRotationAxis) -> None:
        super().__init__(setpoint=axis.setpoint, readback=axis.position)
        self.axis = axis

    async def check_move(self, new_position: float) -> None:
        """Validate the target before issuing a command."""
        self.axis.to_steps(new_position)

    async def calculate_timeout(
        self, old_position: float, new_position: float
    ) -> float:
        """Use the configured move timeout."""
        return self.axis.timeout

    async def move(self, new_position: float, timeout: TimeoutCalculator) -> None:
        """Wait for the step-counter-based move to finish."""
        await self.setpoint.set(new_position, timeout=timeout())

    async def stop(self) -> None:
        """Request a decelerated firmware stop."""
        if self.axis.simulated:
            return
        state = await self.axis.mcu.rotation("stop")
        while state.moving:
            await asyncio.sleep(0.05)
            state = await self.axis.mcu.rotation("read")


class PvaRotationAxis(StandardReadable, StandardMovable[float]):
    """Bluesky movable rotation axis with an assumed, configurable degree scale.

    Position is the MCU's commanded step counter, not an encoder measurement.
    The default scale comes from the EMMsh roll/pitch convention and the sign is
    reversed for this installation. Positions are unwrapped.
    """

    def __init__(
        self,
        prefix: str,
        *,
        name: str = "",
        steps_per_revolution: float = 2048,
        gear_ratio: float = 1,
        direction: int = -1,
        timeout: float = 120,
    ) -> None:
        if not math.isfinite(steps_per_revolution) or steps_per_revolution <= 0:
            raise ValueError("steps per revolution must be positive")
        if not math.isfinite(gear_ratio) or gear_ratio <= 0 or direction not in (-1, 1):
            raise ValueError("positive gear ratio and direction +1 or -1 required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive timeout required")
        self.steps_per_turn = steps_per_revolution * gear_ratio
        self.direction = direction
        self.timeout = timeout
        self.simulated = False
        self._simulated_steps = 0
        self.mcu = RotationClient(prefix)
        self.position = soft_signal_rw(
            float,
            0.0,
            units="deg",
            precision=3,
            getter=self.read_position,
        )
        self.setpoint = soft_signal_rw(float, 0.0, units="deg", setter=self._move)
        super().__init__(name=name)
        self.add_readables([self.position], StandardReadableFormat.HINTED_SIGNAL)

    async def connect(
        self,
        mock: bool | DeviceMock[Any] = False,
        timeout: float = DEFAULT_TIMEOUT,
        force_reconnect: bool = False,
    ) -> None:
        """Use a local step counter when the entire service is mocked."""
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        self.simulated = bool(mock)

    def to_steps(self, degrees: float) -> int:
        """Convert degrees to the nearest assumed physical step."""
        if not math.isfinite(degrees):
            raise ValueError("rotation target must be finite")
        steps = round(self.direction * degrees * self.steps_per_turn / 360)
        if not -(2**31) <= steps < 2**31:
            raise ValueError("rotation target exceeds the firmware counter")
        return steps

    def to_degrees(self, steps: int) -> float:
        """Convert the software step count to assumed degrees."""
        return self.direction * steps * 360 / self.steps_per_turn

    async def read_position(self) -> float:
        """Read the firmware's software step counter in assumed degrees."""
        if self.simulated:
            return self.to_degrees(self._simulated_steps)
        return self.to_degrees((await self.mcu.rotation("read")).position)

    async def _move(self, degrees: float | None) -> float:
        if degrees is None:
            raise ValueError("rotation target required")
        target = self.to_steps(degrees)
        if self.simulated:
            self._simulated_steps = target
            position = self.to_degrees(target)
            await self.position.set(position)
            return position
        await self.mcu.reserve()
        try:
            return await self._move_reserved(target)
        finally:
            if self.mcu.reservation_held:
                state = await self.mcu.rotation("read")
                if not state.moving:
                    await self.mcu.release_reservation()

    async def _move_reserved(self, target: int) -> float:
        state = await self.mcu.rotation("read")
        await self.position.set(self.to_degrees(state.position))
        if not state.enabled:
            await self.mcu.rotation("enable")
        state = await self.mcu.rotation("move", target)
        await self.position.set(self.to_degrees(state.position))
        deadline = asyncio.get_running_loop().time() + self.timeout
        while state.moving:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("rotation move timed out; motor may still be moving")
            await asyncio.sleep(0.05)
            state = await self.mcu.rotation("read")
            await self.position.set(self.to_degrees(state.position))
        if state.position != target:
            raise RuntimeError("rotation stopped before reaching its commanded target")
        return self.to_degrees(state.position)

    @cached_property
    def movable_logic(self) -> RotationLogic:
        """Expose the standard Redsun movable and locatable axis interface."""
        return RotationLogic(self)

    async def zero(self) -> None:
        """Set the stationary software counter to zero without homing."""
        if self.simulated:
            self._simulated_steps = 0
            await self.position.set(0.0)
            return
        state = await self.mcu.rotation("read")
        if state.moving:
            raise RuntimeError("cannot zero a moving rotation axis")
        await self.mcu.rotation("zero")
        await self.position.set(0.0)

    async def disable(self) -> None:
        """Remove coil drive; position may change without being detected."""
        if self.simulated:
            return
        await self.mcu.rotation("disable")


class PvaRotationMotor(StandardReadable):
    """Single-axis motor suitable for Redsun's motor protocol."""

    def __init__(
        self,
        prefix: str,
        *,
        name: str = "",
        steps_per_revolution: float = 2048,
        gear_ratio: float = 1,
        direction: int = -1,
        timeout: float = 120,
        enable_controller_jog: bool = False,
    ) -> None:
        if type(enable_controller_jog) is not bool:
            raise ValueError("enable_controller_jog must be a boolean")
        self.enable_controller_jog = enable_controller_jog
        self.rotation = PvaRotationAxis(
            prefix,
            steps_per_revolution=steps_per_revolution,
            gear_ratio=gear_ratio,
            direction=direction,
            timeout=timeout,
        )
        super().__init__(name=name)
        self.add_readables([self.rotation])

    @property
    def axis(self) -> dict[str, PvaRotationAxis]:
        """Return the rotation axis by its stable name."""
        return {"rotation": self.rotation}
