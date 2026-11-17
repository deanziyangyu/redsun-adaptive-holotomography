"""Ophyd-async client for the native ASI FastCS stage service."""

from __future__ import annotations

import json
from typing import Annotated as A
from typing import TypeVar

from ophyd_async.core import (
    SignalR,
    SignalRW,
    SignalW,
    StandardReadable,
    StandardReadableFormat,
    callback_on_mock_put,
    set_mock_value,
    soft_signal_rw,
)
from ophyd_async.fastcs.core import fastcs_connector
from redsun.log import Loggable

from redsun_aht.robotic.device.containers import ReadableDeviceMap
from redsun_aht.robotic.device.mmcore._stage import MMAxis, MMStage

_MOCK_TRAVEL_UM = 1000.0


def _bounded_mock_position(value: float, *, enabled: bool = True) -> float:
    if enabled and not -_MOCK_TRAVEL_UM <= value <= _MOCK_TRAVEL_UM:
        raise ValueError("Mock LX4000 target exceeds the startup-relative travel envelope")
    return value

class AsiMotion(StandardReadable, Loggable):
    """Service controls and terminal state for bounded native ASI moves.

    ``traverse`` accepts an absolute ``[x_um, y_um]`` endpoint in the same
    coordinate frame as the stage service. ``busy`` and ``outcome`` are
    service-side state, not controller fault codes.
    """

    travel_bounds: SignalW[bool]
    stop_: SignalW[bool]
    z_limits: SignalR[str]
    xy_limits: SignalR[str]
    z_counts_per_um: SignalR[float]
    busy: A[SignalR[bool], StandardReadableFormat.CONFIG_SIGNAL]
    outcome: A[SignalR[str], StandardReadableFormat.CONFIG_SIGNAL]
    traverse: SignalW[str]
    move_relative: SignalW[str]
    telemetry_json: A[SignalR[str], StandardReadableFormat.CONFIG_SIGNAL]
    profile: A[SignalRW[str], StandardReadableFormat.CONFIG_SIGNAL]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(
            name=name,
            connector=fastcs_connector(f"{prefix}Motion:", self),
        )


AxisT = TypeVar("AxisT", bound=MMAxis)


class AsiAxisMap(ReadableDeviceMap[AxisT]):
    """The positioner groups XYZ under this map and stops them together."""

    _halt: SignalW[bool] | None = None

    async def stop(self, success: bool = False) -> None:
        if self._halt is None:
            raise RuntimeError("ASI motion stop is not bound")
        await self._halt.set(True)


class AsiStageAxes(MMStage):
    axis: A[AsiAxisMap[MMAxis], StandardReadableFormat.CHILD]


class AsiSerialStage(StandardReadable, Loggable):
    """Client for stage position and motion signals from the native ASI service.

    The service prefix is shared with its ``Stage`` and ``Motion`` controllers.
    Axis point moves reuse :class:`MMStage`'s settled move/readback behavior;
    ``motion.traverse`` exposes the service's bounded endpoint request.
    """

    xy: A[AsiStageAxes, StandardReadableFormat.CHILD]
    motion: A[AsiMotion, StandardReadableFormat.CHILD]

    @property
    def axis(self) -> ReadableDeviceMap[MMAxis]:
        """Expose the native XY axes through the shared stage-plan protocol."""
        return self.xy.axis

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name)
        with self.add_children_as_readables(StandardReadableFormat.CHILD):
            self.xy = AsiStageAxes(prefix, name="xy")
            self.motion = AsiMotion(prefix, name="motion")


    async def connect(self, *args, **kwargs) -> None:
        mock = kwargs.get("mock", args[0] if args else False)
        if mock:
            for axis in ("x", "y", "z"):
                if axis not in self.xy.axis:
                    motor = MMAxis()
                    with motor.add_children_as_readables(StandardReadableFormat.HINTED_SIGNAL):
                        motor.position = soft_signal_rw(
                            float, 0.0, units="um", precision=1,
                        )
                    self.xy.axis[axis] = motor
        await super().connect(*args, **kwargs)
        if mock:
            bounds_enabled = True

            def configure_bounds(enabled: bool) -> None:
                nonlocal bounds_enabled
                bounds_enabled = enabled

            callback_on_mock_put(self.motion.travel_bounds, configure_bounds)
            for motor in self.xy.axis.values():
                callback_on_mock_put(motor.position, lambda value: _bounded_mock_position(value, enabled=bounds_enabled))
            set_mock_value(self.motion.z_limits, json.dumps([-_MOCK_TRAVEL_UM, _MOCK_TRAVEL_UM]))
            set_mock_value(self.motion.xy_limits, json.dumps({axis: [-_MOCK_TRAVEL_UM, _MOCK_TRAVEL_UM] for axis in ("x", "y")}))
            set_mock_value(self.motion.z_counts_per_um, 10.0)
            set_mock_value(self.motion.profile, json.dumps({
                axis: {"backlash_mm": distance}
                for axis, distance in zip("xyz", (0.016, 0.016, 0.010), strict=True)
            }))
        object.__setattr__(self.xy.axis, "_halt", self.motion.stop_)


__all__ = ["AsiMotion", "AsiSerialStage"]


class AsiFilterWheel(StandardReadable):
    """Wheel PVs on the shared LX4000 controller."""

    busy: SignalR[bool]
    last_slot: SignalR[int]
    slot: SignalW[int]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(f"{prefix}Wheel:", self))

    async def connect(self, *args, **kwargs) -> None:
        await super().connect(*args, **kwargs)
        if not kwargs.get("mock", args[0] if args else False):
            return

        def select_slot(slot: int) -> None:
            if not 0 <= slot <= 7:
                raise ValueError("Filter slot must be in 0..7")
            set_mock_value(self.last_slot, slot)

        set_mock_value(self.busy, False)
        set_mock_value(self.last_slot, -1)
        callback_on_mock_put(self.slot, select_slot)
