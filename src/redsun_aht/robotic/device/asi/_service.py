"""Ophyd-async client for the native ASI FastCS stage service."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Annotated as A

from ophyd_async.core import (
    SignalR,
    SignalRW,
    SignalW,
    StandardReadable,
    StandardReadableFormat,
)
from ophyd_async.fastcs.core import fastcs_connector
from redsun.log import Loggable

from redsun_aht.robotic.device.mmcore._stage import MMStage

if TYPE_CHECKING:
    from redsun_aht.robotic.device.containers import ReadableDeviceMap
    from redsun_aht.robotic.device.mmcore._stage import MMAxis


class AsiMotion(StandardReadable, Loggable):
    """Service controls and terminal state for bounded native ASI moves.

    ``traverse`` accepts an absolute ``[x_um, y_um]`` endpoint in the same
    coordinate frame as the stage service. ``busy`` and ``outcome`` are
    service-side state, not controller fault codes.
    """

    stop_: SignalW[bool]
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


class AsiSerialStage(StandardReadable, Loggable):
    """Client for stage position and motion signals from the native ASI service.

    The service prefix is shared with its ``Stage`` and ``Motion`` controllers.
    Axis point moves reuse :class:`MMStage`'s settled move/readback behavior;
    ``motion.traverse`` exposes the service's bounded endpoint request.
    """

    xy: A[MMStage, StandardReadableFormat.CHILD]
    motion: A[AsiMotion, StandardReadableFormat.CHILD]

    @property
    def axis(self) -> ReadableDeviceMap[MMAxis]:
        """Expose the native XY axes through the shared stage-plan protocol."""
        return self.xy.axis

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name)
        with self.add_children_as_readables(StandardReadableFormat.CHILD):
            self.xy = MMStage(prefix, name="xy")
            self.motion = AsiMotion(prefix, name="motion")


__all__ = ["AsiMotion", "AsiSerialStage"]
