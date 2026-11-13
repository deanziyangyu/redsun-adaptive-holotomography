"""PVAccess clients for the ARH illumination and legacy MCU services."""

from ophyd_async.core import SignalR, SignalRW, SignalW, StandardReadable
from ophyd_async.fastcs.core import fastcs_connector


class LumencorSpectra(StandardReadable):
    shutter_open: SignalRW[bool]
    channel: SignalRW[str]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))


class UnifiedEmmsh(StandardReadable):
    commanded_steps: SignalR[str]
    stop_: SignalW[bool]
    blank: SignalW[bool]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))
