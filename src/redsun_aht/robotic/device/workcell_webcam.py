"""PVAccess client for the sample-handler workcell recorder."""

from ophyd_async.core import SignalR, SignalW, StandardReadable
from ophyd_async.fastcs.core import fastcs_connector


class WorkcellWebcam(StandardReadable):
    start: SignalW[bool]
    snap: SignalW[bool]
    finish: SignalW[bool]
    status: SignalR[str]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))
