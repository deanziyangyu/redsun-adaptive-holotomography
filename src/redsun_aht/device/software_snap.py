"""Explicit one-shot service interface; caller must own the camera exclusively."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ophyd_async.core import Device, SignalR, SignalW
from ophyd_async.fastcs.core import fastcs_connector

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SoftwareSnapCamera(Device):
    """Request exactly one software snap from a deferred-frame camera service.

    A failed or timed-out request has an uncertain outcome and is never retried.
    Frame and count are separate PVA reads and require exclusive camera ownership.
    """

    snap_request: SignalW[bool]
    frame_count: SignalR[int]
    buffer: SignalR[np.ndarray]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))

    async def snap(self) -> tuple[int, NDArray[Any]]:
        """Send once and return the service count and an owned pixel copy."""
        before = await self.frame_count.get_value(cached=False)
        await self.snap_request.set(True)
        after = await self.frame_count.get_value(cached=False)
        if after != before + 1:
            raise RuntimeError("snap frame count mismatch; outcome uncertain")
        frame = await self.buffer.get_value(cached=False)
        return after, np.array(frame, copy=True)
