"""Ophyd-async client for the managed AA AOTF shutter service."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from typing import Annotated as A
from uuid import uuid4

from ophyd_async.core import (
    AsyncStatus,
    DeviceMock,
    SignalR,
    SignalRW,
    SignalW,
    StandardReadable,
    StandardReadableFormat,
)
from ophyd_async.fastcs.core import fastcs_connector
from redsun.log import Loggable

if TYPE_CHECKING:
    from collections.abc import Mapping


class AAAOTF(StandardReadable, Loggable):
    """Named AOTF shutter profiles and service readbacks."""

    profile: A[SignalRW[str], StandardReadableFormat.CONFIG_SIGNAL]
    active_profile: A[SignalR[str], StandardReadableFormat.CONFIG_SIGNAL]
    status: A[SignalR[str], StandardReadableFormat.CONFIG_SIGNAL]
    fault: A[SignalR[str], StandardReadableFormat.CONFIG_SIGNAL]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))


class AttachedAAAOTF(AAAOTF):
    """A Redsun device whose session claims writes on a persistent service."""

    writer_command: SignalW[str]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        self._token: str | None = None
        self._heartbeat: asyncio.Task[None] | None = None
        super().__init__(prefix, name=name)

    async def connect(
        self,
        mock: bool | DeviceMock = False,
        timeout: float = 10.0,
        force_reconnect: bool = False,
    ) -> None:
        """Connect the PVA device and claim writes for this session."""
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        if self._token is not None:
            return
        token = uuid4().hex
        await self._send("claim", token)
        self._token = token
        self._heartbeat = asyncio.create_task(self._renew(token))

    async def _send(self, action: str, token: str, **fields: object) -> None:
        await self.writer_command.set(
            json.dumps({"action": action, "token": token, **fields}), timeout=2.0
        )

    async def _renew(self, token: str) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                await self._send("renew", token)
            except Exception:
                self.logger.exception("AOTF writer renewal failed")
                return

    @AsyncStatus.wrap
    async def set(self, value: str | Mapping[str, int | float | str]) -> None:
        """Set a named profile or a batch of MMCore properties."""
        if self._token is None or self._heartbeat is None or self._heartbeat.done():
            raise RuntimeError("AOTF writer claim is unavailable")
        if isinstance(value, str):
            await self._send("profile", self._token, profile=value)
        else:
            await self._send("properties", self._token, properties=dict(value))

    async def shutdown(self) -> None:
        """Release the writer claim after the service verifies all slots OFF."""
        token, self._token = self._token, None
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
            self._heartbeat = None
        if token is not None:
            await self._send("release", token)
