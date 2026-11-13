"""Experimental read-only ESP300 PVA service; never commands motion."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

from fastcs.attributes import AttrR, AttrW
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, String
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from oxiserial import serial_for_url
from redsun.services import serve_controller

from redsun_aht.device.esp300_readonly import Esp300ReadOnly

if TYPE_CHECKING:
    from redsun_aht.device.esp300_readonly import ControllerSnapshot


class Esp300ReadOnlyController(Controller):  # type: ignore[misc]
    """Publish complete serialized snapshots; no motion surface exists."""

    def __init__(self, owner: Esp300ReadOnly) -> None:
        super().__init__()
        self.owner = owner
        self.snapshot = AttrR(
            String(),
            initial_value="",
            description="Controller-reported XYZ snapshot JSON",
        )
        self.status = AttrR(
            String(),
            initial_value="uninitialized",
            description="Snapshot acquisition state",
        )
        self.refresh = AttrW(Bool(), description="Request a fresh read-only snapshot")
        self.refresh.set_on_put_callback(self._refresh)
        self._task: asyncio.Task[ControllerSnapshot] | None = None
        self._busy = False

    async def initialise(self) -> None:
        """Require one valid identity and position snapshot at startup."""
        await self._read()

    async def _refresh(self, attr: AttrW, value: bool) -> None:
        del attr
        if value:
            await self._read()

    async def _read(self) -> None:
        if self._busy or (self._task is not None and not self._task.done()):
            raise RuntimeError("ESP300 snapshot already in progress")
        self._busy = True
        start = time.monotonic_ns()
        self._task = asyncio.create_task(asyncio.to_thread(self.owner.snapshot))
        try:
            await self.status.update("reading")
            value = await asyncio.shield(self._task)
            end = time.monotonic_ns()
            await self.snapshot.update(
                json.dumps(
                    {
                        "schema": "redsun.esp300-readonly/v1",
                        "clock": "host_monotonic_ns",
                        "start_ns": start,
                        "end_ns": end,
                        **asdict(value),
                    }
                )
            )
            await self.status.update("ready")
        except BaseException:
            await self.snapshot.update("")
            await self.status.update("uncertain")
            raise
        finally:
            self._busy = False

    async def drain(self) -> None:
        """Wait for any in-flight thread before the serial endpoint closes."""
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)


async def _serve(port_name: str, prefix: str) -> None:
    if not port_name.strip() or "<" in port_name or ">" in port_name:
        raise ValueError("explicit serial endpoint required")
    port = serial_for_url(
        port_name,
        baudrate=19200,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=1,
        write_timeout=1,
        rtscts=True,
        xonxoff=False,
        dsrdtr=False,
    )
    controller = Esp300ReadOnlyController(Esp300ReadOnly(port))
    try:
        await serve_controller(controller, EpicsPVATransport(), prefix)
    finally:
        await controller.drain()
        port.close()


def main() -> int:
    """Serve an explicit port and prefix until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--prefix", required=True)
    options = parser.parse_args()
    try:
        asyncio.run(_serve(options.port, options.prefix))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
