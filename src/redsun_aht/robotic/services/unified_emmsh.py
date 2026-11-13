"""Serve status, stop and illumination blanking for the legacy ARH MCU."""

import argparse
import asyncio
from dataclasses import dataclass
import json
import re

from fastcs.attributes import AttrR, AttrW, AttributeIO, AttributeIORef
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, String
from redsun.services import configure_logging

from ..device.ludl._transport import OxiSerialSession
from ._process import controller_id, identity_arguments, run, serve


@dataclass
class EmmshRef(AttributeIORef):
    action: str = ""


class EmmshIO(AttributeIO):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    async def update(self, attr: AttrR):
        await attr.update(await self.controller.snapshot())

    async def send(self, attr: AttrW, value: bool):
        if value:
            await self.controller.safe_action(attr.io_ref.action)


class EmmshController(Controller):
    """One serial owner. Firmware counts report targets, with modulo wrap."""

    def __init__(self, session, firmware_revision):
        self.session = session
        self.firmware_revision = firmware_revision
        self.lock = asyncio.Lock()
        self.closed = False
        io = EmmshIO(self)
        super().__init__(ios=[io])
        self.commanded_steps = AttrR(String(), io_ref=EmmshRef(update_period=1.0))
        self.stop = AttrW(Bool(), io_ref=EmmshRef(action="stop"))
        self.blank = AttrW(Bool(), io_ref=EmmshRef(action="blank"))

    async def _exchange(self, command):
        reply = await self.session.exchange(command + b"\n", terminator=b"\n")
        # The downstream Z controller can emit notices before or after the
        # MCU's local reply. Read past them without resending the command.
        for _ in range(8):
            if not reply.endswith(b"\n"):
                raise RuntimeError("Incomplete MCU response")
            line = reply.decode("ascii").strip()
            if not line.startswith("Z "):
                return line
            reply = await self.session.read_line()
        raise RuntimeError("Too many downstream Z notices before MCU response")

    async def snapshot(self):
        async with self.lock:
            if self.closed:
                raise RuntimeError("MCU service is closed")
            targets = {}
            for axis, command in (("roll", b"?r"), ("pitch", b"?t"), ("yaw", b"?y")):
                reply = await self._exchange(command)
                if not re.fullmatch(r"[0-9]{1,4}", reply) or int(reply) >= 2048:
                    raise RuntimeError(f"Unexpected MCU {axis} target reply: {reply!r}")
                targets[axis] = int(reply)
        return json.dumps({
            "provenance": "commanded_target_modulo_2048",
            "configured_firmware_revision": self.firmware_revision,
            "steps": targets,
        })

    async def _safe_action(self, action):
        command, expected = {
            "stop": (b"k", "Stopped local axes"),
            "blank": (b"q", "Exited Fixed Mode (Ready)."),
        }[action]
        if await self._exchange(command) != expected:
            raise RuntimeError(f"MCU did not acknowledge {action}")

    async def safe_action(self, action):
        async with self.lock:
            if self.closed:
                raise RuntimeError("MCU service is closed")
            await self._safe_action(action)

    async def connect(self):
        await self.snapshot()
        await self.safe_action("blank")
        await super().connect()

    async def disconnect(self):
        try:
            await super().disconnect()
        finally:
            await self.close()

    async def close(self):
        async with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                try:
                    await self._safe_action("stop")
                finally:
                    await self._safe_action("blank")
            finally:
                self.session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baudrate", type=int, required=True)
    parser.add_argument("--firmware-revision", required=True)
    identity_arguments(parser, "unified-emmsh")
    options = parser.parse_args(argv)
    if any(not value or "REPLACE_" in value for value in (options.port, options.firmware_revision)):
        parser.error("Supply the MCU serial port and installed firmware revision")
    configure_logging()

    async def run_service():
        session = OxiSerialSession.open(
            options.port, baudrate=options.baudrate, stopbits=1, timeout=0.5,
        )
        controller = EmmshController(session, options.firmware_revision)
        try:
            await serve(controller, controller_id(options))
        finally:
            await controller.close()

    return run(run_service())


if __name__ == "__main__":
    raise SystemExit(main())
