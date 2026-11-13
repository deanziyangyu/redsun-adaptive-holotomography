"""Fake stalled MCU endpoint for shutdown lifecycle test."""

import asyncio
import sys
from pathlib import Path

from test_unified_mcu import FakePort

from redsun_aht.services import unified_mcu

trace = Path(sys.argv[2])


def record(message):
    with trace.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


class TracingPort(FakePort):
    def __init__(self):
        super().__init__(complete=False, repeat=True)

    async def write(self, data):
        command = data.decode("ascii").strip()
        record("WRITE " + command)
        if not self.is_open:
            raise OSError("port already closed")
        if command.startswith("ABORT ") and sys.argv[3] == "blocked":
            await asyncio.Event().wait()
        result = await super().write(data)
        record("DONE " + command)
        return result

    def close(self):
        record("CLOSE")
        super().close()


port = TracingPort()
unified_mcu.open_unified_port = lambda _: port
asyncio.run(unified_mcu._serve("fake", sys.argv[1]))
