"""Verify that cancelled ASI polls do not leave replies for later commands."""

import asyncio
import unittest

from redsun_aht.robotic.device.ludl._transport import OxiSerialSession


class PollPort:
    def __init__(self) -> None:
        self.commands: list[bytes] = []
        self.poll_started = asyncio.Event()
        self.release_reply = asyncio.Event()

    async def write(self, request: bytes) -> int:
        self.commands.append(request)
        return len(request)

    async def read_until(self, *, expected: bytes) -> bytes:
        if self.commands[-1] == b"3F?\r":
            self.poll_started.set()
            await self.release_reply.wait()
            return b"0" + expected
        return b":A" + expected


class TestSerialCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_poll_is_drained_before_halt(self) -> None:
        port = PollPort()
        session = OxiSerialSession(port)
        poll = asyncio.create_task(session.exchange(b"3F?\r", terminator=b"\x03"))
        halt = None
        try:
            await asyncio.wait_for(port.poll_started.wait(), 1)
            poll.cancel()
            halt = asyncio.create_task(session.exchange(b"2HHALT\r", terminator=b"\x03"))
            await asyncio.sleep(0)
            self.assertEqual(port.commands, [b"3F?\r"])
            self.assertFalse(poll.done())
            port.release_reply.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(poll, 1)
            self.assertEqual(await asyncio.wait_for(halt, 1), b":A\x03")
            self.assertEqual(port.commands, [b"3F?\r", b"2HHALT\r"])
        finally:
            port.release_reply.set()
            await asyncio.gather(*([poll, halt] if halt else [poll]), return_exceptions=True)
