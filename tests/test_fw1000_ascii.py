"""Rehearse FW-1000-SA serial framing and failure behavior without a port."""

import unittest
from collections import deque
from unittest.mock import patch

from redsun_aht.robotic.device.fw1000_ascii import FW1000ASCII, FW1000ProtocolError, LX4000FilterWheel


class SerialScript:
    def __init__(self, *steps: tuple[bytes, bytes]) -> None:
        self.steps = deque(steps)
        self.reply = b""

    def write(self, data: bytes) -> int:
        expected, self.reply = self.steps.popleft()
        if data != expected:
            raise AssertionError((expected, data))
        return len(data)

    def read(self, size: int) -> bytes:
        return self.reply[:size]

    def read_until(self, expected: bytes, size: int) -> bytes:
        return self.reply[:size]


class TestFW1000ASCII(unittest.TestCase):
    def test_move_polls_until_idle(self) -> None:
        serial = SerialScript(
            (b"?", b"0"),
            (b"FW 0\n\r", b"0\n\r0>"),
            (b"MP 7\n\r", b"\n\r0>"),
            (b"?", b"3"),
            (b"?", b"0"),
        )
        FW1000ASCII(serial).move(7)
        self.assertFalse(serial.steps)

    def test_rejected_wheel_never_moves(self) -> None:
        serial = SerialScript((b"?", b"0"), (b"FW 1\n\r", b"ERR\n\r1>"))
        with self.assertRaises(FW1000ProtocolError):
            FW1000ASCII(serial, wheel=1).move(2)
        self.assertFalse(serial.steps)

    def test_invalid_slot_never_writes(self) -> None:
        with self.assertRaises(ValueError):
            FW1000ASCII(SerialScript()).move(8)

    def test_incomplete_prompt_times_out(self) -> None:
        serial = SerialScript((b"?", b"0"), (b"FW 0\n\r", b"0\n\r"))
        with self.assertRaises(TimeoutError):
            FW1000ASCII(serial).move(2)

    def test_busy_error_is_not_idle(self) -> None:
        with self.assertRaises(FW1000ProtocolError):
            FW1000ASCII(SerialScript((b"?", b"5"))).busy()

    def test_move_settle_timeout(self) -> None:
        serial = SerialScript(
            (b"?", b"0"),
            (b"FW 0\n\r", b"0\n\r0>"),
            (b"MP 2\n\r", b"\n\r0>"),
            (b"?", b"3"),
            (b"?", b"3"),
        )
        with patch("redsun_aht.robotic.device.fw1000_ascii.monotonic", side_effect=(0.0, 1.0)):
            with self.assertRaises(TimeoutError):
                FW1000ASCII(serial).move(2, settle_timeout=0.001)


class LX4000Script:
    def __init__(self, *steps: tuple[bytes, bytes]) -> None:
        self.steps = deque(steps)

    async def exchange(self, request: bytes, *, terminator: bytes) -> bytes:
        expected, reply = self.steps.popleft()
        if request != expected or terminator != b"\x03":
            raise AssertionError((expected, request, terminator))
        return reply


class TestLX4000FilterWheel(unittest.IsolatedAsyncioTestCase):
    async def test_cr_requests_and_etx_replies(self) -> None:
        session = LX4000Script(
            (b"3F?\r", b"0\r\n\x03"),
            (b"3FFW 0\r", b"0\r\n\x03"),
            (b"3FMP 7\r", b"7\r\n\x03"),
            (b"3F?\r", b"3\x03"),
            (b"3F?\r", b"0\x03"),
        )
        await LX4000FilterWheel(session).move(7)
        self.assertFalse(session.steps)

    async def test_wrong_slot_ack_stops_transaction(self) -> None:
        session = LX4000Script(
            (b"3F?\r", b"0\x03"),
            (b"3FFW 1\r", b"1\x03"),
            (b"3FMP 7\r", b"6\x03"),
        )
        with self.assertRaises(FW1000ProtocolError):
            await LX4000FilterWheel(session, wheel=1).move(7)
        self.assertFalse(session.steps)

    async def test_truncated_idle_reply_is_rejected(self) -> None:
        session = LX4000Script((b"3F?\r", b"0"))
        with self.assertRaises(TimeoutError):
            await LX4000FilterWheel(session).busy()

    async def test_acknowledged_slot_with_truncated_settle_reply_fails(self) -> None:
        session = LX4000Script(
            (b"3F?\r", b"0\x03"),
            (b"3FFW 0\r", b"0\x03"),
            (b"3FMP 2\r", b"2\x03"),
            (b"3F?\r", b"0"),
        )
        with self.assertRaises(TimeoutError):
            await LX4000FilterWheel(session).move(2)
        self.assertFalse(session.steps)
