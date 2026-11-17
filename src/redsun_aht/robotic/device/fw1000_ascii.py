"""Prompt-mode ASCII driver for an ASI FW-1000-SA on a dedicated serial link."""

from __future__ import annotations

import asyncio
import math
import re
from threading import Lock
from time import monotonic, sleep
from typing import Protocol

from .asi._client import EOT


class WheelSerial(Protocol):
    """An exclusively owned serial connection with finite read/write timeouts."""

    def write(self, data: bytes) -> int: ...
    def read(self, size: int) -> bytes: ...
    def read_until(self, expected: bytes, size: int) -> bytes: ...


class FW1000ProtocolError(RuntimeError):
    """A controller reported a fault or returned an unexpected response."""


class FW1000ASCII:
    """Select a wheel and wait for an idle controller after commanding a slot.

    Idle status does not independently measure the slot. The controller must
    already be homed and use prompt-enabled output. The caller owns the port.
    """

    def __init__(
        self, serial: WheelSerial, *, wheel: int = 0, slots: int = 8
    ) -> None:
        if type(wheel) is not int or wheel not in (0, 1):
            raise ValueError("wheel must be 0 or 1")
        if type(slots) is not int or slots not in (6, 8):
            raise ValueError("slots must be 6 or 8")
        self._serial = serial
        self._wheel = wheel
        self._slots = slots
        self._lock = Lock()

    def _write(self, data: bytes) -> None:
        if self._serial.write(data) != len(data):
            raise FW1000ProtocolError("Incomplete serial write")

    def _command(self, command: bytes) -> bytes:
        self._write(command + b"\n\r")
        reply = self._serial.read_until(b">", size=256)
        if not reply.endswith(b">"):
            raise TimeoutError(f"No complete prompt for {command!r}")
        match = re.fullmatch(rb"(.*)[\r\n]+([01])>", reply, re.DOTALL)
        if match is None or match[2] != str(self._wheel).encode("ascii"):
            raise FW1000ProtocolError(f"Unexpected wheel prompt: {reply!r}")
        payload = match[1].strip()
        if re.search(rb"\b(?:ERR(?:OR)?|INVALID|UNKNOWN|NOT RESPONDING)\b", payload, re.I):
            raise FW1000ProtocolError(f"{command!r} rejected: {payload!r}")
        return payload

    def _status(self) -> int:
        self._write(b"?")
        reply = self._serial.read(1)
        if not reply:
            raise TimeoutError("No FW1000 busy response")
        if reply not in (b"0", b"1", b"2", b"3"):
            raise FW1000ProtocolError(f"FW1000 busy fault or malformed response: {reply!r}")
        return int(reply)

    def busy(self) -> bool:
        """Query whether either wheel is in motion."""
        with self._lock:
            return self._status() != 0

    def move(self, position: int, *, settle_timeout: float = 2.0) -> None:
        """Command a zero-based slot and wait until the controller reports idle."""
        if type(position) is not int or not 0 <= position < self._slots:
            raise ValueError(f"position must be in 0..{self._slots - 1}")
        if not math.isfinite(settle_timeout) or settle_timeout <= 0:
            raise ValueError("settle_timeout must be finite and positive")
        with self._lock:
            if self._status():
                raise FW1000ProtocolError("Wheel must be idle before a move")
            wheel = str(self._wheel).encode("ascii")
            if self._command(b"FW " + wheel) != wheel:
                raise FW1000ProtocolError("Controller did not acknowledge wheel selection")
            self._command(f"MP {position}".encode("ascii"))
            deadline = monotonic() + settle_timeout
            while self._status():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("FW1000 did not settle before the deadline")
                sleep(min(0.02, remaining))


class LX4000Serial(Protocol):
    """The shared LX4000 serial session after its ETX handshake."""

    async def exchange(self, request: bytes, *, terminator: bytes) -> bytes: ...


class LX4000FilterWheel:
    """FW1000 wheel routed through the LX4000's 3F channel.

    Share the serial session with the stage clients and use one wheel client
    per session. The session owner handles the handshake and port lifetime.
    """

    def __init__(self, session: LX4000Serial, *, wheel: int = 0) -> None:
        if type(wheel) is not int or wheel not in (0, 1):
            raise ValueError("wheel must be 0 or 1")
        self._session = session
        self._wheel = str(wheel).encode("ascii")
        self._lock = asyncio.Lock()

    async def _query(self, command: bytes) -> bytes:
        reply = await self._session.exchange(b"3F" + command + b"\r", terminator=EOT)
        if not reply.endswith(EOT):
            raise TimeoutError(f"Incomplete LX4000 wheel reply to {command!r}")
        return reply[:-1].strip()

    async def _busy(self) -> bool:
        reply = await self._query(b"?")
        if reply not in (b"0", b"1", b"2", b"3"):
            raise FW1000ProtocolError(f"Unexpected LX4000 wheel status: {reply!r}")
        return reply != b"0"

    async def busy(self) -> bool:
        """Read the wheel status without selecting a slot."""
        async with self._lock:
            return await self._busy()

    async def move(self, position: int, *, settle_timeout: float = 2.0) -> None:
        """Command a slot and wait for idle within the configured deadline."""
        if type(position) is not int or not 0 <= position < 8:
            raise ValueError("position must be in 0..7")
        if not math.isfinite(settle_timeout) or settle_timeout <= 0:
            raise ValueError("settle_timeout must be finite and positive")
        async with self._lock:
            if await self._busy():
                raise FW1000ProtocolError("Wheel must be idle before a move")
            if await self._query(b"FW " + self._wheel) != self._wheel:
                raise FW1000ProtocolError("LX4000 did not acknowledge wheel selection")
            slot = str(position).encode("ascii")
            if await self._query(b"MP " + slot) != slot:
                raise FW1000ProtocolError("LX4000 did not acknowledge the requested slot")
            async with asyncio.timeout(settle_timeout):
                while await self._busy():
                    await asyncio.sleep(0.02)
