"""Chained ASCII transactions for the volatile dynamic pattern bank."""

from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class DynamicPatternError(RuntimeError):
    """An invalid or unexpected dynamic pattern response."""


@dataclass(frozen=True)
class DynamicPattern:
    """A published dynamic slot and provenance for the bytes uploaded."""

    slot: int
    generation: int
    crc32: int
    sha256: str


class DynamicPatternLink:
    """Protocol over a serial link whose caller holds the operation lock."""

    def __init__(
        self,
        write: Callable[[str], Awaitable[None]],
        read: Callable[[float], Awaitable[str]],
        cleanup: Callable[[str], Awaitable[None]],
        timeout_s: float,
    ) -> None:
        self.write = write
        self.read = read
        self.cleanup = cleanup
        self.timeout_s = timeout_s

    async def exchange(
        self, command: str, expected: str, replay: set[str], *, retries: int = 1
    ) -> str:
        # A timeout can occur after the firmware has processed the command. Retry
        # once and ignore the extra copy if both replies subsequently arrive.
        for attempt in range(retries + 1):
            await self.write(command)
            deadline = asyncio.get_running_loop().time() + self.timeout_s
            try:
                while True:
                    response = await self.read(deadline - asyncio.get_running_loop().time())
                    if response == expected:
                        if attempt:
                            replay.add(response)
                        return response
                    if response not in replay:
                        raise DynamicPatternError(f"unexpected response to {command}: {response!r}")
            except TimeoutError:
                if attempt == retries:
                    raise
        raise AssertionError("unreachable retry state")

    async def capabilities(self) -> tuple[int, int]:
        await self.write("DPCAPS?")
        response = await self.read(self.timeout_s)
        match = re.fullmatch(r"DPCAPS 1 (61|91) 5 ([1-9][0-9]?) 1", response)
        if match is None or int(match[2]) > 20:
            raise DynamicPatternError(f"unsupported dynamic pattern protocol: {response!r}")
        return int(match[1]), int(match[2])

    async def upload_dynamic_pattern(self, slot: int, pixels: bytes) -> DynamicPattern:
        """Publish a single-color map atomically over one chained serial operation."""
        if type(slot) is not int or not 1 <= slot <= 5 or not isinstance(pixels, bytes):
            raise ValueError("a slot in 1..5 and immutable pixel bytes are required")
        pixel_count, chunk_size = await self.capabilities()
        if len(pixels) != pixel_count:
            raise ValueError(f"expected {pixel_count} pixel bytes")
        await self.write(f"DPSLOT? {slot}")
        response = await self.read(self.timeout_s)
        match = re.fullmatch(rf"DPSLOT {slot} ([0-9]{{1,10}}) ([0-9]{{1,10}})", response)
        if match is None or int(match[1]) >= 0xFFFFFFFF or int(match[2]) > 0xFFFFFFFF:
            raise DynamicPatternError(f"invalid slot state: {response!r}")
        generation = int(match[1])
        transaction = 1 + secrets.randbelow(0xFFFFFFFF)
        checksum = zlib.crc32(pixels)
        replay: set[str] = set()
        try:
            await self.exchange(
                f"DPBEGIN {transaction} {slot} {generation} {checksum}",
                f"DPBEGIN {transaction}", replay,
            )
            for offset in range(0, pixel_count, chunk_size):
                chunk = pixels[offset : offset + chunk_size].hex()
                await self.exchange(
                    f"DPWRITE {transaction} {offset} {chunk}",
                    f"DPWRITE {transaction} {offset}", replay,
                )
            commit = f"DPCOMMIT {transaction} {slot} {generation + 1} {checksum}"
            try:
                await self.exchange(f"DPCOMMIT {transaction}", commit, replay)
            except TimeoutError:
                published = f"DPSLOT {slot} {generation + 1} {checksum}"
                replay.add(commit)
                await self.exchange(f"DPSLOT? {slot}", published, replay)
            # DPABORT on a committed transaction is a no-op and provides a
            # differently tagged barrier to drain any remaining replay reply.
            await self.exchange(
                f"DPABORT {transaction}", f"DPABORT {transaction}", replay, retries=0
            )
        except BaseException:
            await self.cleanup(f"DPABORT {transaction}")
            raise
        return DynamicPattern(slot, generation + 1, checksum, hashlib.sha256(pixels).hexdigest())
