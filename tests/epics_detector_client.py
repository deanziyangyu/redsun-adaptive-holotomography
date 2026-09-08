"""Fresh-process client for the full Caproto/Ophyd detector integration test."""

from __future__ import annotations

import asyncio
import sys

from redsun_aht.acquisition import EpicsCameraDetector


async def exercise(service_id: str, prefix: str) -> None:
    """Acquire, copy, and acknowledge one descriptor through ophyd-async."""
    detector = EpicsCameraDetector(service_id, prefix)
    try:
        await detector.connect()
        assert await detector.configure({}) == "camera-ioc-fixed-configuration-v1"
        await detector.arm()
        await detector.trigger()
        frame = await detector.read()
        copied = await detector.copy(frame.sequence)
        assert copied.shape == frame.array.shape
        await detector.acknowledge(frame.sequence)
    finally:
        await detector.disconnect()


if __name__ == "__main__":
    asyncio.run(exercise(sys.argv[1], sys.argv[2]))
