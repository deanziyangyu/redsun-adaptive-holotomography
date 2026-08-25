from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np
import pytest

from redsun_aht.buffers import (
    BufferFullError,
    BufferStateError,
    SharedMemoryFrameRing,
)
from redsun_aht.domain import BufferUsage


def test_acquisition_ring_requires_durable_acknowledgement() -> None:
    ring = SharedMemoryFrameRing(
        service_id="detector-a",
        shape=(2, 2),
        dtype="uint16",
        slots=2,
        usage=BufferUsage.ACQUISITION,
    )
    try:
        first = ring.publish(np.ones((2, 2), dtype=np.uint16), run_id="r", frame_id="0")
        ring.publish(np.full((2, 2), 2, dtype=np.uint16), run_id="r", frame_id="1")

        with pytest.raises(BufferFullError, match="acknowledgement"):
            ring.publish(
                np.full((2, 2), 3, dtype=np.uint16),
                run_id="r",
                frame_id="2",
            )

        lease = ring.acquire(first)
        copied = ring.copy(lease, acknowledge=True)
        ring.release(lease)
        np.testing.assert_array_equal(copied, np.ones((2, 2), dtype=np.uint16))

        replacement = ring.publish(
            np.full((2, 2), 3, dtype=np.uint16),
            run_id="r",
            frame_id="2",
        )
        assert replacement.slot == first.slot
        with pytest.raises(BufferStateError, match="overwritten"):
            ring.acquire(first)
    finally:
        ring.close()


def test_direct_acknowledgement_releases_exact_lossless_slot() -> None:
    with SharedMemoryFrameRing(
        service_id="detector-a",
        shape=(1, 1),
        dtype="uint8",
        slots=1,
        usage=BufferUsage.ACQUISITION,
    ) as ring:
        descriptor = ring.publish(
            np.ones((1, 1), dtype=np.uint8), run_id="r", frame_id="0"
        )
        ring.acknowledge(descriptor)
        with pytest.raises(BufferStateError, match="already acknowledged"):
            ring.acknowledge(descriptor)
        replacement = ring.publish(
            np.zeros((1, 1), dtype=np.uint8), run_id="r", frame_id="1"
        )
        assert replacement.sequence == 1


def test_preview_ring_is_lossy_and_invalidates_stale_lease() -> None:
    ring = SharedMemoryFrameRing(
        service_id="detector-a",
        shape=(1, 2),
        dtype="uint8",
        slots=1,
        usage=BufferUsage.PREVIEW,
    )
    try:
        first = ring.publish(
            np.array([[1, 2]], dtype=np.uint8), run_id="r", frame_id="0"
        )
        lease = ring.acquire(first)
        second = ring.publish(
            np.array([[3, 4]], dtype=np.uint8),
            run_id="r",
            frame_id="1",
        )

        with pytest.raises(BufferStateError, match="unknown"):
            ring.copy(lease)
        current = ring.acquire(second)
        with pytest.raises(BufferStateError, match="do not accept"):
            ring.copy(current, acknowledge=True)
        ring.release(current)
    finally:
        ring.close()


def test_expired_lease_and_owner_cleanup() -> None:
    ring = SharedMemoryFrameRing(
        service_id="detector-a",
        shape=(1, 1),
        dtype="uint8",
        slots=1,
        usage=BufferUsage.PREVIEW,
        lease_timeout_s=0.001,
    )
    name = ring.shared_memory_name
    descriptor = ring.publish(np.ones((1, 1), dtype=np.uint8), run_id="r", frame_id="0")
    lease = ring.acquire(descriptor)
    assert ring.reap_expired(now_ns=lease.expires_ns) == 1
    with pytest.raises(BufferStateError, match="unknown"):
        ring.copy(lease)

    ring.close()
    ring.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_ring_rejects_invalid_dimensions_and_frame_contract() -> None:
    with pytest.raises(ValueError, match="slots"):
        SharedMemoryFrameRing(
            service_id="detector-a",
            shape=(1, 1),
            dtype="uint8",
            slots=0,
            usage=BufferUsage.PREVIEW,
        )

    with (
        SharedMemoryFrameRing(
            service_id="detector-a",
            shape=(2, 2),
            dtype="uint16",
            slots=1,
            usage=BufferUsage.PREVIEW,
        ) as ring,
        pytest.raises(ValueError, match="expected frame"),
    ):
        ring.publish(np.ones((3, 3), dtype=np.uint16), run_id="r", frame_id="0")
