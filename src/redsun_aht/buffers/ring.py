"""Generation-aware local shared-memory frame rings."""

from __future__ import annotations

import hashlib
import time
from multiprocessing import shared_memory
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from redsun_aht.domain import BufferUsage, FrameBufferDescriptor, FrameLease

if TYPE_CHECKING:
    import numpy.typing as npt


class BufferStateError(RuntimeError):
    """Raised when a descriptor or lease no longer names readable data."""


class BufferFullError(RuntimeError):
    """Raised when lossless acquisition slots await acknowledgement."""


class SharedMemoryFrameRing:
    """Own a fixed-size local frame ring with explicit consumer leases."""

    def __init__(
        self,
        *,
        service_id: str,
        shape: tuple[int, ...],
        dtype: str,
        slots: int,
        usage: BufferUsage,
        lease_timeout_s: float = 1.0,
        generation: int | None = None,
    ) -> None:
        if slots <= 0:
            raise ValueError("slots must be positive")
        if lease_timeout_s <= 0:
            raise ValueError("lease_timeout_s must be positive")
        self.service_id = service_id
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.slots = slots
        self.usage = usage
        self.lease_timeout_ns = int(lease_timeout_s * 1_000_000_000)
        self.generation = time.monotonic_ns() if generation is None else generation
        self._frame_bytes = int(np.prod(shape, dtype=np.int64)) * self.dtype.itemsize
        self._memory = shared_memory.SharedMemory(
            create=True,
            size=self._frame_bytes * slots,
        )
        self._descriptors: dict[int, FrameBufferDescriptor] = {}
        self._leases: dict[str, FrameLease] = {}
        self._pending_acknowledgement: set[int] = set()
        self._next_sequence = 0
        self._closed = False
        self._lock = RLock()

    @property
    def shared_memory_name(self) -> str:
        """OS-level shared-memory identity published to local consumers."""
        return self._memory.name

    def publish(
        self,
        frame: npt.NDArray[Any],
        *,
        run_id: str,
        frame_id: str,
        timestamp_ns: int | None = None,
    ) -> FrameBufferDescriptor:
        """Commit a frame and return its generation-aware descriptor."""
        with self._lock:
            self.ensure_writable()
            array = np.asarray(frame)
            if array.shape != self.shape or array.dtype != self.dtype:
                raise ValueError(
                    f"expected frame {self.shape}/{self.dtype}, "
                    f"got {array.shape}/{array.dtype}"
                )
            sequence = self._next_sequence
            slot = sequence % self.slots
            self._invalidate_slot_leases(slot)
            view = self._slot_view(slot)
            np.copyto(view, array)
            checksum = hashlib.sha256(view.tobytes(order="C")).hexdigest()
            descriptor = FrameBufferDescriptor(
                service_id=self.service_id,
                run_id=run_id,
                frame_id=frame_id,
                sequence=sequence,
                generation=self.generation,
                shared_memory_name=self.shared_memory_name,
                slot=slot,
                shape=self.shape,
                dtype=self.dtype.str,
                byte_order=self.dtype.byteorder,
                timestamp_ns=time.time_ns() if timestamp_ns is None else timestamp_ns,
                checksum=checksum,
                usage=self.usage,
                lease_timeout_ns=self.lease_timeout_ns,
            )
            self._descriptors[slot] = descriptor
            if self.usage is BufferUsage.ACQUISITION:
                self._pending_acknowledgement.add(slot)
            self._next_sequence += 1
            return descriptor

    def ensure_writable(self) -> None:
        """Reject before acquisition when the next lossless slot is occupied."""
        with self._lock:
            self._require_open()
            self.reap_expired()
            slot = self._next_sequence % self.slots
            if (
                self.usage is BufferUsage.ACQUISITION
                and slot in self._pending_acknowledgement
            ):
                raise BufferFullError(
                    f"acquisition slot {slot} is awaiting acknowledgement"
                )

    def acquire(self, descriptor: FrameBufferDescriptor) -> FrameLease:
        """Issue a bounded lease after validating descriptor freshness."""
        with self._lock:
            self._require_current(descriptor)
            now = time.monotonic_ns()
            lease = FrameLease(
                lease_id=str(uuid4()),
                descriptor=descriptor,
                expires_ns=now + self.lease_timeout_ns,
            )
            self._leases[lease.lease_id] = lease
            return lease

    def copy(self, lease: FrameLease, *, acknowledge: bool = False) -> npt.NDArray[Any]:
        """Copy leased data, optionally acknowledging durable acquisition."""
        with self._lock:
            current = self._leases.get(lease.lease_id)
            if current != lease:
                raise BufferStateError("lease is unknown or already released")
            if time.monotonic_ns() >= lease.expires_ns:
                self._leases.pop(lease.lease_id, None)
                raise BufferStateError("lease has expired")
            self._require_current(lease.descriptor)
            copied = self._slot_view(lease.descriptor.slot).copy()
            checksum = hashlib.sha256(copied.tobytes(order="C")).hexdigest()
            if checksum != lease.descriptor.checksum:
                raise BufferStateError("frame checksum changed while leased")
            if acknowledge:
                if lease.descriptor.usage is not BufferUsage.ACQUISITION:
                    raise BufferStateError(
                        "preview frames do not accept durable acknowledgements"
                    )
                self._pending_acknowledgement.discard(lease.descriptor.slot)
            return copied

    def release(self, lease: FrameLease) -> None:
        """Release a consumer lease without implying durable acknowledgement."""
        with self._lock:
            self._leases.pop(lease.lease_id, None)

    def acknowledge(self, descriptor: FrameBufferDescriptor) -> None:
        """Release a lossless slot after a designated consumer persisted it."""
        with self._lock:
            self._require_current(descriptor)
            if descriptor.usage is not BufferUsage.ACQUISITION:
                raise BufferStateError("preview frames do not accept acknowledgements")
            if descriptor.slot not in self._pending_acknowledgement:
                raise BufferStateError("acquisition frame is already acknowledged")
            self._pending_acknowledgement.remove(descriptor.slot)

    def reap_expired(self, *, now_ns: int | None = None) -> int:
        """Drop expired leases after consumer death or timeout."""
        with self._lock:
            now = now_ns or time.monotonic_ns()
            expired = [
                lease_id
                for lease_id, lease in self._leases.items()
                if now >= lease.expires_ns
            ]
            for lease_id in expired:
                self._leases.pop(lease_id, None)
            return len(expired)

    def close(self, *, unlink: bool = True) -> None:
        """Close the owner handle and unlink its process-local resource."""
        with self._lock:
            if self._closed:
                return
            self._leases.clear()
            self._pending_acknowledgement.clear()
            self._memory.close()
            if unlink:
                self._memory.unlink()
            self._closed = True

    def _slot_view(self, slot: int) -> npt.NDArray[Any]:
        offset = slot * self._frame_bytes
        return np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self._memory.buf,
            offset=offset,
        )

    def _require_current(self, descriptor: FrameBufferDescriptor) -> None:
        self._require_open()
        if (
            descriptor.shared_memory_name != self.shared_memory_name
            or descriptor.generation != self.generation
        ):
            raise BufferStateError("descriptor belongs to another buffer generation")
        if self._descriptors.get(descriptor.slot) != descriptor:
            raise BufferStateError("descriptor has been overwritten or is unknown")

    def _require_open(self) -> None:
        if self._closed:
            raise BufferStateError("shared-memory ring is closed")

    def _invalidate_slot_leases(self, slot: int) -> None:
        stale = [
            lease_id
            for lease_id, lease in self._leases.items()
            if lease.descriptor.slot == slot
        ]
        for lease_id in stale:
            self._leases.pop(lease_id, None)

    def __enter__(self) -> SharedMemoryFrameRing:
        """Return this owner for bounded context-managed use."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release and unlink the owner resource."""
        self.close()


__all__ = ["BufferFullError", "BufferStateError", "SharedMemoryFrameRing"]
