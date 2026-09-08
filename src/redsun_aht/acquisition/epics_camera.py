"""Lossless DPCT detector adapter for one isolated EPICS camera service."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from ophyd_async.core import wait_for_value

from redsun_aht.device.camera import CameraServiceDevice
from redsun_aht.domain import ArrayReference, Frame, ServiceConnectionState

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from redsun_aht.domain.models import JsonValue


@dataclass(slots=True)
class EpicsCameraDetector:
    """Adapt a camera IOC through the RedSun ophyd-async device contract.

    Caproto remains the isolated IOC implementation. Application-side
    acquisition uses :class:`CameraServiceDevice` for all control-plane
    commands and descriptor reads; this adapter owns only the AHT-specific
    shared-memory copy and checksum boundary.
    """

    service_id: str
    prefix: str
    timeout: float = 15.0
    poll_interval: float = 0.005
    _connected: bool = field(default=False, init=False, repr=False)
    _device: CameraServiceDevice | None = field(default=None, init=False, repr=False)
    _latest_frame: Frame | None = field(default=None, init=False, repr=False)
    _latest_memory_name: str | None = field(default=None, init=False, repr=False)
    _latest_slot: int | None = field(default=None, init=False, repr=False)
    _copy_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Reject an incomplete IOC endpoint before a hardware command."""
        if not self.service_id:
            raise ValueError("camera service_id must not be empty")
        if not self.prefix.endswith(":"):
            raise ValueError("camera IOC prefix must end with ':'")
        if self.timeout <= 0:
            raise ValueError("camera IOC timeout must be positive")
        if self.poll_interval <= 0:
            raise ValueError("camera IOC poll interval must be positive")

    async def connect(self) -> None:
        """Connect and verify that the service ID matches this composition."""
        if self._connected:
            return
        device = CameraServiceDevice(self.prefix, name=f"{self.service_id}_service")
        self._device = device
        try:
            await device.connect(timeout=self.timeout)
            await device.request_connect()
            await self._await_ready(device)
            status = await device.health()
            if status.service_id != self.service_id:
                await device.shutdown()
                raise RuntimeError(
                    f"camera IOC service ID {status.service_id!r} does not match "
                    f"{self.service_id!r}"
                )
            if status.connection_state is not ServiceConnectionState.READY:
                raise RuntimeError("camera IOC did not enter its ready state")
            self._connected = True
        except BaseException:
            self._connected = False
            self._device = None
            raise

    async def configure(self, settings: Mapping[str, JsonValue]) -> str:
        """Reject settings the current camera IOC cannot apply faithfully."""
        if settings:
            raise ValueError(
                "camera IOC configuration is not implemented; "
                "DPCT detector_settings must be empty"
            )
        return "camera-ioc-fixed-configuration-v1"

    async def arm(self) -> None:
        """Arm one already admitted camera service."""
        device = self._require_device()
        await device.arm()
        await wait_for_value(device.acquisition_state, "armed", timeout=self.timeout)

    async def trigger(self) -> None:
        """Trigger exactly one frame and retain its immutable descriptor."""
        device = self._require_device()
        published = await device.frames_published.get_value()
        await device.trigger()
        await wait_for_value(
            device.frames_published,
            lambda value: value >= published + 1,
            timeout=self.timeout,
        )
        (
            generation,
            sequence,
            slot,
            memory_name,
            run_id,
            frame_id,
            shape_text,
            dtype,
            byte_order,
            timestamp_ns,
            checksum_hi,
            checksum_lo,
        ) = await self._await_descriptor(device)
        generation = int(generation)
        sequence = int(sequence)
        slot = int(slot)
        memory_name = str(memory_name)
        run_id = str(run_id)
        frame_id = str(frame_id)
        shape_text = str(shape_text)
        dtype = str(dtype)
        byte_order = str(byte_order)
        timestamp_ns = int(timestamp_ns)
        checksum_hi = str(checksum_hi)
        checksum_lo = str(checksum_lo)
        if sequence < 0:
            raise RuntimeError("camera IOC published an invalid frame sequence")
        shape = tuple(int(item) for item in shape_text.split(","))
        if not shape or any(item <= 0 for item in shape):
            raise RuntimeError("camera IOC published an invalid frame shape")
        if byte_order not in {"=", "<", ">", "|"}:
            raise RuntimeError("camera IOC published an invalid byte order")
        try:
            np.dtype(dtype)
        except TypeError as error:
            raise RuntimeError("camera IOC published an invalid dtype") from error
        checksum = checksum_hi + checksum_lo
        if not all((memory_name, frame_id, run_id)) or len(checksum) != 64:
            raise RuntimeError("camera IOC published an incomplete frame descriptor")
        if timestamp_ns < 0:
            raise RuntimeError("camera IOC published an invalid frame timestamp")
        if slot < 0:
            raise RuntimeError("camera IOC published an invalid frame slot")
        self._latest_frame = Frame(
            run_id=run_id,
            frame_id=frame_id,
            detector_id=self.service_id,
            channel_id=self.service_id,
            sequence=sequence,
            exposure_started_ns=timestamp_ns,
            exposure_ended_ns=timestamp_ns,
            array=ArrayReference(
                uri=(
                    f"shm://{memory_name}?generation={generation}&slot={slot}"
                    f"&sequence={sequence}"
                ),
                checksum=checksum,
                shape=shape,
                dtype=dtype,
                byte_order=byte_order,
            ),
            configuration_revision="camera-ioc-fixed-configuration-v1",
            quality_flags=frozenset({"exposure-window-unavailable"}),
        )
        self._latest_memory_name = memory_name
        self._latest_slot = slot

    async def read(self) -> Frame:
        """Return metadata for the most recently committed frame."""
        if self._latest_frame is None:
            raise RuntimeError("camera IOC has not committed a frame")
        return self._latest_frame

    async def copy(self, sequence: int) -> npt.NDArray[Any]:
        """Copy and checksum-verify the exact unacknowledged shared-memory slot."""
        return await asyncio.to_thread(self._copy, sequence)

    async def acknowledge(self, sequence: int) -> None:
        """Release precisely the frame slot copied by the caller."""
        self._require_latest(sequence)
        device = self._require_device()
        await device.acknowledge(sequence)
        await wait_for_value(
            device.command_state, "frame_acknowledged", timeout=self.timeout
        )
        self._latest_frame = None
        self._latest_memory_name = None
        self._latest_slot = None

    async def stop(self) -> None:
        """Return ordinary camera acquisition to idle."""
        if self._connected:
            device = self._require_device()
            await device.stop()
            await wait_for_value(device.acquisition_state, "idle", timeout=self.timeout)

    async def disconnect(self) -> None:
        """Release the isolated camera service and invalidate local metadata."""
        device, self._device = self._device, None
        try:
            if device is not None:
                await device.shutdown()
        finally:
            self._connected = False
            self._latest_frame = None
            self._latest_memory_name = None
            self._latest_slot = None

    def _copy(self, sequence: int) -> npt.NDArray[Any]:
        with self._copy_lock:
            frame = self._require_latest(sequence)
            memory_name = self._latest_memory_name
            slot = self._latest_slot
            if memory_name is None or slot is None:  # pragma: no cover - invariant
                raise RuntimeError("camera IOC frame slot metadata is unavailable")
            byte_order = cast(
                "Literal['=', '<', '>', '|']",
                frame.array.byte_order,
            )
            dtype = np.dtype(frame.array.dtype).newbyteorder(byte_order)
            frame_bytes = (
                int(np.prod(frame.array.shape, dtype=np.int64)) * dtype.itemsize
            )
            memory = shared_memory.SharedMemory(name=memory_name)
            try:
                copied = np.ndarray(
                    frame.array.shape,
                    dtype=dtype,
                    buffer=memory.buf,
                    offset=slot * frame_bytes,
                ).copy()
            finally:
                memory.close()
            checksum = hashlib.sha256(copied.tobytes(order="C")).hexdigest()
            if checksum != frame.array.checksum:
                raise ValueError("camera shared-memory frame checksum mismatch")
            return copied

    def _require_device(self) -> CameraServiceDevice:
        if not self._connected or self._device is None:
            raise RuntimeError("camera IOC detector is not connected")
        return self._device

    async def _await_ready(self, device: CameraServiceDevice) -> None:
        """Wait for backend admission and surface an IOC failure immediately."""
        deadline = asyncio.get_running_loop().time() + self.timeout
        while True:
            state = str(await device.connection_state.get_value(cached=False))
            if state == ServiceConnectionState.READY.value:
                return
            if state == ServiceConnectionState.FAULTED.value:
                error = str(await device.last_error.get_value(cached=False))
                detail = error or "the IOC did not report a backend error"
                raise RuntimeError(f"camera IOC backend connection failed: {detail}")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("camera IOC did not enter its ready state")
            await asyncio.sleep(self.poll_interval)

    async def _await_descriptor(
        self, device: CameraServiceDevice
    ) -> tuple[int, int, int, str, str, str, str, str, str, int, str, str]:
        """Read the atomically published descriptor after the trigger command.

        The Caproto IOC writes each descriptor PV before incrementing
        ``FRAMES_PUBLISHED``.  Explicit ophyd-async reads avoid accepting an
        older monitor-cache value while Channel Access delivers that update.
        """
        deadline = asyncio.get_running_loop().time() + self.timeout
        while True:
            values = await asyncio.gather(
                device.descriptor_generation.get_value(cached=False),
                device.descriptor_sequence.get_value(cached=False),
                device.descriptor_slot.get_value(cached=False),
                device.descriptor_name.get_value(cached=False),
                device.descriptor_run_id.get_value(cached=False),
                device.descriptor_frame_id.get_value(cached=False),
                device.descriptor_shape.get_value(cached=False),
                device.descriptor_dtype.get_value(cached=False),
                device.descriptor_byte_order.get_value(cached=False),
                device.descriptor_timestamp_ns.get_value(cached=False),
                device.descriptor_checksum_hi.get_value(cached=False),
                device.descriptor_checksum_lo.get_value(cached=False),
            )
            if values[6]:
                return cast(
                    "tuple[int, int, int, str, str, str, str, str, str, int, str, str]",
                    values,
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("camera IOC did not publish a descriptor")
            await asyncio.sleep(self.poll_interval)

    def _require_latest(self, sequence: int) -> Frame:
        frame = self._latest_frame
        if frame is None or frame.sequence != sequence:
            raise ValueError("camera IOC sequence does not match the latest frame")
        return frame


__all__ = ["EpicsCameraDetector"]
