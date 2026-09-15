"""Ophyd-async device for the isolated camera-service control plane."""

from __future__ import annotations

import asyncio
import hashlib
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Literal, cast
from typing import Annotated as A

import numpy as np
from ophyd_async.core import AsyncStatus, SignalR, SignalW, TriggerableCommand
from ophyd_async.epics.core import EpicsDevice, PvSuffix

from redsun_aht.domain import (
    AcquisitionState,
    ArrayReference,
    Frame,
    HardwareServiceStatus,
    ServiceConnectionState,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from redsun_aht.domain.models import JsonValue


class CameraServiceDevice(EpicsDevice):
    """Typed application-side view of one camera service IOC."""

    service_id: A[SignalR[str], PvSuffix("SERVICE_ID")]
    schema_generation: A[SignalR[int], PvSuffix("SCHEMA_GENERATION")]
    heartbeat: A[SignalR[float], PvSuffix("HEARTBEAT")]
    connection_state: A[SignalR[str], PvSuffix("CONNECTION_STATE")]
    acquisition_state: A[SignalR[str], PvSuffix("ACQUISITION_STATE")]
    command_state: A[SignalR[str], PvSuffix("COMMAND_STATE")]
    frames_published: A[SignalR[int], PvSuffix("FRAMES_PUBLISHED")]
    frames_dropped: A[SignalR[int], PvSuffix("FRAMES_DROPPED")]
    last_error: A[SignalR[str], PvSuffix("LAST_ERROR")]
    descriptor_generation: A[SignalR[int], PvSuffix("DESCRIPTOR_GENERATION")]
    descriptor_sequence: A[SignalR[int], PvSuffix("DESCRIPTOR_SEQUENCE")]
    descriptor_slot: A[SignalR[int], PvSuffix("DESCRIPTOR_SLOT")]
    descriptor_name: A[SignalR[str], PvSuffix("DESCRIPTOR_NAME")]
    descriptor_run_id: A[SignalR[str], PvSuffix("DESCRIPTOR_RUN_ID")]
    descriptor_frame_id: A[SignalR[str], PvSuffix("DESCRIPTOR_FRAME_ID")]
    descriptor_shape: A[SignalR[str], PvSuffix("DESCRIPTOR_SHAPE")]
    descriptor_dtype: A[SignalR[str], PvSuffix("DESCRIPTOR_DTYPE")]
    descriptor_byte_order: A[SignalR[str], PvSuffix("DESCRIPTOR_BYTE_ORDER")]
    descriptor_timestamp_ns: A[SignalR[str], PvSuffix("DESCRIPTOR_TIMESTAMP_NS")]
    descriptor_checksum_hi: A[SignalR[str], PvSuffix("DESCRIPTOR_CHECKSUM_HI")]
    descriptor_checksum_lo: A[SignalR[str], PvSuffix("DESCRIPTOR_CHECKSUM_LO")]
    live_state: A[SignalR[str], PvSuffix("LIVE_STATE")]
    live_publish_hz: A[SignalR[float], PvSuffix("LIVE_PUBLISH_HZ")]
    live_sequence: A[SignalR[int], PvSuffix("LIVE_SEQUENCE")]
    live_frames_drained: A[SignalR[int], PvSuffix("LIVE_FRAMES_DRAINED")]
    live_frames_published: A[SignalR[int], PvSuffix("LIVE_FRAMES_PUBLISHED")]
    live_frames_skipped: A[SignalR[int], PvSuffix("LIVE_FRAMES_SKIPPED")]
    live_buffer_overruns: A[SignalR[int], PvSuffix("LIVE_BUFFER_OVERRUNS")]

    command_connect: A[TriggerableCommand, PvSuffix("COMMAND:CONNECT")]
    command_arm: A[TriggerableCommand, PvSuffix("COMMAND:ARM")]
    command_trigger: A[TriggerableCommand, PvSuffix("COMMAND:TRIGGER")]
    command_start_live: A[TriggerableCommand, PvSuffix("COMMAND:START_LIVE")]
    command_stop: A[TriggerableCommand, PvSuffix("COMMAND:STOP")]
    command_disconnect: A[TriggerableCommand, PvSuffix("COMMAND:DISCONNECT")]
    command_acknowledge: A[SignalW[int], PvSuffix("COMMAND:ACKNOWLEDGE")]

    def __init__(self, prefix: str, *, name: str) -> None:
        if not prefix.endswith(":"):
            raise ValueError("camera service prefix must end with ':'")
        self.epics_prefix = prefix
        super().__init__(prefix=prefix, name=name)

    async def request_connect(self) -> None:
        """Ask the service to connect its isolated backend."""
        await self.command_connect.trigger()

    async def arm(self) -> None:
        """Ask the service to arm acquisition."""
        await self.command_arm.trigger()

    async def start_live(self) -> None:
        """Ask the service to start its sequence-backed live-view loop."""
        await self.command_start_live.trigger()

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        """Ask the service to publish one acquisition descriptor."""
        await self.command_trigger.trigger()

    @AsyncStatus.wrap
    async def stop(self, *, success: bool = False) -> None:
        """Ask the service to enter its idle acquisition state."""
        del success
        await self.command_stop.trigger()

    async def request_disconnect(self) -> None:
        """Ask the service to disconnect its isolated backend."""
        await self.command_disconnect.trigger()

    async def acknowledge(self, sequence: int) -> None:
        """Acknowledge one exact lossless descriptor after durable copy."""
        if sequence < 0:
            raise ValueError("acknowledgement sequence must be non-negative")
        await self.command_acknowledge.set(sequence)

    async def health(self) -> HardwareServiceStatus:
        """Read one typed health snapshot from the EPICS control plane."""
        (
            service_id,
            generation,
            heartbeat,
            connection,
            acquisition,
            command,
            published,
            dropped,
            error,
        ) = await asyncio.gather(
            self.service_id.get_value(),
            self.schema_generation.get_value(),
            self.heartbeat.get_value(),
            self.connection_state.get_value(),
            self.acquisition_state.get_value(),
            self.command_state.get_value(),
            self.frames_published.get_value(),
            self.frames_dropped.get_value(),
            self.last_error.get_value(),
        )
        return HardwareServiceStatus(
            service_id=service_id,
            epics_prefix=self.epics_prefix,
            schema_generation=generation,
            heartbeat_ns=int(heartbeat * 1_000_000_000),
            connection_state=ServiceConnectionState(connection),
            acquisition_state=AcquisitionState(acquisition),
            command_state=command,
            frames_published=published,
            frames_dropped=dropped,
            last_error=error or None,
        )


class OphydAsyncCameraDetector:
    """Adapt a composed camera-service device to AHT's lossless detector port."""

    def __init__(self, service_id: str, control: CameraServiceDevice) -> None:
        if not service_id:
            raise ValueError("camera service_id must not be empty")
        self._service_id = service_id
        self.control = control
        self._latest_frame: Frame | None = None
        self._latest_memory_name: str | None = None
        self._latest_slot: int | None = None

    @property
    def service_id(self) -> str:
        """Return the stable detector identity, distinct from the service PV."""
        return self._service_id

    async def connect(self) -> None:
        """Connect the service-owned hardware and verify its declared identity."""
        await self.control.connect()
        await self.control.request_connect()
        observed = await self.control.service_id.get_value()
        if observed != self._service_id:
            await self.control.request_disconnect()
            raise RuntimeError(
                f"camera IOC service ID {observed!r} does not match "
                f"{self._service_id!r}"
            )

    async def configure(self, settings: Mapping[str, JsonValue]) -> str:
        """Reject settings not represented by the current typed IOC schema."""
        if settings:
            raise ValueError(
                "camera IOC configuration is not implemented; "
                "DPCT detector_settings must be empty"
            )
        return "camera-ioc-fixed-configuration-v1"

    async def arm(self) -> None:
        """Arm the service through its ophyd-async command signal."""
        await self.control.arm()

    async def trigger(self) -> None:
        """Trigger one frame and retain its immutable descriptor."""
        await self.control.trigger()
        values = await asyncio.gather(
            self.control.descriptor_generation.get_value(),
            self.control.descriptor_sequence.get_value(),
            self.control.descriptor_slot.get_value(),
            self.control.descriptor_name.get_value(),
            self.control.descriptor_run_id.get_value(),
            self.control.descriptor_frame_id.get_value(),
            self.control.descriptor_shape.get_value(),
            self.control.descriptor_dtype.get_value(),
            self.control.descriptor_byte_order.get_value(),
            self.control.descriptor_timestamp_ns.get_value(),
            self.control.descriptor_checksum_hi.get_value(),
            self.control.descriptor_checksum_lo.get_value(),
        )
        (
            generation,
            sequence,
            slot,
            memory_name,
            run_id,
            frame_id,
            raw_shape,
            dtype,
            byte_order,
            timestamp_ns,
            checksum_hi,
            checksum_lo,
        ) = values
        shape = tuple(int(item) for item in str(raw_shape).split(","))
        checksum = f"{checksum_hi}{checksum_lo}"
        if (
            int(sequence) < 0
            or int(slot) < 0
            or not shape
            or any(item <= 0 for item in shape)
            or not all((memory_name, run_id, frame_id))
            or len(checksum) != 64
        ):
            raise RuntimeError("camera IOC published an invalid frame descriptor")
        byte_order = str(byte_order)
        if byte_order not in {"=", "<", ">", "|"}:
            raise RuntimeError("camera IOC published an invalid byte order")
        try:
            np.dtype(str(dtype))
        except TypeError as error:
            raise RuntimeError("camera IOC published an invalid dtype") from error
        timestamp = int(str(timestamp_ns))
        self._latest_frame = Frame(
            run_id=str(run_id),
            frame_id=str(frame_id),
            detector_id=self._service_id,
            channel_id=self._service_id,
            sequence=int(sequence),
            exposure_started_ns=timestamp,
            exposure_ended_ns=timestamp,
            array=ArrayReference(
                uri=(
                    f"shm://{memory_name}?generation={generation}&slot={slot}"
                    f"&sequence={sequence}"
                ),
                checksum=checksum,
                shape=shape,
                dtype=str(dtype),
                byte_order=byte_order,
            ),
            configuration_revision="camera-ioc-fixed-configuration-v1",
            quality_flags=frozenset({"exposure-window-unavailable"}),
        )
        self._latest_memory_name = str(memory_name)
        self._latest_slot = int(slot)

    async def read(self) -> Frame:
        """Return metadata for the most recently committed frame."""
        if self._latest_frame is None:
            raise RuntimeError("camera IOC has not committed a frame")
        return self._latest_frame

    async def copy(self, sequence: int) -> npt.NDArray[Any]:
        """Copy and checksum-verify one retained shared-memory frame."""
        return await asyncio.to_thread(self._copy, sequence)

    def _copy(self, sequence: int) -> npt.NDArray[Any]:
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
        frame_bytes = int(np.prod(frame.array.shape, dtype=np.int64)) * dtype.itemsize
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

    async def acknowledge(self, sequence: int) -> None:
        """Release one frame only after the caller has persisted it."""
        self._require_latest(sequence)
        await self.control.acknowledge(sequence)

    async def stop(self) -> None:
        """Return the camera service to idle."""
        await self.control.stop()

    async def disconnect(self) -> None:
        """Release hardware while leaving EPICS connection ownership to RedSun."""
        try:
            await self.control.request_disconnect()
        finally:
            self._latest_frame = None
            self._latest_memory_name = None
            self._latest_slot = None

    def _require_latest(self, sequence: int) -> Frame:
        frame = self._latest_frame
        if frame is None or frame.sequence != sequence:
            raise ValueError("camera IOC sequence does not match the latest frame")
        return frame


__all__ = ["CameraServiceDevice", "OphydAsyncCameraDetector"]
