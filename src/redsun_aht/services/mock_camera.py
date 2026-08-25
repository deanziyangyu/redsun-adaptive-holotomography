"""Deterministic camera-service simulator for lifecycle contract tests."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from redsun_aht.buffers import BufferFullError, SharedMemoryFrameRing
from redsun_aht.domain import (
    AcquisitionState,
    ArrayReference,
    BufferUsage,
    DetectorCapabilities,
    Frame,
    HardwareServiceStatus,
    ServiceConnectionState,
)
from redsun_aht.domain.models import thaw_json

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import numpy.typing as npt

    from redsun_aht.domain import FrameBufferDescriptor
    from redsun_aht.domain.models import JsonValue


class CameraServiceStateError(RuntimeError):
    """Raised when a camera command is invalid for the current lifecycle."""


class _Unchanged:
    pass


_UNCHANGED = _Unchanged()


class MockCameraService:
    """Model one isolated camera service without constructing hardware SDKs."""

    def __init__(
        self,
        service_id: str,
        epics_prefix: str,
        *,
        shape: tuple[int, int] = (8, 8),
        acquisition_slots: int = 4,
        fail_connect_attempts: int = 0,
        run_id: str = "simulation",
    ) -> None:
        self._service_id = service_id
        self._epics_prefix = epics_prefix
        self.shape = shape
        self.acquisition_slots = acquisition_slots
        self._fail_connect_attempts = fail_connect_attempts
        self._run_id = run_id
        self._preview: SharedMemoryFrameRing | None = None
        self._acquisition: SharedMemoryFrameRing | None = None
        self._latest_descriptor: FrameBufferDescriptor | None = None
        self._latest_frame: Frame | None = None
        self._configuration_revision = "unconfigured"
        self._sequence = 0
        self._status = HardwareServiceStatus(
            service_id=service_id,
            epics_prefix=epics_prefix,
            schema_generation=0,
            heartbeat_ns=time.monotonic_ns(),
            connection_state=ServiceConnectionState.DISCONNECTED,
            acquisition_state=AcquisitionState.IDLE,
            command_state="idle",
            frames_published=0,
            frames_dropped=0,
        )

    @property
    def service_id(self) -> str:
        """Stable service identity."""
        return self._service_id

    @property
    def epics_prefix(self) -> str:
        """Unique control-plane prefix reserved for this service."""
        return self._epics_prefix

    @property
    def status(self) -> HardwareServiceStatus:
        """Latest immutable health snapshot."""
        return self._status

    @property
    def latest_descriptor(self) -> FrameBufferDescriptor:
        """Latest lossless acquisition descriptor."""
        if self._latest_descriptor is None:
            raise CameraServiceStateError("no frame has been acquired")
        return self._latest_descriptor

    async def discover(self) -> DetectorCapabilities:
        """Return deterministic capabilities without connecting hardware."""
        return DetectorCapabilities(
            trigger_modes=frozenset({"software", "external"}),
            exposure_range_s=(0.000_1, 10.0),
            pixel_formats=frozenset({"mono16"}),
            supports_hardware_timestamps=True,
            supports_roi=True,
            binning=(1, 2, 4),
        )

    async def connect(self) -> None:
        """Start a new independently generated service/buffer generation."""
        if self._status.connection_state is ServiceConnectionState.READY:
            return
        self._update(
            connection_state=ServiceConnectionState.CONNECTING,
            command_state="connect",
            last_error=None,
        )
        if self._fail_connect_attempts:
            self._fail_connect_attempts -= 1
            self._update(
                connection_state=ServiceConnectionState.FAULTED,
                command_state="connect_failed",
                last_error="injected connection failure",
            )
            raise ConnectionError("injected connection failure")
        generation = self._status.schema_generation + 1
        preview = SharedMemoryFrameRing(
            service_id=self.service_id,
            shape=self.shape,
            dtype="uint16",
            slots=2,
            usage=BufferUsage.PREVIEW,
            generation=generation,
        )
        try:
            acquisition = SharedMemoryFrameRing(
                service_id=self.service_id,
                shape=self.shape,
                dtype="uint16",
                slots=self.acquisition_slots,
                usage=BufferUsage.ACQUISITION,
                generation=generation,
            )
        except BaseException:
            preview.close()
            raise
        self._preview = preview
        self._acquisition = acquisition
        self._latest_descriptor = None
        self._latest_frame = None
        self._update(
            schema_generation=generation,
            connection_state=ServiceConnectionState.READY,
            acquisition_state=AcquisitionState.IDLE,
            command_state="connected",
        )

    async def configure(self, settings: Mapping[str, JsonValue]) -> str:
        """Validate settings and return a content-derived revision."""
        self._require_ready()
        exposure_s = settings.get("exposure_s", 0.01)
        if not isinstance(exposure_s, (int, float)) or not 0.000_1 <= exposure_s <= 10:
            raise ValueError("exposure_s is outside the discovered range")
        serialized = json.dumps(
            thaw_json(settings), sort_keys=True, separators=(",", ":")
        )
        self._configuration_revision = hashlib.sha256(serialized.encode()).hexdigest()
        self._update(command_state="configured")
        return self._configuration_revision

    async def arm(self) -> None:
        """Arm software acquisition."""
        self._require_ready()
        self._update(acquisition_state=AcquisitionState.ARMED, command_state="armed")

    async def trigger(self) -> None:
        """Publish distinct preview and acquisition descriptors."""
        if self._status.acquisition_state is not AcquisitionState.ARMED:
            raise CameraServiceStateError("camera must be armed before trigger")
        preview, acquisition = self._require_rings()
        self._update(
            acquisition_state=AcquisitionState.ACQUIRING,
            command_state="trigger",
        )
        sequence = self._sequence
        frame_id = str(uuid4())
        run_id = self._run_id
        timestamp_ns = time.time_ns()
        array = self._frame_data(sequence)
        preview.publish(
            array,
            run_id=run_id,
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
        )
        try:
            descriptor = acquisition.publish(
                array,
                run_id=run_id,
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
            )
        except BufferFullError:
            self._update(
                acquisition_state=AcquisitionState.ARMED,
                command_state="buffer_full",
                frames_dropped=self._status.frames_dropped + 1,
                last_error="lossless acquisition buffer requires acknowledgement",
            )
            raise
        self._latest_descriptor = descriptor
        self._latest_frame = Frame(
            run_id=run_id,
            frame_id=frame_id,
            detector_id=self.service_id,
            channel_id=self.service_id,
            sequence=sequence,
            exposure_started_ns=timestamp_ns - 1_000_000,
            exposure_ended_ns=timestamp_ns,
            array=ArrayReference(
                uri=(
                    f"shm://{descriptor.shared_memory_name}"
                    f"?generation={descriptor.generation}&slot={descriptor.slot}"
                    f"&sequence={descriptor.sequence}"
                ),
                checksum=descriptor.checksum,
                shape=descriptor.shape,
                dtype=descriptor.dtype,
                byte_order=descriptor.byte_order,
            ),
            configuration_revision=self._configuration_revision,
        )
        self._sequence += 1
        self._update(
            acquisition_state=AcquisitionState.ARMED,
            command_state="frame_committed",
            frames_published=self._status.frames_published + 1,
            last_error=None,
        )

    async def read(self) -> Frame:
        """Return metadata for the latest committed acquisition frame."""
        if self._latest_frame is None:
            raise CameraServiceStateError("no frame has been acquired")
        return self._latest_frame

    async def acknowledge(self, sequence: int) -> None:
        """Release the exact latest frame after a durable consumer copy."""
        descriptor = self.latest_descriptor
        if sequence != descriptor.sequence:
            raise ValueError(
                f"acknowledgement sequence {sequence} does not match "
                f"latest sequence {descriptor.sequence}"
            )
        _, acquisition = self._require_rings()
        acquisition.acknowledge(descriptor)

    async def copy(self, sequence: int) -> npt.NDArray[Any]:
        """Copy the exact retained frame without acknowledging its slot."""
        descriptor = self.latest_descriptor
        if sequence != descriptor.sequence:
            raise ValueError(
                f"copy sequence {sequence} does not match "
                f"latest sequence {descriptor.sequence}"
            )
        return self.copy_latest(acknowledge=False)

    def copy_latest(self, *, acknowledge: bool = True) -> npt.NDArray[Any]:
        """Act as the designated writer by copying and acknowledging a frame."""
        _, acquisition = self._require_rings()
        descriptor = self.latest_descriptor
        lease = acquisition.acquire(descriptor)
        try:
            return acquisition.copy(lease, acknowledge=acknowledge)
        finally:
            acquisition.release(lease)

    async def stop(self) -> None:
        """Return acquisition to an idle safe state."""
        if self._status.connection_state is ServiceConnectionState.READY:
            self._update(
                acquisition_state=AcquisitionState.IDLE,
                command_state="stopped",
            )

    async def disconnect(self) -> None:
        """Close and unlink service-owned shared-memory resources."""
        self._close_rings()
        self._latest_descriptor = None
        self._latest_frame = None
        self._update(
            connection_state=ServiceConnectionState.DISCONNECTED,
            acquisition_state=AcquisitionState.IDLE,
            command_state="disconnected",
        )

    async def inject_disconnect(self, reason: str = "injected service death") -> None:
        """Simulate process death and invalidate the current generation."""
        self._close_rings()
        self._update(
            connection_state=ServiceConnectionState.FAULTED,
            acquisition_state=AcquisitionState.IDLE,
            command_state="service_lost",
            last_error=reason,
        )

    def _frame_data(self, sequence: int) -> npt.NDArray[np.uint16]:
        identity = sum(self.service_id.encode()) % 256
        return np.full(self.shape, identity + sequence, dtype=np.uint16)

    def _require_ready(self) -> None:
        if self._status.connection_state is not ServiceConnectionState.READY:
            raise CameraServiceStateError("camera service is not ready")

    def _require_rings(self) -> tuple[SharedMemoryFrameRing, SharedMemoryFrameRing]:
        self._require_ready()
        if self._preview is None or self._acquisition is None:  # pragma: no cover
            raise CameraServiceStateError("camera service buffers are unavailable")
        return self._preview, self._acquisition

    def _close_rings(self) -> None:
        for ring in (self._preview, self._acquisition):
            if ring is not None:
                ring.close()
        self._preview = None
        self._acquisition = None

    def _update(
        self,
        *,
        schema_generation: int | None = None,
        connection_state: ServiceConnectionState | None = None,
        acquisition_state: AcquisitionState | None = None,
        command_state: str | None = None,
        frames_published: int | None = None,
        frames_dropped: int | None = None,
        last_error: str | _Unchanged | None = _UNCHANGED,
    ) -> None:
        previous = self._status
        self._status = HardwareServiceStatus(
            service_id=previous.service_id,
            epics_prefix=previous.epics_prefix,
            schema_generation=(
                previous.schema_generation
                if schema_generation is None
                else schema_generation
            ),
            heartbeat_ns=time.monotonic_ns(),
            connection_state=(
                previous.connection_state
                if connection_state is None
                else connection_state
            ),
            acquisition_state=(
                previous.acquisition_state
                if acquisition_state is None
                else acquisition_state
            ),
            command_state=(
                previous.command_state if command_state is None else command_state
            ),
            frames_published=(
                previous.frames_published
                if frames_published is None
                else frames_published
            ),
            frames_dropped=(
                previous.frames_dropped if frames_dropped is None else frames_dropped
            ),
            last_error=(
                previous.last_error
                if isinstance(last_error, _Unchanged)
                else last_error
            ),
        )


__all__ = ["CameraServiceStateError", "MockCameraService"]
