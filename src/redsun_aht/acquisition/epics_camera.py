"""Lossless DPCT detector adapter for one isolated EPICS camera service."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from caproto.threading.client import Context

from redsun_aht.domain import ArrayReference, Frame

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from redsun_aht.domain.models import JsonValue


@dataclass(slots=True)
class EpicsCameraDetector:
    """Adapt one camera IOC to the lossless detector contract used by DPCT.

    The IOC owns the camera SDK and shared-memory slots.  This client only
    commands the documented EPICS control plane, copies a committed slot, and
    acknowledges it after its caller has durably retained the payload.
    Camera settings are deliberately not accepted here: the current IOC has no
    typed configuration command, so a recipe must use its admitted camera
    configuration rather than silently ignoring requested settings.
    """

    service_id: str
    prefix: str
    timeout: float = 15.0
    poll_interval: float = 0.005
    _connected: bool = field(default=False, init=False, repr=False)
    _latest_frame: Frame | None = field(default=None, init=False, repr=False)
    _latest_memory_name: str | None = field(default=None, init=False, repr=False)
    _latest_slot: int | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _context: Context | None = field(default=None, init=False, repr=False)
    _pvs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

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
        await asyncio.to_thread(self._connect)

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
        await asyncio.to_thread(self._arm)

    async def trigger(self) -> None:
        """Trigger exactly one frame and retain its immutable descriptor."""
        await asyncio.to_thread(self._trigger)

    async def read(self) -> Frame:
        """Return metadata for the most recently committed frame."""
        if self._latest_frame is None:
            raise RuntimeError("camera IOC has not committed a frame")
        return self._latest_frame

    async def copy(self, sequence: int) -> npt.NDArray[Any]:
        """Copy and checksum-verify the exact unacknowledged shared-memory slot."""
        return await asyncio.to_thread(self._copy, sequence)

    async def acknowledge(self, sequence: int) -> None:
        """Release precisely the frame slot that was copied by the caller."""
        await asyncio.to_thread(self._acknowledge, sequence)

    async def stop(self) -> None:
        """Return ordinary camera acquisition to idle."""
        await asyncio.to_thread(self._stop)

    async def disconnect(self) -> None:
        """Release the isolated camera service and invalidate local metadata."""
        await asyncio.to_thread(self._disconnect)

    def _connect(self) -> None:
        with self._lock:
            if self._connected:
                return
            self._open_client()
            try:
                self._write("COMMAND:CONNECT", 1)
                self._await("CONNECTION_STATE", "ready")
                observed = self._text("SERVICE_ID")
                if observed == self.service_id:
                    self._connected = True
                    return
                try:
                    self._write("COMMAND:DISCONNECT", 1)
                finally:
                    self._connected = False
                raise RuntimeError(
                    f"camera IOC service ID {observed!r} does not match "
                    f"{self.service_id!r}"
                )
            except BaseException:
                self._close_client()
                raise

    def _arm(self) -> None:
        with self._lock:
            self._require_connected()
            self._write("COMMAND:ARM", 1)
            self._await("ACQUISITION_STATE", "armed")

    def _trigger(self) -> None:
        with self._lock:
            self._require_connected()
            published = self._integer("FRAMES_PUBLISHED")
            self._write("COMMAND:TRIGGER", 1)
            self._await("FRAMES_PUBLISHED", published + 1)
            sequence = self._integer("DESCRIPTOR_SEQUENCE")
            if sequence < 0:
                raise RuntimeError("camera IOC published an invalid frame sequence")
            shape = tuple(
                int(item) for item in self._text("DESCRIPTOR_SHAPE").split(",")
            )
            if not shape or any(item <= 0 for item in shape):
                raise RuntimeError("camera IOC published an invalid frame shape")
            dtype = self._text("DESCRIPTOR_DTYPE")
            byte_order = self._text("DESCRIPTOR_BYTE_ORDER")
            if byte_order not in {"=", "<", ">", "|"}:
                raise RuntimeError("camera IOC published an invalid byte order")
            try:
                np.dtype(dtype)
            except TypeError as error:
                raise RuntimeError("camera IOC published an invalid dtype") from error
            memory_name = self._text("DESCRIPTOR_NAME")
            frame_id = self._text("DESCRIPTOR_FRAME_ID")
            run_id = self._text("DESCRIPTOR_RUN_ID")
            checksum = self._text("DESCRIPTOR_CHECKSUM_HI") + self._text(
                "DESCRIPTOR_CHECKSUM_LO"
            )
            if not all((memory_name, frame_id, run_id)) or len(checksum) != 64:
                raise RuntimeError(
                    "camera IOC published an incomplete frame descriptor"
                )
            timestamp_ns = self._integer("DESCRIPTOR_TIMESTAMP_NS")
            if timestamp_ns < 0:
                raise RuntimeError("camera IOC published an invalid frame timestamp")
            slot = self._integer("DESCRIPTOR_SLOT")
            if slot < 0:
                raise RuntimeError("camera IOC published an invalid frame slot")
            generation = self._integer("DESCRIPTOR_GENERATION")
            self._latest_frame = Frame(
                run_id=run_id,
                frame_id=frame_id,
                detector_id=self.service_id,
                channel_id=self.service_id,
                sequence=sequence,
                # The IOC publishes the committed-frame timestamp, not the
                # camera's exposure interval. Keep that limitation explicit.
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

    def _copy(self, sequence: int) -> npt.NDArray[Any]:
        with self._lock:
            frame = self._require_latest(sequence)
            memory_name = self._latest_memory_name
            slot = self._latest_slot
            if memory_name is None or slot is None:  # pragma: no cover - invariant
                raise RuntimeError("camera IOC frame slot metadata is unavailable")
            byte_order = cast(
                Literal["=", "<", ">", "|"],  # noqa: TC006 - NumPy needs Literal.
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

    def _acknowledge(self, sequence: int) -> None:
        with self._lock:
            self._require_latest(sequence)
            self._write("COMMAND:ACKNOWLEDGE", sequence)
            self._await("COMMAND_STATE", "frame_acknowledged")

    def _stop(self) -> None:
        with self._lock:
            if self._connected:
                self._write("COMMAND:STOP", 1)
                self._await("ACQUISITION_STATE", "idle")

    def _disconnect(self) -> None:
        with self._lock:
            try:
                if self._connected:
                    self._write("COMMAND:DISCONNECT", 1)
                    self._await("CONNECTION_STATE", "disconnected")
            finally:
                self._connected = False
                self._latest_frame = None
                self._latest_memory_name = None
                self._latest_slot = None
                self._close_client()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("camera IOC detector is not connected")

    def _require_latest(self, sequence: int) -> Frame:
        frame = self._latest_frame
        if frame is None or frame.sequence != sequence:
            raise ValueError("camera IOC sequence does not match the latest frame")
        return frame

    def _await(self, suffix: str, expected: object) -> None:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                value = self._read(suffix)
                if isinstance(value, bytes):
                    value = value.decode()
                if value == expected:
                    return
            except Exception:
                pass
            time.sleep(self.poll_interval)
        raise TimeoutError(f"{self.prefix}{suffix} did not become {expected!r}")

    def _integer(self, suffix: str) -> int:
        return int(self._text(suffix))

    def _text(self, suffix: str) -> str:
        value = self._read(suffix)
        return value.decode() if isinstance(value, bytes) else str(value)

    def _read(self, suffix: str) -> object:
        return self._pv(suffix).read(timeout=self.timeout).data[0]

    def _write(self, suffix: str, value: int) -> None:
        self._pv(suffix).write(
            value,
            notify=True,
            timeout=self.timeout,
        )

    def _open_client(self) -> None:
        if self._context is not None:
            return
        suffixes = (
            "SERVICE_ID",
            "CONNECTION_STATE",
            "ACQUISITION_STATE",
            "COMMAND_STATE",
            "FRAMES_PUBLISHED",
            "DESCRIPTOR_GENERATION",
            "DESCRIPTOR_SEQUENCE",
            "DESCRIPTOR_SLOT",
            "DESCRIPTOR_NAME",
            "DESCRIPTOR_RUN_ID",
            "DESCRIPTOR_FRAME_ID",
            "DESCRIPTOR_SHAPE",
            "DESCRIPTOR_DTYPE",
            "DESCRIPTOR_BYTE_ORDER",
            "DESCRIPTOR_TIMESTAMP_NS",
            "DESCRIPTOR_CHECKSUM_HI",
            "DESCRIPTOR_CHECKSUM_LO",
            "COMMAND:CONNECT",
            "COMMAND:ARM",
            "COMMAND:TRIGGER",
            "COMMAND:STOP",
            "COMMAND:DISCONNECT",
            "COMMAND:ACKNOWLEDGE",
        )
        context = Context(timeout=self.timeout)
        pvs = context.get_pvs(
            *(f"{self.prefix}{suffix}" for suffix in suffixes), timeout=self.timeout
        )
        self._context = context
        self._pvs = dict(zip(suffixes, pvs, strict=True))

    def _close_client(self) -> None:
        context, self._context = self._context, None
        self._pvs.clear()
        if context is not None:
            context.disconnect()

    def _pv(self, suffix: str) -> Any:
        try:
            return self._pvs[suffix]
        except KeyError as exc:
            raise RuntimeError(
                f"camera IOC PV suffix is not configured: {suffix}"
            ) from exc


__all__ = ["EpicsCameraDetector"]
