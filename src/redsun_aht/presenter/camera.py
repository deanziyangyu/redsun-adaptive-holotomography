"""One-frame camera-service acquisition for an optional Qt client."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Protocol

import numpy as np
from caproto.sync.client import read, write


class CameraGuiClient(Protocol):
    """Small, synchronous acquisition surface suitable for a Qt button action."""

    def acquire_one(self) -> AcquiredCameraFrame:
        """Acquire, verify, acknowledge, and detach one frame."""


@dataclass(frozen=True, slots=True)
class AcquiredCameraFrame:
    """Detached frame returned only after its service lease is acknowledged."""

    service_id: str
    run_id: str
    frame_id: str
    sequence: int
    timestamp_ns: int
    checksum: str
    array: np.ndarray[Any, Any]


class EpicsCameraGuiClient:
    """Use the existing lossless EPICS/shm service contract for one GUI capture.

    The client owns no camera SDK and does not construct a Micro-Manager core.
    It is intentionally single-frame and synchronous so a GUI test can make
    the exact same acknowledgement and cleanup assertions as the headless
    physical gate.
    """

    def __init__(self, prefix: str, *, timeout: float = 15.0) -> None:
        if not prefix.endswith(":"):
            raise ValueError("camera service prefix must end with ':'")
        if timeout <= 0:
            raise ValueError("camera service timeout must be positive")
        self.prefix = prefix
        self.timeout = timeout

    def acquire_one(self) -> AcquiredCameraFrame:
        """Capture one frame and always request ordinary service cleanup."""
        connected = False
        try:
            self._write("COMMAND:CONNECT", 1)
            connected = True
            self._await("CONNECTION_STATE", "ready")
            self._write("COMMAND:ARM", 1)
            self._await("ACQUISITION_STATE", "armed")
            published_before = self._integer("FRAMES_PUBLISHED")
            self._write("COMMAND:TRIGGER", 1)
            self._await("FRAMES_PUBLISHED", published_before + 1)
            frame = self._copy_verified_frame()
            self._write("COMMAND:ACKNOWLEDGE", frame.sequence)
            self._await("COMMAND_STATE", "frame_acknowledged")
            return frame
        finally:
            if connected:
                try:
                    self._write("COMMAND:STOP", 1)
                finally:
                    self._write("COMMAND:DISCONNECT", 1)

    def _copy_verified_frame(self) -> AcquiredCameraFrame:
        sequence = self._integer("DESCRIPTOR_SEQUENCE")
        if sequence < 0:
            raise RuntimeError("camera service did not publish a frame descriptor")
        shape = tuple(int(item) for item in self._text("DESCRIPTOR_SHAPE").split(","))
        dtype = np.dtype(self._text("DESCRIPTOR_DTYPE"))
        slot = self._integer("DESCRIPTOR_SLOT")
        name = self._text("DESCRIPTOR_NAME")
        if not name:
            raise RuntimeError("camera service published an empty shared-memory name")
        expected_checksum = self._text("DESCRIPTOR_CHECKSUM_HI") + self._text(
            "DESCRIPTOR_CHECKSUM_LO"
        )
        frame_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        memory = shared_memory.SharedMemory(name=name)
        try:
            array = np.ndarray(
                shape,
                dtype=dtype,
                buffer=memory.buf,
                offset=slot * frame_bytes,
            ).copy()
        finally:
            memory.close()
        checksum = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        if checksum != expected_checksum:
            raise ValueError("camera shared-memory frame checksum mismatch")
        return AcquiredCameraFrame(
            service_id=self._text("SERVICE_ID"),
            run_id=self._text("DESCRIPTOR_RUN_ID"),
            frame_id=self._text("DESCRIPTOR_FRAME_ID"),
            sequence=sequence,
            timestamp_ns=self._integer("DESCRIPTOR_TIMESTAMP_NS"),
            checksum=checksum,
            array=array,
        )

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
            time.sleep(0.05)
        raise TimeoutError(f"{self.prefix}{suffix} did not become {expected!r}")

    def _text(self, suffix: str) -> str:
        value = self._read(suffix)
        return value.decode() if isinstance(value, bytes) else str(value)

    def _integer(self, suffix: str) -> int:
        return int(self._text(suffix))

    def _read(self, suffix: str) -> object:
        return read(f"{self.prefix}{suffix}", timeout=0.5, repeater=False).data[0]

    def _write(self, suffix: str, value: int) -> None:
        write(
            f"{self.prefix}{suffix}",
            value,
            notify=True,
            repeater=False,
            timeout=self.timeout,
        )


__all__ = ["AcquiredCameraFrame", "CameraGuiClient", "EpicsCameraGuiClient"]
