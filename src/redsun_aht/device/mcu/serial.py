"""Serial framing adapter for an already-open MCU port."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol


class SerialPort(Protocol):
    """Subset of a binary pyserial-compatible port used by the adapter."""

    def write(self, data: bytes) -> int:
        """Write bytes and return the number accepted."""

    def flush(self) -> None:
        """Wait until buffered output has been written."""

    def read_until(self, expected: bytes, size: int | None = None) -> bytes:
        """Read through *expected* or until timeout/size exhaustion."""


@dataclass(slots=True)
class DelimitedSerialTransport:
    """Serialize request/response exchanges on one COBS-framed serial port."""

    port: SerialPort
    max_response_bytes: int = 8192
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the bounded read policy."""
        if self.max_response_bytes < 2:
            raise ValueError("max_response_bytes must be at least 2")

    def exchange(self, frame: bytes) -> bytes:
        """Write exactly one frame and read exactly one delimited response."""
        if not frame.endswith(b"\x00") or b"\x00" in frame[:-1]:
            raise ValueError("request must contain exactly one delimited frame")
        with self._lock:
            written = self.port.write(frame)
            if written != len(frame):
                raise OSError(f"short serial write: {written} of {len(frame)} bytes")
            self.port.flush()
            response = self.port.read_until(b"\x00", self.max_response_bytes)
        if not response.endswith(b"\x00"):
            raise TimeoutError("timed out waiting for a complete MCU response")
        return response
