"""Read-only ESP300 shared-port admission for the experimental survey path.

This module does not open a port or provide move, enable, home, or stop
commands. The caller owns one correctly configured serial endpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import Lock
from typing import Protocol


class SerialEndpoint(Protocol):
    """Bounded blocking serial methods; timeout is configured by the caller."""

    def write(self, data: bytes) -> int:
        """Return the number of bytes accepted."""
        ...

    def flush(self) -> None:
        """Flush accepted command bytes."""
        ...

    def readline(self, size: int = -1) -> bytes:
        """Return a bounded response line."""
        ...


@dataclass(frozen=True, slots=True)
class AxisSnapshot:
    """Controller-reported values in micrometres, not independent position."""

    axis: int
    identity: str
    resolution_um: float
    lower_um: float
    upper_um: float
    actual_um: float
    target_um: float
    motion_done: bool


@dataclass(frozen=True, slots=True)
class ControllerSnapshot:
    """One serialized series of queries; axes are not sampled simultaneously."""

    version: str
    axes: tuple[AxisSnapshot, ...]
    quality: str = "controller_reported"


class Esp300ReadOnly:
    """One shared serial owner; an uncertain reply poisons this instance."""

    def __init__(self, endpoint: SerialEndpoint) -> None:
        self._endpoint = endpoint
        self._lock = Lock()
        self._uncertain = False

    def _query(self, command: str) -> str:
        data = (command + "?\r").encode("ascii")
        if self._endpoint.write(data) != len(data):
            raise OSError("short ESP300 serial write")
        self._endpoint.flush()
        raw = self._endpoint.readline(512)
        if not raw or len(raw) >= 512 or not raw.endswith(b"\n"):
            raise RuntimeError("missing, truncated, or oversized ESP300 response")
        try:
            text = raw.decode("ascii", "strict").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError("non-ASCII ESP300 response") from exc
        if not text or any(ord(char) < 32 for char in text):
            raise RuntimeError("invalid ESP300 response")
        return text

    def _number(self, axis: int, query: str) -> float:
        try:
            value = float(self._query(f"{axis}{query}"))
        except ValueError as exc:
            raise RuntimeError("invalid ESP300 numeric response") from exc
        if not math.isfinite(value):
            raise RuntimeError("nonfinite ESP300 numeric response")
        return value

    def snapshot(self) -> ControllerSnapshot:
        """Query identity and axes; never enable, home, move, stop, or clear errors."""
        with self._lock:
            if self._uncertain:
                raise RuntimeError("serial stream uncertain; reopen and reconcile")
            try:
                version = self._query("VE")
                if "ESP300" not in version.upper():
                    raise RuntimeError("unexpected controller identity")
                axes = []
                for axis in (1, 2, 3):
                    identity = self._query(f"{axis}ID")
                    if self._query(f"{axis}SN") != "2":
                        raise RuntimeError("axis unit is not millimetres")
                    resolution = self._number(axis, "SU") * 1000
                    low = self._number(axis, "SL") * 1000
                    high = self._number(axis, "SR") * 1000
                    actual = self._number(axis, "TP") * 1000
                    target = self._number(axis, "DP") * 1000
                    done = self._query(f"{axis}MD")
                    if (
                        resolution <= 0
                        or low >= high
                        or not low <= actual <= high
                        or done not in ("0", "1")
                    ):
                        raise RuntimeError("invalid axis configuration or status")
                    axes.append(
                        AxisSnapshot(
                            axis,
                            identity,
                            resolution,
                            low,
                            high,
                            actual,
                            target,
                            done == "1",
                        )
                    )
                return ControllerSnapshot(version, tuple(axes))
            except BaseException:
                self._uncertain = True
                raise
