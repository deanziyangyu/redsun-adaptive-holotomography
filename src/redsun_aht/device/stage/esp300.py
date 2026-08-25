"""Newport ESP300 ASCII transport and axis backend.

The controller object does not open or own a serial port. Deployment code must
construct an explicit 19200-baud, 8-N-1, RTS/CTS pyserial-compatible endpoint
and close it after all composed axes disconnect.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from typing import Protocol

from redsun_aht.device.stage.composite import AxisSpec, CompositeStageDevice

ESP300_BAUD_RATE = 19_200
ESP300_DATA_BITS = 8
ESP300_PARITY = "N"
ESP300_STOP_BITS = 1
ESP300_RTS_CTS = True
ESP300_TIMEOUT_SECONDS = 1.0

ESP300_UNITS = {
    0: "encoder_count",
    1: "motor_step",
    2: "mm",
    3: "um",
    4: "in",
    5: "mil",
    6: "micro_in",
    7: "deg",
    8: "grad",
    9: "rad",
    10: "mrad",
    11: "urad",
}


class Esp300SerialPort(Protocol):
    """Binary serial subset used by the ESP300 controller adapter."""

    def write(self, data: bytes) -> int:
        """Write command bytes and return the number accepted."""

    def flush(self) -> None:
        """Wait until buffered output has been written."""

    def readline(self, size: int = -1) -> bytes:
        """Read one controller response line."""


@dataclass(frozen=True, slots=True)
class Esp300AxisIdentity:
    """Read-only identity, unit, resolution, limit, and position snapshot."""

    axis: int
    stage_id: str
    unit_code: int
    unit: str
    encoder_resolution: float
    lower_limit: float
    upper_limit: float
    position: float


@dataclass(frozen=True, slots=True)
class Esp300Identity:
    """Read-only controller and configured-axis snapshot."""

    version: str
    axes: tuple[Esp300AxisIdentity, ...]


@dataclass(slots=True)
class Esp300Controller:
    """Serialize bounded ASCII commands over an already-open serial port."""

    port: Esp300SerialPort
    axis_count: int = 3
    max_response_bytes: int = 512
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate controller topology and response bound."""
        if not 1 <= self.axis_count <= 3:
            raise ValueError("ESP300 axis_count must be in the range 1..3")
        if self.max_response_bytes < 2:
            raise ValueError("max_response_bytes must be at least 2")

    def read_identity(self) -> Esp300Identity:
        """Read controller identity and donor-proven axis configuration."""
        version = self.query("VE")
        if "ESP300" not in version.upper():
            raise RuntimeError(f"unexpected Newport controller identity: {version!r}")
        axes = tuple(
            self._read_axis_identity(axis) for axis in range(1, self.axis_count + 1)
        )
        return Esp300Identity(version=version, axes=axes)

    def _read_axis_identity(self, axis: int) -> Esp300AxisIdentity:
        stage_id = self.query("ID", axis=axis)
        unit_code = self._query_int("SN", axis)
        try:
            unit = ESP300_UNITS[unit_code]
        except KeyError as exc:
            raise RuntimeError(f"unknown ESP300 unit code: {unit_code}") from exc
        resolution = self._query_float("SU", axis)
        lower_limit = self._query_float("SL", axis)
        upper_limit = self._query_float("SR", axis)
        position = self._query_float("TP", axis)
        values = (resolution, lower_limit, upper_limit, position)
        if not all(isfinite(value) for value in values):
            raise RuntimeError(f"ESP300 axis {axis} reports a non-finite value")
        if resolution <= 0:
            raise RuntimeError(
                f"ESP300 axis {axis} reports a non-positive encoder resolution"
            )
        if lower_limit >= upper_limit:
            raise RuntimeError(
                f"ESP300 axis {axis} reports invalid limits "
                f"[{lower_limit}, {upper_limit}]"
            )
        if not lower_limit <= position <= upper_limit:
            raise RuntimeError(
                f"ESP300 axis {axis} position is outside its reported limits"
            )
        return Esp300AxisIdentity(
            axis=axis,
            stage_id=stage_id,
            unit_code=unit_code,
            unit=unit,
            encoder_resolution=resolution,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            position=position,
        )

    def query(self, command: str, *, axis: int | None = None) -> str:
        """Issue one query and return a non-empty bounded ASCII response."""
        wire = self._wire_command(command, axis=axis, query=True)
        with self._lock:
            self._write(wire)
            response = self.port.readline(self.max_response_bytes)
        if not response:
            raise TimeoutError(f"ESP300 query {wire!r} timed out")
        if len(response) >= self.max_response_bytes and not response.endswith(
            (b"\r", b"\n")
        ):
            raise RuntimeError("ESP300 response exceeds configured bound")
        try:
            decoded = response.rstrip(b"\r\n").decode("ascii")
        except UnicodeDecodeError as exc:
            raise RuntimeError("ESP300 response is not ASCII") from exc
        if not decoded:
            raise RuntimeError("ESP300 returned an empty response line")
        return decoded

    def write_axis(self, axis: int, command: str) -> None:
        """Issue one non-query axis command."""
        wire = self._wire_command(command, axis=axis, query=False)
        with self._lock:
            self._write(wire)

    def enable_axis(self, axis: int) -> None:
        """Enable one admitted axis before an explicit motion request.

        This is an ordinary controller command, not a homing or safety-reset
        operation. Callers remain responsible for explicit movement authority.
        """
        self.write_axis(axis, "MO")

    def _query_float(self, command: str, axis: int) -> float:
        response = self.query(command, axis=axis)
        try:
            return float(response)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid ESP300 {command} response: {response!r}"
            ) from exc

    def _query_int(self, command: str, axis: int) -> int:
        response = self.query(command, axis=axis)
        try:
            return int(response)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid ESP300 {command} response: {response!r}"
            ) from exc

    def _write(self, wire: bytes) -> None:
        written = self.port.write(wire)
        if written != len(wire):
            raise OSError(f"short ESP300 write: {written} of {len(wire)} bytes")
        self.port.flush()

    def _wire_command(self, command: str, *, axis: int | None, query: bool) -> bytes:
        if axis is not None and not 1 <= axis <= self.axis_count:
            raise ValueError(f"ESP300 axis must be in range 1..{self.axis_count}")
        normalized = command.strip().upper()
        if not normalized or not normalized.isascii():
            raise ValueError("ESP300 command must be non-empty ASCII")
        if any(value in normalized for value in ("\r", "\n", "?")):
            raise ValueError("ESP300 command contains framing characters")
        prefix = str(axis) if axis is not None else ""
        suffix = "?" if query else ""
        return f"{prefix}{normalized}{suffix}\r".encode("ascii")


@dataclass(slots=True)
class Esp300AxisBackend:
    """Adapt one ESP300 axis to the composite-stage backend contract."""

    controller: Esp300Controller
    axis: int

    def __post_init__(self) -> None:
        """Validate the axis without issuing a hardware command."""
        if not 1 <= self.axis <= self.controller.axis_count:
            raise ValueError("axis is outside the configured ESP300 topology")

    async def move_to(self, position: float) -> None:
        """Start an absolute move in the controller's preconfigured units."""
        await asyncio.to_thread(
            self.controller.write_axis, self.axis, f"PA{position:.12g}"
        )

    async def read_position(self) -> float:
        """Read the actual position in the controller's configured units."""
        return await asyncio.to_thread(self.controller._query_float, "TP", self.axis)

    async def motion_complete(self) -> bool:
        """Read the motion-done status (`1` means complete)."""
        response = await asyncio.to_thread(self.controller.query, "MD", axis=self.axis)
        if response not in {"0", "1"}:
            raise RuntimeError(f"invalid ESP300 motion status: {response!r}")
        return response == "1"

    async def stop(self) -> None:
        """Issue the documented ordinary axis stop command."""
        await asyncio.to_thread(self.controller.write_axis, self.axis, "ST")


def build_esp300_composite_stage(
    controller: Esp300Controller,
    identity: Esp300Identity,
    *,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    coordinate_frame: str = "aht-sample-stage",
    settle_timeout: float = 10.0,
    settle_poll_interval: float = 0.02,
) -> CompositeStageDevice:
    """Compose admitted ESP300 axes using their reported units and limits."""
    if len(identity.axes) != controller.axis_count:
        raise ValueError("identity axis count does not match controller topology")
    if len(axis_names) != len(identity.axes) or len(set(axis_names)) != len(axis_names):
        raise ValueError("axis_names must uniquely name every admitted ESP300 axis")

    specs: list[AxisSpec] = []
    backends: dict[str, Esp300AxisBackend] = {}
    for name, axis_identity in zip(axis_names, identity.axes, strict=True):
        tolerance = axis_identity.encoder_resolution * 2
        specs.append(
            AxisSpec(
                name=name,
                unit=axis_identity.unit,
                lower_limit=axis_identity.lower_limit,
                upper_limit=axis_identity.upper_limit,
                settle_tolerance=tolerance,
                uncertainty=axis_identity.encoder_resolution / 2,
            )
        )
        backends[name] = Esp300AxisBackend(controller, axis_identity.axis)
    return CompositeStageDevice(
        specs=tuple(specs),
        backends=backends,
        coordinate_frame=coordinate_frame,
        source="newport-esp300",
        settle_timeout=settle_timeout,
        settle_poll_interval=settle_poll_interval,
    )
