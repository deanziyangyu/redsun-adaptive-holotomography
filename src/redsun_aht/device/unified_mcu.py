"""Experimental single-owner transport for the unified LED/X-stack firmware.

This is an unqualified, newline-ASCII transport, not a camera acquisition or
an emergency stop. One instance must exclusively own the serial endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from oxiserial.aio import serial_for_url

from redsun_aht.device.dynamic_patterns import DynamicPattern, DynamicPatternLink

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from redsun_aht.device.arduino.service import AsyncSerialPort

IDENTITY = "ARH-UNIFIED-STACK/0.18"
DUE_IDENTITY = "ARH-DUE ZDT-X protocol=1"
logger = logging.getLogger(__name__)


def open_unified_port(port_name: str) -> AsyncSerialPort:
    """Open the unified endpoint at 115200 with DTR/RTS set before open."""
    if not port_name.strip() or "<" in port_name or ">" in port_name:
        raise ValueError("an explicit serial port name is required")
    port = serial_for_url(
        port_name,
        do_not_open=True,
        baudrate=115200,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=0.25,
        write_timeout=1.0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    port.dtr = False
    port.rts = False
    port.open()
    return cast("AsyncSerialPort", port)


class UnifiedMcuError(RuntimeError):
    """The operation failed or its outcome cannot be established."""


@dataclass(frozen=True)
class RotationState:
    """Software step counter and motion state, not encoder feedback."""

    position: int
    target: int
    moving: bool
    enabled: bool


@dataclass(frozen=True, slots=True)
class XControllerState:
    """Raw controller-reported X state, not an independent position reading."""

    target_tenths: int
    position_tenths: int
    speed_tenths_rpm: int
    error_hundredths_degree: int
    home_flags: int
    motor_flags: int
    raw_hex: str
    quality: str = "controller_reported"


def parse_x_state(frame: bytes) -> XControllerState:
    """Validate and decode the X-family aggregate state reply."""
    if (
        len(frame) != 37
        or frame[:4] != bytes((1, 0x43, 37, 12))
        or frame[-1] != 0x6B
        or any(frame[index] not in (0, 1) for index in (14, 19, 22, 27))
    ):
        raise UnifiedMcuError("invalid X-controller state")

    def signed(sign: int, start: int, end: int) -> int:
        value = int.from_bytes(frame[start:end], "big")
        return -value if frame[sign] else value

    return XControllerState(
        signed(14, 15, 19),
        signed(22, 23, 27),
        signed(19, 20, 22),
        signed(27, 28, 32),
        frame[34],
        frame[35],
        frame.hex(),
    )


@dataclass(frozen=True, slots=True)
class AngularStack:
    """One finite X stack, in 0.1-degree units."""

    run_id: int
    first: int
    step: int
    count: int
    lower: int
    upper: int
    trigger_mask: int = 1

    def __post_init__(self) -> None:
        """Check the firmware's finite session envelope before any I/O."""
        values = (
            self.run_id,
            self.first,
            self.step,
            self.count,
            self.lower,
            self.upper,
            self.trigger_mask,
        )
        if any(type(value) is not int for value in values):
            raise ValueError("integer stack parameters required")
        if not 1 <= self.run_id <= 2_147_483_647 or not 1 <= self.count <= 32:
            raise ValueError("invalid run ID or plane count")
        if not 1 <= self.trigger_mask <= 7:
            raise ValueError("trigger mask must select outputs 1..3")
        if not -2_147_483_648 <= self.step <= 2_147_483_647:
            raise ValueError("step exceeds firmware integer range")
        if not -3600 <= self.lower <= self.upper <= 3600:
            raise ValueError("stack limits exceed session envelope")
        last = self.first + self.step * (self.count - 1)
        if (
            not self.lower <= self.first <= self.upper
            or not self.lower <= last <= self.upper
        ):
            raise ValueError("stack targets exceed bounds")
        self.commit_line()  # Validate the firmware's line-length limit before I/O.

    def commit_line(self) -> str:
        """Return the currently installed provisional command."""
        line = (
            f"COMMIT {self.run_id} {self.first} {self.step} "
            f"{self.count} {self.lower} {self.upper} 1"
        )
        if len(line) > 48:
            raise ValueError("stack command exceeds firmware line length")
        return line


@dataclass(frozen=True, slots=True)
class StackAttempt:
    """Software-reported GPIO pulse attempts, not camera exposures."""

    run_id: int
    targets_tenths: tuple[int, ...]
    quality: str = "controller_reported"


@dataclass(frozen=True, slots=True)
class StackProgress:
    """Validated events observed so far; incomplete is not a stop confirmation."""

    run_id: int
    targets_tenths: tuple[int, ...]
    complete: bool = False
    quality: str = "controller_reported"


@dataclass(frozen=True, slots=True)
class AutomaticShotSummary:
    """Final software pulse count, not verified GPIO edges or camera frames."""

    attempted_pulses: int


class UnifiedMcu:
    """One explicit owner of both illumination and the experimental stack port."""

    def __init__(self, port: AsyncSerialPort, *, line_timeout_s: float = 1.0) -> None:
        if line_timeout_s <= 0:
            raise ValueError("line timeout must be positive")
        self._port = port
        self._line_timeout_s = line_timeout_s
        self._dynamic = DynamicPatternLink(self._write, self._line, self._dynamic_cleanup, line_timeout_s)
        self._operation = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._identified = False
        self._uncertain = False
        self._active_id: int | None = None
        self._abort_sent = False
        self._illumination_held = False
        self._progress: StackProgress | None = None

    @property
    def progress(self) -> StackProgress | None:
        """Last observed events, including after an uncertain or aborted run."""
        return self._progress

    @property
    def uncertain(self) -> bool:
        """Whether serial outcomes require external reconciliation."""
        return self._uncertain

    async def _write(self, line: str) -> None:
        payload = (line + "\n").encode("ascii")
        async with self._write_lock:
            count = await asyncio.wait_for(
                self._port.write(payload), self._line_timeout_s
            )
            if count != len(payload):
                raise UnifiedMcuError("short serial write; outcome uncertain")
            await asyncio.wait_for(self._port.flush(), self._line_timeout_s)

    async def _line(self, timeout_s: float | None = None) -> str:
        timeout = self._line_timeout_s if timeout_s is None else timeout_s
        deadline = asyncio.get_running_loop().time() + timeout
        line = bytearray()
        while (remaining := deadline - asyncio.get_running_loop().time()) > 0:
            value = await asyncio.wait_for(self._port.read_until(b"\n"), remaining)
            if value:
                line.extend(value)
                if len(line) > 256 or b"\n" in line[:-1]:
                    raise UnifiedMcuError("invalid serial line")
                if line.endswith(b"\n"):
                    return line.decode("ascii", "strict").strip()
        raise TimeoutError("serial response timed out")

    async def _exchange(self, command: str, expected: str) -> None:
        await self._write(command)
        response = await self._line()
        if response != expected:
            raise UnifiedMcuError(f"unexpected response to {command}: {response!r}")

    async def _dynamic_cleanup(self, command: str) -> None:
        self._uncertain = True
        try:
            await asyncio.shield(asyncio.wait_for(self._write(command), 1.0))
        except BaseException:
            logger.exception("Could not send dynamic cleanup command")

    async def upload_dynamic_pattern(self, slot: int, pixels: bytes) -> DynamicPattern:
        """Publish one dynamic slot under the serial operation lock."""
        self._admit()
        async with self._operation:
            return await self._dynamic.upload_dynamic_pattern(slot, pixels)

    async def dynamic_shots(
        self,
        patterns: Sequence[DynamicPattern],
        repeats: int,
        snap: Callable[[int, DynamicPattern], Awaitable[None]],
        *,
        camera_exclusive: bool,
        illumination_ready: bool,
        snap_timeout_s: float = 20.0,
    ) -> None:
        """Run a stationary recipe, counting frames only after snap returns."""
        patterns = tuple(patterns)
        if not 1 <= len(patterns) <= 64 or type(repeats) is not int or not 1 <= repeats <= 32:
            raise ValueError("expected 1..64 steps and 1..32 repeats")
        if camera_exclusive is not True or illumination_ready is not True:
            raise ValueError("exclusive camera and illumination readiness required")
        if not 0 < snap_timeout_s < 30:
            raise ValueError("snap timeout must be positive and below the firmware lease")
        self._admit()
        async with self._operation:
            await self._dynamic.capabilities()
            for pattern in dict.fromkeys(patterns):
                if (
                    type(pattern.slot) is not int or not 1 <= pattern.slot <= 5
                    or type(pattern.generation) is not int or not 1 <= pattern.generation <= 0xFFFFFFFF
                    or type(pattern.crc32) is not int or not 0 <= pattern.crc32 <= 0xFFFFFFFF
                ):
                    raise ValueError("invalid dynamic pattern receipt")
                await self._exchange(
                    f"DPSLOT? {pattern.slot}",
                    f"DPSLOT {pattern.slot} {pattern.generation} {pattern.crc32}",
                )
            run = 1 + secrets.randbelow(0xFFFFFFFF)
            replay: set[str] = set()
            total = len(patterns) * repeats
            try:
                await self._dynamic.exchange(
                    f"DPCFG {run} {len(patterns)} {repeats}", f"DPCFG {run}", replay
                )
                for index, pattern in enumerate(patterns):
                    await self._dynamic.exchange(
                        f"DPSTEP {run} {index} {pattern.slot} {pattern.generation}",
                        f"DPSTEP {run} {index}", replay,
                    )
                await self._dynamic.exchange(f"DPARM {run}", f"DPARM {run} {total}", replay)
                for ordinal in range(total):
                    pattern = patterns[ordinal % len(patterns)]
                    token = ordinal + 1
                    await self._dynamic.exchange(
                        f"DPNEXT {run} {ordinal}",
                        f"DPREADY {run} {ordinal} {pattern.slot} {pattern.generation} {token}",
                        replay,
                    )
                    await asyncio.wait_for(snap(ordinal, pattern), snap_timeout_s)
                    await self._dynamic.exchange(
                        f"DPACK {run} {ordinal} {token}", f"DPACK {run} {ordinal} {token}", replay
                    )
                await self._dynamic.exchange(
                    f"DPSTATUS? {run}", f"DPSTATUS {run} 4 {total} 0", replay
                )
                await self._dynamic.exchange(
                    f"DPSTOP {run}", f"DPSTOP {run}", replay, retries=0
                )
                self._illumination_held = False
            except BaseException:
                await self._dynamic_cleanup(f"DPSTOP {run}")
                raise

    async def _preflight(
        self, command: str, prefix: str, function: int, fields: int
    ) -> bytes:
        await self._write(command)
        if await self._line() != "ZA pending":
            raise UnifiedMcuError("controller preflight not admitted")
        line = await self._line()
        if not line.startswith(prefix):
            raise UnifiedMcuError("controller preflight failed")
        try:
            frame = bytes.fromhex(line.removeprefix(prefix))
        except ValueError as exc:
            raise UnifiedMcuError("invalid preflight hex") from exc
        if (
            len(frame) != 37
            or frame[:4] != bytes((1, function, 37, fields))
            or frame[-1] != 0x6B
        ):
            raise UnifiedMcuError("invalid preflight envelope")
        return frame

    async def _wait_stationary_preflight(self, definition: AngularStack) -> None:
        """Wait briefly between runs for fresh controller-reported readiness."""
        deadline = asyncio.get_running_loop().time() + 1.0
        while True:
            state = parse_x_state(
                await self._preflight("zasyncxstate", "ZA state_hex=", 0x43, 12)
            )
            if (
                state.home_flags & 3 != 3
                or state.home_flags & 0x3C
                or not state.motor_flags & 1
                or state.motor_flags & 0x8C
                or not definition.lower <= state.position_tenths <= definition.upper
            ):
                raise UnifiedMcuError("controller fault or position outside bounds")
            if (
                state.motor_flags & 2
                and state.speed_tenths_rpm == 0
                and abs(state.error_hundredths_degree) <= 10
            ):
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise UnifiedMcuError("controller not stationary and ready")
            await asyncio.sleep(0.025)

    def _admit(self) -> None:
        if not self._identified or self._uncertain or self._operation.locked():
            raise UnifiedMcuError("unidentified, uncertain or busy unified MCU")

    async def read_x_state(self) -> XControllerState:
        """Query one serialized controller state; failed reads quarantine the port."""
        self._admit()
        async with self._operation:
            try:
                frame = await self._preflight("zasyncxstate", "ZA state_hex=", 0x43, 12)
                return parse_x_state(frame)
            except BaseException:
                self._uncertain = True
                raise

    async def rotation_command(
        self, action: str, steps: int | None = None
    ) -> RotationState:
        """Send a rotation command and read the firmware's open-loop counter."""
        expected = {
            "read": None,
            "enable": "ROT enabled",
            "disable": "ROT disabled",
            "zero": "ROT zeroed",
            "stop": "ROT stopping",
        }
        if action == "move":
            if type(steps) is not int or not -(2**31) <= steps < 2**31:
                raise ValueError("rotation target must be a signed 32-bit step count")
            command, reply = f"rot move {steps}", f"ROT target={steps}"
        elif action in expected and steps is None:
            command = "rot?" if action == "read" else f"rot {action}"
            reply = expected[action]
        else:
            raise ValueError("invalid rotation command")
        self._admit()
        async with self._operation:
            try:
                if reply is not None:
                    await self._exchange(command, reply)
                await self._write("rot?")
                line = await self._line()
                match = re.fullmatch(
                    r"ROT pos=(-?\d+) target=(-?\d+) moving=([01]) enabled=([01])", line
                )
                if match is None:
                    raise UnifiedMcuError(f"invalid rotation state: {line!r}")
                position, target, moving, enabled = map(int, match.groups())
                return RotationState(position, target, bool(moving), bool(enabled))
            except BaseException:
                self._uncertain = True
                raise

    async def identify(self, *, due: bool = False) -> None:
        """Require the selected firmware identity before any operation."""
        if self._operation.locked() or self._uncertain:
            raise UnifiedMcuError("unified MCU busy or uncertain")
        async with self._operation:
            try:
                await self._write("ID?" if due else "identity")
                identity = await self._line()
                if identity != (DUE_IDENTITY if due else IDENTITY):
                    raise UnifiedMcuError(f"unexpected firmware identity: {identity!r}")
                self._identified = True
            except BaseException:
                self._uncertain = True
                raise

    async def motion_profile(
        self, speed_tenths_rpm: int | None = None, acceleration_rpm_s: int | None = None
    ) -> tuple[int, int]:
        """Read or set the volatile speed and symmetric acceleration profile."""
        values = (speed_tenths_rpm, acceleration_rpm_s)
        if values == (None, None):
            command = "zprofile?"
        elif (
            type(speed_tenths_rpm) is int
            and type(acceleration_rpm_s) is int
            and 1 <= speed_tenths_rpm <= 30000
            and 1 <= acceleration_rpm_s <= 65535
        ):
            command = f"zprofile {speed_tenths_rpm} {acceleration_rpm_s}"
        else:
            raise ValueError("valid speed and acceleration required")
        self._admit()
        async with self._operation:
            try:
                await self._write(command)
                line = await self._line()
                match = re.fullmatch(
                    r"ZPROFILE speed_tenths_rpm=(\d+) accel_rpm_s=(\d+)", line
                )
                if not match:
                    raise UnifiedMcuError(f"invalid profile response: {line!r}")
                result = (int(match.group(1)), int(match.group(2)))
                if not (1 <= result[0] <= 30000 and 1 <= result[1] <= 65535):
                    raise UnifiedMcuError("invalid profile values")
                if command != "zprofile?" and result != values:
                    raise UnifiedMcuError("profile readback mismatch")
                return result
            except BaseException:
                self._uncertain = True
                raise

    async def readiness_error_limit(self, hundredths: int | None = None) -> int:
        """Read or set the volatile per-plane error limit; zero disables it."""
        if hundredths is None:
            command = "zreadyerr?"
        elif type(hundredths) is int and 0 <= hundredths <= 65535:
            command = f"zreadyerr {hundredths}"
        else:
            raise ValueError("error limit must be 0..65535 hundredths of a degree")
        self._admit()
        async with self._operation:
            try:
                await self._write(command)
                line = await self._line()
                match = re.fullmatch(r"ZREADYERR hundredths=(\d+)", line)
                if not match:
                    raise UnifiedMcuError(f"invalid readiness response: {line!r}")
                result = int(match.group(1))
                if result > 65535 or (hundredths is not None and result != hundredths):
                    raise UnifiedMcuError("readiness error limit mismatch")
                return result
            except BaseException:
                self._uncertain = True
                raise

    async def illumination(self, command: str) -> None:
        """Send one supported LED command; pattern commands also pulse GPIO13."""
        if command not in {
            "00",
            "01",
            "11",
            "12",
            "13",
            "14",
            "15",
            "51",
            "52",
            "53",
            "61",
            "62",
            "63",
            "64",
        }:
            raise ValueError("unsupported illumination command")
        self._admit()
        async with self._operation:
            try:
                await self._exchange(command, str(int(command)))
                if command == "00":
                    self._illumination_held = False
            except BaseException:
                self._uncertain = True
                raise

    async def set_auto_pulses(self, enabled: bool) -> None:
        """Set c0/c1; acknowledgement is not an electrical measurement."""
        if type(enabled) is not bool:
            raise ValueError("automatic trigger setting must be boolean")
        self._admit()
        async with self._operation:
            try:
                await self._exchange(
                    "c1" if enabled else "c0",
                    "Camera trigger enabled."
                    if enabled
                    else "Camera trigger disabled.",
                )
            except BaseException:
                self._uncertain = True
                raise

    async def pulse_output(self, device: int) -> None:
        """Request d1/d2/d3 once; a reply does not prove an electrical edge."""
        if type(device) is not int or device not in (1, 2, 3):
            raise ValueError("trigger output must be 1, 2 or 3")
        self._admit()
        async with self._operation:
            try:
                await self._exchange(
                    f"d{device}",
                    ("Triggered Cam", "Triggered Dev2", "Triggered Dev3")[device - 1],
                )
            except BaseException:
                self._uncertain = True
                raise

    async def abort(self, run_id: int) -> None:
        """Send one matching abort request; its outcome is read by the stack task."""
        if run_id != self._active_id or self._abort_sent:
            raise UnifiedMcuError("no matching abortable run")
        self._abort_sent = True
        try:
            await self._write(f"ABORT {run_id}")
        except BaseException:
            self._uncertain = True
            raise

    async def software_shots(
        self,
        definition: AngularStack,
        shots_per_plane: int,
        patterns: Sequence[int],
        snap: Callable[[int, int, int, int], Awaitable[None]],
        *,
        camera_exclusive: bool,
        illumination_ready: bool,
        snap_timeout_s: float = 20.0,
        bind_stage: bool = True,
        hold_illumination: bool = False,
    ) -> None:
        """Run one configured MCU shot sequence with one host snap per ready event.

        The caller owns the camera and supplies a snap that returns only after the
        frame has been received. Readiness is controller-reported, not exposure
        evidence; any uncertain outcome inhibits further owner operations.
        """
        if camera_exclusive is not True or illumination_ready is not True:
            raise ValueError("exclusive camera and illumination readiness required")
        if type(bind_stage) is not bool:
            raise ValueError("stage binding must be boolean")
        if type(hold_illumination) is not bool:
            raise ValueError("illumination hold must be boolean")
        if type(shots_per_plane) is not int or not 1 <= shots_per_plane <= 64:
            raise ValueError("shots per plane must be 1..64")
        if not 0 < snap_timeout_s < 30:
            raise ValueError("snap timeout must be positive and below firmware timeout")
        patterns = tuple(patterns)
        if len(patterns) != definition.count * shots_per_plane or any(
            type(value) is not int or not 1 <= value <= 5 for value in patterns
        ):
            raise ValueError("one valid pattern per shot is required")
        config_line = (
            f"SHOTCFG {definition.run_id} {definition.first} {definition.step} "
            f"{definition.count} {shots_per_plane} "
            f"{definition.lower} {definition.upper}"
        )
        if len(config_line) > 48:
            raise ValueError("shot configuration exceeds firmware line length")
        self._admit()
        async with self._operation:
            armed = False
            config_sent = False
            try:
                await self._exchange(
                    f"SHOTHOLD {int(hold_illumination)}",
                    f"SHOT hold={int(hold_illumination)}",
                )
                await self._exchange(
                    f"SHOTBIND {int(bind_stage)}", f"SHOT binding={int(bind_stage)}"
                )
                config_sent = True  # The response can be lost after configuration.
                await self._exchange(config_line, "SHOT configured")
                if bind_stage:
                    config = await self._preflight(
                        "zasyncxconfig", "ZA config_hex=", 0x42, 24
                    )
                    if (config[24], config[25], config[26]) != (0, 1, 0):
                        raise UnifiedMcuError("unsupported controller configuration")
                    await self._wait_stationary_preflight(definition)
                armed = True  # The command may be accepted even if its reply is lost.
                await self._exchange(
                    f"SHOTARM {definition.run_id}", "SHOT armed_host_asserted"
                )
                for index, pattern in enumerate(patterns):
                    plane = index // shots_per_plane
                    target = definition.first + definition.step * plane
                    await self._write(f"SHOT {definition.run_id} {index} {pattern}")
                    event = await self._line(timeout_s=9.0)
                    if (
                        event
                        != f"SHOT READY {definition.run_id} {index} {target} {pattern}"
                    ):
                        raise UnifiedMcuError(f"unexpected shot readiness: {event!r}")
                    await asyncio.wait_for(
                        snap(index, plane, pattern, target), snap_timeout_s
                    )
                await self._exchange(
                    f"SHOTFIN {definition.run_id} {len(patterns) - 1}", "SHOT complete"
                )
                self._illumination_held = hold_illumination
            except BaseException:
                self._uncertain = True
                if armed:
                    try:
                        await asyncio.shield(
                            asyncio.wait_for(
                                self._write(f"SHOTABORT {definition.run_id}"), 1.0
                            )
                        )
                    except BaseException:
                        logger.exception(
                            "Software-shot abort request was not confirmed"
                        )
                elif config_sent or self._illumination_held:
                    try:
                        await asyncio.shield(asyncio.wait_for(self._write("00"), 1.0))
                    except BaseException:
                        logger.exception(
                            "Software-shot blank request was not confirmed"
                        )
                raise

    async def automatic_shots(
        self,
        definition: AngularStack,
        shots_per_plane: int,
        pattern_mask: int,
        interval_ms: int,
        *,
        camera_armed: bool,
        illumination_ready: bool,
        bind_stage: bool = True,
        before_start: Callable[[], Awaitable[None]] | None = None,
    ) -> AutomaticShotSummary:
        """Run a shared-config hardware-trigger sequence with a final count only.

        Camera arming is asserted by the caller; this cannot prove an electrical
        pulse, exposure, received frame, physical position or safe stop.
        """
        if camera_armed is not True or illumination_ready is not True:
            raise ValueError("armed camera and illumination readiness required")
        if type(bind_stage) is not bool:
            raise ValueError("stage binding must be boolean")
        if type(shots_per_plane) is not int or not 1 <= shots_per_plane <= 64:
            raise ValueError("shots per plane must be 1..64")
        if type(pattern_mask) is not int or not 1 <= pattern_mask <= 31:
            raise ValueError("pattern mask must select array patterns 1..5")
        if type(interval_ms) is not int or not 1 <= interval_ms < 30000:
            raise ValueError("interval must be 1..29999 milliseconds")
        config_line = (
            f"SHOTCFG {definition.run_id} {definition.first} {definition.step} "
            f"{definition.count} {shots_per_plane} "
            f"{definition.lower} {definition.upper}"
        )
        if len(config_line) > 48:
            raise ValueError("shot configuration exceeds firmware line length")
        self._admit()
        async with self._operation:
            armed = False
            config_sent = False
            try:
                await self._exchange(
                    f"SHOTBIND {int(bind_stage)}", f"SHOT binding={int(bind_stage)}"
                )
                config_sent = True
                await self._exchange(config_line, "SHOT configured")
                if bind_stage:
                    config = await self._preflight(
                        "zasyncxconfig", "ZA config_hex=", 0x42, 24
                    )
                    if (config[24], config[25], config[26]) != (0, 1, 0):
                        raise UnifiedMcuError("unsupported controller configuration")
                    await self._wait_stationary_preflight(definition)
                armed = True
                await self._exchange(
                    f"SHOTARM {definition.run_id}", "SHOT armed_host_asserted"
                )
                if before_start is not None:
                    await before_start()
                await self._exchange(
                    f"SHOTAUTO {definition.run_id} {pattern_mask} "
                    f"{definition.trigger_mask} {interval_ms}",
                    "SHOT auto_started_host_asserted",
                )
                total = definition.count * shots_per_plane
                maximum_s = 2.0 + definition.count * 8.0 + total * interval_ms / 1000
                event = await self._line(timeout_s=maximum_s)
                if event != f"SHOT complete attempts={total}":
                    raise UnifiedMcuError(f"unexpected shot completion: {event!r}")
                return AutomaticShotSummary(attempted_pulses=total)
            except BaseException:
                self._uncertain = True
                try:
                    if armed:
                        await asyncio.shield(
                            asyncio.wait_for(
                                self._write(f"SHOTABORT {definition.run_id}"), 1.0
                            )
                        )
                    elif config_sent:
                        await asyncio.shield(asyncio.wait_for(self._write("00"), 1.0))
                except BaseException:
                    logger.exception("Automatic-shot cleanup request was not confirmed")
                raise

    async def stack(self, definition: AngularStack) -> StackAttempt:
        """Submit one stack and collect ordered controller-reported pulse attempts.

        ARM asserts camera and illumination readiness; this method does not
        configure a camera or verify electrical pulses. Never replay a start
        after a timeout or disconnected link.
        """
        self._admit()
        self._progress = None
        async with self._operation:
            self._progress = StackProgress(definition.run_id, ())
            sent_start = False
            self._abort_sent = False
            try:
                await self._exchange(definition.commit_line(), "ZSTACK committed")
                if definition.trigger_mask != 1:
                    await self._exchange(
                        f"TRIGMASK {definition.run_id} {definition.trigger_mask}",
                        f"ZSTACK trigger_mask={definition.trigger_mask}",
                    )
                config = await self._preflight(
                    "zasyncxconfig", "ZA config_hex=", 0x42, 24
                )
                if (config[24], config[25], config[26]) != (0, 1, 0):
                    raise UnifiedMcuError("unsupported controller configuration")
                await self._wait_stationary_preflight(definition)
                await self._exchange(
                    f"ARM {definition.run_id}", "ZSTACK armed_host_asserted"
                )
                self._active_id = definition.run_id
                sent_start = True
                await self._exchange(f"START {definition.run_id}", "ZSTACK started")
                targets: list[int] = []
                while True:
                    event = await self._line(timeout_s=9.0)
                    if event == "ZSTACK stopping" and self._abort_sent:
                        continue
                    if event.startswith("ZSTACK stop_unverified"):
                        raise UnifiedMcuError("stack stop outcome unverified")
                    match = re.fullmatch(r"ZSTACK EDGE (\d+) (-?\d+)", event)
                    if match:
                        plane, target = map(int, match.groups())
                        if (
                            plane != len(targets)
                            or plane >= definition.count
                            or target != definition.first + definition.step * plane
                        ):
                            raise UnifiedMcuError("stack event mismatch")
                        targets.append(target)
                        self._progress = StackProgress(
                            definition.run_id, tuple(targets)
                        )
                        continue
                    if (
                        event == f"ZSTACK complete edges={definition.count}"
                        and len(targets) == definition.count
                    ):
                        if self._abort_sent:
                            raise UnifiedMcuError("completion raced abort; reconcile")
                        self._progress = StackProgress(
                            definition.run_id, tuple(targets), complete=True
                        )
                        return StackAttempt(definition.run_id, tuple(targets))
                    raise UnifiedMcuError(f"unexpected stack event: {event!r}")
            except BaseException:
                self._uncertain = True
                if sent_start and not self._abort_sent:
                    self._abort_sent = True
                    try:
                        await asyncio.shield(
                            asyncio.wait_for(
                                self._write(f"ABORT {definition.run_id}"), 1.0
                            )
                        )
                    except BaseException:
                        logger.exception("Abort request was not confirmed")
                raise
            finally:
                self._active_id = None
