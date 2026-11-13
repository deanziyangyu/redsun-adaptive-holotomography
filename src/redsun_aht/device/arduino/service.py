"""One serialized owner of the MMStudio-compatible 91-LED v1 Arduino port.

This service is the only object in the process that may write the sketch.  Every
write is taken under one :class:`asyncio.Lock`, so two plans cannot interleave
commands even when their tasks are scheduled together.  A pattern write
(``11``..``15``) selects illumination without a camera TTL pulse. Callers
must not compensate for a timeout by writing the same command again.

The dialect speaks the 91-LED v1 layout (``panel91``, revision 1): ``11``..``14``
are the four semicircle patterns and ``15`` is the centre LED.  ``00`` blanks and
``01`` opens the shutter state; ``10`` (array off) is
deliberately unsupported.  LED indices live in firmware, not here.

The transport follows the validated Micro-Manager settings: 9600 baud, eight
data bits, no parity, one stop bit, neither software nor hardware flow control,
and DTR/RTS deasserted.  The MM adapter ignored responses, so an echo is
diagnostic only.  No response is required for success: a command whose bytes
were accepted and flushed is ``sent`` even when the sketch stays silent.

A timeout, a short write or an interrupted transfer is ``uncertain``.  The
service records that outcome and returns it; it does not retry, because the
command may already have changed the illumination state.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Final, Protocol, cast

from oxiserial import PortNotOpenError, SerialException, SerialTimeoutException
from oxiserial.aio import serial_for_url

from .commands import (
    BLANK,
    PANEL_PROFILE,
    PATTERN_COMMANDS,
    SUPPORTED,
    SUPPORTED_ORDER,
    CommandOutcome,
    CommandStatus,
    EchoDrain,
    PanelProfile,
    blank,
    drain_echo,
    encode,
)

logger = logging.getLogger(__name__)

BAUDRATE: Final = 9600
"""The line speed the validated MMStudio configuration uses."""

READ_TIMEOUT_S: Final = 0.25
"""Port read timeout, also the budget for draining a diagnostic echo."""

WRITE_TIMEOUT_S: Final = 2.0
"""Upper bound on accepting and flushing one two-digit command."""

IDENTITY_COMMAND: Final = b"90\r\n"
IDENTITY_REPLY: Final = "AHT-MM-ASCII-91/1"


class AsyncSerialPort(Protocol):
    """Narrow view of the ``oxiserial.aio.Serial`` methods this service needs."""

    @property
    def is_open(self) -> bool:
        """Whether the underlying device is open."""
        ...

    def write(self, data: bytes) -> Awaitable[int]:
        """Accept bytes and report how many were accepted."""
        ...

    def flush(self) -> Awaitable[None]:
        """Wait until accepted bytes have been transmitted."""
        ...

    def read_until(self, expected: bytes = b"\n") -> Awaitable[bytes]:
        """Read one line, or return empty bytes when the port read times out."""
        ...

    def close(self) -> None:
        """Release the underlying device."""
        ...


@dataclass(frozen=True, slots=True)
class ArduinoLedProfile:
    """Explicit serial endpoint for one MMStudio-compatible Arduino.

    There is no default port.  COM19 is historical and machine-specific, so a
    profile must name the endpoint this process is allowed to own.
    """

    port: str
    baudrate: int = BAUDRATE
    read_timeout_s: float = READ_TIMEOUT_S
    write_timeout_s: float = WRITE_TIMEOUT_S

    def __post_init__(self) -> None:
        """Reject empty ports, placeholders and non-positive settings."""
        if not self.port.strip() or "<" in self.port or ">" in self.port:
            raise ValueError("An explicit serial port name is required")
        if self.baudrate <= 0:
            raise ValueError("Baudrate must be positive")
        if self.read_timeout_s <= 0 or self.write_timeout_s <= 0:
            raise ValueError("Serial read and write timeouts must be positive")


PortOpener = Callable[[ArduinoLedProfile], AsyncSerialPort]


def open_serial_port(profile: ArduinoLedProfile) -> AsyncSerialPort:
    """Open *profile* as 8N1 with no flow control and DTR/RTS deasserted."""
    port = serial_for_url(
        profile.port,
        baudrate=profile.baudrate,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=profile.read_timeout_s,
        write_timeout=profile.write_timeout_s,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    port.dtr = False
    port.rts = False
    return cast("AsyncSerialPort", port)


class ArduinoLedService:
    """Serialized command transport for one Arduino LED controller."""

    def __init__(
        self,
        port: AsyncSerialPort,
        profile: ArduinoLedProfile,
        *,
        opener: PortOpener = open_serial_port,
    ) -> None:
        self._port = port
        self._profile = profile
        self._opener = opener
        self._lock = asyncio.Lock()
        self._closed = False
        self._last_outcome: CommandOutcome | None = None
        self._last_detail = ""
        self._last_pattern: int | None = None

    @classmethod
    def open(
        cls,
        profile: ArduinoLedProfile,
        *,
        opener: PortOpener = open_serial_port,
    ) -> ArduinoLedService:
        """Open *profile* and return the service that owns the resulting port."""
        return cls(opener(profile), profile, opener=opener)

    @property
    def profile(self) -> ArduinoLedProfile:
        """The endpoint currently owned by this service."""
        return self._profile

    @property
    def port_name(self) -> str:
        """Name of the serial port currently owned by this service."""
        return self._profile.port

    @property
    def baudrate(self) -> int:
        """Line speed currently owned by this service."""
        return self._profile.baudrate

    @property
    def last_outcome(self) -> CommandOutcome | None:
        """The most recent attempt, or ``None`` before the first command."""
        return self._last_outcome

    @property
    def last_command(self) -> str:
        """Two-digit command of the most recent attempt, or an empty string."""
        return "" if self._last_outcome is None else self._last_outcome.command

    @property
    def last_detail(self) -> str:
        """Diagnostics for the most recent attempt."""
        return self._last_detail

    @property
    def last_pattern(self) -> int | None:
        """Last DPCT group known to have been delivered, or ``None``."""
        return self._last_pattern

    @property
    def is_closed(self) -> bool:
        """Whether :meth:`close` has released the port."""
        return self._closed

    @property
    def panel_profile(self) -> PanelProfile:
        """The panel this dialect believes it is driving; indices live in firmware."""
        return PANEL_PROFILE

    @property
    def led_count(self) -> int:
        """Number of LEDs in the installed panel (91 for panel91 v1)."""
        return PANEL_PROFILE.led_count

    def capabilities(self) -> tuple[str, ...]:
        """Return exactly the commands this panel dialect advertises, in order."""
        return SUPPORTED_ORDER

    def supports(self, command: str) -> bool:
        """Return whether *command* belongs to this service's advertised set."""
        return command in SUPPORTED

    async def send(self, command: str) -> CommandOutcome:
        """Send one supported command exactly once and return its typed outcome.

        Unsupported commands, a closed service and an already-closed port are
        ``failed`` without touching the transport.  A timeout or short write is
        ``uncertain``; this method never resends.
        """
        if command not in SUPPORTED:
            outcome = CommandOutcome(
                command,
                CommandStatus.FAILED,
                f"unsupported Arduino command {command!r}; "
                f"this service advertises {', '.join(SUPPORTED_ORDER)}",
            )
            self._record(outcome)
            return outcome

        async with self._lock:
            if self._closed:
                outcome = CommandOutcome(
                    command,
                    CommandStatus.FAILED,
                    "service is closed; no bytes were written",
                )
                self._record(outcome)
                return outcome
            return await self._send_locked(command)

    async def all_off(self) -> CommandOutcome:
        """Blank every output without emitting a camera trigger."""
        return await self.send(BLANK)

    async def verify_firmware(self) -> None:
        """Require the panel91 MM sketch's read-only identity response."""
        async with self._lock:
            if self._closed or not self._port.is_open:
                raise RuntimeError(
                    "Arduino port is closed; firmware cannot be verified"
                )
            await self._verify_port(self._port)

    async def _verify_port(self, port: AsyncSerialPort) -> None:
        status, detail = await self._write_payload(port, IDENTITY_COMMAND)
        if status is not CommandStatus.SENT:
            raise RuntimeError(f"Arduino firmware identity request failed: {detail}")
        answer = (
            (await port.read_until(b"\n")).decode("ascii", errors="replace").strip()
        )
        if answer != IDENTITY_REPLY:
            raise RuntimeError(
                f"Arduino firmware identity mismatch: expected {IDENTITY_REPLY!r}, "
                f"received {answer!r}"
            )

    async def open_shutter(self) -> CommandOutcome:
        """Clear the shutter-closed flag without emitting a camera trigger."""
        return await self.send("01")

    async def reconfigure(
        self, *, port: str | None = None, baudrate: int | None = None
    ) -> ArduinoLedProfile:
        """Open a replacement endpoint, blank it, and release the old one.

        The old port is blanked best-effort before it is closed.  If the new
        endpoint cannot be opened or blanked, the old endpoint stays owned and
        the error is raised; no accepted command is replayed.
        """
        async with self._lock:
            if self._closed:
                raise RuntimeError("service is closed; cannot reconfigure its port")
            profile = replace(
                self._profile,
                port=self._profile.port if port is None else port,
                baudrate=self._profile.baudrate if baudrate is None else baudrate,
            )
            if profile == self._profile:
                return profile

            new_port = self._opener(profile)
            try:
                status, detail = await self._write_payload(new_port, blank())
                if status is not CommandStatus.SENT:
                    raise RuntimeError(
                        f"could not blank the new Arduino port {profile.port!r}: "
                        f"{detail}"
                    )
                blank_reply = await self._drain(new_port, BLANK)
                if not blank_reply.matched:
                    raise RuntimeError("new Arduino port did not echo startup blank")
                await self._verify_port(new_port)
            except BaseException:
                new_port.close()
                raise

            if self._port.is_open:
                try:
                    await self._write_payload(self._port, blank())
                    await self._drain(self._port, BLANK)
                except OSError:
                    logger.exception(
                        "Could not blank the previous Arduino port before releasing it"
                    )
            old_port = self._port
            self._port = new_port
            self._profile = profile
            old_port.close()
            self._record(
                CommandOutcome(
                    BLANK,
                    CommandStatus.SENT,
                    f"port changed to {profile.port}; outputs were blanked",
                )
            )
            return profile

    async def close(self) -> None:
        """Blank the outputs, then always release the port.

        ``00`` is attempted under the same lock as every other command, so it
        cannot race a plan.  A failed or timed-out blank is recorded and never
        retried; releasing the port still happens.  Calling ``close`` again is
        a no-op.
        """
        async with self._lock:
            if self._closed:
                return
            try:
                if self._port.is_open:
                    try:
                        await self._send_locked(BLANK)
                    except Exception:
                        logger.exception(
                            "Failed to blank Arduino outputs during close; "
                            "releasing the port anyway"
                        )
            finally:
                self._closed = True
                if self._port.is_open:
                    self._port.close()

    async def _send_locked(self, command: str) -> CommandOutcome:
        """Write and flush once, then drain a diagnostic echo if one arrives."""
        payload = encode(command)
        status, detail = await self._write_payload(self._port, payload)
        echo: str | None = None

        if status is CommandStatus.SENT:
            try:
                drained = await drain_echo(self._read_line, command)
            except OSError as error:
                detail = f"command sent, but draining its echo failed: {error}"
            else:
                if drained.matched:
                    status = CommandStatus.ECHOED
                    echo = drained.line
                elif drained.unexpected:
                    detail = (
                        "command sent; tolerated unexpected serial "
                        f"reply {drained.unexpected!r}"
                    )

        outcome = CommandOutcome(command, status, detail, echo)
        self._record(outcome)
        return outcome

    async def _write_payload(
        self, port: AsyncSerialPort, payload: bytes
    ) -> tuple[CommandStatus, str]:
        """Write and flush *payload* once; classify a short or timed-out write."""
        if not port.is_open:
            return (
                CommandStatus.FAILED,
                "serial port is not open; no bytes were written",
            )
        try:
            written = await port.write(payload)
        except TimeoutError as error:
            return (
                CommandStatus.UNCERTAIN,
                f"serial write timed out ({error}); delivery unknown; not retried",
            )
        except SerialTimeoutException as error:
            return (
                CommandStatus.UNCERTAIN,
                f"serial write timed out ({error}); delivery unknown; not retried",
            )
        except PortNotOpenError as error:
            return (
                CommandStatus.FAILED,
                f"serial port was closed before the write ({error}); "
                "no bytes were written",
            )
        except SerialException as error:
            return (
                CommandStatus.UNCERTAIN,
                f"serial write failed ({error}); delivery unknown; not retried",
            )
        except OSError as error:
            return (
                CommandStatus.UNCERTAIN,
                f"serial write failed ({error}); delivery unknown; not retried",
            )

        if written != len(payload):
            return (
                CommandStatus.UNCERTAIN,
                f"short serial write: {written}/{len(payload)} bytes accepted; "
                "delivery unknown; not retried",
            )

        try:
            await port.flush()
        except OSError as error:
            return (
                CommandStatus.UNCERTAIN,
                f"serial flush failed after {written} bytes were accepted "
                f"({error}); delivery unknown; not retried",
            )
        return CommandStatus.SENT, ""

    async def _drain(self, port: AsyncSerialPort, command: str) -> EchoDrain:
        """Drain one echo from *port* without writing anything."""

        async def read_line() -> bytes:
            return await port.read_until(b"\n")

        return await drain_echo(read_line, command)

    async def _read_line(self) -> bytes:
        """Read one line from the owned port for the echo drain."""
        return await self._port.read_until(b"\n")

    def _record(self, outcome: CommandOutcome) -> None:
        """Remember the last attempt without turning a failure into a latch."""
        self._last_outcome = outcome
        self._last_detail = outcome.detail
        if outcome.delivered and outcome.command in PATTERN_COMMANDS:
            self._last_pattern = int(outcome.command[1])


__all__ = [
    "BAUDRATE",
    "READ_TIMEOUT_S",
    "WRITE_TIMEOUT_S",
    "ArduinoLedProfile",
    "ArduinoLedService",
    "AsyncSerialPort",
    "PortOpener",
    "open_serial_port",
]
