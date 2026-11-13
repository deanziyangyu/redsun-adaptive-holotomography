"""MMStudio-compatible commands for the 91-LED v1 panel.

The Micro-Manager setup sends two ASCII digits followed by CRLF to the
``aht_mm_ascii_91`` sketch at 9600 baud, 8N1, with no flow control.
The sketch does not use an address prefix, so :data:`PREFIX` is empty; keeping it
explicit makes a future addressable dialect a visible change rather than an
implicit rewrite of every command.

This dialect drives the **91-LED v1 layout** (``panel91``, revision 1).  ``11``
through ``14`` select the four semicircle patterns defined by the firmware's
``group1``..``group4`` tables, and ``15`` selects the centre LED pattern.  The
LED indices and the exact pattern tables live in the firmware; the host only
names the profile it believes it is driving.

A pattern command selects illumination without a camera TTL pulse. The sketch
echoes accepted commands. The MM adapter ignores responses, so an echo
is diagnostic evidence about the serial link, never proof that a frame was
captured or that the illumination was optically correct.  A missing echo is not
a failure, and a timeout leaves delivery uncertain: callers must not resend.

``00`` blanks every output and ``01`` clears the shutter-closed flag. ``10``
blanks the array in the MM sketch but is omitted from this service surface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

PREFIX: Final = b""
"""Bytes every command carries before its two digits; empty in the validated setup."""

TERMINATOR: Final = b"\r\n"
"""Every command ends with CRLF, as the MMStudio configuration specifies."""

ENCODING: Final = "ascii"
"""The sketch accepts ASCII command characters only."""

BLANK: Final = "00"
"""Blank every output without emitting a camera trigger."""

OPEN_SHUTTER: Final = "01"
"""Clear the shutter-closed flag without emitting a camera trigger."""

ARRAY_OFF: Final = "10"
"""MM state-device off command, omitted from this service surface."""

PATTERN_BASE: Final = 10
"""First pattern command number; ``10 + n`` selects panel pattern ``n``."""

PATTERN_NUMBERS: Final = (1, 2, 3, 4, 5)
"""Panel pattern ids: four semicircles plus the centre."""

DPCT_PATTERN_NUMBERS: Final = (1, 2, 3, 4)
"""Pattern ids the four-frame DPCT plan uses; the centre is a separate choice."""

CENTER_PATTERN: Final = 5
"""Pattern id of the 91-LED v1 centre LED, encoded as ``15``."""

PATTERN_COMMANDS: Final = ("11", "12", "13", "14", "15")
"""Encoded illumination patterns: four semicircles then the centre."""

SUPPORTED: Final = frozenset({BLANK, OPEN_SHUTTER, *PATTERN_COMMANDS})
"""Exactly the commands this 91-LED v1 panel dialect advertises."""

SUPPORTED_ORDER: Final = (BLANK, OPEN_SHUTTER, *PATTERN_COMMANDS)
"""Supported commands in presentation order."""

ECHO_TIMEOUT_S: Final = 0.25
"""How long to wait for a diagnostic echo before treating it as absent."""

MAX_ECHO_LINES: Final = 4
"""Bounded number of unexpected lines consumed while draining an echo."""


@dataclass(frozen=True, slots=True)
class PanelProfile:
    """The illumination panel this command dialect believes it is driving.

    This is a host-side name only.  The LED indices and the semicircle/centre
    pattern tables live in the firmware (from ``led_triple_mode_ttl.ino``); the
    host records the profile it is speaking to so a run can be interpreted.
    """

    name: str
    revision: int
    led_count: int
    pattern_count: int
    centre_pattern: int
    dpct_patterns: tuple[int, ...]
    pattern_commands: tuple[str, ...]


PANEL_PROFILE: Final = PanelProfile(
    name="panel91",
    revision=1,
    led_count=91,
    pattern_count=5,
    centre_pattern=CENTER_PATTERN,
    dpct_patterns=DPCT_PATTERN_NUMBERS,
    pattern_commands=PATTERN_COMMANDS,
)
"""The installed 91-LED v1 panel, revision 1."""


class CommandStatus(StrEnum):
    """Transport outcome of one command.

    ``SENT`` means the bytes were accepted and flushed; ``ECHOED`` means the
    sketch also echoed the command.  ``UNCERTAIN`` means delivery or effect is
    unknown, usually a timeout or a short write, and must never be retried
    blindly.  ``FAILED`` means no command was delivered, for example because
    the request was unsupported or the port was already closed.
    """

    SENT = "sent"
    ECHOED = "echoed"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Typed result of one command attempt.

    Parameters
    ----------
    command:
        Two-digit command that was attempted, or the rejected text.
    status:
        Transport outcome; see :class:`CommandStatus`.
    detail:
        Human-readable diagnostics.  Empty when nothing needs explaining.
    echo:
        The decoded echo when the sketch returned one, otherwise ``None``.
    """

    command: str
    status: CommandStatus
    detail: str = ""
    echo: str | None = None

    @property
    def delivered(self) -> bool:
        """Whether the command bytes are known to have reached the sketch."""
        return self.status in (CommandStatus.SENT, CommandStatus.ECHOED)

    @property
    def uncertain(self) -> bool:
        """Whether delivery or physical effect is unknown."""
        return self.status is CommandStatus.UNCERTAIN

    @property
    def is_pattern(self) -> bool:
        """Whether this command selects an illumination pattern."""
        return self.command in PATTERN_COMMANDS


@dataclass(frozen=True, slots=True)
class EchoDrain:
    """What a bounded echo read found.

    ``matched`` is true only when the sketch echoed exactly the command sent.
    ``unexpected`` records other non-empty lines that were tolerated and
    discarded, so they can be reported without being mistaken for an ACK.
    """

    matched: bool
    line: str | None = None
    unexpected: tuple[str, ...] = ()


def pattern_command(number: int) -> str:
    """Return the two-digit command for panel pattern *number* (``1..5``).

    Patterns ``1..4`` are the four semicircles; pattern ``5`` is the centre.

    Raises
    ------
    ValueError
        If *number* is not one of the five patterns of the panel91 layout.
    """
    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError(f"pattern must be an integer, got {type(number).__name__}")
    if number not in PATTERN_NUMBERS:
        raise ValueError(
            f"pattern must be one of {PATTERN_NUMBERS} for the panel91 layout "
            "(four semicircles plus centre), "
            f"got {number}"
        )
    return f"{PATTERN_BASE + number:02d}"


def encode(command: str) -> bytes:
    """Encode a supported command as ASCII digits followed by CRLF.

    Raises
    ------
    ValueError
        If *command* is not one of :data:`SUPPORTED`.
    """
    if command not in SUPPORTED:
        raise ValueError(
            f"unsupported Arduino command {command!r}; "
            f"this service advertises {', '.join(SUPPORTED_ORDER)}"
        )
    return PREFIX + command.encode(ENCODING) + TERMINATOR


def blank() -> bytes:
    """Return the bytes that blank every output."""
    return encode(BLANK)


def open_shutter() -> bytes:
    """Return the bytes that clear the shutter-closed flag."""
    return encode(OPEN_SHUTTER)


def pattern(number: int) -> bytes:
    """Return the bytes for panel pattern *number*.

    Patterns ``1..4`` are the semicircles; ``5`` is the centre LED.
    """
    return encode(pattern_command(number))


def centre() -> bytes:
    """Return the ``15`` bytes that light the centre."""
    return encode(pattern_command(CENTER_PATTERN))


def decode(payload: bytes) -> str:
    """Decode a response line, tolerating non-ASCII bytes in diagnostics.

    The result is stripped of CRLF and surrounding whitespace.  Decoding never
    raises: an unreadable reply is reported as replacement characters instead
    of breaking an acquisition, because an echo is not required evidence.
    """
    return payload.decode(ENCODING, errors="replace").strip()


def is_echo(payload: bytes, command: str) -> bool:
    """Return whether *payload* is exactly the echo of *command*."""
    return decode(payload) == command


async def drain_echo(
    read_line: Callable[[], Awaitable[bytes]],
    command: str,
    *,
    timeout: float = ECHO_TIMEOUT_S,
) -> EchoDrain:
    """Read and discard at most one command echo within *timeout*.

    The helper tolerates three validated behaviours: a matching echo, no echo
    at all, and unrelated diagnostic lines.  It never writes, so it cannot
    trigger or resend anything.  A missing echo returns ``matched=False`` and is
    not an error; callers map that to :attr:`CommandStatus.SENT`.

    Parameters
    ----------
    read_line:
        Awaitable that returns one CRLF-terminated line, or empty bytes when the
        port read timed out.
    command:
        The command just written, used only to identify its echo.
    timeout:
        Overall budget for the drain, in seconds.
    """
    unexpected: list[str] = []
    deadline = asyncio.get_running_loop().time() + timeout
    while len(unexpected) < MAX_ECHO_LINES:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(read_line(), timeout=remaining)
        except TimeoutError:
            break
        if not raw:
            break
        line = decode(raw)
        if not line:
            continue
        if line == command:
            return EchoDrain(matched=True, line=line, unexpected=tuple(unexpected))
        unexpected.append(line)
    return EchoDrain(matched=False, unexpected=tuple(unexpected))


__all__ = [
    "ARRAY_OFF",
    "BLANK",
    "CENTER_PATTERN",
    "DPCT_PATTERN_NUMBERS",
    "ECHO_TIMEOUT_S",
    "ENCODING",
    "MAX_ECHO_LINES",
    "OPEN_SHUTTER",
    "PANEL_PROFILE",
    "PATTERN_BASE",
    "PATTERN_COMMANDS",
    "PATTERN_NUMBERS",
    "PREFIX",
    "SUPPORTED",
    "SUPPORTED_ORDER",
    "TERMINATOR",
    "CommandOutcome",
    "CommandStatus",
    "EchoDrain",
    "PanelProfile",
    "blank",
    "centre",
    "decode",
    "drain_echo",
    "encode",
    "is_echo",
    "open_shutter",
    "pattern",
    "pattern_command",
]
