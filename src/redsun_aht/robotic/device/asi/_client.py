"""Typed ASI legacy-ASCII client for an owned serial session.

The ASI MS2000/LX4000 command set used here is the subset exercised by the
operator-supplied donor script: paired ``2H`` commands for the XY axes, a
backspace handshake that enables the end-of-text terminator, and 0.1 um
(tenth-micron) position units. The client never retries and never changes a
controller setting on construction.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redsun_aht.robotic.device.ludl._transport import OxiSerialSession

#: The controller reports and accepts XY positions in tenths of a micrometre.
TENTHS_PER_UM = 10.0

#: ASI terminates every response with EOT after the CRLF payload.
EOT = b"\x03"

_SPEED_VALUE = re.compile(r"=\s*([-+]?\d*\.?\d+)")
_RAMP_VALUE = re.compile(r"=\s*([-+]?\d*\.?\d+)")
_BACKLASH_VALUE = re.compile(r"=\s*([-+]?\d*\.?\d+)")


class AsiXYClient:
    """Paired XY reads and moves over one sole-owned ASI serial session."""

    def __init__(self, session: OxiSerialSession) -> None:
        self._session = session

    def close(self) -> None:
        """Release the serial session during the owning service shutdown."""
        self._session.close()

    async def connect(self) -> None:
        """Run the ASI backspace handshake that enables EOT-framed replies.

        The controller answers the handshake with BEL+EOT; a BEL is expected
        here and is not a command rejection.
        """
        await self._session.exchange(b"\x08", terminator=EOT)

    async def read_xy_tenths(self) -> tuple[float, float]:
        """Read both raw tenth-micron coordinates in X, Y order."""
        return (
            await self._read_axis_tenths("X"),
            await self._read_axis_tenths("Y"),
        )

    async def read_axis_tenths(self, axis: str) -> float:
        """Read one raw tenth-micron coordinate."""
        return await self._read_axis_tenths(_axis(axis))

    async def move_xy_tenths(self, x: int, y: int) -> None:
        """Issue X then Y absolute moves and require each acknowledgement.

        The donor script uses per-axis ``2HM`` commands, so this client keeps
        that verified shape. A failure after the first axis has been accepted is
        a partial command; the caller must treat the outcome as unknown.
        """
        _require_int(x, "x tenths")
        _require_int(y, "y tenths")
        try:
            await self._move_axis_tenths("X", x)
            await self._move_axis_tenths("Y", y)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "ASI paired move was not fully acknowledged; axis outcome "
                f"unknown: {exc}"
            ) from exc

    async def move_axis_tenths(self, axis: str, tenths: int) -> None:
        """Command one axis only.

        A paired move that re-states an unchanged axis is not free: this controller's backlash compensation treats
        a sub-play command (including zero) as a reversal and drives the axis about two backlashes before returning.
        Measured 2026-10-26 on the stage alone: a 0.16 um y command produced a 31.0 um excursion with 16 um of
        compensation configured, 0.00 um with it disabled, and 0.00 um when the y axis was not commanded at all.
        """
        await self._move_axis_tenths(axis, tenths)

    async def move_relative_xy_tenths(self, x: int, y: int) -> None:
        """Issue X then Y relative moves and require each acknowledgement."""
        _require_int(x, "x relative tenths")
        _require_int(y, "y relative tenths")
        try:
            await self._query(f"2HR X={x:d}")
            await self._query(f"2HR Y={y:d}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "ASI relative move was not fully acknowledged; axis outcome "
                f"unknown: {exc}"
            ) from exc

    async def is_busy_xy(self) -> bool:
        """Read the controller's paired XY busy flag."""
        response = await self._query("2H/")
        text = response.strip().upper()
        if text.startswith("B"):
            return True
        if text.startswith("N"):
            return False
        raise ValueError(f"unexpected ASI XY status reply: {response!r}")

    async def halt_xy(self) -> None:
        """Issue the controller's paired XY halt."""
        reply = await self._query("2HHALT")
        if reply != ":A":
            raise RuntimeError(f"unexpected ASI XY halt reply: {reply!r}")

    async def read_speed_mm_s(self, axis: str) -> float:
        """Read the configured run speed in mm/s."""
        axis = _axis(axis)
        response = await self._query(f"2HSPEED {axis}?")
        match = _SPEED_VALUE.search(response)
        if match is None:
            raise ValueError(f"expected ASI {axis} speed reply, got {response!r}")
        return float(match.group(1))

    async def set_speed_mm_s(self, axis: str, speed_mm_s: float) -> None:
        """Set one axis run speed in mm/s."""
        axis = _axis(axis)
        if not _finite_positive(speed_mm_s):
            raise ValueError("speed must be finite and positive")
        await self._query(f"2HSPEED {axis}= {speed_mm_s:1.4f}")

    async def read_ramp_ms(self, axis: str) -> float:
        """Read the configured acceleration/ramp time in milliseconds."""
        axis = _axis(axis)
        response = await self._query(f"2HAC {axis}?")
        match = _RAMP_VALUE.search(response)
        if match is None:
            raise ValueError(f"expected ASI {axis} ramp reply, got {response!r}")
        return float(match.group(1))

    async def set_ramp_ms(self, axis: str, ramp_ms: float) -> None:
        """Set one axis acceleration/ramp time in milliseconds."""
        axis = _axis(axis)
        if not _finite_nonnegative(ramp_ms):
            raise ValueError("ramp time must be finite and non-negative")
        await self._query(f"2HAC {axis}= {ramp_ms:1.4f}")

    async def read_backlash_mm(self, axis: str) -> float:
        """Read the configured anti-backlash distance in millimetres."""
        axis = "Z" if axis in ("z", "Z") else _axis(axis)
        card = 1 if axis == "Z" else 2
        response = await self._query(f"{card}HB {axis}?")
        match = _BACKLASH_VALUE.search(response)
        if match is None:
            raise ValueError(f"expected ASI {axis} backlash reply, got {response!r}")
        return float(match.group(1))

    async def set_backlash_xy_mm(self, x_mm: float, y_mm: float) -> None:
        """Set both anti-backlash distances in millimetres.

        The paired write is the command shape verified on the ix71 controller;
        per-axis reads remain the source of truth for a captured profile.
        """
        for value, name in ((x_mm, "X backlash"), (y_mm, "Y backlash")):
            if not _finite_nonnegative(value) or value > 1.0:
                raise ValueError(
                    f"{name} must be finite, non-negative and at most 1 mm"
                )
        await self._query(f"2HB X= {x_mm:1.4f} Y= {y_mm:1.4f}")

    async def _read_axis_tenths(self, axis: str) -> float:
        """Read one axis position, tolerating a single out-of-step reply.

        Measured on the ix71 controller, the serial stream can be one reply behind after a
        connect, so a ``2HW`` read can return the acknowledgement of an earlier command
        instead of a position. Two shapes were observed: ``:X=0.016000 A`` (a backlash
        value, not ``:A``-prefixed) and ``:A Y=0.200000`` (a speed echo, ``:A``-prefixed
        but not numeric). Both are skipped and the query repeated **once**; a second
        mismatch stays fatal, because two consecutive non-position replies mean the
        controller is not answering position queries at all and guessing a coordinate from
        that would be unsafe.
        """
        stale: str | None = None
        for attempt in range(2):
            response = await self._query(f"2HW {axis}")
            if response.startswith(":A"):
                try:
                    return float(response[2:].strip())
                except ValueError:
                    stale = response
            else:
                stale = response
            if attempt == 0:
                continue
        raise ValueError(
            f"unexpected ASI {axis} position reply: {stale!r}"
            " (one re-query attempted after an out-of-step reply)"
        )

    async def _move_axis_tenths(self, axis: str, tenths: int) -> None:
        await self._query(f"2HM {axis}= {tenths:d}")

    async def _query(self, command: str) -> str:
        response = await self._session.exchange(
            command.encode("ascii") + b"\r", terminator=EOT
        )
        text = response.replace(EOT, b"").decode("ascii", errors="replace").strip()
        if not text:
            raise TimeoutError(f"empty response to ASI command {command!r}")
        self._require_no_bell(text, command)
        if text.startswith(":AERR"):
            raise RuntimeError(f"ASI controller error for {command!r}: {text}")
        return text

    @staticmethod
    def _require_no_bell(response: bytes | str, command: str) -> None:
        text = (
            response.decode("ascii", errors="replace")
            if isinstance(response, bytes)
            else response
        )
        if "\x07" in text:
            raise ValueError(f"ASI controller rejected {command!r}")


def _axis(axis: str) -> str:
    if not isinstance(axis, str) or axis.upper() not in {"X", "Y"}:
        raise ValueError("axis must be 'X' or 'Y'")
    return axis.upper()


def _finite_positive(value: float) -> bool:
    import math

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _finite_nonnegative(value: float) -> bool:
    import math

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _require_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


__all__ = ["EOT", "TENTHS_PER_UM", "AsiXYClient"]
