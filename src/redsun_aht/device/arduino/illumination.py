"""Plan-facing illumination adapter over the PVA-served Arduino device.

:class:`ArduinoIllumination` turns one :class:`ArduinoLedDevice` into the
:class:`redsun_aht.contracts.Illumination` protocol.  It writes the PVA signals
and then reports what the service said happened; it never fabricates optical
evidence and never resends a command whose readback is ``uncertain``.

A pattern write (``11``..``15``) selects illumination in the installed
firmware without emitting a camera trigger. The adapter writes the illumination
pattern and returns a :class:`~redsun_aht.contracts.CommandReceipt` that names
the pattern the plan requested. The service reports no clock, so receipts leave
``sent_monotonic_ns`` unset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from redsun_aht.contracts import (
    CommandReceipt,
    CommandStatus,
    IlluminationAction,
    IlluminationCapability,
)

from .commands import (
    BLANK,
    CENTER_PATTERN,
    OPEN_SHUTTER,
    PANEL_PROFILE,
    PATTERN_NUMBERS,
    pattern_command,
)

if TYPE_CHECKING:
    from .device import ArduinoLedDevice


CAPABILITY = IlluminationCapability(
    pattern_ids=PATTERN_NUMBERS,
    center_pattern_id=CENTER_PATTERN,
    panel_profile=PANEL_PROFILE.name,
    panel_revision=PANEL_PROFILE.revision,
    led_count=PANEL_PROFILE.led_count,
    blank_supported=True,
    open_supported=True,
    echo_expected=False,
    dialect="mm-ascii",
)
"""Capability declaration for the installed 91-LED v1 panel."""


class ArduinoIllumination:
    """Adapter satisfying :class:`redsun_aht.contracts.Illumination`."""

    def __init__(self, device: ArduinoLedDevice) -> None:
        self._device = device

    @property
    def capability(self) -> IlluminationCapability:
        """Return the 91-LED v1 capability, derived from :data:`PANEL_PROFILE`."""
        return CAPABILITY

    async def blank(self) -> CommandReceipt:
        """Send ``00`` to blank every output without triggering a frame."""
        await self._device.all_off.set(True)
        return await self._receipt(IlluminationAction.BLANK, BLANK)

    async def open(self) -> CommandReceipt:
        """Send ``01`` to clear the shutter-closed state without a trigger."""
        await self._device.open_shutter.set(True)
        return await self._receipt(IlluminationAction.OPEN, OPEN_SHUTTER)

    async def pattern(self, pattern_id: int) -> CommandReceipt:
        """Send the pattern command for *pattern_id* without a camera trigger.

        Raises
        ------
        ValueError
            If *pattern_id* is not one of the five 91-LED v1 patterns.  The
            refusal happens before any signal is written.
        """
        command = pattern_command(pattern_id)
        await self._device.pattern.set(pattern_id)
        return await self._receipt(
            IlluminationAction.PATTERN,
            command,
            pattern_id=pattern_id,
        )

    async def _receipt(
        self,
        action: IlluminationAction,
        expected_command: str,
        *,
        pattern_id: int | None = None,
    ) -> CommandReceipt:
        """Read the service's report and map it into a plan-facing receipt.

        The status word is converted by value, never by identity, so the
        contracts enum and the service enum need not be the same type.  An
        ``uncertain`` report is returned as-is; this adapter never writes a
        replacement command.
        """
        reported_command = (await self._device.last_command.get_value()).strip()
        command = expected_command
        outcome = (await self._device.last_outcome.get_value()).strip()
        detail = (await self._device.last_detail.get_value()).strip()
        try:
            status = CommandStatus(outcome)
        except ValueError as error:
            raise RuntimeError(
                f"illumination service reported unknown outcome {outcome!r}"
            ) from error

        if reported_command and reported_command != expected_command:
            mismatch = (
                f"service reported command {reported_command!r}, "
                f"expected {expected_command!r}"
            )
            detail = f"{detail}; {mismatch}" if detail else mismatch

        echo = command if status == CommandStatus.ECHOED else None
        return CommandReceipt.from_transport(
            action=action,
            command=command,
            status=status,
            detail=detail,
            echo=echo,
            pattern_id=pattern_id,
        )


__all__ = ["CAPABILITY", "ArduinoIllumination"]
