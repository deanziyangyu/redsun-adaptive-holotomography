"""Ophyd-async device for the PVA-served 91-LED v1 Arduino controller.

The device is a typed client of :mod:`redsun_aht.services.arduino_led`; it owns
no serial port and no camera.  Writing :attr:`ArduinoLedDevice.pattern` sends
the MMStudio pattern command. The AHT MM ASCII 91 sketch selects illumination
without a camera TTL pulse. Patterns ``1..4`` are the semicircles and pattern
``5`` is the centre LED. The software DPCT plan reads a fresh camera frame
after each pattern write.

The PVA prefix must carry the ``pva://`` scheme so ophyd-async selects its PVA
backend.  Attribute names are FastCS's ``snake_to_pascal`` spellings.
"""

from __future__ import annotations

from typing import Annotated, Final

from ophyd_async.core import (  # noqa: TC002 - runtime signal annotations
    SignalR,
    SignalRW,
    SignalW,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix

from .commands import CENTER_PATTERN

PATTERN_MIN: Final = 1
"""First 91-LED v1 semicircle pattern."""

PATTERN_MAX: Final = CENTER_PATTERN
"""Centre LED pattern id; the four semicircles use 1..4."""


def pva_prefix(prefix: str) -> str:
    """Return *prefix* with an explicit ``pva://`` scheme and one trailing colon."""
    bare = prefix.split("://", 1)[-1]
    if not bare.endswith(":"):
        bare = f"{bare}:"
    return f"pva://{bare}"


class ArduinoLedDevice(EpicsDevice):
    """Pattern selection and transport diagnostics for one Arduino LED service."""

    pattern: Annotated[SignalRW[int], PvSuffix("Pattern")]
    all_off: Annotated[SignalW[bool], PvSuffix("AllOff")]
    open_shutter: Annotated[SignalW[bool], PvSuffix("OpenShutter")]
    last_command: Annotated[SignalR[str], PvSuffix("LastCommand")]
    last_outcome: Annotated[SignalR[str], PvSuffix("LastOutcome")]
    last_detail: Annotated[SignalR[str], PvSuffix("LastDetail")]
    panel_profile: Annotated[SignalR[str], PvSuffix("PanelProfile")]
    led_count: Annotated[SignalR[int], PvSuffix("LedCount")]

    def __init__(self, prefix: str, name: str = "") -> None:
        super().__init__(prefix=pva_prefix(prefix), name=name)

    async def set_pattern(self, number: int) -> None:
        """Select panel pattern *number* (``1..5``).

        Patterns ``1..4`` are the four 91-LED v1 semicircles and ``5`` is the
        centre.  The method refuses anything outside that range before touching
        the transport. The firmware emits no camera TTL pulse.

        Raises
        ------
        ValueError
            If *number* is not one of ``1..5``.
        """
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError(f"pattern must be an integer, got {type(number).__name__}")
        if not PATTERN_MIN <= number <= PATTERN_MAX:
            raise ValueError(
                f"pattern must be between {PATTERN_MIN} and {PATTERN_MAX}, "
                f"got {number}; panel91 has four semicircles plus a centre"
            )
        await self.pattern.set(number)

    async def set_centre(self) -> None:
        """Select the centre LED (``15``)."""
        await self.set_pattern(CENTER_PATTERN)


__all__ = [
    "PATTERN_MAX",
    "PATTERN_MIN",
    "ArduinoLedDevice",
    "pva_prefix",
]
