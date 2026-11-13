"""Experimental UniMMCore focus adapter over an admitted ESP300 transport.

The injected transport owns controller communication and must verify move completion.
This adapter does not open a port, establish ownership, or verify a physical stop.
"""

from __future__ import annotations

import math
from threading import Lock
from typing import Protocol

from pymmcore_plus.experimental.unicore import StageDevice


class Esp300ZTransport(Protocol):
    """Single-owner, blocking axis-3 capability; position units are millimetres."""

    def move_and_settle_mm(self, target_mm: float) -> None:
        """Submit one move and verify controller-reported completion."""
        ...

    def position_mm(self) -> float:
        """Read the controller-reported position."""
        ...

    def stop(self) -> None:
        """Request a stop without asserting that physical motion stopped."""
        ...


class Esp300UniStage(StageDevice):
    """Bounded, non-sequenced Z stage; ownership is a caller assertion."""

    def __init__(
        self,
        transport: Esp300ZTransport,
        *,
        lower_um: float,
        upper_um: float,
        resources_exclusive: bool,
    ) -> None:
        super().__init__()
        if resources_exclusive is not True:
            raise ValueError("exclusive ESP300 ownership must be asserted")
        if (
            not all(math.isfinite(v) for v in (lower_um, upper_um))
            or lower_um > upper_um
        ):
            raise ValueError("finite ordered bounds are required")
        self.transport = transport
        self.lower_um, self.upper_um = lower_um, upper_um
        self._motion_guard = Lock()
        self._uncertain = False

    def set_position_um(self, val: float) -> None:
        """Prevalidate and submit one software-stepped move."""
        if not math.isfinite(val) or not self.lower_um <= val <= self.upper_um:
            raise ValueError("target outside configured bounds")
        with self._motion_guard:
            if self._uncertain:
                raise RuntimeError("ESP300 state uncertain; reconciliation required")
            try:
                self.transport.move_and_settle_mm(val / 1000.0)
            except BaseException:
                self._uncertain = True
                raise

    def get_position_um(self) -> float:
        """Return a fresh controller reading, not independent metrology."""
        with self._motion_guard:
            if self._uncertain:
                raise RuntimeError("ESP300 state uncertain; reconciliation required")
            try:
                position = self.transport.position_mm() * 1000.0
                if (
                    not math.isfinite(position)
                    or not self.lower_um <= position <= self.upper_um
                ):
                    raise RuntimeError("controller position outside configured bounds")
                return position
            except BaseException:
                self._uncertain = True
                raise

    def stop(self) -> None:
        """Request stop; subsequent use requires reconciliation."""
        self._uncertain = True
        self.transport.stop()

    def home(self) -> None:
        """Refuse implicit reference changes."""
        raise NotImplementedError("homing is not supported")

    def set_origin(self) -> None:
        """Refuse implicit reference changes."""
        raise NotImplementedError("origin changes are not supported")
