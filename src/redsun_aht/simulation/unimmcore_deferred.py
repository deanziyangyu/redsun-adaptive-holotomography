"""Simulated UniMMCore deferred stage and custom MDA sequence ordering.

This module cannot open a physical port, move a real stage or arm a camera.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pymmcore_plus.experimental.unicore import StageDevice
from pymmcore_plus.mda import MDAEngine

if TYPE_CHECKING:
    from collections.abc import Callable


class DeferredSimulatedStage(StageDevice):
    """Fake stage whose ordinary sequence start arms without moving."""

    def __init__(self, on_release: Callable[[tuple[float, ...]], None]) -> None:
        super().__init__()
        self._on_release = on_release
        self.position = 0.0
        self.sequence: tuple[float, ...] = ()
        self.armed = False
        self.released = False

    def home(self) -> None:
        """Home the simulated position."""
        self.position = 0.0

    def stop(self) -> None:
        """Disarm the simulated sequence."""
        self.armed = False

    def set_origin(self) -> None:
        """Reset the simulated coordinate."""
        self.position = 0.0

    def set_position_um(self, val: float) -> None:
        """Set a finite simulated position."""
        if not math.isfinite(val):
            raise ValueError("finite simulated position required")
        self.position = val

    def get_position_um(self) -> float:
        """Return the simulated coordinate."""
        return self.position

    def get_sequence_max_length(self) -> int:
        """Return the fake sequence capacity."""
        return 32

    def send_sequence(self, sequence: tuple[float, ...]) -> None:
        """Load a finite synthetic sequence."""
        if (
            not sequence
            or len(sequence) > 32
            or any(not math.isfinite(x) for x in sequence)
        ):
            raise ValueError("invalid simulated sequence")
        self.sequence = sequence
        self.armed = False
        self.released = False

    def start_sequence(self) -> None:
        """Arm without issuing a fake trigger."""
        if not self.sequence or self.armed or self.released:
            raise RuntimeError("simulated sequence not available")
        self.armed = True

    def stop_sequence(self) -> None:
        """Disarm without releasing."""
        self.armed = False

    def release_after_camera_start(self) -> None:
        """Release once after the engine's camera-start call returned."""
        if not self.armed or self.released:
            raise RuntimeError("simulated sequence not armed")
        self.armed = False
        self.released = True
        self._on_release(self.sequence)
        self.position = self.sequence[-1]


class DeferredSimulatedEngine(MDAEngine):
    """Custom hook paired only with the supplied fake stage."""

    def __init__(self, mmc: Any, stage: DeferredSimulatedStage) -> None:
        super().__init__(mmc)
        self.stage = stage

    def post_sequence_started(self, event: Any) -> None:
        """Release the fake stage only for an event carrying a Z sequence."""
        if event.z_sequence:
            self.stage.release_after_camera_start()
