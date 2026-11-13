"""Shared outcome of a motion request that a stop ended before its target.

A stop command is not an error the controller reports: the axis simply comes to
rest short of the target. A bounded move that waits for that target would
otherwise keep waiting until its own timeout, leaving the service busy long after
the stage has stopped, so the device facade raises :class:`MoveHalted` with the
position it actually reached. The owning service turns that into a
``stopped`` outcome instead of an unknown one.
"""

from __future__ import annotations

from typing import Final

#: Outcome label the service publishes for a stop-ended move.
HALTED_OUTCOME: Final = "stopped"


class MoveHalted(RuntimeError):
    """A stop request ended a move before its target; the position is known.

    ``position_xy_um`` is the settled coordinate the axis reported after the
    stop, so a caller can record where the move actually ended rather than
    guessing from the requested target.
    """

    def __init__(self, position_xy_um: tuple[float, float], message: str) -> None:
        super().__init__(message)
        self.position_xy_um = (float(position_xy_um[0]), float(position_xy_um[1]))


__all__ = ["HALTED_OUTCOME", "MoveHalted"]
