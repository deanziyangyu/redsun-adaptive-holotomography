"""In-process survey rehearsal over a synthetic composite stage.

This is not a physical motion coordinator: it has no hardware stop, timing,
PVA authorization, asynchronous motion, or collision checking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from redsun_aht.operator_input.xinput import (
    ControllerState,
    SurveyController,
    SurveyIntent,
)

if TYPE_CHECKING:
    from redsun_aht.simulation.mcu_stack import SimulatedStage


@dataclass(frozen=True, slots=True)
class SurveyResult:
    """One simulated input tick and the positions after any movement."""

    intent: SurveyIntent
    positions_um: dict[str, float]
    moved: bool


class SimulatedSurvey:
    """Apply small, quantized manual increments to reserved synthetic axes."""

    def __init__(self, stage: SimulatedStage, *, max_step_um: float = 1.0) -> None:
        if not math.isfinite(max_step_um) or max_step_um <= 0:
            raise ValueError("maximum step must be positive and finite")
        if not {"survey_x", "survey_y", "coarse_z"} <= set(stage.axes):
            raise ValueError("survey axes are missing")
        self.stage = stage
        self.max_step_um = max_step_um
        self.input = SurveyController()

    def update(self, state: ControllerState | None) -> SurveyResult:
        """Apply at most one bounded move; disconnect/release makes no move."""
        intent = self.input.update(state)
        if intent.stop_requested:
            return SurveyResult(intent, dict(self.stage.positions), False)
        owner = "simulated-survey"
        try:
            self.stage.reserve(owner)
        except RuntimeError:
            self.input.disarm()
            raise
        try:
            targets = {}
            for name, amount in zip(
                ("survey_x", "survey_y", "coarse_z"),
                (intent.x, intent.y, intent.z),
                strict=True,
            ):
                axis = self.stage.axes[name]
                # Round toward zero so quantization never exceeds the per-tick cap.
                increments = math.trunc(amount * self.max_step_um / axis.increment_um)
                if increments:
                    targets[name] = (
                        self.stage.positions[name] + increments * axis.increment_um
                    )
            if targets:
                self.stage.move(targets, owner=owner)
            return SurveyResult(intent, dict(self.stage.positions), bool(targets))
        except Exception:
            self.input.disarm()
            raise
        finally:
            self.stage.release(owner)
