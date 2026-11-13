"""Xbox-to-synthetic-axis rehearsal; no controller or stage hardware."""

import pytest

from redsun_aht.operator_input.xinput import BUTTON_A, DPAD_UP, ControllerState
from redsun_aht.simulation.mcu_stack import Axis, SimulatedMcu, SimulatedStage
from redsun_aht.simulation.survey import SimulatedSurvey


def setup_stage():
    axes = tuple(
        Axis(name, -2, 2, 0.25)
        for name in ("survey_x", "survey_y", "coarse_z", "fast_z")
    )
    stage = SimulatedStage(axes)
    return stage, SimulatedSurvey(stage, max_step_um=1.0)


def test_survey_moves_three_synthetic_axes_and_release_stops() -> None:
    stage, survey = setup_stage()
    survey.update(ControllerState(0, 0, 0))
    result = survey.update(ControllerState(BUTTON_A | DPAD_UP, 32767, -32768))
    assert result.positions_um == {
        "survey_x": 1.0,
        "survey_y": -1.0,
        "coarse_z": 1.0,
        "fast_z": 0.0,
    }
    assert result.moved
    result = survey.update(None)
    assert result.intent.stop_requested and not result.moved
    assert stage.owner is None


def test_survey_conflict_and_limits_leave_axes_unchanged() -> None:
    stage, survey = setup_stage()
    survey.update(ControllerState(0, 0, 0))
    stage.reserve("stack")
    with pytest.raises(RuntimeError, match="reserved"):
        survey.update(ControllerState(BUTTON_A, 32767, 0))
    assert stage.owner == "stack" and stage.positions["survey_x"] == 0
    stage.release("stack")
    stage.move({"survey_x": 2.0})
    survey.update(ControllerState(0, 0, 0))
    with pytest.raises(ValueError, match="outside"):
        survey.update(ControllerState(BUTTON_A, 32767, 32767))
    assert stage.positions["survey_x"] == 2.0 and stage.positions["survey_y"] == 0
    assert stage.owner is None


def test_stack_reservation_blocks_survey_until_release() -> None:
    stage, survey = setup_stage()
    mcu = SimulatedMcu(stage.axes["fast_z"], stage=stage)
    survey.update(ControllerState(0, 0, 0))
    mcu.reserve("stack")
    with pytest.raises(RuntimeError, match="reserved"):
        survey.update(ControllerState(BUTTON_A, 32767, 0))
    mcu.cancel("stack")
    mcu.release("stack")
    assert not survey.update(ControllerState(BUTTON_A, 32767, 0)).moved
    survey.update(ControllerState(0, 0, 0))
    assert survey.update(ControllerState(BUTTON_A, 32767, 0)).moved
