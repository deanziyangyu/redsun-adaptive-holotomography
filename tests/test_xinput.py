"""XInput mapping tests use injected data and never open a controller."""

import ctypes

import pytest

from redsun_aht.operator_input.xinput import (
    BUTTON_A,
    BUTTON_B,
    DPAD_DOWN,
    DPAD_UP,
    ControllerState,
    XInputPoller,
    _Gamepad,
    _State,
    survey_intent,
)


def test_xinput_structure_layout() -> None:
    assert ctypes.sizeof(_Gamepad) == 12
    assert ctypes.sizeof(_State) == 16


def test_deadman_stop_and_disconnect_neutralize_all_axes() -> None:
    for state in (
        None,
        ControllerState(0, 32767, -32768),
        ControllerState(BUTTON_A | BUTTON_B, 32767, -32768),
    ):
        intent = survey_intent(state)
        assert not intent.enabled and intent.stop_requested
        assert (intent.x, intent.y, intent.z) == (0, 0, 0)


def test_axes_are_bounded_deadzone_and_direction_configurable() -> None:
    intent = survey_intent(
        ControllerState(BUTTON_A | DPAD_UP, -32768, 32767), invert_x=True
    )
    assert (intent.x, intent.y, intent.z) == (1, 1, 1)
    neutral = survey_intent(ControllerState(BUTTON_A | DPAD_UP | DPAD_DOWN, 100, -100))
    assert (neutral.x, neutral.y, neutral.z) == (0, 0, 0)
    assert not neutral.stop_requested
    assert (
        survey_intent(ControllerState(BUTTON_A | DPAD_DOWN, 0, 0), invert_z=True).z == 1
    )
    with pytest.raises(ValueError):
        survey_intent(None, deadzone=1.0)


def test_poll_success_and_error_are_fail_closed() -> None:
    def fake(index, ptr):
        state = ctypes.cast(ptr, ctypes.POINTER(_State)).contents
        state.gamepad.buttons = BUTTON_A
        state.gamepad.left_x = 32767
        return 0

    sample = XInputPoller(get_state=fake).read()
    assert sample is not None and survey_intent(sample).x == 1
    assert survey_intent(XInputPoller(get_state=lambda *_: 1167).read()).stop_requested
    with pytest.raises(ValueError):
        XInputPoller(index=4, get_state=fake)


def test_reconnect_and_stop_require_enable_release_before_rearming() -> None:
    from redsun_aht.operator_input.xinput import SurveyController

    controller = SurveyController()
    held = ControllerState(BUTTON_A, 32767, 0)
    released = ControllerState(0, 0, 0)
    assert controller.update(held).stop_requested
    assert controller.update(released).stop_requested
    assert controller.update(held).enabled
    assert controller.update(None).stop_requested
    assert controller.update(held).stop_requested
    controller.update(released)
    assert controller.update(held).enabled
    assert controller.update(
        ControllerState(BUTTON_A | BUTTON_B, 32767, 0)
    ).stop_requested
    assert controller.update(held).stop_requested
    controller.update(released)
    assert controller.update(held).enabled
