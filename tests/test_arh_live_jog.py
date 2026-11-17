from unittest.mock import Mock

from redsun_aht.operator_input.live_jog import LiveJog
from redsun_aht.operator_input.xinput import BUTTON_A, BUTTON_B, DPAD_LEFT, DPAD_RIGHT, DPAD_UP, ControllerState


def test_live_jog_requires_neutral_and_halts_on_release_or_disconnect():
    read = Mock(side_effect=[
        ControllerState(BUTTON_A, 32767, 0),
        ControllerState(0, 0, 0),
        ControllerState(BUTTON_A, 32767, 10000),
        ControllerState(BUTTON_A | DPAD_UP, 0, 0),
        ControllerState(0, 0, 0),
        ControllerState(BUTTON_A, 0, -32768),
        None,
        ControllerState(BUTTON_A, 32767, 0),
    ])
    move, halt = Mock(), Mock()
    jog = LiveJog(read, move, halt)
    for _ in range(8):
        jog.poll()
    assert [entry.args for entry in move.call_args_list] == [
        ("x", 1.0), ("zdt", 1.0), ("y", -1.0),
    ]
    assert halt.call_count == 2


def test_stop_disarms_and_halts_only_once():
    read = Mock(side_effect=[
        ControllerState(0, 0, 0),
        ControllerState(BUTTON_A, 32767, 0),
        ControllerState(BUTTON_A | BUTTON_B, 32767, 0),
        ControllerState(BUTTON_A, 32767, 0),
    ])
    move, halt = Mock(), Mock()
    jog = LiveJog(read, move, halt)
    for _ in range(4):
        jog.poll()
    jog.stop()
    move.assert_called_once_with("x", 1.0)
    halt.assert_called_once()


def test_shoulders_select_one_step_per_press_and_clamp():
    from redsun_aht.operator_input.xinput import LEFT_SHOULDER, RIGHT_SHOULDER

    move, halt, changed = Mock(), Mock(), Mock()
    read = Mock()
    jog = LiveJog(read, move, halt, changed)

    def poll(buttons=0, x=0):
        read.return_value = ControllerState(buttons, x, 0)
        jog.poll()

    poll()
    for _ in range(4):
        poll(RIGHT_SHOULDER)
        poll(RIGHT_SHOULDER)
        poll()
    poll(BUTTON_A, -32768)
    poll(LEFT_SHOULDER)
    poll(BUTTON_A | DPAD_UP)
    assert [call.args[0] for call in changed.call_args_list] == [5, 20, 100, 20]
    assert [call.args for call in move.call_args_list] == [("x", -100), ("zdt", 20)]


def test_horizontal_dpad_jogs_focus_only_after_neutral():
    read = Mock(side_effect=[
        ControllerState(DPAD_LEFT, 0, 0),
        ControllerState(BUTTON_A | DPAD_LEFT, 0, 0),
        ControllerState(0, 0, 0),
        ControllerState(BUTTON_A | DPAD_LEFT, 0, 0),
        ControllerState(BUTTON_A | DPAD_RIGHT, 0, 0),
    ])
    move = Mock()
    jog = LiveJog(read, move, Mock())
    for _ in range(5):
        jog.poll()
    assert [call.args for call in move.call_args_list] == [("z", -1), ("z", 1)]


def test_rotary_controls_require_neutral_and_stop_on_release():
    read = Mock(side_effect=[
        ControllerState(0, 0, 0, right_x=32767),
        ControllerState(BUTTON_A, 0, 0, right_x=32767),
        ControllerState(0, 0, 0),
        ControllerState(BUTTON_A, 0, 0, right_x=32767),
        ControllerState(BUTTON_A, 0, 0, right_y=-32768),
        ControllerState(BUTTON_A, 0, 0, right_trigger=255),
        ControllerState(0, 0, 0),
    ])
    move, halt_mcu = Mock(), Mock()
    jog = LiveJog(read, move, Mock(), halt_mcu=halt_mcu)
    for _ in range(7):
        jog.poll()
    assert [call.args for call in move.call_args_list] == [
        ("roll", 0.2), ("pitch", -0.2), ("yaw", 0.2),
    ]
    halt_mcu.assert_called_once()


def test_holospim_mapping_uses_horizontal_dpad_for_zdt_focus():
    read = Mock(side_effect=[
        ControllerState(0, 0, 0),
        ControllerState(BUTTON_A | DPAD_LEFT, 0, 0),
        ControllerState(BUTTON_A | DPAD_UP, 0, 0),
    ])
    move = Mock()
    jog = LiveJog(read, move, Mock(), zdt_is_focus=True)
    for _ in range(3):
        jog.poll()
    assert [call.args for call in move.call_args_list] == [("zdt", -1), ("z", 1)]


def test_holospim_survey_jog_ignores_unavailable_axes():
    read = Mock(side_effect=[
        ControllerState(0, 0, 0),
        ControllerState(BUTTON_A | DPAD_LEFT, 0, 0, right_x=32767),
        ControllerState(BUTTON_A, 18000, 0, right_y=32767),
        ControllerState(BUTTON_A | DPAD_UP, 0, 0),
        ControllerState(0, 0, 0),
    ])
    move, halt, halt_mcu = Mock(), Mock(), Mock()
    jog = LiveJog(
        read, move, halt, halt_mcu=halt_mcu, zdt_is_focus=True,
        supported_axes={"x", "y", "z"},
    )
    for _ in range(5):
        jog.poll()
    assert [call.args for call in move.call_args_list] == [("x", 1), ("z", 1)]
    halt.assert_called_once()
    halt_mcu.assert_not_called()


def test_holospim_positioner_stops_only_unsettled_jogs(monkeypatch):
    from types import MethodType, SimpleNamespace

    from redsun.utils.devices import DevicePath

    from redsun_aht.configurations import holospim

    monkeypatch.setattr(holospim, "XInputPoller", Mock())
    monkeypatch.setattr(holospim, "QtCore", SimpleNamespace(QTimer=Mock()))
    monkeypatch.setattr(holospim, "QtWidgets", SimpleNamespace(QApplication=Mock()))
    view = Mock(spec=holospim.HolospimPositionerView)
    for name in (
        "configure_jog", "_move_jog", "_stop_jog", "set_moving", "set_locked",
    ):
        setattr(view, name, MethodType(getattr(holospim.HolospimPositionerView, name), view))
    view._groups = {"survey": Mock()}
    view._paths = {}
    view._held_boxes = set()
    view._configuration_tab = None
    view._lock_entries = Mock()
    view._paths["survey"] = {
        "z": DevicePath("survey", ("axis", "z"), ("survey", "survey-axis", "survey-axis-z")),
    }
    view.configure_jog({"z": ("survey", "z"), "zdt": ("mcu", "x28"), "roll": ("rotation", "rotation")})
    moves, stops = [], []
    view.sig_move.emit.side_effect = lambda *args: moves.append(args)
    view.sig_stop_device.emit.side_effect = stops.append

    view._move_jog("z", 1)
    view.set_moving("survey", True)
    view._move_jog("z", 1)
    view.set_moving("survey", False)
    view._stop_jog()
    assert moves == [("survey", "z", 1)]
    assert stops == []

    view._jog.jogging = True
    view._move_jog("z", 1)
    view.set_locked(frozenset({"survey-axis-z"}))
    view._move_jog("z", 1)
    assert len(moves) == 2
    assert stops == ["survey"]
    view.set_locked(frozenset())
    poller = holospim.XInputPoller.return_value
    poller.read.return_value = ControllerState(BUTTON_A | DPAD_UP, 0, 0)
    view._jog.poll()
    assert len(moves) == 2
    poller.read.return_value = ControllerState(0, 0, 0)
    view._jog.poll()
    poller.read.return_value = ControllerState(BUTTON_A | DPAD_UP, 0, 0)
    view._jog.poll()
    assert len(moves) == 3
    view.set_moving("survey", False)
    poller.read.return_value = ControllerState(0, 0, 0)
    view._jog.poll()
    poller.read.return_value = ControllerState(BUTTON_A | DPAD_LEFT, 0, 0)
    view._jog.poll()
    assert moves[-1] == ("mcu", "x28", -1)
    poller.read.return_value = ControllerState(0, 0, 0)
    view._jog.poll()
    assert stops[-1] == "mcu"
    poller.read.return_value = ControllerState(BUTTON_A, 0, 0, right_x=32767)
    view._jog.poll()
    assert moves[-1] == ("rotation", "rotation", 0.2)
    poller.read.return_value = None
    view._jog.poll()
    assert stops[-1] == "rotation"
