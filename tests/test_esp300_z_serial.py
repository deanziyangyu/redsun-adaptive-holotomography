"""Fake-only ESP300 PA/MD/TP transport checks."""

from collections import deque

import pytest

from redsun_aht.device.esp300_readonly import AxisSnapshot
from redsun_aht.device.esp300_z_serial import Esp300ZSerial


class Port:
    def __init__(self, replies=()):
        self.replies = deque(replies)
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        pass

    def readline(self, size=-1):
        return self.replies.popleft()


def snapshot(**changes):
    data = dict(
        axis=3,
        identity="stage",
        resolution_um=0.1,
        lower_um=0,
        upper_um=12500,
        actual_um=0,
        target_um=0,
        motion_done=True,
    )
    data.update(changes)
    return AxisSnapshot(**data)


def make(port, state=None, **kwargs):
    return Esp300ZSerial(
        port,
        state or snapshot(),
        expected_identity="stage",
        lower_um=0,
        upper_um=10,
        **kwargs,
    )


def test_move_and_settle_uses_historical_axis_commands():
    port = Port([b"1\r\n", b"0\r\n", b"0\r\n", b"0\r\n", b"1\r\n", b"0.005\r\n"])
    stage = make(port)
    stage.move_and_settle_mm(0.005)
    assert port.writes == [
        b"3MD?\r",
        b"3TP?\r",
        b"3DP?\r",
        b"3PA0.005\r",
        b"3MD?\r",
        b"3MD?\r",
        b"3TP?\r",
    ]


def test_bad_admission_and_out_of_bounds_never_move():
    port = Port()
    for state in (snapshot(axis=2), snapshot(motion_done=False), snapshot(actual_um=1)):
        with pytest.raises(ValueError):
            make(port, state)
    stage = make(port)
    with pytest.raises(ValueError):
        stage.move_and_settle_mm(0.02)
    assert port.writes == []


def test_wrong_final_position_latches_uncertainty_without_replay():
    port = Port([b"1\r\n", b"0\r\n", b"0\r\n", b"1\r\n", b"0.004\r\n"])
    stage = make(port)
    with pytest.raises(RuntimeError, match="match target"):
        stage.move_and_settle_mm(0.005)
    with pytest.raises(RuntimeError, match="uncertain"):
        stage.move_and_settle_mm(0.005)
    assert port.writes.count(b"3PA0.005\r") == 1


def test_stop_inhibits_further_requests():
    port = Port()
    stage = make(port)
    stage.stop()
    with pytest.raises(RuntimeError, match="uncertain"):
        stage.move_and_settle_mm(0.005)
    assert port.writes == [b"3ST\r"]


def test_registered_unimmcore_stage_uses_fake_serial_transport():
    from pymmcore_plus.experimental.unicore import UniMMCore

    from redsun_aht.device.unimmcore_esp300 import Esp300UniStage

    port = Port([b"1\r\n", b"0\r\n", b"0\r\n", b"1\r\n", b"0.005\r\n", b"0.005\r\n"])
    transport = make(port)
    stage = Esp300UniStage(transport, lower_um=0, upper_um=10, resources_exclusive=True)
    core = UniMMCore()
    core.loadPyDevice("esp-z", stage)
    try:
        core.initializeDevice("esp-z")
        core.setFocusDevice("esp-z")
        core.setPosition(5)
        assert core.getPosition() == 5
        assert port.writes == [
            b"3MD?\r",
            b"3TP?\r",
            b"3DP?\r",
            b"3PA0.005\r",
            b"3MD?\r",
            b"3TP?\r",
            b"3TP?\r",
        ]
    finally:
        core.unloadAllDevices()


@pytest.mark.parametrize("replies", [[b"0\r\n"], [b"1\r\n", b"0\r\n", b"0.001\r\n"]])
def test_pre_move_unexpected_state_refuses_absolute_move(replies):
    port = Port(replies)
    stage = make(port)
    with pytest.raises(RuntimeError):
        stage.move_and_settle_mm(0.005)
    assert all(not command.startswith(b"3PA") for command in port.writes)
    with pytest.raises(RuntimeError, match="uncertain"):
        stage.move_and_settle_mm(0.005)


def test_factory_reads_admission_from_same_endpoint():
    class RespondingPort(Port):
        def readline(self, size=-1):
            cmd = self.writes[-1].decode().removesuffix("?\r")
            value = {
                "VE": "ESP300 v3",
                "ID": "stage",
                "SN": "2",
                "SU": "0.0001",
                "SL": "0",
                "SR": "12.5",
                "TP": "0",
                "DP": "0",
                "MD": "1",
            }
            return (value["VE" if cmd == "VE" else cmd[1:]] + "\r\n").encode()

    port = RespondingPort()
    stage = Esp300ZSerial.admit(
        port, expected_identity="stage", lower_um=0, upper_um=10
    )
    assert stage.position_mm() == 0
    assert port.writes[0] == b"VE?\r" and port.writes[-1] == b"3TP?\r"
    assert all(b"?" in command for command in port.writes)
