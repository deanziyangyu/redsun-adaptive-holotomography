"""Fake-only ESP300 PA/MD/TP transport checks."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from redsun_aht.device.esp300_readonly import AxisSnapshot
from redsun_aht.device.esp300_z_serial import Esp300SurveySerial, Esp300ZSerial


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


def test_wrong_final_position_latches_uncertainty_without_replay(monkeypatch):
    ticks = iter([0.0, 11.0])
    monkeypatch.setattr(
        "redsun_aht.device.esp300_z_serial.time",
        type("Timer", (), {"monotonic": staticmethod(lambda: next(ticks))}),
    )
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


def test_failed_stop_write_can_be_retried_without_unlatching():
    class ShortWritePort(Port):
        def write(self, data):
            count = super().write(data)
            return count - 1 if len(self.writes) == 1 else count

    port = ShortWritePort()
    stage = make(port)
    with pytest.raises(OSError, match="short"):
        stage.stop()
    with pytest.raises(RuntimeError, match="uncertain"):
        stage.move_and_settle_mm(0.005)
    stage.stop()
    stage.stop()
    assert port.writes == [b"3ST\r", b"3ST\r"]


def xyz_snapshot_replies():
    return [b"ESP300\r\n"] + [
        line + b"\r\n"
        for index in range(1, 4)
        for line in (
            str(index).encode(),
            b"2",
            b"0.0001",
            b"0",
            b"12.5",
            b"0",
            b"0",
            b"1",
        )
    ]


@pytest.mark.parametrize("stop_during_preflight", (False, True))
def test_stop_during_serial_query_cancels_current_and_queued_moves(
    stop_during_preflight,
):
    entered, release = Event(), Event()

    class BlockingPort(Port):
        def readline(self, size=-1):
            command = self.writes[-1]
            if stop_during_preflight:
                interrupt = command == b"VE?\r" and self.writes.count(command) == 2
            else:
                interrupt = command == b"3MD?\r" and b"3PA0.005\r" in self.writes
            if interrupt:
                entered.set()
                assert release.wait(3)
            return super().readline(size)

    port = BlockingPort(
        xyz_snapshot_replies() * 2 + [b"1\r\n", b"1\r\n", b"0\r\n", b"0\r\n", b"0\r\n"]
    )
    survey = Esp300SurveySerial(port, (("1", 0, 10), ("2", 0, 10), ("3", 0, 10)))
    with ThreadPoolExecutor(max_workers=3) as threads:
        move = threads.submit(survey.move_um, 3, 5)
        try:
            assert entered.wait(3)
            queued = threads.submit(survey.move_um, 1, 5)
            stop = threads.submit(survey.stop)
            assert survey._uncertain.wait(3)
        finally:
            release.set()
        stop.result(3)
        with pytest.raises(RuntimeError, match=r"stop requested|uncertain"):
            move.result(3)
        with pytest.raises(RuntimeError, match="uncertain"):
            queued.result(3)
    expected = [] if stop_during_preflight else [b"3PA0.005\r", b"3ST\r"]
    assert [line for line in port.writes if not line.endswith(b"?\r")] == expected


@pytest.mark.parametrize("axis", [1, 2, 3])
@pytest.mark.parametrize("settled", [True, False])
def test_xyz_owner_moves_only_selected_axis_and_stops_on_unverified_readback(
    axis, settled, monkeypatch
):
    snapshot_replies = xyz_snapshot_replies()
    initial_poll = [b"1\r\n", b"0\r\n"]
    final_poll = [b"1\r\n", b"0.005\r\n"] if settled else []
    if not settled:
        ticks = iter([0.0, 11.0])
        monkeypatch.setattr(
            "redsun_aht.device.esp300_z_serial.time",
            type("Timer", (), {"monotonic": staticmethod(lambda: next(ticks))}),
        )
    port = Port(
        snapshot_replies * 2
        + [b"1\r\n", b"1\r\n", b"0\r\n", b"0\r\n"]
        + initial_poll
        + final_poll
    )
    survey = Esp300SurveySerial(port, (("1", 0, 10), ("2", 0, 10), ("3", 0, 10)))
    before = list(port.writes)
    with pytest.raises(ValueError, match="bounds"):
        survey.move_um(axis, 20)
    assert port.writes == before
    if settled:
        survey.move_um(axis, 5)
        expected = [f"{axis}PA0.005\r".encode()]
    else:
        with pytest.raises(RuntimeError, match="match target"):
            survey.move_um(axis, 5)
        expected = [f"{axis}PA0.005\r".encode(), f"{axis}ST\r".encode()]
        with pytest.raises(RuntimeError, match="uncertain"):
            survey.snapshot()
    assert [line for line in port.writes if not line.endswith(b"?\r")] == expected


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
