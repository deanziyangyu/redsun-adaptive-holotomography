"""ESP300 read-only shared-port tests; no serial hardware."""

import pytest

from redsun_aht.device.esp300_readonly import Esp300ReadOnly


class Port:
    def __init__(self):
        self.commands = []
        self.bad = None

    def write(self, data):
        self.commands.append(data.decode("ascii"))
        return len(data)

    def flush(self):
        pass

    def readline(self, size=-1):
        command = self.commands[-1].removesuffix("?\r")
        if command == self.bad:
            return b""
        if command == "VE":
            return b"ESP300 Version 3.08\r\n"
        reply = {
            "ID": "CMA",
            "SN": "2",
            "SU": "0.0001",
            "SL": "0",
            "SR": "12.5",
            "TP": "0.002",
            "DP": "0.003",
            "MD": "1",
        }[command[1:]]
        return (reply + "\r\n").encode()


def test_xyz_snapshot_converts_units_and_uses_queries_only():
    port = Port()
    result = Esp300ReadOnly(port).snapshot()
    assert result.quality == "controller_reported"
    assert [a.axis for a in result.axes] == [1, 2, 3]
    assert all(
        (a.actual_um, a.target_um, a.resolution_um) == (2, 3, 0.1) for a in result.axes
    )
    assert all(
        (a.lower_um, a.upper_um, a.motion_done) == (0, 12500, True) for a in result.axes
    )
    assert all(command.endswith("?\r") for command in port.commands)
    assert port.commands == ["VE?\r"] + [
        f"{axis}{command}?\r"
        for axis in (1, 2, 3)
        for command in ("ID", "SN", "SU", "SL", "SR", "TP", "DP", "MD")
    ]


def test_timeout_quarantines_stream_without_retry():
    port = Port()
    port.bad = "2TP"
    owner = Esp300ReadOnly(port)
    with pytest.raises(RuntimeError, match="missing"):
        owner.snapshot()
    commands = list(port.commands)
    with pytest.raises(RuntimeError, match="uncertain"):
        owner.snapshot()
    assert port.commands == commands
