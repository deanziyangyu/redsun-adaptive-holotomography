from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

import pytest

from redsun_aht.device.stage import (
    ESP300_BAUD_RATE,
    ESP300_DATA_BITS,
    ESP300_PARITY,
    ESP300_RTS_CTS,
    ESP300_STOP_BITS,
    ESP300_TIMEOUT_SECONDS,
    Esp300AxisBackend,
    Esp300Controller,
    build_esp300_composite_stage,
)


@dataclass
class _FakeEspSerial:
    responses: list[bytes]
    writes: list[bytes] = field(default_factory=list)
    flushed: int = 0
    short_write: bool = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data) - 1 if self.short_write else len(data)

    def flush(self) -> None:
        self.flushed += 1

    def readline(self, size: int = -1) -> bytes:
        assert size == 64
        return self.responses.pop(0)


def test_read_only_identity_uses_controller_and_axis_queries() -> None:
    port = _FakeEspSerial(
        [
            b"ESP300 Version 3.0.1\r\n",
            b"M-UTM25PP,123\r\n",
            b"2\r\n",
            b"0.0001\r\n",
            b"-12.5\r\n",
            b"12.5\r\n",
            b"1.25\r\n",
        ]
    )
    controller = Esp300Controller(port, axis_count=1, max_response_bytes=64)

    identity = controller.read_identity()

    assert identity.version == "ESP300 Version 3.0.1"
    assert identity.axes[0].stage_id == "M-UTM25PP,123"
    assert identity.axes[0].unit == "mm"
    assert identity.axes[0].encoder_resolution == 0.0001
    assert identity.axes[0].lower_limit == -12.5
    assert identity.axes[0].upper_limit == 12.5
    assert identity.axes[0].position == 1.25
    assert port.writes == [
        b"VE?\r",
        b"1ID?\r",
        b"1SN?\r",
        b"1SU?\r",
        b"1SL?\r",
        b"1SR?\r",
        b"1TP?\r",
    ]
    assert port.flushed == 7


def test_admitted_identity_builds_limit_checked_composite() -> None:
    port = _FakeEspSerial(
        [
            b"ESP300 Version 3.0.1\r\n",
            b"M-UTM25PP,123\r\n",
            b"2\r\n",
            b"0.0001\r\n",
            b"-12.5\r\n",
            b"12.5\r\n",
            b"1.25\r\n",
        ]
    )
    controller = Esp300Controller(port, axis_count=1, max_response_bytes=64)
    identity = controller.read_identity()

    stage = build_esp300_composite_stage(controller, identity, axis_names=("x",))

    assert stage.axes == {"x": "mm"}
    assert stage.specs[0].lower_limit == -12.5
    assert stage.specs[0].upper_limit == 12.5
    assert stage.specs[0].settle_tolerance == 0.0002


def test_axis_backend_uses_documented_absolute_status_and_stop_commands() -> None:
    port = _FakeEspSerial([b"1.25\r\n", b"1\r\n"])
    controller = Esp300Controller(port, max_response_bytes=64)
    axis = Esp300AxisBackend(controller, axis=2)

    async def exercise() -> tuple[float, bool]:
        await axis.move_to(1.25)
        position = await axis.read_position()
        complete = await axis.motion_complete()
        await axis.stop()
        return position, complete

    assert asyncio.run(exercise()) == (1.25, True)
    assert port.writes == [b"2PA1.25\r", b"2TP?\r", b"2MD?\r", b"2ST\r"]


def test_controller_enables_an_admitted_axis_before_motion() -> None:
    port = _FakeEspSerial([])
    controller = Esp300Controller(port, max_response_bytes=64)

    controller.enable_axis(2)

    assert port.writes == [b"2MO\r"]


@pytest.mark.parametrize("response", [b"", b"\xff\r\n", b"\r\n"])
def test_query_rejects_invalid_responses(response: bytes) -> None:
    controller = Esp300Controller(_FakeEspSerial([response]), max_response_bytes=64)

    with pytest.raises((TimeoutError, RuntimeError)):
        controller.query("TP", axis=1)


def test_transport_rejects_bad_identity_short_write_and_axis() -> None:
    bad_identity = Esp300Controller(
        _FakeEspSerial([b"not an ESP controller\r\n"]), max_response_bytes=64
    )
    with pytest.raises(RuntimeError, match="unexpected Newport"):
        bad_identity.read_identity()

    short_port = _FakeEspSerial([], short_write=True)
    with pytest.raises(OSError, match="short ESP300 write"):
        Esp300Controller(short_port, max_response_bytes=64).write_axis(1, "ST")

    with pytest.raises(ValueError, match="range"):
        Esp300Controller(_FakeEspSerial([]), max_response_bytes=64).query("TP", axis=4)

    with pytest.raises(ValueError, match="axis_count"):
        Esp300Controller(_FakeEspSerial([]), axis_count=0)
    with pytest.raises(ValueError, match="max_response_bytes"):
        Esp300Controller(_FakeEspSerial([]), max_response_bytes=1)


def test_identity_rejects_unknown_units_and_invalid_axis_values() -> None:
    unknown_unit = _FakeEspSerial([b"ESP300 Version 3\r\n", b"stage\r\n", b"99\r\n"])
    with pytest.raises(RuntimeError, match="unit code"):
        Esp300Controller(
            unknown_unit, axis_count=1, max_response_bytes=64
        ).read_identity()

    bad_resolution = _FakeEspSerial(
        [
            b"ESP300 Version 3\r\n",
            b"stage\r\n",
            b"2\r\n",
            b"0\r\n",
            b"-1\r\n",
            b"1\r\n",
            b"0\r\n",
        ]
    )
    with pytest.raises(RuntimeError, match="non-positive"):
        Esp300Controller(
            bad_resolution, axis_count=1, max_response_bytes=64
        ).read_identity()


def test_query_and_axis_backend_reject_malformed_values() -> None:
    overlong = Esp300Controller(_FakeEspSerial([b"x" * 64]), max_response_bytes=64)
    with pytest.raises(RuntimeError, match="exceeds"):
        overlong.query("TP", axis=1)

    with pytest.raises(ValueError, match="ASCII"):
        Esp300Controller(_FakeEspSerial([]), max_response_bytes=64).query("")
    with pytest.raises(ValueError, match="framing"):
        Esp300Controller(_FakeEspSerial([]), max_response_bytes=64).query("TP?")

    controller = Esp300Controller(
        _FakeEspSerial([b"not-a-number\r\n", b"2\r\n"]),
        max_response_bytes=64,
    )
    axis = Esp300AxisBackend(controller, 1)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="invalid ESP300 TP"):
            await axis.read_position()
        with pytest.raises(RuntimeError, match="motion status"):
            await axis.motion_complete()

    asyncio.run(exercise())

    with pytest.raises(ValueError, match="outside"):
        Esp300AxisBackend(controller, 4)


@pytest.mark.hardware
def test_real_esp300_read_only_exact_admission() -> None:
    if os.getenv("AHT_TEST_REAL_ESP300") != "1":
        pytest.skip("set AHT_TEST_REAL_ESP300=1 for read-only COM admission")

    port_name = os.environ["AHT_ESP300_PORT"]
    expected_version = os.environ["AHT_ESP300_EXPECTED_VERSION"]
    expected_axis_ids = tuple(
        value.strip() for value in os.environ["AHT_ESP300_EXPECTED_AXIS_IDS"].split("|")
    )
    if not expected_axis_ids or any(not value for value in expected_axis_ids):
        raise ValueError("AHT_ESP300_EXPECTED_AXIS_IDS must be pipe-delimited")

    import serial

    with serial.Serial(
        port=port_name,
        baudrate=ESP300_BAUD_RATE,
        bytesize=ESP300_DATA_BITS,
        parity=ESP300_PARITY,
        stopbits=ESP300_STOP_BITS,
        timeout=ESP300_TIMEOUT_SECONDS,
        write_timeout=ESP300_TIMEOUT_SECONDS,
        rtscts=ESP300_RTS_CTS,
    ) as port:
        identity = Esp300Controller(
            port, axis_count=len(expected_axis_ids)
        ).read_identity()

    assert identity.version == expected_version
    assert tuple(axis.stage_id for axis in identity.axes) == expected_axis_ids
    assert all(axis.unit == "mm" for axis in identity.axes)
