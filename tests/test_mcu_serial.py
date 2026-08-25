from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from redsun_aht.device.mcu import DelimitedSerialTransport


@dataclass
class _FakeSerial:
    response: bytes
    written: list[bytes] = field(default_factory=list)
    flushed: bool = False
    short_write: bool = False

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data) - 1 if self.short_write else len(data)

    def flush(self) -> None:
        self.flushed = True

    def read_until(self, expected: bytes, size: int | None = None) -> bytes:
        assert expected == b"\x00"
        assert size == 64
        return self.response


def test_serial_transport_exchanges_one_delimited_frame() -> None:
    port = _FakeSerial(response=b"response\x00")
    transport = DelimitedSerialTransport(port, max_response_bytes=64)

    assert transport.exchange(b"request\x00") == b"response\x00"
    assert port.written == [b"request\x00"]
    assert port.flushed


def test_serial_transport_rejects_timeout_and_partial_write() -> None:
    timeout_port = _FakeSerial(response=b"partial")
    with pytest.raises(TimeoutError, match="complete MCU response"):
        DelimitedSerialTransport(timeout_port, max_response_bytes=64).exchange(
            b"request\x00"
        )

    short_port = _FakeSerial(response=b"unused", short_write=True)
    with pytest.raises(OSError, match="short serial write"):
        DelimitedSerialTransport(short_port, max_response_bytes=64).exchange(
            b"request\x00"
        )


@pytest.mark.parametrize("frame", [b"", b"missing", b"two\x00frames\x00"])
def test_serial_transport_rejects_non_single_frames(frame: bytes) -> None:
    transport = DelimitedSerialTransport(
        _FakeSerial(response=b"unused"), max_response_bytes=64
    )

    with pytest.raises(ValueError, match="exactly one"):
        transport.exchange(frame)
