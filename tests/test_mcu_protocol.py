from __future__ import annotations

import pytest

from redsun_aht.mcu import (
    MAX_PAYLOAD_BYTES,
    ChecksumMismatchError,
    MessageType,
    Packet,
    PacketFlag,
    ProtocolDecodeError,
    cobs_decode,
    cobs_encode,
    crc16_ccitt,
    decode_packet,
    encode_packet,
)


@pytest.mark.parametrize(
    "payload",
    [b"", b"\x00", b"\x00\x00", b"abc", bytes(range(256)), b"x" * 512],
)
def test_cobs_round_trip(payload: bytes) -> None:
    encoded = cobs_encode(payload)

    assert encoded
    assert b"\x00" not in encoded
    assert cobs_decode(encoded) == payload


def test_crc16_ccitt_standard_check_value() -> None:
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_packet_golden_vector_and_round_trip() -> None:
    packet = Packet(
        message_type=MessageType.BEGIN_PATTERN,
        flags=PacketFlag.RESPONSE_REQUIRED,
        sequence=0x1234,
        payload=b"\x00\x01\xff",
    )

    frame = encode_packet(packet)

    assert frame.hex() == "07011101341203010501ffeebb00"
    assert decode_packet(frame) == packet


def test_packet_rejects_checksum_corruption() -> None:
    frame = bytearray(
        encode_packet(Packet(message_type=MessageType.ALL_OFF, sequence=7))
    )
    frame[-2] ^= 0x01

    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        decode_packet(bytes(frame))


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (b"\x01", "missing its zero delimiter"),
        (b"\x00", "COBS payload is empty"),
        (b"\x01\x00\x01\x00", "more than one packet"),
        (b"\x02\x01\x00", "shorter than its envelope"),
        (b"\x05\x01\x02\x00", "COBS block exceeds payload"),
    ],
)
def test_packet_rejects_malformed_frames(frame: bytes, message: str) -> None:
    with pytest.raises(ProtocolDecodeError, match=message):
        decode_packet(frame)


def test_packet_validation_is_bounded() -> None:
    with pytest.raises(ValueError, match="unsigned 16-bit"):
        Packet(message_type=MessageType.STOP, sequence=0x10000)
    with pytest.raises(ValueError, match="payload exceeds"):
        Packet(
            message_type=MessageType.WRITE_PATTERN_CHUNK,
            sequence=1,
            payload=b"x" * (MAX_PAYLOAD_BYTES + 1),
        )
    with pytest.raises(ValueError, match="unknown packet flags"):
        Packet(message_type=MessageType.STOP, sequence=1, flags=PacketFlag(0x80))
