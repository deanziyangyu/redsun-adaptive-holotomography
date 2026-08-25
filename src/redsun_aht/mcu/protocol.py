"""Bounded v1 wire framing for the AHT MCU host protocol.

The dense MCU data plane uses a fixed little-endian envelope protected by a
CRC-16/CCITT-FALSE checksum. Each envelope is COBS encoded and terminated by a
zero byte so a serial reader can recover packet boundaries after corruption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from struct import Struct

PROTOCOL_VERSION = 1
MAX_PAYLOAD_BYTES = 4096

_HEADER = Struct("<BBBHH")
_CHECKSUM = Struct("<H")
_MIN_PACKET_BYTES = _HEADER.size + _CHECKSUM.size


class MessageType(IntEnum):
    """Message identifiers reserved by the AHT v1 host protocol."""

    GET_CAPABILITIES = 0x01
    HEARTBEAT = 0x02

    ALL_OFF = 0x10
    BEGIN_PATTERN = 0x11
    WRITE_PATTERN_CHUNK = 0x12
    COMMIT_PATTERN = 0x13
    SHOW_PATTERN = 0x14
    DEFINE_SEQUENCE = 0x15
    ARM_SEQUENCE = 0x16
    START_SEQUENCE = 0x17
    STOP = 0x18

    ACK = 0x70
    ERROR = 0x71
    TELEMETRY = 0x72


class PacketFlag(IntFlag):
    """Flags that qualify a request or response packet."""

    NONE = 0
    RESPONSE_REQUIRED = 1 << 0
    RETRY = 1 << 1
    UNSOLICITED = 1 << 2


_KNOWN_FLAGS = PacketFlag.RESPONSE_REQUIRED | PacketFlag.RETRY | PacketFlag.UNSOLICITED


class ProtocolDecodeError(ValueError):
    """Base error for a malformed MCU wire frame."""


class ChecksumMismatchError(ProtocolDecodeError):
    """Raised when a packet's declared and calculated checksums differ."""


@dataclass(frozen=True, slots=True)
class Packet:
    """Decoded MCU packet before command-specific payload interpretation."""

    message_type: MessageType
    sequence: int
    payload: bytes = b""
    flags: PacketFlag = PacketFlag.NONE
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        """Validate envelope fields before encoding."""
        if self.protocol_version != PROTOCOL_VERSION:
            msg = f"unsupported protocol version: {self.protocol_version}"
            raise ValueError(msg)
        if not 0 <= self.sequence <= 0xFFFF:
            raise ValueError("sequence must fit in an unsigned 16-bit integer")
        if len(self.payload) > MAX_PAYLOAD_BYTES:
            msg = f"payload exceeds {MAX_PAYLOAD_BYTES} bytes"
            raise ValueError(msg)
        unknown_flags = int(self.flags) & ~int(_KNOWN_FLAGS)
        if unknown_flags:
            msg = f"unknown packet flags: 0x{unknown_flags:02x}"
            raise ValueError(msg)


def crc16_ccitt(data: bytes) -> int:
    """Return the CRC-16/CCITT-FALSE checksum for *data*."""
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else crc << 1
    return crc


def cobs_encode(data: bytes) -> bytes:
    """Encode one payload with Consistent Overhead Byte Stuffing."""
    encoded = bytearray(b"\x00")
    code_index = 0
    code = 1

    for value in data:
        if value == 0:
            encoded[code_index] = code
            code_index = len(encoded)
            encoded.append(0)
            code = 1
        else:
            encoded.append(value)
            code += 1
            if code == 0xFF:
                encoded[code_index] = code
                code_index = len(encoded)
                encoded.append(0)
                code = 1

    encoded[code_index] = code
    return bytes(encoded)


def cobs_decode(data: bytes) -> bytes:
    """Decode one COBS payload without its trailing frame delimiter."""
    if not data:
        raise ProtocolDecodeError("COBS payload is empty")
    if 0 in data:
        raise ProtocolDecodeError("COBS payload contains a zero byte")

    decoded = bytearray()
    index = 0
    while index < len(data):
        code = data[index]
        index += 1
        block_end = index + code - 1
        if block_end > len(data):
            raise ProtocolDecodeError("COBS block exceeds payload")
        decoded.extend(data[index:block_end])
        index = block_end
        if code != 0xFF and index < len(data):
            decoded.append(0)
    return bytes(decoded)


def encode_packet(packet: Packet) -> bytes:
    """Encode one packet as a delimiter-terminated serial frame."""
    header = _HEADER.pack(
        packet.protocol_version,
        packet.message_type,
        packet.flags,
        packet.sequence,
        len(packet.payload),
    )
    body = header + packet.payload
    decoded = body + _CHECKSUM.pack(crc16_ccitt(body))
    return cobs_encode(decoded) + b"\x00"


def decode_packet(frame: bytes) -> Packet:
    """Decode exactly one delimiter-terminated serial frame."""
    if not frame.endswith(b"\x00"):
        raise ProtocolDecodeError("frame is missing its zero delimiter")
    encoded = frame[:-1]
    if b"\x00" in encoded:
        raise ProtocolDecodeError("frame contains more than one packet")
    decoded = cobs_decode(encoded)
    if len(decoded) < _MIN_PACKET_BYTES:
        raise ProtocolDecodeError("decoded packet is shorter than its envelope")

    body = decoded[: -_CHECKSUM.size]
    declared_checksum = _CHECKSUM.unpack(decoded[-_CHECKSUM.size :])[0]
    calculated_checksum = crc16_ccitt(body)
    if declared_checksum != calculated_checksum:
        msg = (
            f"checksum mismatch: declared 0x{declared_checksum:04x}, "
            f"calculated 0x{calculated_checksum:04x}"
        )
        raise ChecksumMismatchError(msg)

    version, message_value, flags_value, sequence, payload_length = _HEADER.unpack(
        body[: _HEADER.size]
    )
    payload = body[_HEADER.size :]
    if payload_length != len(payload):
        msg = f"payload length is {len(payload)} bytes, expected {payload_length}"
        raise ProtocolDecodeError(msg)
    if payload_length > MAX_PAYLOAD_BYTES:
        msg = f"payload exceeds {MAX_PAYLOAD_BYTES} bytes"
        raise ProtocolDecodeError(msg)

    try:
        message_type = MessageType(message_value)
    except ValueError as exc:
        msg = f"unknown message type: 0x{message_value:02x}"
        raise ProtocolDecodeError(msg) from exc
    try:
        flags = PacketFlag(flags_value)
        return Packet(
            protocol_version=version,
            message_type=message_type,
            flags=flags,
            sequence=sequence,
            payload=payload,
        )
    except ValueError as exc:
        raise ProtocolDecodeError(str(exc)) from exc
