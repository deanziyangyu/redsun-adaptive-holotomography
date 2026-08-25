"""Typed command payloads for the AHT MCU v1 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from struct import Struct
from zlib import crc32

from redsun_aht.mcu.protocol import (
    MAX_PAYLOAD_BYTES,
    MessageType,
    Packet,
    ProtocolDecodeError,
)

_BEGIN_PATTERN = Struct("<HHBHI")
_PATTERN_CHUNK = Struct("<HHH")
_PATTERN_ID = Struct("<H")
_CAPABILITIES = Struct("<HHHHB")
_ACK = Struct("<B")
_ERROR = Struct("<BH")
_SEQUENCE_HEADER = Struct("<HH")
_SEQUENCE_STEP = Struct("<HIB")
MAX_PATTERN_CHUNK_BYTES = MAX_PAYLOAD_BYTES - _PATTERN_CHUNK.size


class PatternEncoding(IntEnum):
    """Pattern encodings understood by v1 firmware."""

    RAW_U8 = 1


class ErrorCode(IntEnum):
    """Typed MCU command failure categories."""

    MALFORMED_REQUEST = 1
    UNSUPPORTED_COMMAND = 2
    SEQUENCE_CONFLICT = 3
    INVALID_STATE = 4
    PATTERN_NOT_FOUND = 5
    COVERAGE_INCOMPLETE = 6
    CHECKSUM_MISMATCH = 7
    PANEL_REVISION_MISMATCH = 8
    BOUNDS = 9


@dataclass(frozen=True, slots=True)
class Capabilities:
    """MCU identity and bounded pattern capacity."""

    firmware_version: str
    target: str
    panel_profile: str
    panel_revision: int
    led_count: int
    max_pattern_bytes: int
    max_sequence_steps: int
    supported_encodings: int = 1 << PatternEncoding.RAW_U8


@dataclass(frozen=True, slots=True)
class BeginPattern:
    """Declaration that opens a transactional pattern upload."""

    pattern_id: int
    panel_revision: int
    encoding: PatternEncoding
    led_count: int
    checksum: int


@dataclass(frozen=True, slots=True)
class PatternChunk:
    """One bounded region of a staged pattern."""

    pattern_id: int
    offset: int
    values: bytes


@dataclass(frozen=True, slots=True)
class CommandError:
    """Decoded typed error returned by an MCU."""

    request_type: MessageType
    code: ErrorCode
    detail: str


@dataclass(frozen=True, slots=True)
class SequenceStep:
    """Reference to a committed pattern with dwell and trigger metadata."""

    pattern_id: int
    dwell_us: int
    trigger: bool = True


@dataclass(frozen=True, slots=True)
class SequenceDefinition:
    """Bounded sequence of committed pattern references."""

    sequence_id: int
    steps: tuple[SequenceStep, ...]


def pattern_checksum(values: bytes) -> int:
    """Return the unsigned CRC-32 recorded by a pattern declaration."""
    return crc32(values) & 0xFFFFFFFF


def encode_begin_pattern(command: BeginPattern) -> bytes:
    """Encode a begin-pattern declaration."""
    _require_u16("pattern_id", command.pattern_id)
    _require_u16("panel_revision", command.panel_revision)
    _require_u16("led_count", command.led_count)
    if not 0 <= command.checksum <= 0xFFFFFFFF:
        raise ValueError("checksum must fit in an unsigned 32-bit integer")
    return _BEGIN_PATTERN.pack(
        command.pattern_id,
        command.panel_revision,
        command.encoding,
        command.led_count,
        command.checksum,
    )


def decode_begin_pattern(payload: bytes) -> BeginPattern:
    """Decode a begin-pattern declaration."""
    if len(payload) != _BEGIN_PATTERN.size:
        raise ProtocolDecodeError("begin-pattern payload has the wrong size")
    pattern_id, panel_revision, encoding_value, led_count, checksum = (
        _BEGIN_PATTERN.unpack(payload)
    )
    try:
        encoding = PatternEncoding(encoding_value)
    except ValueError as exc:
        raise ProtocolDecodeError("unsupported pattern encoding") from exc
    return BeginPattern(pattern_id, panel_revision, encoding, led_count, checksum)


def encode_pattern_chunk(command: PatternChunk) -> bytes:
    """Encode a pattern chunk including its declared byte count."""
    _require_u16("pattern_id", command.pattern_id)
    _require_u16("offset", command.offset)
    _require_u16("chunk length", len(command.values))
    return (
        _PATTERN_CHUNK.pack(command.pattern_id, command.offset, len(command.values))
        + command.values
    )


def decode_pattern_chunk(payload: bytes) -> PatternChunk:
    """Decode and length-check a pattern chunk."""
    if len(payload) < _PATTERN_CHUNK.size:
        raise ProtocolDecodeError("pattern-chunk payload is truncated")
    pattern_id, offset, count = _PATTERN_CHUNK.unpack(payload[: _PATTERN_CHUNK.size])
    values = payload[_PATTERN_CHUNK.size :]
    if len(values) != count:
        raise ProtocolDecodeError("pattern-chunk byte count does not match payload")
    return PatternChunk(pattern_id, offset, values)


def encode_pattern_id(pattern_id: int) -> bytes:
    """Encode a pattern identifier."""
    _require_u16("pattern_id", pattern_id)
    return _PATTERN_ID.pack(pattern_id)


def decode_pattern_id(payload: bytes) -> int:
    """Decode a pattern identifier."""
    if len(payload) != _PATTERN_ID.size:
        raise ProtocolDecodeError("pattern-id payload has the wrong size")
    return int(_PATTERN_ID.unpack(payload)[0])


def encode_sequence(command: SequenceDefinition) -> bytes:
    """Encode a sequence definition with fixed-width steps."""
    _require_u16("sequence_id", command.sequence_id)
    _require_u16("step count", len(command.steps))
    encoded = bytearray(_SEQUENCE_HEADER.pack(command.sequence_id, len(command.steps)))
    for step in command.steps:
        _require_u16("pattern_id", step.pattern_id)
        if not 0 <= step.dwell_us <= 0xFFFFFFFF:
            raise ValueError("dwell_us must fit in an unsigned 32-bit integer")
        encoded.extend(
            _SEQUENCE_STEP.pack(step.pattern_id, step.dwell_us, step.trigger)
        )
    return bytes(encoded)


def decode_sequence(payload: bytes) -> SequenceDefinition:
    """Decode a sequence and reject size or boolean representation errors."""
    if len(payload) < _SEQUENCE_HEADER.size:
        raise ProtocolDecodeError("sequence payload is truncated")
    sequence_id, step_count = _SEQUENCE_HEADER.unpack(payload[: _SEQUENCE_HEADER.size])
    expected_size = _SEQUENCE_HEADER.size + step_count * _SEQUENCE_STEP.size
    if len(payload) != expected_size:
        raise ProtocolDecodeError("sequence step count does not match payload")
    steps: list[SequenceStep] = []
    offset = _SEQUENCE_HEADER.size
    for _ in range(step_count):
        pattern_id, dwell_us, trigger_value = _SEQUENCE_STEP.unpack(
            payload[offset : offset + _SEQUENCE_STEP.size]
        )
        if trigger_value not in {0, 1}:
            raise ProtocolDecodeError("sequence trigger flag is not boolean")
        steps.append(SequenceStep(pattern_id, dwell_us, bool(trigger_value)))
        offset += _SEQUENCE_STEP.size
    return SequenceDefinition(sequence_id, tuple(steps))


def encode_capabilities(capabilities: Capabilities) -> bytes:
    """Encode bounded capabilities and three length-prefixed identity strings."""
    fixed = _CAPABILITIES.pack(
        capabilities.panel_revision,
        capabilities.led_count,
        capabilities.max_pattern_bytes,
        capabilities.max_sequence_steps,
        capabilities.supported_encodings,
    )
    return fixed + b"".join(
        _encode_short_text(value)
        for value in (
            capabilities.firmware_version,
            capabilities.target,
            capabilities.panel_profile,
        )
    )


def decode_capabilities(payload: bytes) -> Capabilities:
    """Decode a capabilities response body."""
    if len(payload) < _CAPABILITIES.size:
        raise ProtocolDecodeError("capabilities payload is truncated")
    panel_revision, led_count, max_pattern, max_steps, encodings = _CAPABILITIES.unpack(
        payload[: _CAPABILITIES.size]
    )
    remainder = payload[_CAPABILITIES.size :]
    values: list[str] = []
    for _ in range(3):
        value, remainder = _decode_short_text(remainder)
        values.append(value)
    if remainder:
        raise ProtocolDecodeError("capabilities payload has trailing bytes")
    return Capabilities(
        firmware_version=values[0],
        target=values[1],
        panel_profile=values[2],
        panel_revision=panel_revision,
        led_count=led_count,
        max_pattern_bytes=max_pattern,
        max_sequence_steps=max_steps,
        supported_encodings=encodings,
    )


def acknowledge(request: Packet, body: bytes = b"") -> Packet:
    """Build an ACK correlated to one request packet."""
    return Packet(
        message_type=MessageType.ACK,
        sequence=request.sequence,
        payload=_ACK.pack(request.message_type) + body,
    )


def decode_ack(packet: Packet, expected_type: MessageType) -> bytes:
    """Validate an ACK and return its command-specific body."""
    if packet.message_type != MessageType.ACK:
        raise ProtocolDecodeError("response is not an ACK")
    if len(packet.payload) < _ACK.size:
        raise ProtocolDecodeError("ACK payload is truncated")
    request_value = _ACK.unpack(packet.payload[: _ACK.size])[0]
    if request_value != expected_type:
        raise ProtocolDecodeError("ACK identifies a different request type")
    return packet.payload[_ACK.size :]


def error_response(request: Packet, code: ErrorCode, detail: str = "") -> Packet:
    """Build a bounded typed error correlated to one request."""
    return Packet(
        message_type=MessageType.ERROR,
        sequence=request.sequence,
        payload=_ERROR.pack(request.message_type, code) + _encode_short_text(detail),
    )


def decode_error(packet: Packet) -> CommandError:
    """Decode a typed MCU error response."""
    if packet.message_type != MessageType.ERROR:
        raise ProtocolDecodeError("response is not an error")
    if len(packet.payload) < _ERROR.size:
        raise ProtocolDecodeError("error payload is truncated")
    request_value, code_value = _ERROR.unpack(packet.payload[: _ERROR.size])
    detail, remainder = _decode_short_text(packet.payload[_ERROR.size :])
    if remainder:
        raise ProtocolDecodeError("error payload has trailing bytes")
    try:
        return CommandError(
            request_type=MessageType(request_value),
            code=ErrorCode(code_value),
            detail=detail,
        )
    except ValueError as exc:
        raise ProtocolDecodeError("error payload contains an unknown enum") from exc


def _require_u16(name: str, value: int) -> None:
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{name} must fit in an unsigned 16-bit integer")


def _encode_short_text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFF:
        raise ValueError("text field exceeds 255 UTF-8 bytes")
    return bytes((len(encoded),)) + encoded


def _decode_short_text(payload: bytes) -> tuple[str, bytes]:
    if not payload:
        raise ProtocolDecodeError("length-prefixed text is truncated")
    length = payload[0]
    if len(payload) < length + 1:
        raise ProtocolDecodeError("length-prefixed text is truncated")
    try:
        value = payload[1 : length + 1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolDecodeError("text field is not valid UTF-8") from exc
    return value, payload[length + 1 :]
