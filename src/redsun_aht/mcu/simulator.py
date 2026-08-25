"""Deterministic in-process model of the AHT MCU command boundary."""

from __future__ import annotations

from dataclasses import dataclass, field

from redsun_aht.mcu.commands import (
    BeginPattern,
    Capabilities,
    ErrorCode,
    SequenceDefinition,
    acknowledge,
    decode_begin_pattern,
    decode_pattern_chunk,
    decode_pattern_id,
    decode_sequence,
    encode_capabilities,
    error_response,
    pattern_checksum,
)
from redsun_aht.mcu.protocol import (
    MessageType,
    Packet,
    ProtocolDecodeError,
    decode_packet,
    encode_packet,
)


@dataclass(slots=True)
class _StagedPattern:
    declaration: BeginPattern
    values: bytearray
    written: bytearray


@dataclass(slots=True)
class SimulatedMcu:
    """Execute v1 pattern commands with transactional and retry semantics."""

    capabilities: Capabilities
    committed_patterns: dict[int, bytes] = field(default_factory=dict)
    sequences: dict[int, SequenceDefinition] = field(default_factory=dict)
    displayed_pattern_id: int | None = None
    armed_sequence_id: int | None = None
    running: bool = False
    display_count: int = 0
    start_count: int = 0
    _staged: _StagedPattern | None = None
    _responses: dict[int, tuple[bytes, Packet]] = field(default_factory=dict)

    def handle(self, request: Packet) -> Packet:
        """Return one correlated response without repeating duplicate effects."""
        fingerprint = bytes((request.message_type,)) + request.payload
        previous = self._responses.get(request.sequence)
        if previous is not None:
            previous_fingerprint, response = previous
            if fingerprint == previous_fingerprint:
                return response
            return error_response(
                request,
                ErrorCode.SEQUENCE_CONFLICT,
                "sequence was already used for a different request",
            )

        try:
            response = self._execute(request)
        except ProtocolDecodeError as exc:
            response = error_response(request, ErrorCode.MALFORMED_REQUEST, str(exc))
        self._responses[request.sequence] = (fingerprint, response)
        return response

    def _execute(self, request: Packet) -> Packet:
        if request.message_type == MessageType.GET_CAPABILITIES:
            if request.payload:
                raise ProtocolDecodeError("GET_CAPABILITIES payload must be empty")
            return acknowledge(request, encode_capabilities(self.capabilities))
        if request.message_type == MessageType.BEGIN_PATTERN:
            return self._begin_pattern(request)
        if request.message_type == MessageType.WRITE_PATTERN_CHUNK:
            return self._write_pattern_chunk(request)
        if request.message_type == MessageType.COMMIT_PATTERN:
            return self._commit_pattern(request)
        if request.message_type == MessageType.SHOW_PATTERN:
            return self._show_pattern(request)
        if request.message_type == MessageType.DEFINE_SEQUENCE:
            return self._define_sequence(request)
        if request.message_type == MessageType.ARM_SEQUENCE:
            return self._arm_sequence(request)
        if request.message_type == MessageType.START_SEQUENCE:
            if request.payload:
                raise ProtocolDecodeError("START_SEQUENCE payload must be empty")
            if self.armed_sequence_id is None:
                return error_response(
                    request, ErrorCode.INVALID_STATE, "no sequence is armed"
                )
            self.running = True
            self.start_count += 1
            return acknowledge(request)
        if request.message_type in {MessageType.ALL_OFF, MessageType.STOP}:
            if request.payload:
                raise ProtocolDecodeError("stop command payload must be empty")
            self.running = False
            if request.message_type == MessageType.ALL_OFF:
                self.displayed_pattern_id = None
            return acknowledge(request)
        return error_response(
            request, ErrorCode.UNSUPPORTED_COMMAND, "command is not implemented"
        )

    def _begin_pattern(self, request: Packet) -> Packet:
        declaration = decode_begin_pattern(request.payload)
        if declaration.panel_revision != self.capabilities.panel_revision:
            return error_response(
                request,
                ErrorCode.PANEL_REVISION_MISMATCH,
                "pattern targets a different panel revision",
            )
        encoding_bit = 1 << declaration.encoding
        if not self.capabilities.supported_encodings & encoding_bit:
            return error_response(
                request, ErrorCode.UNSUPPORTED_COMMAND, "encoding is unavailable"
            )
        if declaration.led_count != self.capabilities.led_count:
            return error_response(
                request, ErrorCode.BOUNDS, "pattern LED count does not match panel"
            )
        if declaration.led_count > self.capabilities.max_pattern_bytes:
            return error_response(
                request, ErrorCode.BOUNDS, "pattern exceeds declared capacity"
            )
        self._staged = _StagedPattern(
            declaration=declaration,
            values=bytearray(declaration.led_count),
            written=bytearray(declaration.led_count),
        )
        return acknowledge(request)

    def _write_pattern_chunk(self, request: Packet) -> Packet:
        chunk = decode_pattern_chunk(request.payload)
        staged = self._staged
        if staged is None or chunk.pattern_id != staged.declaration.pattern_id:
            return error_response(
                request, ErrorCode.INVALID_STATE, "pattern is not being staged"
            )
        end = chunk.offset + len(chunk.values)
        if end > len(staged.values):
            return error_response(
                request, ErrorCode.BOUNDS, "pattern chunk exceeds declared length"
            )
        staged.values[chunk.offset : end] = chunk.values
        staged.written[chunk.offset : end] = b"\x01" * len(chunk.values)
        return acknowledge(request)

    def _commit_pattern(self, request: Packet) -> Packet:
        pattern_id = decode_pattern_id(request.payload)
        staged = self._staged
        if staged is None or pattern_id != staged.declaration.pattern_id:
            return error_response(
                request, ErrorCode.INVALID_STATE, "pattern is not being staged"
            )
        if not all(staged.written):
            return error_response(
                request, ErrorCode.COVERAGE_INCOMPLETE, "pattern has unwritten bytes"
            )
        values = bytes(staged.values)
        if pattern_checksum(values) != staged.declaration.checksum:
            return error_response(
                request, ErrorCode.CHECKSUM_MISMATCH, "pattern checksum differs"
            )
        self.committed_patterns[pattern_id] = values
        self._staged = None
        return acknowledge(request)

    def _show_pattern(self, request: Packet) -> Packet:
        pattern_id = decode_pattern_id(request.payload)
        if pattern_id not in self.committed_patterns:
            return error_response(
                request, ErrorCode.PATTERN_NOT_FOUND, "pattern is not committed"
            )
        self.displayed_pattern_id = pattern_id
        self.display_count += 1
        return acknowledge(request)

    def _define_sequence(self, request: Packet) -> Packet:
        sequence = decode_sequence(request.payload)
        if not sequence.steps:
            return error_response(
                request, ErrorCode.BOUNDS, "sequence must contain at least one step"
            )
        if len(sequence.steps) > self.capabilities.max_sequence_steps:
            return error_response(
                request, ErrorCode.BOUNDS, "sequence exceeds declared capacity"
            )
        missing = [
            step.pattern_id
            for step in sequence.steps
            if step.pattern_id not in self.committed_patterns
        ]
        if missing:
            return error_response(
                request,
                ErrorCode.PATTERN_NOT_FOUND,
                f"sequence references uncommitted pattern {missing[0]}",
            )
        self.sequences[sequence.sequence_id] = sequence
        return acknowledge(request)

    def _arm_sequence(self, request: Packet) -> Packet:
        sequence_id = decode_pattern_id(request.payload)
        if sequence_id not in self.sequences:
            return error_response(
                request, ErrorCode.INVALID_STATE, "sequence is not defined"
            )
        self.armed_sequence_id = sequence_id
        return acknowledge(request)


@dataclass(slots=True)
class SimulatedMcuTransport:
    """Frame transport that connects a host client to :class:`SimulatedMcu`."""

    endpoint: SimulatedMcu

    def exchange(self, frame: bytes) -> bytes:
        """Decode, execute, and encode one complete frame."""
        return encode_packet(self.endpoint.handle(decode_packet(frame)))
