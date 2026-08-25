"""Transport-facing host adapter for the semantic AHT MCU protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from redsun_aht.mcu.commands import (
    MAX_PATTERN_CHUNK_BYTES,
    BeginPattern,
    Capabilities,
    CommandError,
    PatternChunk,
    PatternEncoding,
    SequenceDefinition,
    SequenceStep,
    decode_ack,
    decode_capabilities,
    decode_error,
    encode_begin_pattern,
    encode_pattern_chunk,
    encode_pattern_id,
    encode_sequence,
    pattern_checksum,
)
from redsun_aht.mcu.protocol import (
    MessageType,
    Packet,
    PacketFlag,
    decode_packet,
    encode_packet,
)


class FrameTransport(Protocol):
    """Exchange one complete serial frame with an MCU endpoint."""

    def exchange(self, frame: bytes) -> bytes:
        """Write one frame and return its correlated response frame."""


class McuCommandFailure(RuntimeError):
    """Typed command rejection returned by the MCU endpoint."""

    def __init__(self, error: CommandError) -> None:
        self.error = error
        detail = f": {error.detail}" if error.detail else ""
        message = f"MCU rejected {error.request_type.name}: {error.code.name}{detail}"
        super().__init__(message)


@dataclass(slots=True)
class McuHostClient:
    """Typed host API independent of a concrete serial implementation."""

    transport: FrameTransport
    retry_count: int = 1
    chunk_bytes: int = 512
    _next_sequence: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        """Validate local transport policy."""
        if self.retry_count < 0:
            raise ValueError("retry_count cannot be negative")
        if not 1 <= self.chunk_bytes <= MAX_PATTERN_CHUNK_BYTES:
            msg = f"chunk_bytes must be in the range 1..{MAX_PATTERN_CHUNK_BYTES}"
            raise ValueError(msg)

    def get_capabilities(self) -> Capabilities:
        """Read immutable endpoint and panel capabilities."""
        response = self._request(MessageType.GET_CAPABILITIES)
        body = decode_ack(response, MessageType.GET_CAPABILITIES)
        return decode_capabilities(body)

    def upload_pattern(self, pattern_id: int, values: bytes) -> None:
        """Stage chunks and atomically commit one raw 8-bit LED map."""
        capabilities = self.get_capabilities()
        declaration = BeginPattern(
            pattern_id=pattern_id,
            panel_revision=capabilities.panel_revision,
            encoding=PatternEncoding.RAW_U8,
            led_count=len(values),
            checksum=pattern_checksum(values),
        )
        self._expect_ack(MessageType.BEGIN_PATTERN, encode_begin_pattern(declaration))
        for offset in range(0, len(values), self.chunk_bytes):
            chunk = PatternChunk(
                pattern_id=pattern_id,
                offset=offset,
                values=values[offset : offset + self.chunk_bytes],
            )
            self._expect_ack(
                MessageType.WRITE_PATTERN_CHUNK, encode_pattern_chunk(chunk)
            )
        self._expect_ack(MessageType.COMMIT_PATTERN, encode_pattern_id(pattern_id))

    def show_pattern(self, pattern_id: int) -> None:
        """Atomically select a previously committed pattern."""
        self._expect_ack(MessageType.SHOW_PATTERN, encode_pattern_id(pattern_id))

    def define_sequence(
        self, sequence_id: int, steps: tuple[SequenceStep, ...]
    ) -> None:
        """Define a bounded sequence that references committed patterns."""
        payload = encode_sequence(SequenceDefinition(sequence_id, steps))
        self._expect_ack(MessageType.DEFINE_SEQUENCE, payload)

    def arm_sequence(self, sequence_id: int) -> None:
        """Prepare a defined sequence without starting execution."""
        self._expect_ack(MessageType.ARM_SEQUENCE, encode_pattern_id(sequence_id))

    def start_sequence(self) -> None:
        """Start the armed sequence."""
        self._expect_ack(MessageType.START_SEQUENCE)

    def stop(self) -> None:
        """Stop active sequence execution without changing stored patterns."""
        self._expect_ack(MessageType.STOP)

    def all_off(self) -> None:
        """Stop execution and clear the currently displayed pattern."""
        self._expect_ack(MessageType.ALL_OFF)

    def _expect_ack(self, message_type: MessageType, payload: bytes = b"") -> None:
        response = self._request(message_type, payload)
        body = decode_ack(response, message_type)
        if body:
            raise RuntimeError(f"unexpected ACK body for {message_type.name}")

    def _request(self, message_type: MessageType, payload: bytes = b"") -> Packet:
        sequence = self._allocate_sequence()
        for attempt in range(self.retry_count + 1):
            flags = PacketFlag.RESPONSE_REQUIRED
            if attempt:
                flags |= PacketFlag.RETRY
            request = Packet(
                message_type=message_type,
                sequence=sequence,
                payload=payload,
                flags=flags,
            )
            try:
                response_frame = self.transport.exchange(encode_packet(request))
                response = decode_packet(response_frame)
            except TimeoutError:
                if attempt == self.retry_count:
                    raise
                continue
            if response.sequence != sequence:
                raise RuntimeError("MCU response sequence does not match request")
            if response.message_type == MessageType.ERROR:
                raise McuCommandFailure(decode_error(response))
            return response
        raise AssertionError("retry loop completed without a response")

    def _allocate_sequence(self) -> int:
        if self._next_sequence > 0xFFFF:
            raise RuntimeError("MCU sequence space exhausted; reconnect the transport")
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence
