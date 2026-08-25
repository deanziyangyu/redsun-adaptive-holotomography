from __future__ import annotations

from dataclasses import dataclass

import pytest

from redsun_aht.device.mcu import McuCommandFailure, McuHostClient
from redsun_aht.mcu import (
    MAX_PATTERN_CHUNK_BYTES,
    MessageType,
    Packet,
    PacketFlag,
    SimulatedMcu,
    SimulatedMcuTransport,
)
from redsun_aht.mcu.commands import (
    BeginPattern,
    Capabilities,
    ErrorCode,
    PatternChunk,
    PatternEncoding,
    SequenceDefinition,
    SequenceStep,
    decode_ack,
    decode_error,
    encode_begin_pattern,
    encode_pattern_chunk,
    encode_pattern_id,
    encode_sequence,
    pattern_checksum,
)


@pytest.fixture
def capabilities() -> Capabilities:
    return Capabilities(
        firmware_version="1.0.0-sim",
        target="simulator",
        panel_profile="aht-61",
        panel_revision=3,
        led_count=8,
        max_pattern_bytes=8,
        max_sequence_steps=32,
    )


def test_host_upload_show_and_all_off(capabilities: Capabilities) -> None:
    endpoint = SimulatedMcu(capabilities)
    host = McuHostClient(SimulatedMcuTransport(endpoint), chunk_bytes=3)
    values = bytes((0, 32, 64, 96, 128, 160, 192, 255))

    assert host.get_capabilities() == capabilities
    host.upload_pattern(17, values)
    host.show_pattern(17)

    assert endpoint.committed_patterns == {17: values}
    assert endpoint.displayed_pattern_id == 17
    assert endpoint.display_count == 1

    host.all_off()
    assert endpoint.displayed_pattern_id is None


def test_commit_is_atomic_until_coverage_and_checksum_pass(
    capabilities: Capabilities,
) -> None:
    endpoint = SimulatedMcu(capabilities)
    values = bytes(range(8))
    begin = Packet(
        message_type=MessageType.BEGIN_PATTERN,
        sequence=1,
        payload=encode_begin_pattern(
            BeginPattern(
                pattern_id=9,
                panel_revision=3,
                encoding=PatternEncoding.RAW_U8,
                led_count=8,
                checksum=pattern_checksum(values),
            )
        ),
    )
    decode_ack(endpoint.handle(begin), MessageType.BEGIN_PATTERN)
    decode_ack(
        endpoint.handle(
            Packet(
                message_type=MessageType.WRITE_PATTERN_CHUNK,
                sequence=2,
                payload=encode_pattern_chunk(PatternChunk(9, 0, values[:4])),
            )
        ),
        MessageType.WRITE_PATTERN_CHUNK,
    )

    incomplete = endpoint.handle(
        Packet(
            message_type=MessageType.COMMIT_PATTERN,
            sequence=3,
            payload=encode_pattern_id(9),
        )
    )

    assert decode_error(incomplete).code == ErrorCode.COVERAGE_INCOMPLETE
    assert endpoint.committed_patterns == {}


def test_duplicate_retry_does_not_repeat_display_side_effect(
    capabilities: Capabilities,
) -> None:
    endpoint = SimulatedMcu(capabilities, committed_patterns={5: b"\x00" * 8})
    first = Packet(
        message_type=MessageType.SHOW_PATTERN,
        sequence=42,
        payload=encode_pattern_id(5),
        flags=PacketFlag.RESPONSE_REQUIRED,
    )
    retry = Packet(
        message_type=MessageType.SHOW_PATTERN,
        sequence=42,
        payload=encode_pattern_id(5),
        flags=PacketFlag.RESPONSE_REQUIRED | PacketFlag.RETRY,
    )

    assert endpoint.handle(retry) == endpoint.handle(first)
    assert endpoint.display_count == 1


def test_reused_sequence_with_different_payload_is_rejected(
    capabilities: Capabilities,
) -> None:
    endpoint = SimulatedMcu(capabilities, committed_patterns={5: b"\x00" * 8})
    first = Packet(
        message_type=MessageType.SHOW_PATTERN,
        sequence=42,
        payload=encode_pattern_id(5),
    )
    second = Packet(
        message_type=MessageType.SHOW_PATTERN,
        sequence=42,
        payload=encode_pattern_id(6),
    )

    decode_ack(endpoint.handle(first), MessageType.SHOW_PATTERN)
    assert decode_error(endpoint.handle(second)).code == ErrorCode.SEQUENCE_CONFLICT


def test_panel_revision_mismatch_is_typed(capabilities: Capabilities) -> None:
    endpoint = SimulatedMcu(capabilities)
    request = Packet(
        message_type=MessageType.BEGIN_PATTERN,
        sequence=1,
        payload=encode_begin_pattern(BeginPattern(1, 99, PatternEncoding.RAW_U8, 8, 0)),
    )

    assert decode_error(endpoint.handle(request)).code == (
        ErrorCode.PANEL_REVISION_MISMATCH
    )


@dataclass
class _DropFirstResponse:
    endpoint: SimulatedMcu
    calls: int = 0

    def exchange(self, frame: bytes) -> bytes:
        from redsun_aht.mcu import decode_packet, encode_packet

        self.calls += 1
        response = self.endpoint.handle(decode_packet(frame))
        if self.calls == 1:
            raise TimeoutError("simulated lost ACK")
        return encode_packet(response)


def test_host_retries_same_sequence_after_lost_response(
    capabilities: Capabilities,
) -> None:
    endpoint = SimulatedMcu(capabilities, committed_patterns={5: b"\x00" * 8})
    transport = _DropFirstResponse(endpoint)
    host = McuHostClient(transport)

    host.show_pattern(5)

    assert transport.calls == 2
    assert endpoint.display_count == 1


def test_host_surfaces_typed_command_error(capabilities: Capabilities) -> None:
    host = McuHostClient(SimulatedMcuTransport(SimulatedMcu(capabilities)))

    with pytest.raises(McuCommandFailure) as caught:
        host.show_pattern(404)

    assert caught.value.error.code == ErrorCode.PATTERN_NOT_FOUND


def test_sequence_references_committed_patterns_and_starts_once(
    capabilities: Capabilities,
) -> None:
    endpoint = SimulatedMcu(capabilities, committed_patterns={5: b"\x00" * 8})
    host = McuHostClient(SimulatedMcuTransport(endpoint))

    host.define_sequence(2, (SequenceStep(5, dwell_us=250, trigger=True),))
    host.arm_sequence(2)
    host.start_sequence()

    assert endpoint.armed_sequence_id == 2
    assert endpoint.running
    assert endpoint.start_count == 1

    host.stop()
    assert not endpoint.running


def test_host_chunk_size_is_bounded(capabilities: Capabilities) -> None:
    transport = SimulatedMcuTransport(SimulatedMcu(capabilities))

    with pytest.raises(ValueError, match="chunk_bytes"):
        McuHostClient(transport, chunk_bytes=MAX_PATTERN_CHUNK_BYTES + 1)

    with pytest.raises(ValueError, match="retry_count"):
        McuHostClient(transport, retry_count=-1)


@pytest.mark.parametrize(
    ("message_type", "payload", "expected"),
    [
        (MessageType.GET_CAPABILITIES, b"unexpected", ErrorCode.MALFORMED_REQUEST),
        (MessageType.START_SEQUENCE, b"", ErrorCode.INVALID_STATE),
        (MessageType.HEARTBEAT, b"", ErrorCode.UNSUPPORTED_COMMAND),
        (
            MessageType.WRITE_PATTERN_CHUNK,
            encode_pattern_chunk(PatternChunk(1, 0, b"x")),
            ErrorCode.INVALID_STATE,
        ),
        (MessageType.ARM_SEQUENCE, encode_pattern_id(7), ErrorCode.INVALID_STATE),
        (
            MessageType.DEFINE_SEQUENCE,
            encode_sequence(SequenceDefinition(1, (SequenceStep(7, 10),))),
            ErrorCode.PATTERN_NOT_FOUND,
        ),
    ],
)
def test_simulator_returns_typed_state_and_payload_errors(
    capabilities: Capabilities,
    message_type: MessageType,
    payload: bytes,
    expected: ErrorCode,
) -> None:
    endpoint = SimulatedMcu(capabilities)

    response = endpoint.handle(
        Packet(message_type=message_type, sequence=1, payload=payload)
    )

    assert decode_error(response).code == expected
