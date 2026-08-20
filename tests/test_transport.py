from __future__ import annotations

import pytest

from redsun_aht.transport import ServiceEnvelope, decode_envelope, encode_envelope


def test_service_envelope_msgpack_round_trip() -> None:
    envelope = ServiceEnvelope(
        message_type="frame.committed",
        producer="mock-camera",
        timestamp_ns=123,
        correlation_id="command-1",
        run_id="run-1",
        frame_id="frame-1",
        payload={"shape": [8, 16], "units": "count"},
    )

    assert decode_envelope(encode_envelope(envelope)) == envelope


def test_service_envelope_rejects_unknown_version() -> None:
    envelope = ServiceEnvelope(
        schema_version=2,
        message_type="future",
        producer="test",
        timestamp_ns=123,
        payload={},
    )

    with pytest.raises(ValueError, match="unsupported schema version"):
        encode_envelope(envelope)
