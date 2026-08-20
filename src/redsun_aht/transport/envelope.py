"""MsgPack codec for the shared AHT service envelope."""

from typing import Any

import msgspec

SCHEMA_VERSION = 1


class ServiceEnvelope(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Transport-neutral, correlation-aware service message."""

    message_type: str
    producer: str
    timestamp_ns: int
    payload: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    correlation_id: str | None = None
    run_id: str | None = None
    frame_id: str | None = None
    error: str | None = None


_DECODER = msgspec.msgpack.Decoder(type=ServiceEnvelope)


def encode_envelope(envelope: ServiceEnvelope) -> bytes:
    """Encode one envelope without coupling callers to MsgPack APIs."""
    if envelope.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {envelope.schema_version}")
    return msgspec.msgpack.encode(envelope)


def decode_envelope(data: bytes) -> ServiceEnvelope:
    """Decode and reject unsupported envelope schema versions."""
    envelope = _DECODER.decode(data)
    if envelope.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {envelope.schema_version}")
    return envelope
