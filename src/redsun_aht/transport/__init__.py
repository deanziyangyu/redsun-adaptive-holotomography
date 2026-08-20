"""Versioned transport-neutral service messages."""

from .envelope import SCHEMA_VERSION, ServiceEnvelope, decode_envelope, encode_envelope

__all__ = ["SCHEMA_VERSION", "ServiceEnvelope", "decode_envelope", "encode_envelope"]
