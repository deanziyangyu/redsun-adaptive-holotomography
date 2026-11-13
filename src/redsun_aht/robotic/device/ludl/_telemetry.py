"""Shared constants for the native Ludl live telemetry payload."""

from __future__ import annotations

LIVE_TELEMETRY_SCHEMA = "redsun.ludl-stage-live-telemetry/v1"
LIVE_TELEMETRY_CLOCK = "python.time.monotonic_ns"
TELEMETRY_TIMESTAMP_SEMANTICS = (
    "host monotonic serial read-call brackets; not UART or exposure time"
)

__all__ = [
    "LIVE_TELEMETRY_CLOCK",
    "LIVE_TELEMETRY_SCHEMA",
    "TELEMETRY_TIMESTAMP_SEMANTICS",
]
