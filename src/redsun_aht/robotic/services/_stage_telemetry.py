"""Bounded PVA batches for sampled stage-position observations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Any

from fastcs.attributes import AttributeIO, AttributeIORef, AttrR

from redsun_aht.robotic.device.ludl._telemetry import (
    LIVE_TELEMETRY_CLOCK,
    LIVE_TELEMETRY_SCHEMA,
    TELEMETRY_TIMESTAMP_SEMANTICS,
)

LIVE_TELEMETRY_PERIOD = float(os.environ.get("ASI_TELEMETRY_PERIOD_S", "0.02"))
LIVE_TELEMETRY_CAPACITY = 512

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRef(AttributeIORef):
    """Select the bounded live stage-position batch signal."""


class StageTelemetryIO(AttributeIO[str, TelemetryRef]):
    """Publish sampled stage observations over a bounded JSON PV."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: deque[dict[str, Any]] = deque()
        self._sequence = 0
        self._overflow = False
        self._run_id: str | None = None
        self._source = "Ludl stage telemetry"
        self._schema = LIVE_TELEMETRY_SCHEMA
        self._timestamp_semantics = TELEMETRY_TIMESTAMP_SEMANTICS
        self._axis_device_numbers: tuple[int, int] | None = None
        self._calibration_metadata: dict[str, dict[str, int | float]] | None = None
        self._attr: AttrR[str, TelemetryRef] | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._flush_lock = asyncio.Lock()

    def start_run(
        self,
        run_id: str,
        *,
        source: str = "Ludl stage telemetry",
        schema: str = LIVE_TELEMETRY_SCHEMA,
        timestamp_semantics: str = TELEMETRY_TIMESTAMP_SEMANTICS,
        axis_device_numbers: tuple[int, int] | None = None,
        calibration_metadata: dict[str, dict[str, int | float]] | None = None,
    ) -> None:
        """Set provenance for samples from one service connection."""
        self._pending.clear()
        self._sequence = 0
        self._overflow = False
        self._run_id = run_id
        self._source = source
        self._schema = schema
        self._timestamp_semantics = timestamp_semantics
        self._axis_device_numbers = axis_device_numbers
        self._calibration_metadata = calibration_metadata

    def publish(self, record: dict[str, Any]) -> None:
        """Queue one report without blocking the stage communication path."""
        if len(self._pending) >= LIVE_TELEMETRY_CAPACITY:
            self._overflow = True
            return
        self._sequence += 1
        self._pending.append({"sample_id": self._sequence, **record})
        self._ensure_flusher()

    def _ensure_flusher(self) -> None:
        """Start an independent flush task.

        PVA updates then do not depend on the controller's command/update
        serialisation.
        """
        if self._attr is None:
            return
        if self._flush_task is not None and not self._flush_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._flush_task = loop.create_task(
            self._flush_loop(), name="stage-telemetry-flusher"
        )

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(LIVE_TELEMETRY_PERIOD)
            attr = self._attr
            if attr is None or not self._pending:
                continue
            try:
                await self._publish_pending(attr)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("stage telemetry flush failed")

    async def _publish_pending(self, attr: AttrR[str, TelemetryRef]) -> None:
        async with self._flush_lock:
            count = len(self._pending)
            if count == 0:
                return
            payload = json.dumps(
                {
                    "schema": self._schema,
                    "run_id": self._run_id,
                    "source": self._source,
                    "clock": LIVE_TELEMETRY_CLOCK,
                    "timestamp_semantics": self._timestamp_semantics,
                    "axis_device_numbers": self._axis_device_numbers,
                    "position_calibration": self._calibration_metadata,
                    "overflow": self._overflow,
                    "published_records": self._sequence,
                    "records": list(self._pending),
                },
                separators=(",", ":"),
            )
            await attr.update(payload)
            for _ in range(count):
                self._pending.popleft()

    async def update(self, attr: AttrR[str, TelemetryRef]) -> None:
        """Publish queued observations and retain them if PVA update fails."""
        self._attr = attr
        self._ensure_flusher()
        await self._publish_pending(attr)


__all__ = [
    "LIVE_TELEMETRY_CAPACITY",
    "LIVE_TELEMETRY_PERIOD",
    "StageTelemetryIO",
    "TelemetryRef",
]
