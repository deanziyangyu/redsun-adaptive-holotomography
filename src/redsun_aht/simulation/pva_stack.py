"""Typed client of the experimental simulated-stack PVA service."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from uuid import uuid4

from ophyd_async.core import Device, SignalR, SignalW
from ophyd_async.fastcs.core import fastcs_connector

from .mcu_stack import Edge, Stack


@dataclass(frozen=True, slots=True)
class StackEvidence:
    """Separately generated synthetic edge and camera-counter streams."""

    edges: tuple[Edge, ...]
    frame_sequences: tuple[int, ...]


class PvaSimulatedStack(Device):
    """Submit one fake stack to a service; this device cannot move hardware."""

    reservation: SignalW[str]
    reservation_status: SignalR[str]
    request: SignalW[str]
    result: SignalR[str]
    abort: SignalW[str]
    jog: SignalW[str]
    manual: SignalW[str]
    manual_result: SignalR[str]
    positions: SignalR[str]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))
        self._token: str | None = None
        self._reservation_uncertain = False

    async def reserve_service(self) -> None:
        """Acquire a non-expiring cooperative fake-service reservation."""
        if self._token is not None or self._reservation_uncertain:
            raise RuntimeError("reservation held or uncertain")
        token = uuid4().hex
        self._token = token
        try:
            await self.reservation.set(f"acquire {token}")
        except BaseException:
            self._reservation_uncertain = True
            raise

    async def release_service(self) -> None:
        """Release only a certainly held fake-service reservation."""
        if self._token is None or self._reservation_uncertain:
            raise RuntimeError("reservation absent or uncertain")
        try:
            await self.reservation.set(f"release {self._token}")
        except BaseException:
            self._reservation_uncertain = True
            raise
        self._token = None

    def _payload_with_token(self, key: str, payload: dict) -> str:
        if self._reservation_uncertain:
            raise RuntimeError("reservation uncertain")
        if self._token is None:
            return json.dumps(payload, allow_nan=False)
        return json.dumps(
            {key: payload, "reservation_token": self._token}, allow_nan=False
        )

    async def jog_by(self, increments: dict[str, float]) -> None:
        """Submit one bounded synthetic jog; the service enforces reservation."""
        if (
            not isinstance(increments, dict)
            or not increments
            or not set(increments) <= {"survey_x", "survey_y", "coarse_z"}
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or abs(value) > 1
                or value * 4 != round(value * 4)
                for value in increments.values()
            )
        ):
            raise ValueError("invalid synthetic jog increment")
        await self.jog.set(self._payload_with_token("increments", increments))

    async def jog_positions(self) -> dict[str, float]:
        """Return the last published simulated positions, not a hardware read."""
        positions = json.loads(await self.positions.get_value(cached=False))
        if (
            not isinstance(positions, dict)
            or set(positions) != {"survey_x", "survey_y", "coarse_z", "fast_z"}
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in positions.values()
            )
        ):
            raise ValueError("invalid synthetic position result")
        return positions

    async def abort_stack(self, run_id: str) -> None:
        """Request a run-scoped simulated abort; not an emergency stop."""
        if not run_id:
            raise ValueError("abort requires a run ID")
        await self.abort.set(run_id)

    async def wait_for_aborted_result(self, run_id: str) -> str:
        """Wait for this run's simulated partial result, without resubmitting it.

        The caller must bound the wait. The result is raw, unassociated evidence.
        """
        while True:
            value = await self.result.get_value(cached=False)
            if isinstance(value, str) and value:
                response = json.loads(value)
                if (
                    isinstance(response, dict)
                    and response.get("run_id") == run_id
                    and response.get("quality") == "simulated"
                    and response.get("state") == "aborted"
                    and isinstance(response.get("edges"), list)
                    and isinstance(response.get("camera_frame_sequences"), list)
                ):
                    return value
            await asyncio.sleep(0.02)

    async def execute(self, stack: Stack) -> StackEvidence:
        """Send once, then validate returned simulated evidence for this run.

        A timeout or failed put is propagated and never blindly retried.
        """
        request = {
            "run_id": stack.run_id,
            "targets_um": stack.targets_um,
            "pattern": stack.pattern,
        }
        if self._reservation_uncertain:
            raise RuntimeError("reservation uncertain")
        if self._token is not None:
            request["reservation_token"] = self._token
        payload = json.dumps(request, allow_nan=False)
        await self.request.set(payload)
        response = json.loads(await self.result.get_value(cached=False))
        if not isinstance(response, dict) or response.get("run_id") != stack.run_id:
            raise ValueError("result run identity mismatch")
        if response.get("state") != "complete":
            raise ValueError("synthetic stack did not complete")
        if response.get("quality") != "simulated" or not isinstance(
            response.get("edges"), list
        ):
            raise ValueError("result is not simulated edge evidence")
        records = response["edges"]
        if len(records) != len(stack.targets_um):
            raise ValueError("result edge count mismatch")
        edges = []
        for index, (record, target) in enumerate(
            zip(records, stack.targets_um, strict=True)
        ):
            if not isinstance(record, dict) or record != {
                "run_id": stack.run_id,
                "plane": index,
                "target_um": target,
                "pattern": stack.pattern,
            }:
                raise ValueError("result edge identity mismatch")
            edges.append(Edge(stack.run_id, index, target, stack.pattern))
        frames = response.get("camera_frame_sequences")
        if not isinstance(frames, list) or any(
            type(frame) is not int for frame in frames
        ):
            raise ValueError("invalid synthetic camera counters")
        if frames != list(range(len(edges))):
            raise ValueError("synthetic trigger/frame count or order mismatch")
        positions = response.get("final_positions_um")
        if (
            not isinstance(positions, dict)
            or positions.get("fast_z") != stack.targets_um[-1]
        ):
            raise ValueError("synthetic final fast-Z readback mismatch")
        return StackEvidence(tuple(edges), tuple(frames))
