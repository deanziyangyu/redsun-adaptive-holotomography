"""Experimental Redsun client of the unified MCU PVA service.

Reported pulse attempts are not measured GPIO edges or camera frames. This
client does not implement camera arming, durable leases, recovery or a safety stop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from ophyd_async.core import Device, SignalR, SignalW, wait_for_value
from ophyd_async.fastcs.core import fastcs_connector

from redsun_aht.device.unified_mcu import (
    AngularStack,
    AutomaticShotSummary,
    RotationState,
    StackProgress,
)

logger = logging.getLogger(__name__)


def parse_progress(value: str, stack: AngularStack) -> StackProgress:
    """Validate a controller-reported prefix against the requested sequence."""
    response = json.loads(value)
    if not isinstance(response, dict) or set(response) != {
        "run_id",
        "targets_tenths",
        "complete",
        "quality",
    }:
        raise ValueError("invalid progress fields")
    targets = response["targets_tenths"]
    complete = response["complete"]
    if (
        type(response["run_id"]) is not int
        or response["run_id"] != stack.run_id
        or response["quality"] != "controller_reported"
        or type(complete) is not bool
        or not isinstance(targets, list)
        or any(type(target) is not int for target in targets)
        or len(targets) > stack.count
        or (complete and len(targets) != stack.count)
        or targets
        != [stack.first + stack.step * index for index in range(len(targets))]
    ):
        raise ValueError("invalid or mismatched controller progress")
    return StackProgress(stack.run_id, tuple(targets), complete)


class PvaUnifiedMcu(Device):
    """Sequential transaction client; uncertain outcomes prevent reuse."""

    request: SignalW[str]
    shot_request: SignalW[str]
    automatic_request: SignalW[str]
    automatic_result: SignalR[str]
    shot_ready: SignalR[str]
    shot_confirm: SignalW[str]
    shot_cancel: SignalW[str]
    result: SignalR[str]
    abort: SignalW[str]
    illumination: SignalW[str]
    reserved_illumination: SignalW[str]
    auto_pulses: SignalW[bool]
    auto_status: SignalR[str]
    trigger_output: SignalW[int]
    trigger_result: SignalR[str]
    busy: SignalR[bool]
    reservation: SignalW[str]
    reservation_status: SignalR[str]
    rotation_request: SignalW[str]
    rotation_result: SignalR[str]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))
        self._last_allocated_id = 0
        self._active = False
        self._uncertain = False
        self._reservation_token: str | None = None
        self._reserving = False
        self._reservation_generation = 0
        self._rotation_lock = asyncio.Lock()

    @property
    def reservation_generation(self) -> int:
        """Local generation that changes after each successful reservation."""
        return self._reservation_generation

    async def rotation(self, action: str, steps: int | None = None) -> RotationState:
        """Command or query rotation through the same cooperative MCU owner."""
        async with self._rotation_lock:
            return await self._rotation_transaction(action, steps)

    async def _rotation_transaction(
        self, action: str, steps: int | None
    ) -> RotationState:
        if self._active or self._reserving or self._uncertain:
            raise RuntimeError("client busy or uncertain")
        self._active = True
        request_id = uuid4().hex
        try:
            await self.rotation_request.set(
                json.dumps(
                    {
                        "id": request_id,
                        "action": action,
                        "steps": steps,
                        "token": self._reservation_token,
                    }
                )
            )
            result = json.loads(await self.rotation_result.get_value())
            if not isinstance(result, dict) or result.get("id") != request_id:
                raise ValueError("rotation response mismatch")
            state = result["state"]
            if (
                not isinstance(state, dict)
                or set(state) != {"position", "target", "moving", "enabled"}
                or type(state["position"]) is not int
                or type(state["target"]) is not int
                or type(state["moving"]) is not bool
                or type(state["enabled"]) is not bool
            ):
                raise ValueError("invalid rotation response")
            return RotationState(**state)
        except BaseException:
            self._uncertain = True
            raise
        finally:
            self._active = False

    @property
    def reservation_held(self) -> bool:
        """Whether this client holds a locally acknowledged, certain token."""
        return (
            self._reservation_token is not None
            and not self._reserving
            and not self._uncertain
        )

    async def reserve(self) -> None:
        """Acquire a cooperative reservation; uncertainty quarantines this client."""
        if (
            self._active
            or self._reserving
            or self._uncertain
            or self._reservation_token is not None
        ):
            raise RuntimeError("client busy, uncertain or already reserved")
        token = uuid4().hex
        self._reservation_token = token
        self._reserving = True
        try:
            await self.reservation.set(f"acquire {token}")
            self._last_allocated_id = 0
            self._reservation_generation += 1
        except BaseException:
            self._uncertain = True
            raise
        finally:
            self._reserving = False

    async def release_reservation(self) -> None:
        """Release only a certain, idle reservation; no automatic expiry exists."""
        if (
            self._active
            or self._reserving
            or self._uncertain
            or self._reservation_token is None
        ):
            raise RuntimeError("client busy, uncertain or not reserved")
        self._reserving = True
        try:
            await self.reservation.set(f"release {self._reservation_token}")
        except BaseException:
            self._uncertain = True
            raise
        finally:
            self._reserving = False
        self._reservation_token = None

    async def allocate_run_ids(self, count: int) -> tuple[int, ...]:
        """Allocate local command tags for use during this reservation.

        IDs are reusable after release and are not acquisition identifiers.
        """
        if type(count) is not int or not 1 <= count <= 10_000:
            raise ValueError("run ID count must be 1..10000")
        if not self.reservation_held or self._active:
            raise RuntimeError("an idle, certain reservation is required")
        first = self._last_allocated_id + 1
        last = first + count - 1
        if last > 2_147_483_647:
            raise OverflowError("controller run ID space exhausted")
        self._last_allocated_id = last
        return tuple(range(first, last + 1))

    async def set_reserved_illumination(self, command: str) -> None:
        """Send an LED command while retaining the cooperative reservation."""
        if not isinstance(command, str):
            raise ValueError("illumination command must be a string")
        if not self.reservation_held or self._active:
            raise RuntimeError("an idle, certain reservation is required")
        self._active = True
        try:
            await self.reserved_illumination.set(
                json.dumps({"token": self._reservation_token, "command": command})
            )
        except BaseException:
            self._uncertain = True
            raise
        finally:
            self._active = False

    async def execute(
        self, stack: AngularStack, *, ready: bool, require_reservation: bool = False
    ) -> StackProgress:
        """Submit once after host assertion and require matching complete evidence."""
        if ready is not True:
            raise ValueError("host readiness assertion required")
        if self._active or self._reserving or self._uncertain:
            raise RuntimeError("client busy or uncertain; reconcile before reuse")
        if require_reservation and not self.reservation_held:
            raise RuntimeError("a held MCU reservation is required")
        self._active = True
        try:
            payload: dict[str, Any] = {**asdict(stack), "ready": True}
            if self._reservation_token is not None:
                payload["reservation_token"] = self._reservation_token
            await self.request.set(json.dumps(payload))
            progress = await self.read_progress(stack)
            if not progress.complete:
                raise ValueError("stack result is incomplete")
            return progress
        except BaseException:
            # Lost responses and cancellation cannot establish remote outcome.
            self._uncertain = True
            raise
        finally:
            self._active = False

    async def software_shots(
        self,
        definition: AngularStack,
        shots_per_plane: int,
        patterns: Any,
        snap: Any,
        *,
        camera_exclusive: bool,
        illumination_ready: bool,
        snap_timeout_s: float = 15.0,
        bind_stage: bool = True,
        hold_illumination: bool = False,
    ) -> None:
        """Run token-reserved shots, confirming each only after the caller's snap."""
        if camera_exclusive is not True or illumination_ready is not True:
            raise ValueError("camera and illumination readiness required")
        if type(bind_stage) is not bool:
            raise ValueError("stage binding must be boolean")
        if type(hold_illumination) is not bool:
            raise ValueError("illumination hold must be boolean")
        if not self.reservation_held or self._active:
            raise RuntimeError("an idle reserved MCU client is required")
        if type(shots_per_plane) is not int or not 1 <= shots_per_plane <= 64:
            raise ValueError("invalid shots per plane")
        patterns = tuple(patterns)
        if len(patterns) != definition.count * shots_per_plane or any(
            type(p) is not int or not 1 <= p <= 5 for p in patterns
        ):
            raise ValueError("one valid pattern per shot required")
        if not 0 < snap_timeout_s < 20:
            raise ValueError("snap timeout must be positive and less than 20 seconds")
        token = self._reservation_token
        assert token is not None
        payload = {
            "stack": asdict(definition),
            "shots_per_plane": shots_per_plane,
            "patterns": patterns,
            "reservation_token": token,
            "camera_exclusive": True,
            "illumination_ready": True,
            "bind_stage": bind_stage,
            "hold_illumination": hold_illumination,
        }
        self._active = True
        request_task: asyncio.Task[None] | None = None
        waiter: asyncio.Task[None] | None = None
        try:

            async def submit() -> None:
                await self.shot_request.set(json.dumps(payload))

            request_task = asyncio.create_task(submit())
            for index, pattern in enumerate(patterns):
                plane = index // shots_per_plane
                target = definition.first + definition.step * plane
                expected = {
                    "run_id": definition.run_id,
                    "shot": index,
                    "plane": plane,
                    "pattern": pattern,
                    "target_tenths": target,
                }

                def matches(raw: str, expected: dict[str, int] = expected) -> bool:
                    try:
                        return json.loads(raw) == expected
                    except (TypeError, ValueError):
                        return False

                waiter = asyncio.create_task(
                    wait_for_value(self.shot_ready, matches, timeout=9.0)
                )
                done, _ = await asyncio.wait(
                    {waiter, request_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if request_task in done:
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)
                    request_task.result()
                    raise RuntimeError("shot request ended before readiness")
                await waiter
                await asyncio.wait_for(
                    snap(index, plane, pattern, target), snap_timeout_s
                )
                await self.shot_confirm.set(
                    json.dumps(
                        {"token": token, "run_id": definition.run_id, "shot": index}
                    )
                )
            await request_task
        except BaseException:
            self._uncertain = True
            if waiter is not None and not waiter.done():
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            try:
                await asyncio.shield(
                    asyncio.wait_for(
                        self.shot_cancel.set(
                            json.dumps({"token": token, "run_id": definition.run_id})
                        ),
                        2,
                    )
                )
            except BaseException:
                logger.warning(
                    "Shot cancellation not confirmed; outcome remains uncertain"
                )
            if request_task is not None and not request_task.done():
                request_task.cancel()
            if request_task is not None:
                await asyncio.gather(request_task, return_exceptions=True)
            raise
        finally:
            self._active = False

    async def automatic_shots(
        self,
        definition: AngularStack,
        shots_per_plane: int,
        pattern_mask: int,
        interval_ms: int,
        *,
        camera_armed: bool,
        illumination_ready: bool,
        bind_stage: bool = True,
        before_start: Any = None,
    ) -> AutomaticShotSummary:
        """Submit one autonomous request, never replaying an uncertain outcome."""
        if camera_armed is not True or illumination_ready is not True:
            raise ValueError("armed camera and illumination readiness required")
        if not self.reservation_held or self._active:
            raise RuntimeError("an idle reserved MCU client is required")
        if (
            type(shots_per_plane) is not int
            or not 1 <= shots_per_plane <= 64
            or type(pattern_mask) is not int
            or not 1 <= pattern_mask <= 31
            or type(interval_ms) is not int
            or not 1 <= interval_ms < 30000
            or type(bind_stage) is not bool
        ):
            raise ValueError("invalid automatic shot configuration")
        token = self._reservation_token
        assert token is not None
        payload = {
            "stack": asdict(definition),
            "shots_per_plane": shots_per_plane,
            "pattern_mask": pattern_mask,
            "interval_ms": interval_ms,
            "reservation_token": token,
            "camera_armed": True,
            "illumination_ready": True,
            "bind_stage": bind_stage,
        }
        self._active = True
        sent = False
        try:
            if before_start is not None:
                await before_start()
            sent = True
            # The put completes only after the autonomous stack finishes.  The
            # default signal timeout is too short for longer sequences.
            request_timeout = (
                definition.count * shots_per_plane * interval_ms / 1000
                + definition.count * 8
                + 10
            )
            await self.automatic_request.set(
                json.dumps(payload), timeout=request_timeout
            )
            result = json.loads(await self.automatic_result.get_value(cached=False))
            if not isinstance(result, dict) or result != {
                "run_id": definition.run_id,
                "attempted_pulses": definition.count * shots_per_plane,
                "quality": "controller_reported",
            }:
                raise ValueError("invalid automatic attempt report")
            return AutomaticShotSummary(result["attempted_pulses"])
        except BaseException:
            if sent:
                self._uncertain = True
                try:
                    await asyncio.shield(
                        asyncio.wait_for(
                            self.shot_cancel.set(
                                json.dumps(
                                    {"token": token, "run_id": definition.run_id}
                                )
                            ),
                            2,
                        )
                    )
                except BaseException:
                    logger.warning(
                        "Automatic cancellation not confirmed; outcome uncertain"
                    )
            raise
        finally:
            self._active = False

    async def read_progress(self, stack: AngularStack) -> StackProgress:
        """Read matching observed progress; incomplete does not establish stop."""
        return parse_progress(await self.result.get_value(cached=False), stack)

    async def abort_stack(self, run_id: int) -> None:
        """Request a matching abort; delivery does not prove physical stopping."""
        if type(run_id) is not int or not 1 <= run_id <= 2_147_483_647:
            raise ValueError("invalid run ID")
        await self.abort.set(str(run_id))
