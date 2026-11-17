"""Experimental PVA surface for one unified LED and X-stack serial owner.

Host-asserted readiness is not a camera interlock. An abort is not a verified
motor stop. Do not share this port with any other service or process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import deque
from dataclasses import asdict

from fastcs.attributes import AttrR, AttrW
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, Int, String
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.services import serve_controller

from redsun_aht.device.dynamic_patterns import DynamicPattern
from redsun_aht.device.unified_mcu import AngularStack, UnifiedMcu, open_unified_port

logger = logging.getLogger(__name__)


class UnifiedMcuController(Controller):  # type: ignore[misc]
    """Experimental service; reservation is cooperative, not authorization."""

    def __init__(
        self,
        owner: UnifiedMcu,
        *,
        require_holospim_panel: bool = False,
        roll_bounds_steps: tuple[int, int] | None = None,
        x28_jog_bounds_tenths: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.owner = owner
        self._require_holospim_panel = require_holospim_panel
        if roll_bounds_steps is not None:
            lower, upper = roll_bounds_steps
            if (
                type(lower) is not int
                or type(upper) is not int
                or not -(2**31) <= lower < upper < 2**31
            ):
                raise ValueError("roll bounds must be increasing signed step counts")
        self._roll_bounds_steps = roll_bounds_steps
        self._x28_jog_bounds = x28_jog_bounds_tenths
        self._x_move_task: tuple[str, asyncio.Task] | None = None
        self._cancelled_x_moves: deque[str] = deque(maxlen=32)
        if x28_jog_bounds_tenths is not None:
            lower, upper = x28_jog_bounds_tenths
            if (
                not require_holospim_panel
                or type(lower) is not int
                or type(upper) is not int
                or not -3600 <= lower < upper <= 3600
            ):
                raise ValueError("X28 jog requires HoloSPIM and valid angular bounds")
        self._lock = asyncio.Lock()
        self._active_task: asyncio.Task[object] | None = None
        self._closing = False
        self._reservation: str | None = None
        self._shot_waiter: asyncio.Future[None] | None = None
        self._shot_pending: tuple[str, int, int] | None = None
        self._shot_context: tuple[str, int] | None = None
        self.request = AttrW(String(), description="Finite angular stack JSON")
        self.dynamic_pattern_request = AttrW(
            String(), description="Reserved 91-pixel map upload"
        )
        self.dynamic_pattern_result = AttrR(
            String(), initial_value="", description="Correlated dynamic map receipt"
        )
        self.dynamic_shot_request = AttrW(
            String(), description="Reserved dynamic software-shot recipe"
        )
        self.shot_request = AttrW(
            String(), description="Finite software-shot request JSON"
        )
        self.automatic_request = AttrW(
            String(), description="Autonomous shot request JSON"
        )
        self.automatic_result = AttrR(
            String(), initial_value="", description="Autonomous pulse attempt report"
        )
        self.shot_ready = AttrR(
            String(), initial_value="", description="Current host-shot readiness"
        )
        self.shot_confirm = AttrW(
            String(), description="Confirm one captured host shot"
        )
        self.shot_cancel = AttrW(String(), description="Cancel active host-shot run")
        self.abort = AttrW(String(), description="Run ID to abort; stop unverified")
        self.illumination = AttrW(String(), description="Unified numeric LED command")
        self.reserved_illumination = AttrW(
            String(), description="Reserved numeric LED command JSON"
        )
        self.auto_pulses = AttrW(
            Bool(), description="Enable or disable automatic camera pulses"
        )
        self.auto_status = AttrR(
            String(),
            initial_value="unknown",
            description="Last acknowledged automatic pulse setting",
        )
        self.trigger_output = AttrW(
            Int(min=1, max=3), description="Request one pulse on output 1, 2 or 3"
        )
        self.trigger_result = AttrR(
            String(), initial_value="", description="Last pulse command acknowledgement"
        )
        self.result = AttrR(String(), initial_value="", description="Reported attempts")
        self.x_state_request = AttrW(
            String(), description="Read-only X state request ID"
        )
        self.x_state_result = AttrR(
            String(), initial_value="", description="Controller-reported X state"
        )
        self.x_move_request = AttrW(String(), description="Bounded X28 move JSON")
        self.x_move_result = AttrR(
            String(), initial_value="", description="Correlated settled X28 state"
        )
        self.x_move_cancel = AttrW(String(), description="Cancel a matching X28 move")
        self.rotation_request = AttrW(
            String(), description="Open-loop rotation command JSON"
        )
        self.rotation_result = AttrR(
            String(),
            initial_value="",
            description="Correlated open-loop rotation state",
        )
        self.busy = AttrR(Bool(), initial_value=False, description="Operation active")
        self.reservation = AttrW(
            String(), description="Acquire/release a cooperative token"
        )
        self.reservation_status = AttrR(
            String(), initial_value="free", description="Cooperative reservation state"
        )
        self.request.set_on_put_callback(self._run)
        self.shot_request.set_on_put_callback(self._run_shots)
        self.automatic_request.set_on_put_callback(self._run_automatic)
        self.shot_confirm.set_on_put_callback(self._confirm_shot)
        self.shot_cancel.set_on_put_callback(self._cancel_shots)
        self.abort.set_on_put_callback(self._abort)
        self.illumination.set_on_put_callback(self._illumination)
        self.reserved_illumination.set_on_put_callback(self._reserved_illumination)
        self.auto_pulses.set_on_put_callback(self._auto_pulses)
        self.trigger_output.set_on_put_callback(self._trigger_output)
        self.reservation.set_on_put_callback(self._reservation_command)
        self.x_state_request.set_on_put_callback(self._x_state)
        self.x_move_request.set_on_put_callback(self._x_move)
        self.x_move_cancel.set_on_put_callback(self._cancel_x_move)
        self.rotation_request.set_on_put_callback(self._rotation)
        self.dynamic_pattern_request.set_on_put_callback(self._upload_dynamic_pattern)
        self.dynamic_shot_request.set_on_put_callback(self._run_dynamic_shots)

    async def initialise(self) -> None:
        """Verify dialect and blank before accepting operations."""
        await self.owner.identify(require_holospim_panel=self._require_holospim_panel)
        await self.owner.illumination("00")

    async def _run(self, attr: AttrW, value: str) -> None:
        del attr
        if len(value.encode("utf-8")) > 1024:
            raise ValueError("request too large")
        payload = json.loads(value)
        fields = {"run_id", "first", "step", "count", "lower", "upper", "ready"}
        if (
            not isinstance(payload, dict)
            or not fields <= set(payload)
            or set(payload) - fields - {"trigger_mask", "reservation_token"}
        ):
            raise ValueError("invalid request fields")
        token = payload.pop("reservation_token", None)
        if payload.pop("ready") is not True:
            raise ValueError(
                "host must explicitly assert camera and illumination ready"
            )
        if self.auto_status.get() == "disabled":
            raise RuntimeError("automatic camera pulses disabled")
        definition = AngularStack(**payload)
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if token != self._reservation:
                raise RuntimeError("reservation mismatch")
            self._active_task = asyncio.current_task()
            try:
                await self.busy.update(True)
                previous_progress = self.owner.progress
                try:
                    # Clear old evidence before attempting a new request.
                    await self.result.update("")
                    await self.owner.stack(definition)
                finally:
                    progress = self.owner.progress
                    try:
                        if progress is not None and progress is not previous_progress:
                            await self.result.update(json.dumps(asdict(progress)))
                    finally:
                        await self.busy.update(False)
            finally:
                self._active_task = None

    async def _run_automatic(self, attr: AttrW, value: str) -> None:
        """Run one autonomous MCU sequence under a cooperative reservation."""
        del attr
        if len(value.encode("utf-8")) > 2048:
            raise ValueError("automatic request too large")
        payload = json.loads(value)
        fields = {
            "stack",
            "shots_per_plane",
            "pattern_mask",
            "interval_ms",
            "reservation_token",
            "camera_armed",
            "illumination_ready",
            "bind_stage",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("invalid automatic request fields")
        token = payload["reservation_token"]
        if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}", token) is None:
            raise ValueError("reservation token required")
        if (
            payload["camera_armed"] is not True
            or payload["illumination_ready"] is not True
        ):
            raise ValueError("camera and illumination assertions required")
        if (
            not isinstance(payload["stack"], dict)
            or type(payload["bind_stage"]) is not bool
        ):
            raise ValueError("invalid automatic stack")
        definition = AngularStack(**payload["stack"])
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if self._reservation != token:
                raise RuntimeError("reservation mismatch")
            self._active_task = asyncio.current_task()
            self._shot_context = (token, definition.run_id)
            try:
                await self.busy.update(True)
                await self.automatic_result.update("")
                summary = await self.owner.automatic_shots(
                    definition,
                    payload["shots_per_plane"],
                    payload["pattern_mask"],
                    payload["interval_ms"],
                    camera_armed=True,
                    illumination_ready=True,
                    bind_stage=payload["bind_stage"],
                )
                await self.automatic_result.update(
                    json.dumps(
                        {
                            "run_id": definition.run_id,
                            "attempted_pulses": summary.attempted_pulses,
                            "quality": "controller_reported",
                        }
                    )
                )
            finally:
                self._shot_context = None
                await self.busy.update(False)
                self._active_task = None

    async def _run_shots(self, attr: AttrW, value: str) -> None:
        """Own one configured sequence while a token holder confirms each frame."""
        del attr
        if len(value.encode("utf-8")) > 16384:
            raise ValueError("shot request too large")
        payload = json.loads(value)
        fields = {
            "stack",
            "shots_per_plane",
            "patterns",
            "reservation_token",
            "camera_exclusive",
            "illumination_ready",
            "bind_stage",
        }
        if not isinstance(payload, dict) or set(payload) not in (
            fields,
            fields | {"hold_illumination"},
        ):
            raise ValueError("invalid shot request fields")
        hold_illumination = payload.get("hold_illumination", False)
        if type(hold_illumination) is not bool:
            raise ValueError("illumination hold must be boolean")
        token = payload["reservation_token"]
        if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}", token) is None:
            raise ValueError("reservation token required")
        if (
            payload["camera_exclusive"] is not True
            or payload["illumination_ready"] is not True
        ):
            raise ValueError("camera and illumination assertions required")
        if type(payload["bind_stage"]) is not bool:
            raise ValueError("stage binding must be boolean")
        if not isinstance(payload["stack"], dict) or not isinstance(
            payload["patterns"], list
        ):
            raise ValueError("invalid shot definition")
        definition = AngularStack(**payload["stack"])
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if self._reservation != token:
                raise RuntimeError("reservation mismatch")
            self._active_task = asyncio.current_task()
            self._shot_context = (token, definition.run_id)
            try:
                await self.busy.update(True)
                await self.shot_ready.update("")

                async def ready(
                    shot: int, plane: int, pattern: int, target: int
                ) -> None:
                    await self._publish_ready(
                        token,
                        definition.run_id,
                        shot,
                        {
                            "plane": plane,
                            "pattern": pattern,
                            "target_tenths": target,
                        },
                    )

                await self.owner.software_shots(
                    definition,
                    payload["shots_per_plane"],
                    payload["patterns"],
                    ready,
                    camera_exclusive=True,
                    illumination_ready=True,
                    bind_stage=payload["bind_stage"],
                    hold_illumination=hold_illumination,
                )
            finally:
                self._shot_waiter = None
                self._shot_pending = None
                self._shot_context = None
                await self.busy.update(False)
                self._active_task = None

    async def _publish_ready(
        self, token: str, run_id: int, shot: int, fields: dict[str, int | str]
    ) -> None:
        waiter = asyncio.get_running_loop().create_future()
        self._shot_waiter = waiter
        self._shot_pending = (token, run_id, shot)
        try:
            await self.shot_ready.update(
                json.dumps(
                    {
                        "run_id": run_id,
                        "shot": shot,
                        **fields,
                    }
                )
            )
            await asyncio.wait_for(waiter, 20.0)
        finally:
            self._shot_waiter = None
            self._shot_pending = None

    async def _run_dynamic_shots(self, attr: AttrW, value: str) -> None:
        del attr
        if len(value) > 12288 or not value.isascii():
            raise ValueError("invalid dynamic shot request length or encoding")
        payload = json.loads(value)
        if not isinstance(payload, dict) or set(payload) != {
            "token",
            "run_id",
            "patterns",
            "repeats",
            "ready",
        }:
            raise ValueError("invalid dynamic shot request")
        token, run_id, patterns, repeats, ready = (
            payload[key] for key in ("token", "run_id", "patterns", "repeats", "ready")
        )
        if (
            not isinstance(token, str)
            or re.fullmatch(r"[0-9a-f]{32}", token) is None
            or type(run_id) is not int
            or not 1 <= run_id <= 2_147_483_647
            or not isinstance(patterns, list)
            or not 1 <= len(patterns) <= 64
            or type(repeats) is not int
            or not 1 <= repeats <= 32
            or ready is not True
        ):
            raise ValueError("invalid dynamic shot fields")
        receipts = []
        for receipt in patterns:
            if (
                not isinstance(receipt, dict)
                or set(receipt) != {"slot", "generation", "crc32", "sha256"}
                or type(receipt["slot"]) is not int
                or not 1 <= receipt["slot"] <= 5
                or type(receipt["generation"]) is not int
                or not 1 <= receipt["generation"] <= 0xFFFFFFFF
                or type(receipt["crc32"]) is not int
                or not 0 <= receipt["crc32"] <= 0xFFFFFFFF
                or not isinstance(receipt["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]) is None
            ):
                raise ValueError("invalid dynamic pattern receipt")
            receipts.append(DynamicPattern(**receipt))
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if token != self._reservation:
                raise RuntimeError("reservation mismatch")
            self._active_task = asyncio.current_task()
            self._shot_context = (token, run_id)
            try:
                await self.busy.update(True)
                await self.shot_ready.update("")

                async def ready_frame(shot: int, pattern: DynamicPattern) -> None:
                    await self._publish_ready(
                        token,
                        run_id,
                        shot,
                        {
                            "mode": "dynamic",
                            "slot": pattern.slot,
                            "generation": pattern.generation,
                        },
                    )

                await self.owner.dynamic_shots(
                    receipts,
                    repeats,
                    ready_frame,
                    camera_exclusive=True,
                    illumination_ready=True,
                )
            finally:
                self._shot_context = None
                self._active_task = None
                await self.busy.update(False)

    async def _confirm_shot(self, attr: AttrW, value: str) -> None:
        del attr
        payload = json.loads(value)
        if not isinstance(payload, dict) or set(payload) != {"token", "run_id", "shot"}:
            raise ValueError("invalid shot confirmation")
        if (
            not isinstance(payload["token"], str)
            or type(payload["run_id"]) is not int
            or type(payload["shot"]) is not int
        ):
            raise ValueError("invalid shot confirmation types")
        key = (payload["token"], payload["run_id"], payload["shot"])
        if (
            self._shot_pending != key
            or self._shot_waiter is None
            or self._shot_waiter.done()
        ):
            raise RuntimeError("shot confirmation does not match pending readiness")
        self._shot_waiter.set_result(None)

    async def _cancel_shots(self, attr: AttrW, value: str) -> None:
        del attr
        payload = json.loads(value)
        if not isinstance(payload, dict) or set(payload) != {"token", "run_id"}:
            raise ValueError("invalid shot cancellation")
        if not isinstance(payload["token"], str) or type(payload["run_id"]) is not int:
            raise ValueError("invalid shot cancellation types")
        if self._shot_context != (payload["token"], payload["run_id"]):
            raise RuntimeError("no matching pending shot")
        if self._active_task is not None:
            self._active_task.cancel()

    async def _abort(self, attr: AttrW, value: str) -> None:
        del attr
        if not value.isascii() or not value.isdecimal() or len(value) > 10:
            raise ValueError("abort requires a numeric run ID")
        await self.owner.abort(int(value))

    async def _upload_dynamic_pattern(self, attr: AttrW, value: str) -> None:
        del attr
        if len(value) > 512 or not value.isascii():
            raise ValueError("invalid dynamic map request length or encoding")
        request = json.loads(value)
        if not isinstance(request, dict) or set(request) != {
            "id",
            "token",
            "slot",
            "pixels",
        }:
            raise ValueError("invalid dynamic map request")
        request_id, token, slot, pixels = (
            request[k] for k in ("id", "token", "slot", "pixels")
        )
        if (
            not isinstance(request_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", request_id) is None
            or not isinstance(token, str)
            or re.fullmatch(r"[0-9a-f]{32}", token) is None
            or type(slot) is not int
            or not 1 <= slot <= 5
            or not isinstance(pixels, str)
            or re.fullmatch(r"[0-9a-f]{182}", pixels) is None
        ):
            raise ValueError("invalid dynamic map fields")
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if token != self._reservation:
                raise RuntimeError("reservation mismatch")
            self._active_task = asyncio.current_task()
            try:
                receipt = await self.owner.upload_dynamic_pattern(
                    slot, bytes.fromhex(pixels)
                )
                await self.dynamic_pattern_result.update(
                    json.dumps({"id": request_id, **asdict(receipt)})
                )
            finally:
                self._active_task = None

    async def _rotation(self, attr: AttrW, value: str) -> None:
        del attr
        if len(value.encode("utf-8")) > 512:
            raise ValueError("rotation request too large")
        payload = json.loads(value)
        if not isinstance(payload, dict) or set(payload) != {
            "id",
            "action",
            "steps",
            "token",
        }:
            raise ValueError("invalid rotation request")
        if (
            not isinstance(payload["id"], str)
            or re.fullmatch(r"[a-f0-9]{32}", payload["id"]) is None
        ):
            raise ValueError("invalid rotation request ID")
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if payload["token"] != self._reservation:
                raise RuntimeError("reservation mismatch")
            action = payload["action"]
            if action in ("enable", "move"):
                bounds = self._roll_bounds_steps
                if bounds is None:
                    if self._require_holospim_panel:
                        raise RuntimeError(
                            "configure HoloSPIM roll bounds before motion"
                        )
                else:
                    lower, upper = bounds
                    steps = payload["steps"]
                    if action == "move" and (
                        type(steps) is not int or not lower <= steps <= upper
                    ):
                        raise ValueError("roll target exceeds configured step bounds")
                    state = await self.owner.rotation_command("read")
                    if state.moving or not lower <= state.position <= upper:
                        raise RuntimeError(
                            "roll counter is moving or outside configured bounds"
                        )
            elif action == "zero" and self._roll_bounds_steps is not None:
                raise RuntimeError(
                    "roll zero would invalidate the configured step bounds"
                )
            state = await self.owner.rotation_command(action, payload["steps"])
            await self.rotation_result.update(
                json.dumps({"id": payload["id"], "state": asdict(state)})
            )

    async def _x_state(self, attr: AttrW, value: str) -> None:
        """Publish a correlated state without interleaving serial traffic."""
        del attr
        if re.fullmatch(r"[a-f0-9]{32}", value) is None:
            raise ValueError("X state request ID must be 32 lowercase hex digits")
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            self._active_task = asyncio.current_task()
            try:
                await self.x_state_result.update("")
                state = await self.owner.read_x_state()
                await self.x_state_result.update(
                    json.dumps({"request_id": value, "state": asdict(state)})
                )
            finally:
                self._active_task = None

    async def _x_move(self, attr: AttrW, value: str) -> None:
        del attr
        if len(value) > 128:
            raise ValueError("X28 move request too large")
        request = json.loads(value)
        if (
            not isinstance(request, dict)
            or set(request) != {"request_id", "target"}
            or not isinstance(request["request_id"], str)
            or re.fullmatch(r"[a-f0-9]{32}", request["request_id"]) is None
            or type(request["target"]) is not int
        ):
            raise ValueError("invalid X28 move request")
        bounds = self._x28_jog_bounds
        if bounds is None:
            raise RuntimeError("X28 jog bounds are not configured")
        if not bounds[0] <= request["target"] <= bounds[1]:
            raise ValueError("X28 target outside configured bounds")
        if self._closing or self._lock.locked() or self._reservation is not None:
            raise RuntimeError("unified MCU busy or reserved")
        request_id = request["request_id"]
        async with self._lock:
            if request_id in self._cancelled_x_moves:
                raise RuntimeError("X28 move was cancelled before admission")
            self._active_task = asyncio.current_task()
            assert self._active_task is not None
            move = asyncio.create_task(
                self.owner.move_x_absolute(
                    request["target"], lower=bounds[0], upper=bounds[1]
                )
            )
            self._x_move_task = request_id, move
            try:
                await self.x_move_result.update("")
                try:
                    state = await asyncio.shield(move)
                except asyncio.CancelledError:
                    if request_id in self._cancelled_x_moves:
                        raise RuntimeError(
                            "X28 move cancelled; stop unverified"
                        ) from None
                    raise
                await self.x_move_result.update(
                    json.dumps({
                        "request_id": request["request_id"], "state": asdict(state),
                    })
                )
            finally:
                if not move.done():
                    if not move.cancelling():
                        move.cancel()
                    await asyncio.gather(move, return_exceptions=True)
                self._x_move_task = None
                self._active_task = None

    async def _cancel_x_move(self, attr: AttrW, request_id: str) -> None:
        del attr
        if re.fullmatch(r"[a-f0-9]{32}", request_id) is None:
            raise ValueError("invalid X28 cancellation ID")
        if request_id in self._cancelled_x_moves:
            return
        self._cancelled_x_moves.append(request_id)
        active = self._x_move_task
        if active is not None and active[0] == request_id and not active[1].done():
            active[1].cancel()
            await asyncio.gather(active[1], return_exceptions=True)

    async def _illumination(self, attr: AttrW, value: str) -> None:
        del attr
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if self._reservation is not None:
                raise RuntimeError("unified MCU reserved")
            await self.owner.illumination(value)

    async def _reserved_illumination(self, attr: AttrW, value: str) -> None:
        del attr
        if len(value.encode("utf-8")) > 1024:
            raise ValueError("request too large")
        payload = json.loads(value)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"token", "command"}
            or not isinstance(payload["token"], str)
            or re.fullmatch(r"[0-9a-f]{32}", payload["token"]) is None
            or not isinstance(payload["command"], str)
        ):
            raise ValueError("invalid reserved illumination request")
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if payload["token"] != self._reservation:
                raise RuntimeError("reservation mismatch")
            await self.owner.illumination(payload["command"])

    async def _auto_pulses(self, attr: AttrW, value: bool) -> None:
        del attr
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if self._reservation is not None:
                raise RuntimeError("unified MCU reserved")
            try:
                await self.owner.set_auto_pulses(value)
            except BaseException:
                await self.auto_status.update("uncertain")
                raise
            await self.auto_status.update("enabled" if value else "disabled")

    async def _trigger_output(self, attr: AttrW, value: int) -> None:
        del attr
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        async with self._lock:
            if self._reservation is not None:
                raise RuntimeError("unified MCU reserved")
            await self.trigger_result.update("")
            try:
                await self.owner.pulse_output(value)
            except BaseException:
                await self.trigger_result.update("uncertain")
                raise
            await self.trigger_result.update(f"output {value} acknowledged")

    async def _reservation_command(self, attr: AttrW, value: str) -> None:
        """Reserve or release only while idle; never expire implicitly."""
        del attr
        match = re.fullmatch(r"(acquire|release) ([0-9a-f]{32})", value)
        if match is None:
            raise ValueError("invalid reservation command")
        if self._closing or self._lock.locked():
            raise RuntimeError("unified MCU busy")
        action, token = match.groups()
        async with self._lock:
            if action == "acquire":
                if self._reservation is not None:
                    raise RuntimeError("unified MCU reserved")
                if self.owner.uncertain:
                    raise RuntimeError("unified MCU uncertain; reconciliation required")
                self._reservation = token
                await self.reservation_status.update("held")
            else:
                if self._reservation != token:
                    raise RuntimeError("reservation mismatch")
                if self.owner.uncertain:
                    raise RuntimeError("unified MCU uncertain; reconciliation required")
                await self.reservation_status.update("free")
                self._reservation = None

    async def shutdown(self) -> None:
        """Cancel an active request before port closure; stop remains unverified."""
        self._closing = True
        task = self._active_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(task, return_exceptions=True), timeout=3.0
                )
            except TimeoutError:
                logger.error("Active stack cancellation timed out; stop unverified")


async def _serve(
    port_name: str,
    prefix: str,
    *,
    require_holospim_panel: bool = False,
    roll_bounds_steps: tuple[int, int] | None = None,
    x28_jog_bounds_tenths: tuple[int, int] | None = None,
) -> None:
    port = open_unified_port(port_name)
    controller = UnifiedMcuController(
        UnifiedMcu(port),
        require_holospim_panel=require_holospim_panel,
        roll_bounds_steps=roll_bounds_steps,
        x28_jog_bounds_tenths=x28_jog_bounds_tenths,
    )
    try:
        await serve_controller(controller, EpicsPVATransport(), prefix)
    finally:
        try:
            await controller.shutdown()
        finally:
            # Closing does not establish motor stop or illumination state.
            port.close()


def main(argv: list[str] | None = None) -> int:
    """Serve one explicitly selected endpoint until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--roll-bounds-steps",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="allowed roll range in signed controller step counts",
    )
    parser.add_argument(
        "--require-holospim-panel",
        action="store_true",
        help="require 91-pixel intensity and dynamic pattern banks",
    )
    parser.add_argument("--x28-jog-bounds-tenths", nargs=2, type=int,
                        metavar=("MIN", "MAX"), help="commissioned X28 jog range")
    options = parser.parse_args(argv)
    x28_bounds = options.x28_jog_bounds_tenths
    if x28_bounds is not None and (
        not options.require_holospim_panel
        or not -3600 <= x28_bounds[0] < x28_bounds[1] <= 3600
    ):
        parser.error("X28 jog requires HoloSPIM and valid angular bounds")
    bounds = options.roll_bounds_steps
    if bounds is not None and not -(2**31) <= bounds[0] < bounds[1] < 2**31:
        parser.error("roll bounds must be increasing signed step counts")
    try:
        asyncio.run(
            _serve(
                options.port,
                options.prefix,
                require_holospim_panel=options.require_holospim_panel,
                roll_bounds_steps=(bounds[0], bounds[1])
                if bounds is not None
                else None,
                x28_jog_bounds_tenths=(x28_bounds[0], x28_bounds[1])
                if x28_bounds is not None
                else None,
            )
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
