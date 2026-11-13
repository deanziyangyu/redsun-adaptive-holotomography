"""Experimental PVA wrapper for a simulated MCU stack; never opens hardware."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from dataclasses import asdict

from fastcs.attributes import AttrR, AttrW
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, String
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.services import serve_controller

from redsun_aht.simulation.mcu_stack import (
    Axis,
    SimulatedMcu,
    SimulatedStage,
    SimulatedTriggeredCamera,
    Stack,
    State,
)

MAX_REQUEST_BYTES = 16384
logger = logging.getLogger(__name__)


class SimulatedStackController(Controller):  # type: ignore[misc]
    """One-start local fake sequencer with explicitly simulated output.

    This is a rehearsal transport, not an authentication or physical lease
    system. Requests are serialized within one process; a newly configured
    transaction may reuse its tag. It does not operate a camera, serial port,
    or motor.
    """

    def __init__(self) -> None:
        super().__init__()
        axes = tuple(
            Axis(name, -100.0, 100.0, 0.25)
            for name in ("survey_x", "survey_y", "coarse_z", "fast_z")
        )
        self.stage = SimulatedStage(axes)
        self.mcu = SimulatedMcu(axes[-1], stage=self.stage)
        self._lock = asyncio.Lock()
        self._reservation: str | None = None
        self.reservation = AttrW(
            String(), description="Synthetic cooperative reservation"
        )
        self.reservation_status = AttrR(String(), initial_value="free")
        self.reservation.set_on_put_callback(self._reservation_command)
        self.request = AttrW(
            String(), description="One finite simulated stack JSON request"
        )
        self.result = AttrR(
            String(), initial_value="", description="Simulated trigger evidence"
        )
        self.request.set_on_put_callback(self._run)
        self.busy = AttrR(
            Bool(), initial_value=False, description="Synthetic stack active"
        )
        self.abort = AttrW(String(), description="Run ID of a synthetic stack to abort")
        self.abort.set_on_put_callback(self._abort)
        self.manual = AttrW(String(), description="Synthetic manual absolute targets")
        self.manual_result = AttrR(
            String(), initial_value="", description="Synthetic manual result"
        )
        self.manual.set_on_put_callback(self._manual)
        self.jog = AttrW(String(), description="Bounded synthetic survey increments")
        self.jog.set_on_put_callback(self._jog)
        self.positions = AttrR(
            String(),
            initial_value=json.dumps(self.stage.positions),
            description="Last published synthetic stage positions in micrometres",
        )

    async def _run(self, attr: AttrW, value: str) -> None:
        del attr
        if len(value.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("request too large")
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("invalid request fields")
        token = payload.pop("reservation_token", None)
        self._check_reservation(token)
        if set(payload) != {"run_id", "targets_um", "pattern"}:
            raise ValueError("invalid request fields")
        run_id, targets, pattern = (
            payload["run_id"],
            payload["targets_um"],
            payload["pattern"],
        )
        if (
            not isinstance(run_id, str)
            or not isinstance(targets, list)
            or type(pattern) is not int
        ):
            raise ValueError("invalid request types")
        if any(type(value) not in (int, float) for value in targets):
            raise ValueError("targets must be numeric")
        stack = Stack(run_id, tuple(float(value) for value in targets), pattern)
        if self._lock.locked():
            raise RuntimeError("simulated stack busy")
        async with self._lock:
            owner = "pva-rehearsal"
            self.mcu.reserve(owner)
            try:
                await self.busy.update(True)
                self.mcu.commit(owner, stack)
                camera = SimulatedTriggeredCamera()
                camera.arm()
                self.mcu.arm(owner, camera_ready=camera.armed, referenced=True)
                self.mcu.begin(owner, run_id)
                try:
                    while self.mcu.state is State.RUNNING:
                        await asyncio.sleep(0.05)  # synthetic, not a hardware rate
                        if self.mcu.state is State.RUNNING:
                            try:
                                self.mcu.step(
                                    owner,
                                    camera_ready=camera.armed,
                                    on_edge=camera.receive,
                                )
                            finally:
                                await self._publish_positions()
                except Exception:
                    if self.mcu.state is not State.ABORTED:
                        self.mcu.cancel(owner)
                    try:
                        await self._publish_result(run_id, camera)
                    except Exception:
                        logger.exception("Could not publish partial simulated result")
                    raise
                await self._publish_result(run_id, camera)
                if self.mcu.state is State.ABORTED:
                    raise RuntimeError("simulated stack aborted")
            finally:
                try:
                    await self.busy.update(False)
                finally:
                    if self.mcu.state not in (State.COMPLETE, State.ABORTED):
                        self.mcu.cancel(owner)
                    self.mcu.release(owner)

    async def _publish_result(
        self, run_id: str, camera: SimulatedTriggeredCamera
    ) -> None:
        await self.result.update(
            json.dumps(
                {
                    "run_id": run_id,
                    "quality": "simulated",
                    "state": self.mcu.state.value,
                    "edges": [asdict(edge) for edge in self.mcu.edges],
                    "camera_frame_sequences": camera.frames,
                    "final_positions_um": dict(self.stage.positions),
                }
            )
        )

    async def _publish_positions(self) -> None:
        await self.positions.update(json.dumps(self.stage.positions))

    async def _abort(self, attr: AttrW, value: str) -> None:
        del attr
        if (
            self.mcu.state is not State.RUNNING
            or self.mcu.stack is None
            or self.mcu.stack.run_id != value
        ):
            raise RuntimeError("no matching running simulated stack")
        self.mcu.cancel("pva-rehearsal")

    def _check_reservation(self, token: str | None) -> None:
        if token != self._reservation:
            raise RuntimeError("simulated reservation mismatch")

    async def _reservation_command(self, attr: AttrW, value: str) -> None:
        del attr
        match = re.fullmatch(r"(acquire|release) ([0-9a-f]{32})", value)
        if match is None:
            raise ValueError("invalid reservation command")
        if self._lock.locked() or self.stage.owner is not None:
            raise RuntimeError("simulated stage busy")
        action, token = match.groups()
        if action == "acquire":
            if self._reservation is not None:
                raise RuntimeError("simulated stage reserved")
            self._reservation = token
            await self.reservation_status.update("held")
        else:
            if self._reservation != token:
                raise RuntimeError("simulated reservation mismatch")
            self._reservation = None
            await self.reservation_status.update("free")

    def _manual_payload(self, value: str, key: str) -> dict:
        payload = json.loads(value)
        if isinstance(payload, dict) and set(payload) == {key, "reservation_token"}:
            self._check_reservation(payload["reservation_token"])
            return payload[key]
        self._check_reservation(None)
        return payload

    async def _manual(self, attr: AttrW, value: str) -> None:
        del attr
        if len(value.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("request too large")
        targets = self._manual_payload(value, "targets")
        if (
            not isinstance(targets, dict)
            or not targets
            or not set(targets) <= {"survey_x", "survey_y", "coarse_z"}
        ):
            raise ValueError("invalid synthetic survey axes")
        if any(type(value) not in (int, float) for value in targets.values()):
            raise ValueError("targets must be numeric")
        owner = "manual-rehearsal"
        self.stage.reserve(owner)  # refuses immediately if a stack owns the stage
        try:
            positions = self.stage.move(targets, owner=owner)
            await self._publish_positions()
            await self.manual_result.update(json.dumps(positions))
        finally:
            self.stage.release(owner)

    async def _jog(self, attr: AttrW, value: str) -> None:
        """Apply bounded increments to current positions under one reservation."""
        del attr
        if len(value.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("request too large")
        increments = self._manual_payload(value, "increments")
        if (
            not isinstance(increments, dict)
            or not increments
            or not set(increments) <= {"survey_x", "survey_y", "coarse_z"}
            or any(
                type(step) not in (int, float)
                or not math.isfinite(step)
                or abs(step) > 1.0
                for step in increments.values()
            )
        ):
            raise ValueError("invalid or oversized synthetic jog")
        owner = "jog-rehearsal"
        self.stage.reserve(owner)
        try:
            targets = {
                name: self.stage.positions[name] + step
                for name, step in increments.items()
            }
            positions = self.stage.move(targets, owner=owner)
            await self._publish_positions()
            await self.manual_result.update(json.dumps(positions))
        finally:
            self.stage.release(owner)


def main(argv: list[str] | None = None) -> int:
    """Serve the fake controller on a caller-selected PVA prefix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    options = parser.parse_args(argv)
    asyncio.run(
        serve_controller(
            SimulatedStackController(), EpicsPVATransport(), options.prefix
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
