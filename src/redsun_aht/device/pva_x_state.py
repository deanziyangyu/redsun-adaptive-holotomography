"""Correlated read-only X-controller state from the unified PVA service."""

from __future__ import annotations

import json
from dataclasses import asdict
from uuid import uuid4

from ophyd_async.core import Device, SignalR, SignalW
from ophyd_async.fastcs.core import fastcs_connector

from redsun_aht.device.unified_mcu import XControllerState, parse_x_state


class PvaXState(Device):
    """Read controller-reported state without opening the serial port."""

    x_state_request: SignalW[str]
    x_state_result: SignalR[str]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))

    async def read_state(self) -> XControllerState:
        """Request once and validate correlation, raw bytes and decoded fields."""
        request_id = uuid4().hex
        await self.x_state_request.set(request_id)
        response = json.loads(await self.x_state_result.get_value(cached=False))
        if (
            not isinstance(response, dict)
            or set(response) != {"request_id", "state"}
            or response["request_id"] != request_id
        ):
            raise ValueError("X state response identity mismatch")
        state = response["state"]
        if not isinstance(state, dict) or not isinstance(state.get("raw_hex"), str):
            raise ValueError("invalid X state payload")
        try:
            parsed = parse_x_state(bytes.fromhex(state["raw_hex"]))
        except ValueError as exc:
            raise ValueError("invalid X state hex") from exc
        expected = asdict(parsed)
        if set(state) != set(expected) or any(
            type(state[k]) is not type(v) or state[k] != v for k, v in expected.items()
        ):
            raise ValueError("X state fields disagree with raw reply")
        return parsed
