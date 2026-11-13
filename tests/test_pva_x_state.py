"""Reject stale and inconsistent X-state PVA evidence."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from ophyd_async.core import set_mock_value

from redsun_aht.device.pva_x_state import PvaXState
from redsun_aht.device.unified_mcu import parse_x_state


async def test_x_state_client_rejects_stale_and_inconsistent_results(monkeypatch):
    monkeypatch.setattr(
        "redsun_aht.device.pva_x_state.uuid4", lambda: SimpleNamespace(hex="a" * 32)
    )
    client = PvaXState("TEST:", name="state")
    await client.connect(mock=True)
    frame = bytearray(37)
    frame[:4] = bytes([1, 0x43, 37, 12])
    frame[34:37] = bytes([3, 3, 0x6B])
    state = asdict(parse_x_state(bytes(frame)))
    set_mock_value(
        client.x_state_result, json.dumps({"request_id": "b" * 32, "state": state})
    )
    with pytest.raises(ValueError, match="identity"):
        await client.read_state()
    state["position_tenths"] = 1
    set_mock_value(
        client.x_state_result, json.dumps({"request_id": "a" * 32, "state": state})
    )
    with pytest.raises(ValueError, match="disagree"):
        await client.read_state()
