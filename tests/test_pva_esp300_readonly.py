"""Typed read-only ESP300 PVA client validation."""

import json

import pytest
from ophyd_async.core import get_mock_put, set_mock_value

from redsun_aht.device.pva_esp300_readonly import PvaEsp300ReadOnly, parse_snapshot


def payload():
    return {
        "schema": "redsun.esp300-readonly/v1",
        "clock": "host_monotonic_ns",
        "start_ns": 1,
        "end_ns": 2,
        "version": "ESP300 3.08",
        "quality": "controller_reported",
        "axes": [
            {
                "axis": i,
                "identity": f"axis{i}",
                "resolution_um": 0.1,
                "lower_um": 0,
                "upper_um": 100,
                "actual_um": 1,
                "target_um": 2,
                "motion_done": True,
            }
            for i in (1, 2, 3)
        ],
    }


def test_parser_rejects_untrusted_snapshots():
    data = payload()
    parsed = parse_snapshot(json.dumps(data))
    assert parsed.snapshot.axes[2].actual_um == 1
    for key, value in (
        ("quality", "measured"),
        ("start_ns", True),
        ("clock", "wall"),
        ("end_ns", 0),
    ):
        with pytest.raises(ValueError):
            parse_snapshot(json.dumps({**data, key: value}))
    for key, value in (
        ("axis", True),
        ("motion_done", 1),
        ("actual_um", float("nan")),
        ("resolution_um", 0),
        ("actual_um", 101),
    ):
        bad = payload()
        bad["axes"][0][key] = value
        with pytest.raises(ValueError):
            parse_snapshot(json.dumps(bad))


async def test_client_refresh_and_status_guard():
    client = PvaEsp300ReadOnly("SIM:", name="esp")
    await client.connect(mock=True)
    set_mock_value(client.status, "ready")
    set_mock_value(client.snapshot, json.dumps(payload()))
    result = await client.read_observation()
    assert len(result.snapshot.axes) == 3
    get_mock_put(client.refresh).assert_called_once_with(True)
    set_mock_value(client.status, "uncertain")
    with pytest.raises(RuntimeError, match="not ready"):
        await client.read_observation(refresh=False)
    assert get_mock_put(client.refresh).call_count == 1
