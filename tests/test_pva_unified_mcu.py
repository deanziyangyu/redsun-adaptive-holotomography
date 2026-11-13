"""Typed unified PVA client tests with no hardware."""

import asyncio
import json

import pytest
from ophyd_async.core import get_mock_put, set_mock_value

from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu, parse_progress
from redsun_aht.device.unified_mcu import AngularStack


async def test_one_request_and_strict_result():
    client = PvaUnifiedMcu("SIM:", name="mcu")
    await client.connect(mock=True)
    stack = AngularStack(7, 0, 18, 3, -180, 180)
    result = dict(
        run_id=7,
        targets_tenths=[0, 18, 36],
        complete=True,
        quality="controller_reported",
    )
    set_mock_value(client.result, json.dumps(result))
    progress = await client.execute(stack, ready=True)
    assert progress.complete and progress.targets_tenths == (0, 18, 36)
    assert get_mock_put(client.request).call_count == 1
    with pytest.raises(ValueError, match="readiness"):
        await client.execute(stack, ready=False)
    assert get_mock_put(client.request).call_count == 1
    await client.execute(stack, ready=True)
    assert get_mock_put(client.request).call_count == 2
    set_mock_value(client.result, json.dumps({**result, "run_id": 8}))
    await client.execute(AngularStack(8, 0, 18, 3, -180, 180), ready=True)
    assert get_mock_put(client.request).call_count == 3
    await client.abort_stack(7)
    get_mock_put(client.abort).assert_called_once_with("7")


def test_partial_and_malformed_results():
    stack = AngularStack(7, 0, 18, 3, -180, 180)
    result = dict(
        run_id=7,
        targets_tenths=[0, 18],
        complete=False,
        quality="controller_reported",
    )
    assert not parse_progress(json.dumps(result), stack).complete
    for change in (
        {"run_id": True},
        {"run_id": 8},
        {"targets_tenths": [0, 19]},
        {"targets_tenths": [False]},
        {"complete": True},
        {"complete": 1},
        {"quality": "measured"},
        {"extra": 1},
    ):
        with pytest.raises(ValueError):
            parse_progress(json.dumps({**result, **change}), stack)


async def test_cancelled_request_cannot_be_resubmitted():
    client = PvaUnifiedMcu("SIM:", name="mcu")
    await client.connect(mock=True)
    entered = asyncio.Event()

    async def delayed_put(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    put = get_mock_put(client.request)
    put.side_effect = delayed_put
    stack = AngularStack(7, 0, 18, 3, -180, 180)
    task = asyncio.create_task(client.execute(stack, ready=True))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(RuntimeError, match="uncertain"):
        await client.execute(stack, ready=True)
    with pytest.raises(RuntimeError, match="uncertain"):
        await client.execute(AngularStack(8, 0, 18, 3, -180, 180), ready=True)
    assert put.call_count == 1
    await client.abort_stack(7)
    get_mock_put(client.abort).assert_called_once_with("7")


async def test_uncertain_reserved_request_does_not_release():
    client = PvaUnifiedMcu("SIM:", name="mcu")
    await client.connect(mock=True)
    await client.reserve()
    get_mock_put(client.request).side_effect = TimeoutError("unknown outcome")
    with pytest.raises(TimeoutError):
        await client.execute(AngularStack(7, 0, 18, 3, -180, 180), ready=True)
    with pytest.raises(RuntimeError, match="uncertain"):
        await client.release_reservation()
    assert get_mock_put(client.reservation).call_count == 1


async def test_required_reservation_is_checked_before_request():
    client = PvaUnifiedMcu("SIM:", name="mcu")
    await client.connect(mock=True)
    stack = AngularStack(7, 0, 18, 3, -180, 180)
    with pytest.raises(RuntimeError, match="reservation"):
        await client.execute(stack, ready=True, require_reservation=True)
    get_mock_put(client.request).assert_not_called()
    await client.reserve()
    assert client.reservation_held
    set_mock_value(
        client.result,
        json.dumps(
            dict(
                run_id=7,
                targets_tenths=[0, 18, 36],
                complete=True,
                quality="controller_reported",
            )
        ),
    )
    await client.execute(stack, ready=True, require_reservation=True)
    request = json.loads(get_mock_put(client.request).call_args.args[0])
    assert len(request["reservation_token"]) == 32
    await client.release_reservation()
    assert not client.reservation_held


async def test_reusable_firmware_allocates_from_one_without_watermark():
    client = PvaUnifiedMcu("SIM:", name="mcu")
    await client.connect(mock=True)
    set_mock_value(
        client.result,
        json.dumps(
            dict(
                run_id=1,
                targets_tenths=[0, 18, 36],
                complete=True,
                quality="controller_reported",
            )
        ),
    )
    await client.reserve()
    assert await client.allocate_run_ids(2) == (1, 2)
    await client.execute(AngularStack(1, 0, 18, 3, -180, 180), ready=True)
    await client.execute(AngularStack(1, 0, 18, 3, -180, 180), ready=True)
    await client.release_reservation()
    await client.reserve()
    assert await client.allocate_run_ids(1) == (1,)
    assert client.reservation_generation == 2
    await client.release_reservation()
