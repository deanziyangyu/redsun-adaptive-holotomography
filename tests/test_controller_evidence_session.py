"""Headless coarse-readback/fast-stack controller-evidence plan tests."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from bluesky import RunEngine

from redsun_aht.device.controller_evidence_session import controller_stack_plan
from redsun_aht.device.esp300_readonly import AxisSnapshot, ControllerSnapshot
from redsun_aht.device.pva_esp300_readonly import Esp300Observation
from redsun_aht.device.unified_mcu import AngularStack, StackProgress


def test_plan_records_distinct_coarse_and_fast_evidence():
    coarse = AsyncMock()
    axes = tuple(
        AxisSnapshot(i, f"axis{i}", 0.1, 0, 100, 0, 0, True) for i in (1, 2, 3)
    )
    coarse.read_observation.return_value = Esp300Observation(
        1, 2, ControllerSnapshot("ESP300", axes)
    )
    fast = AsyncMock()
    fast.execute.return_value = StackProgress(7, (0, 18, 36), complete=True)
    documents = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))
    engine(
        controller_stack_plan(
            coarse, fast, AngularStack(7, 0, 18, 3, -180, 180), ready=True
        )
    )
    assert coarse.read_observation.await_count == 1 and fast.execute.await_count == 1
    events = [doc["data"] for name, doc in documents if name == "event"]
    assert (
        json.loads(events[0]["coarse_snapshot_json"])["snapshot"]["quality"]
        == "controller_reported"
    )
    assert json.loads(events[1]["fast_stack_json"])["targets_tenths"] == [0, 18, 36]
    assert (
        next(doc for name, doc in documents if name == "start")["camera_frames"]
        is False
    )
    fast.abort_stack.assert_not_awaited()


def test_failed_coarse_read_never_starts_stack():
    coarse = AsyncMock()
    coarse.read_observation.side_effect = RuntimeError("coarse unavailable")
    fast = AsyncMock()
    with pytest.raises(RuntimeError, match="coarse unavailable"):
        RunEngine({})(
            controller_stack_plan(
                coarse, fast, AngularStack(7, 0, 18, 3, -180, 180), ready=True
            )
        )
    fast.execute.assert_not_awaited()


def test_timeout_requests_single_abort_and_does_not_replay():
    coarse = AsyncMock()
    axes = tuple(
        AxisSnapshot(i, f"axis{i}", 0.1, 0, 100, 0, 0, True) for i in (1, 2, 3)
    )
    coarse.read_observation.return_value = Esp300Observation(
        1, 2, ControllerSnapshot("ESP300", axes)
    )
    fast = AsyncMock()

    async def never(*args, **kwargs):
        await asyncio.Event().wait()

    fast.execute.side_effect = never
    with pytest.raises(TimeoutError):
        RunEngine({})(
            controller_stack_plan(
                coarse,
                fast,
                AngularStack(7, 0, 18, 3, -180, 180),
                ready=True,
                timeout_s=0.02,
            )
        )
    assert fast.execute.await_count == 1
    fast.abort_stack.assert_awaited_once_with(7)
