"""Tests for the simulated stack PVA client's result validation."""

import json

import pytest
from ophyd_async.core import get_mock_put, set_mock_value

from redsun_aht.simulation.mcu_stack import Stack
from redsun_aht.simulation.pva_stack import PvaSimulatedStack, StackEvidence


async def test_client_submits_once_and_checks_result() -> None:
    client = PvaSimulatedStack("SIM:", name="sim")
    await client.connect(mock=True)
    stack = Stack("r1", (0.0, 0.25), 2)
    set_mock_value(
        client.result,
        json.dumps(
            {
                "run_id": "r1",
                "quality": "simulated",
                "state": "complete",
                "camera_frame_sequences": [0, 1],
                "final_positions_um": {"fast_z": 0.25},
                "edges": [
                    {"run_id": "r1", "plane": 0, "target_um": 0.0, "pattern": 2},
                    {"run_id": "r1", "plane": 1, "target_um": 0.25, "pattern": 2},
                ],
            }
        ),
    )
    result = await client.execute(stack)
    assert [edge.target_um for edge in result.edges] == [0.0, 0.25]
    assert get_mock_put(client.request).call_count == 1


async def test_client_refuses_stale_or_mislabelled_result() -> None:
    client = PvaSimulatedStack("SIM:", name="sim")
    await client.connect(mock=True)
    stack = Stack("r1", (0.0,), 2)
    for result in (
        {"run_id": "other", "quality": "simulated", "state": "complete", "edges": []},
        {"run_id": "r1", "quality": "measured", "state": "complete", "edges": []},
        {"run_id": "r1", "quality": "simulated", "state": "complete", "edges": []},
        {
            "run_id": "r1",
            "quality": "simulated",
            "state": "complete",
            "edges": [{"run_id": "r1", "plane": 1, "target_um": 0.0, "pattern": 2}],
        },
    ):
        set_mock_value(client.result, json.dumps(result))
        with pytest.raises(ValueError):
            await client.execute(stack)
    assert get_mock_put(client.request).call_count == 4


def test_pva_stack_plan_submits_one_request_and_emits_simulated_events() -> None:
    from unittest.mock import AsyncMock

    from bluesky import RunEngine

    from redsun_aht.simulation.mcu_stack import Edge
    from redsun_aht.simulation.pva_stack_plan import pva_stack_plan

    stack = Stack("pva-plan", (0.0, 0.25), 2)
    client = AsyncMock()
    client.execute.return_value = StackEvidence(
        (Edge("pva-plan", 0, 0.0, 2), Edge("pva-plan", 1, 0.25, 2)),
        (0, 1),
    )
    documents = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))
    engine(pva_stack_plan(client, stack))
    assert client.execute.await_count == 1
    assert [doc["data"]["plane"] for name, doc in documents if name == "event"] == [
        0,
        1,
    ]


def test_pva_stack_timeout_does_not_resubmit() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from bluesky import RunEngine

    from redsun_aht.simulation.pva_stack_plan import pva_stack_plan

    async def never_reply(_stack):
        await asyncio.sleep(60)

    client = AsyncMock()
    client.execute.side_effect = never_reply
    partial = json.dumps({"run_id": "timeout", "state": "aborted", "edges": [0]})
    client.wait_for_aborted_result.return_value = partial
    documents = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))
    with pytest.raises(TimeoutError):
        engine(pva_stack_plan(client, Stack("timeout", (0.0,), 1), timeout_s=0.01))
    assert client.execute.await_count == 1
    client.abort_stack.assert_awaited_once_with("timeout")
    event = next(doc["data"] for name, doc in documents if name == "event")
    assert event["partial_result_json"] == partial
    assert event["association"] == "unassociated"
    assert (
        next(doc for name, doc in documents if name == "stop")["exit_status"] == "fail"
    )


async def test_partial_result_waits_for_matching_aborted_run() -> None:
    import asyncio

    client = PvaSimulatedStack("SIM:", name="sim")
    await client.connect(mock=True)
    set_mock_value(client.result, json.dumps({"run_id": "older", "state": "aborted"}))
    waiter = asyncio.create_task(client.wait_for_aborted_result("current"))
    await asyncio.sleep(0.03)
    assert not waiter.done()
    partial = {
        "run_id": "current",
        "state": "aborted",
        "quality": "simulated",
        "edges": [],
        "camera_frame_sequences": [],
    }
    set_mock_value(client.result, json.dumps(partial))
    assert json.loads(await asyncio.wait_for(waiter, 1)) == partial


async def test_client_rejects_missing_or_duplicate_camera_counters() -> None:
    """A complete edge list is insufficient when camera evidence disagrees."""
    client = PvaSimulatedStack("SIM:", name="sim")
    await client.connect(mock=True)
    stack = Stack("r1", (0.0, 0.25), 2)
    edges = [
        {"run_id": "r1", "plane": 0, "target_um": 0.0, "pattern": 2},
        {"run_id": "r1", "plane": 1, "target_um": 0.25, "pattern": 2},
    ]
    for frames in ([0], [0, 0], [0, 1, 2]):
        set_mock_value(
            client.result,
            json.dumps(
                {
                    "run_id": "r1",
                    "quality": "simulated",
                    "state": "complete",
                    "edges": edges,
                    "camera_frame_sequences": frames,
                }
            ),
        )
        with pytest.raises(ValueError, match="frame count or order"):
            await client.execute(stack)


def test_pva_stack_plan_persists_synthetic_frames(tmp_path) -> None:
    """Reopen stored pixels and verify event-to-store indices."""
    from unittest.mock import AsyncMock

    import zarr
    from bluesky import RunEngine

    from redsun_aht.simulation.mcu_stack import Edge
    from redsun_aht.simulation.pva_stack_plan import pva_stack_plan
    from redsun_aht.storage.writer import manifest_path

    stack = Stack("store-run", (0.0, 0.25), 2)
    client = AsyncMock()
    client.execute.return_value = StackEvidence(
        (Edge("store-run", 0, 0.0, 2), Edge("store-run", 1, 0.25, 2)), (0, 1)
    )
    path = tmp_path / "stack.zarr"
    docs = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    engine(pva_stack_plan(client, stack, store_uri=str(path)))
    events = [doc["data"] for name, doc in docs if name == "event"]
    array = zarr.open_array(str(path / "simulated"), mode="r")
    manifest = json.loads(
        manifest_path(str(path), "simulated").read_text(encoding="utf-8")
    )
    assert array.shape == (2, 4, 4)
    assert manifest["frame_count"] == 2 and manifest["planned_frames"] == 2
    assert [event["store_index"] for event in events] == [0, 1]
    assert [int(array[event["store_index"], 0, 0]) for event in events] == [0, 1]
    assert all(event["evidence"] == "simulated" for event in events)


def test_pva_stack_timeout_closes_empty_store(tmp_path) -> None:
    """Keep a short manifest when no camera evidence arrives before timeout."""
    import asyncio
    from unittest.mock import AsyncMock

    from bluesky import RunEngine

    from redsun_aht.simulation.pva_stack_plan import pva_stack_plan
    from redsun_aht.storage.writer import manifest_path

    async def never_reply(_stack):
        await asyncio.sleep(60)

    client = AsyncMock()
    client.execute.side_effect = never_reply
    path = tmp_path / "partial.zarr"
    with pytest.raises(TimeoutError):
        RunEngine({})(
            pva_stack_plan(
                client, Stack("r", (0.0,), 1), timeout_s=0.01, store_uri=str(path)
            )
        )
    manifest = json.loads(
        manifest_path(str(path), "simulated").read_text(encoding="utf-8")
    )
    assert manifest["frame_count"] == 0 and manifest["planned_frames"] == 1
    assert client.execute.await_count == 1


async def test_client_rejects_wrong_final_fast_z() -> None:
    """A full trigger/frame count does not excuse the wrong final readback."""
    client = PvaSimulatedStack("SIM:", name="sim")
    await client.connect(mock=True)
    stack = Stack("r", (0.0, 0.25), 2)
    set_mock_value(
        client.result,
        json.dumps(
            {
                "run_id": "r",
                "quality": "simulated",
                "state": "complete",
                "edges": [
                    {"run_id": "r", "plane": 0, "target_um": 0.0, "pattern": 2},
                    {"run_id": "r", "plane": 1, "target_um": 0.25, "pattern": 2},
                ],
                "camera_frame_sequences": [0, 1],
                "final_positions_um": {"fast_z": 0.0},
            }
        ),
    )
    with pytest.raises(ValueError, match="fast-Z readback mismatch"):
        await client.execute(stack)


async def test_typed_client_sends_single_simulated_abort_request() -> None:
    """The abort signal is explicit and does not submit a stack request."""
    client = PvaSimulatedStack("SIM:", name="sim")
    await client.connect(mock=True)
    await client.abort_stack("r")
    get_mock_put(client.abort).assert_called_once_with("r")
    get_mock_put(client.request).assert_not_called()


def test_timeout_preserves_original_error_if_abort_is_refused() -> None:
    """An unconfirmed abort cannot turn a timed-out request into success."""
    import asyncio
    from unittest.mock import AsyncMock

    from bluesky import RunEngine

    from redsun_aht.simulation.pva_stack_plan import pva_stack_plan

    async def never_reply(_stack):
        await asyncio.sleep(60)

    client = AsyncMock()
    client.execute.side_effect = never_reply
    client.abort_stack.side_effect = RuntimeError("no matching run")
    with pytest.raises(TimeoutError):
        RunEngine({})(pva_stack_plan(client, Stack("r", (0.0,), 1), timeout_s=0.01))
    client.abort_stack.assert_awaited_once_with("r")
    assert client.execute.await_count == 1
