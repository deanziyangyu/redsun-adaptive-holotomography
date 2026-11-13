"""Tests of the hardware-free simulated stack service semantics."""

import json

import pytest

from redsun_aht.services.simulated_stack import SimulatedStackController


async def test_one_request_runs_locally_and_tag_can_be_reused() -> None:
    service = SimulatedStackController()
    request = json.dumps({"run_id": "r1", "targets_um": [0, 0.25, 0.5], "pattern": 2})
    await service.request.put(request)
    result = json.loads(service.result.get())
    assert result["quality"] == "simulated"
    assert result["camera_frame_sequences"] == [0, 1, 2]
    assert result["final_positions_um"]["fast_z"] == 0.5
    assert service.stage.owner is None
    assert [edge["target_um"] for edge in result["edges"]] == [0, 0.25, 0.5]
    assert service.mcu.start_count == 1
    await service.request.put(request)
    assert service.mcu.start_count == 2
    assert service.mcu.owner is None


async def test_invalid_targets_do_not_start_or_claim_resources() -> None:
    service = SimulatedStackController()
    for targets in ([0, 1000], [0, 0.1], [True]):
        with pytest.raises(ValueError):
            await service.request.put(
                json.dumps({"run_id": "r", "targets_um": targets, "pattern": 1})
            )
        assert service.mcu.start_count == 0
        assert service.mcu.owner is None


async def test_simulated_camera_drop_is_independent_of_trigger_count() -> None:
    """The fake camera can miss an edge without changing MCU edge count."""
    from redsun_aht.simulation.mcu_stack import (
        Axis,
        SimulatedMcu,
        SimulatedTriggeredCamera,
        Stack,
    )

    mcu = SimulatedMcu(Axis("fast_z", -1, 1, 0.25))
    camera = SimulatedTriggeredCamera(drop_trigger=1)
    camera.arm()
    mcu.reserve("plan")
    mcu.commit("plan", Stack("r", (0.0, 0.25, 0.5), 2))
    mcu.arm("plan", camera_ready=camera.armed, referenced=True)
    edges = mcu.start("plan", "r", on_edge=camera.receive)
    assert len(edges) == camera.triggers == 3
    assert camera.frames == [0, 1]


async def test_camera_error_publishes_partial_evidence_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-trigger failure retains both counts without inventing a frame."""
    from redsun_aht.simulation.mcu_stack import Edge, SimulatedTriggeredCamera

    original_receive = SimulatedTriggeredCamera.receive

    def fail_second(camera: SimulatedTriggeredCamera, edge: Edge) -> None:
        if edge.plane == 1:
            raise RuntimeError("synthetic camera failure")
        original_receive(camera, edge)

    monkeypatch.setattr(SimulatedTriggeredCamera, "receive", fail_second)
    service = SimulatedStackController()
    request = json.dumps(
        {"run_id": "camera-fault", "targets_um": [0, 0.25, 0.5], "pattern": 1}
    )
    with pytest.raises(RuntimeError, match="synthetic camera failure"):
        await service.request.put(request)
    result = json.loads(service.result.get())
    assert result["run_id"] == "camera-fault" and result["state"] == "aborted"
    assert [edge["plane"] for edge in result["edges"]] == [0, 1]
    assert result["camera_frame_sequences"] == [0]
    assert result["final_positions_um"]["fast_z"] == 0.25
    assert json.loads(service.positions.get())["fast_z"] == 0.25
    assert service.stage.owner is None and not service.busy.get()
    monkeypatch.setattr(SimulatedTriggeredCamera, "receive", original_receive)
    await service.request.put(request)
    assert service.mcu.start_count == 2


async def test_camera_not_ready_stops_before_next_move_or_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A camera losing readiness stops the next synthetic plane at admission."""
    from redsun_aht.simulation.mcu_stack import Edge, SimulatedTriggeredCamera

    original_receive = SimulatedTriggeredCamera.receive

    def lose_readiness(camera: SimulatedTriggeredCamera, edge: Edge) -> None:
        original_receive(camera, edge)
        camera.armed = False

    monkeypatch.setattr(SimulatedTriggeredCamera, "receive", lose_readiness)
    service = SimulatedStackController()
    request = json.dumps(
        {"run_id": "not-ready", "targets_um": [0, 0.25, 0.5], "pattern": 1}
    )
    with pytest.raises(RuntimeError, match="camera not ready"):
        await service.request.put(request)
    result = json.loads(service.result.get())
    assert result["state"] == "aborted"
    assert [edge["plane"] for edge in result["edges"]] == [0]
    assert result["camera_frame_sequences"] == [0]
    assert result["final_positions_um"]["fast_z"] == 0.0
    assert service.stage.owner is None and not service.busy.get()


async def test_manual_is_rejected_during_stack_and_not_queued() -> None:
    """A competing fake survey write cannot run after a stack finishes."""
    import asyncio

    service = SimulatedStackController()
    request = json.dumps({"run_id": "long", "targets_um": [0, 0.25, 0.5], "pattern": 1})
    stack_task = asyncio.create_task(service.request.put(request))
    await asyncio.sleep(0.02)
    with pytest.raises(RuntimeError, match="already reserved"):
        await service.manual.put(json.dumps({"survey_x": 1.0}))
    await stack_task
    assert service.stage.positions["survey_x"] == 0.0
    assert service.stage.positions["fast_z"] == 0.5
    assert service.stage.owner is None
    await service.manual.put(json.dumps({"survey_x": 1.0}))
    assert service.stage.positions["survey_x"] == 1.0


async def test_abort_preserves_partial_edges_and_releases_stage() -> None:
    """Abort stops future fake planes and makes the request fail explicitly."""
    import asyncio

    service = SimulatedStackController()
    request = json.dumps(
        {"run_id": "abort", "targets_um": [0, 0.25, 0.5, 0.75], "pattern": 1}
    )
    stack_task = asyncio.create_task(service.request.put(request))
    await asyncio.sleep(0.075)
    with pytest.raises(RuntimeError, match="no matching"):
        await service.abort.put("different-run")
    assert not stack_task.done()
    await service.abort.put("abort")
    with pytest.raises(RuntimeError, match="aborted"):
        await stack_task
    result = json.loads(service.result.get())
    assert result["state"] == "aborted"
    assert 0 < len(result["edges"]) < 4
    assert len(result["edges"]) == len(result["camera_frame_sequences"])
    assert json.loads(service.positions.get()) == service.stage.positions
    assert service.stage.owner is None and not service.busy.get()
    count = len(service.mcu.edges)
    await asyncio.sleep(0.1)
    assert len(service.mcu.edges) == count
    await service.request.put(request)
    assert service.mcu.start_count == 2


async def test_multiaxis_jog_rejects_entire_request_if_one_axis_exceeds_limit():
    service = SimulatedStackController()
    await service.manual.put(json.dumps({"survey_x": 100.0}))
    before = dict(service.stage.positions)
    with pytest.raises(ValueError):
        await service.jog.put(json.dumps({"survey_x": 1, "survey_y": 1, "coarse_z": 1}))
    assert service.stage.positions == before
    assert service.stage.owner is None
    await service.jog.put(json.dumps({"survey_x": -1, "survey_y": 1, "coarse_z": 1}))
    assert service.stage.positions == {
        "survey_x": 99.0,
        "survey_y": 1.0,
        "coarse_z": 1.0,
        "fast_z": 0.0,
    }


async def test_service_reservation_blocks_unowned_operations_and_releases():
    service = SimulatedStackController()
    token = "a" * 32
    await service.reservation.put(f"acquire {token}")
    with pytest.raises(RuntimeError, match="reservation"):
        await service.manual.put(json.dumps({"survey_x": 1}))
    with pytest.raises(RuntimeError, match="reservation"):
        await service.jog.put(json.dumps({"survey_x": 1}))
    request = {"run_id": "run", "targets_um": [0], "pattern": 1}
    with pytest.raises(RuntimeError, match="reservation"):
        await service.request.put(json.dumps(request))
    await service.manual.put(
        json.dumps({"targets": {"survey_x": 1}, "reservation_token": token})
    )
    await service.request.put(json.dumps({**request, "reservation_token": token}))
    with pytest.raises(RuntimeError, match="reservation"):
        await service.reservation.put(f"release {'b' * 32}")
    await service.reservation.put(f"release {token}")
    await service.jog.put(json.dumps({"survey_x": 0.25}))
    assert service.stage.positions["survey_x"] == 1.25
