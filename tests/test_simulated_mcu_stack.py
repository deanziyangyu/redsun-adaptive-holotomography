"""Hardware-free tests of the preliminary local-sequencer model."""

import pytest

from redsun_aht.simulation.mcu_stack import (
    Axis,
    Frame,
    SimulatedMcu,
    SimulatedStage,
    Stack,
    State,
    camera_frames,
    reconcile,
)


def test_finite_stack_and_frame_reconciliation() -> None:
    axes = tuple(
        Axis(name, -100.0, 100.0, 0.25)
        for name in ("survey_x", "survey_y", "coarse_z", "fast_z")
    )
    stage = SimulatedStage(axes)
    assert stage.move({"survey_x": 5.0, "survey_y": -5.0})["survey_x"] == 5.0
    before = dict(stage.positions)
    with pytest.raises(ValueError):
        stage.move({"survey_x": 1.0, "fast_z": 1000.0})
    assert stage.positions == before
    mcu = SimulatedMcu(axes[-1])
    stack = Stack("run-1", (0.0, 0.25, 0.5), 2)
    mcu.reserve("plan")
    with pytest.raises(RuntimeError):
        mcu.reserve("manual")
    mcu.commit("plan", stack)
    with pytest.raises(RuntimeError):
        mcu.arm("plan", camera_ready=False, referenced=True)
    mcu.arm("plan", camera_ready=True, referenced=True)
    edges = mcu.start("plan", "run-1")
    assert mcu.start_count == 1
    assert [e.target_um for e in edges] == list(stack.targets_um)
    assert len(reconcile(edges, camera_frames(edges), stack)) == 3
    with pytest.raises(RuntimeError):
        mcu.start("plan", "run-1")
    mcu.release("plan")


def test_partial_stack_and_bad_frame_evidence_are_rejected() -> None:
    mcu = SimulatedMcu(Axis("fast_z", -10, 10, 0.25))
    stack = Stack("r", (0.0, 0.25, 0.5), 1)
    mcu.reserve("plan")
    mcu.commit("plan", stack)
    mcu.arm("plan", camera_ready=True, referenced=True)
    partial = mcu.start("plan", "r", fail_before_plane=1)
    assert mcu.state is State.ABORTED
    with pytest.raises(ValueError):
        reconcile(partial, camera_frames(partial), stack)
    mcu.release("plan")
    mcu2 = SimulatedMcu(mcu.fast_z)
    mcu2.reserve("plan")
    mcu2.commit("plan", stack)
    mcu2.arm("plan", camera_ready=True, referenced=True)
    edges = mcu2.start("plan", "r")
    for bad in (
        (Frame("r", 0), Frame("r", 0), Frame("r", 2)),
        (Frame("r", 1), Frame("r", 0), Frame("r", 2)),
        (Frame("wrong", 0), Frame("r", 1), Frame("r", 2)),
    ):
        with pytest.raises(ValueError):
            reconcile(edges, bad, stack)


def test_limits_quantization_and_invalid_commit_preserve_state() -> None:
    axis = Axis("fast_z", -1, 1, 0.25)
    mcu = SimulatedMcu(axis)
    mcu.reserve("plan")
    for targets in ((0.0, 1.25), (0.0, 0.1), (float("nan"),)):
        with pytest.raises(ValueError):
            mcu.commit("plan", Stack("r", targets, 1))
        assert mcu.state is State.EMPTY
    with pytest.raises(RuntimeError):
        mcu.commit("other", Stack("r", (0.0,), 1))


def test_bluesky_rehearsal_records_and_cleanup() -> None:
    from bluesky import RunEngine

    from redsun_aht.simulation.stack_plan import simulated_stack_plan

    mcu = SimulatedMcu(Axis("fast_z", -1, 1, 0.25))
    stack = Stack("bluesky-stack", (0.0, 0.25, 0.5), 2)
    documents = []
    engine = RunEngine({})
    engine.subscribe(lambda name, document: documents.append((name, document)))
    engine(simulated_stack_plan(mcu, stack))
    events = [
        doc for name, doc in documents if name == "event" and "plane" in doc["data"]
    ]
    assert [event["data"]["plane"] for event in events] == [0, 1, 2]
    assert (
        len(
            [
                doc
                for name, doc in documents
                if name == "event" and "edge_plane" in doc["data"]
            ]
        )
        == 3
    )
    assert all(event["data"]["evidence"] == "simulated" for event in events)
    assert mcu.start_count == 1 and mcu.owner is None
    assert (
        next(doc for name, doc in documents if name == "stop")["exit_status"]
        == "success"
    )


def test_bluesky_failure_is_recorded_and_owner_released() -> None:
    from bluesky import RunEngine

    from redsun_aht.simulation.stack_plan import simulated_stack_plan

    mcu = SimulatedMcu(Axis("fast_z", -1, 1, 0.25))
    stack = Stack("partial", (0.0, 0.25, 0.5), 2)
    documents = []
    engine = RunEngine({})
    engine.subscribe(lambda name, document: documents.append((name, document)))
    with pytest.raises(ValueError, match="incomplete"):
        engine(simulated_stack_plan(mcu, stack, fail_before_plane=1))
    assert len(mcu.edges) == 1 and mcu.owner is None
    events = [doc for name, doc in documents if name == "event"]
    assert len(events) == 1 and events[0]["data"]["edge_plane"] == 0
    assert events[0]["data"]["edge_evidence"] == "simulated"
    assert (
        next(doc for name, doc in documents if name == "stop")["exit_status"] == "fail"
    )


def test_partial_edges_survive_document_persistence(tmp_path) -> None:
    import json

    from bluesky import RunEngine

    from redsun_aht.simulation.stack_plan import simulated_stack_plan

    path = tmp_path / "documents.jsonl"
    mcu = SimulatedMcu(Axis("fast_z", -1, 1, 0.25))
    stack = Stack("persisted-partial", (0.0, 0.25), 2)
    engine = RunEngine({})
    with path.open("w", encoding="utf-8") as stream:
        engine.subscribe(lambda name, doc: stream.write(json.dumps([name, doc]) + "\n"))
        with pytest.raises(ValueError, match="incomplete"):
            engine(simulated_stack_plan(mcu, stack, fail_before_plane=1))
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    edges = [
        doc for name, doc in docs if name == "event" and "edge_plane" in doc["data"]
    ]
    assert len(edges) == 1 and edges[0]["data"]["edge_plane"] == 0
    assert next(doc for name, doc in docs if name == "stop")["exit_status"] == "fail"
    assert mcu.owner is None


def test_synthetic_pixel_store_keeps_complete_and_partial_runs(tmp_path) -> None:
    import numpy as np
    import zarr
    from bluesky import RunEngine

    from redsun_aht.simulation.stack_plan import simulated_stack_plan
    from redsun_aht.storage.writer import RunFrameStore

    for failure in (None, 1):
        path = tmp_path / f"stack-{failure}.zarr"
        store = RunFrameStore(
            str(path), "simulated", np.zeros((4, 4), dtype=np.uint16), planned_frames=3
        )
        mcu = SimulatedMcu(Axis("fast_z", -1, 1, 0.25))
        stack = Stack(f"run-{failure}", (0.0, 0.25, 0.5), 2)
        engine = RunEngine({})
        documents = []
        engine.subscribe(
            lambda name, doc, captured=documents: captured.append((name, doc))
        )
        if failure is None:
            engine(simulated_stack_plan(mcu, stack, frame_store=store))
            expected = 3
        else:
            with pytest.raises(ValueError, match="incomplete"):
                engine(
                    simulated_stack_plan(
                        mcu, stack, frame_store=store, fail_before_plane=failure
                    )
                )
            expected = 1
        array = zarr.open_array(str(path / "simulated"), mode="r")
        assert store.closed and store.manifest is not None
        assert store.manifest.frame_count == expected
        assert store.manifest.planned_frames == 3
        assert array.shape == (expected, 4, 4)
        assert list(array[:, 0, 0]) == list(range(expected))
        matched = [
            doc["data"]
            for name, doc in documents
            if name == "event" and "store_index" in doc["data"]
        ]
        if failure is None:
            assert [entry["store_index"] for entry in matched] == [0, 1, 2]
            assert [int(array[entry["store_index"], 0, 0]) for entry in matched] == [
                entry["frame_plane"] for entry in matched
            ]
            assert all(entry["evidence"] == "simulated" for entry in matched)
        else:
            assert not matched  # Partial frame remains unassociated.
        assert mcu.owner is None


def test_store_close_failure_still_releases_simulated_axis() -> None:
    from unittest.mock import Mock

    from bluesky import RunEngine

    from redsun_aht.simulation.stack_plan import simulated_stack_plan

    mcu = SimulatedMcu(Axis("fast_z", -1, 1, 0.25))
    store = Mock()
    store.close.side_effect = OSError("simulated storage failure")
    with pytest.raises(OSError, match="simulated storage failure"):
        RunEngine({})(
            simulated_stack_plan(mcu, Stack("r", (0.0,), 1), frame_store=store)
        )
    assert mcu.owner is None


def test_reservation_conflict_closes_store_without_releasing_other_owner() -> None:
    from unittest.mock import Mock

    from bluesky import RunEngine

    from redsun_aht.simulation.stack_plan import simulated_stack_plan

    mcu = SimulatedMcu(Axis("fast_z", -1, 1, 0.25))
    mcu.reserve("other")
    store = Mock()
    with pytest.raises(RuntimeError, match="reserved"):
        RunEngine({})(
            simulated_stack_plan(mcu, Stack("r", (0.0,), 1), frame_store=store)
        )
    store.close.assert_called_once()
    assert mcu.owner == "other"


def test_headless_redsun_session_exposes_simulated_plan() -> None:
    from bluesky import RunEngine

    from redsun_aht.simulation.session import SimulatedStackSession

    session = SimulatedStackSession().build()
    try:
        plan = session.stack.plan_map()["simulated_zstack"]["plan"]
        docs = []
        engine = RunEngine({})
        engine.subscribe(lambda name, doc: docs.append((name, doc)))
        engine(plan())
        events = [
            doc for name, doc in docs if name == "event" and "plane" in doc["data"]
        ]
        assert [event["data"]["plane"] for event in events] == [0, 1, 2]
        assert not session.services
    finally:
        session.shutdown()


def test_shared_sequencer_requires_rearming_but_allows_tag_reuse() -> None:
    mcu = SimulatedMcu(Axis("fast_z", -1, 1, 0.25))
    stack = Stack("reusable", (0.0,), 1)
    mcu.reserve("owner")
    mcu.commit("owner", stack)
    mcu.arm("owner", camera_ready=True, referenced=True)
    mcu.start("owner", stack.run_id)
    with pytest.raises(RuntimeError, match="start refused"):
        mcu.start("owner", stack.run_id)
    mcu.release("owner")
    mcu.reserve("owner")
    mcu.commit("owner", stack)
    mcu.arm("owner", camera_ready=True, referenced=True)
    mcu.start("owner", stack.run_id)
    assert mcu.start_count == 2
    mcu.release("owner")


def test_presenter_shared_controller_conflict_and_second_run() -> None:
    from bluesky import RunEngine

    from redsun_aht.simulation.session import SimulatedStackSession

    session = SimulatedStackSession().build()
    try:
        mcu = session.stack.mcu
        mcu.reserve("other-client")
        with pytest.raises(RuntimeError, match="reserved"):
            RunEngine({})(session.stack.run_stack())
        assert mcu.owner == "other-client"
        mcu.cancel("other-client")
        mcu.release("other-client")
        RunEngine({})(session.stack.run_stack())
        RunEngine({})(session.stack.run_stack())
        assert mcu.start_count == 2
    finally:
        session.shutdown()


def test_shared_stage_stack_ownership_and_partial_position() -> None:
    """A stack reserves the synthetic composite and preserves partial Z."""
    x = Axis("survey_x", -10, 10, 0.25)
    z = Axis("fast_z", -10, 10, 0.25)
    stage = SimulatedStage((x, z))
    stage.move({"survey_x": 1.0})
    mcu = SimulatedMcu(z, stage=stage)
    mcu.reserve("stack-owner")
    with pytest.raises(RuntimeError, match="another caller"):
        stage.move({"survey_x": 2.0})
    mcu.commit("stack-owner", Stack("r", (0.0, 0.25, 0.5), 1))
    mcu.arm("stack-owner", camera_ready=True, referenced=True)
    edges = mcu.start("stack-owner", "r", fail_before_plane=2)
    assert len(edges) == 2 and stage.positions == {"survey_x": 1.0, "fast_z": 0.25}
    mcu.release("stack-owner")
    assert stage.move({"survey_x": 2.0})["survey_x"] == 2.0


def test_mcu_reservation_fails_when_stage_is_held() -> None:
    """Do not partially reserve the MCU if another client owns the stage."""
    z = Axis("fast_z", -1, 1, 0.25)
    stage = SimulatedStage((z,))
    stage.reserve("manual")
    mcu = SimulatedMcu(z, stage=stage)
    with pytest.raises(RuntimeError, match="already reserved"):
        mcu.reserve("stack")
    assert mcu.owner is None and stage.owner == "manual"
