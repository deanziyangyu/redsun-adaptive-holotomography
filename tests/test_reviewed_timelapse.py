"""Hardware-free reviewed-revisit/time-lapse composition."""

import numpy as np
import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps
from test_unimmcore_software_shots import Owner, make_core

from redsun_aht.device.unified_mcu import AngularStack
from redsun_aht.plans.reviewed_timelapse import (
    RevisitedTarget,
    reviewed_revisit_timelapse_chain,
)
from redsun_aht.plans.timelapse import timelapse_plan
from redsun_aht.plans.timelapse_mcu import timelapse_mcu_plan
from redsun_aht.storage.timelapse_sink import TimeLapseFrameSink


def test_actual_revisit_order_drives_separate_timelapse_run():
    targets = (RevisitedTarget("b", 20, 4), RevisitedTarget("a", 3, 7))
    visited, returned, frames, documents = [], [], [], []

    def anchor():
        yield from bps.null()
        return (1, 2)

    def revisit():
        yield from bps.open_run(md={"purpose": "revisit"})
        yield from bps.close_run()
        return targets

    def move(position):
        visited.append(position.target_id)
        yield from bps.null()

    def capture(point, *args):
        frames.append(point.sequence)
        yield from bps.null()

    def gate():
        yield from bps.null()

    def timelapse(actual):
        yield from timelapse_plan(
            actual,
            2,
            exposure_ms=1,
            move_position=move,
            capture=capture,
            open_shutter=gate,
            close_shutter=gate,
        )

    def restore(position):
        returned.append(position)
        yield from bps.null()

    re = RunEngine({})
    re.subscribe(lambda name, doc: documents.append((name, doc)))
    re(
        reviewed_revisit_timelapse_chain(
            read_anchor=anchor,
            revisit=revisit,
            run_timelapse=timelapse,
            return_to_anchor=restore,
        )
    )
    assert visited == ["b", "a"] and frames == [0, 1, 2, 3]
    assert returned == [(1.0, 2.0)]
    assert len([doc for name, doc in documents if name == "start"]) == 2


def test_empty_revisit_refuses_timelapse():
    calls = []

    def anchor():
        yield from bps.null()
        return (0, 0)

    def revisit():
        yield from bps.null()
        return []

    def unexpected(*args):
        calls.append(args)
        yield from bps.null()

    with pytest.raises(ValueError, match="at least one valid target"):
        RunEngine({})(
            reviewed_revisit_timelapse_chain(
                read_anchor=anchor,
                revisit=revisit,
                run_timelapse=unexpected,
                return_to_anchor=unexpected,
            )
        )
    assert calls == []


def test_return_failure_does_not_replace_success(caplog):
    def anchor():
        yield from bps.null()
        return (0, 0)

    def revisit():
        yield from bps.null()
        return [RevisitedTarget("x", 1, 2)]

    def capture(*args):
        yield from bps.null()

    def restore(*args):
        yield from bps.null()
        raise RuntimeError("return failed")

    RunEngine({})(
        reviewed_revisit_timelapse_chain(
            read_anchor=anchor,
            revisit=revisit,
            run_timelapse=capture,
            return_to_anchor=restore,
        )
    )
    assert "anchor return failed" in caplog.text


def test_reviewed_target_ids_survive_mcu_timelapse_store(tmp_path):
    import json

    core, owner = make_core(), Owner()
    path = tmp_path / "frames"
    sink = TimeLapseFrameSink(str(path), np.zeros((4, 4), dtype=np.uint16), count=16)
    targets = (RevisitedTarget("B", 20, 4), RevisitedTarget("A", 3, 7))
    moves = []

    def action(*args):
        yield from bps.null()

    def anchor():
        yield from bps.null()
        return (0, 0)

    def revisit():
        yield from bps.null()
        return targets

    def move(target):
        moves.append(target.target_id)
        yield from bps.null()

    def timelapse(actual):
        yield from timelapse_mcu_plan(
            owner,
            core,
            sink,
            actual,
            2,
            ["red"],
            [1] * 4,
            position_ids=[target.target_id for target in actual],
            stack_for_block=lambda i: AngularStack(10 + i, 0, 0, 1, -180, 180),
            move_position=move,
            configure_channel=action,
            open_shutter=action,
            close_shutter=action,
            shutter_gate="acquisition",
            camera_exclusive=True,
            illumination_ready=True,
            plane_sequence=True,
            camera_internal_triggered=True,
            axis_order="time-position",
        )

    try:
        RunEngine({})(
            reviewed_revisit_timelapse_chain(
                read_anchor=anchor,
                revisit=revisit,
                run_timelapse=timelapse,
                return_to_anchor=action,
            )
        )
        sink.close()
        frames = json.loads((path / "timelapse-index.json").read_text())["frames"]
        assert moves == ["B", "A", "B", "A"]
        assert [frames[i]["position_id"] for i in range(0, 16, 4)] == moves
    finally:
        core.unloadAllDevices()
