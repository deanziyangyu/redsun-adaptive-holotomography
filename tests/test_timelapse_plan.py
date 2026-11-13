"""Hardware-free IX83-derived time-lapse orchestration tests."""

import json
import time
from itertools import pairwise

import numpy as np
import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps

from redsun_aht.plans.timelapse import timelapse_plan
from redsun_aht.storage.timelapse_sink import TimeLapseFrameSink


@pytest.mark.parametrize("gate", ["per-snap", "acquisition"])
def test_multidimensional_order_gating_and_period(gate):
    operations = []
    starts = []
    docs = []

    def move(pos):
        operations.append(("move", pos))
        yield from bps.null()

    def shutter(state):
        operations.append(("shutter", state))
        yield from bps.null()

    def capture(point, z, channel, pattern):
        starts.append(time.perf_counter())
        operations.append(("frame", point, z, channel, pattern))
        yield from bps.null()

    re = RunEngine({})
    re.subscribe(lambda name, doc: docs.append((name, doc)))
    re(
        timelapse_plan(
            ["left", "right"],
            2,
            z_values=[0, 5],
            channels=["red"],
            patterns=[1, 2],
            frame_period_s=0.005,
            exposure_ms=1,
            shutter_gate=gate,
            move_position=move,
            capture=capture,
            open_shutter=lambda: shutter(True),
            close_shutter=lambda: shutter(False),
        )
    )
    frames = [item for item in operations if item[0] == "frame"]
    assert len(frames) == 16
    assert [item[1].sequence for item in frames] == list(range(16))
    assert [(f[1].position, f[1].timepoint, f[1].z, f[1].pattern) for f in frames] == [
        (p, t, z, s)
        for p in range(2)
        for t in range(2)
        for z in range(2)
        for s in range(2)
    ]
    assert len([d for name, d in docs if name == "event"]) == 16
    assert all(b - a >= 0.004 for a, b in pairwise(starts))
    assert operations.count(("shutter", True)) == (16 if gate == "per-snap" else 1)
    assert operations.count(("shutter", False)) == (16 if gate == "per-snap" else 1)


@pytest.mark.parametrize("gate", ["per-snap", "acquisition"])
def test_failure_closes_shutter_and_stops(gate):
    operations = []

    def action(name):
        operations.append(name)
        yield from bps.null()

    def capture(point, *args):
        operations.append(point.sequence)
        if point.sequence == 1:
            raise RuntimeError("capture failure")
        yield from bps.null()

    re = RunEngine({})
    with pytest.raises(RuntimeError, match="capture failure"):
        re(
            timelapse_plan(
                [0],
                3,
                exposure_ms=1,
                shutter_gate=gate,
                move_position=lambda _: action("move"),
                capture=capture,
                open_shutter=lambda: action("open"),
                close_shutter=lambda: action("close"),
            )
        )
    assert [x for x in operations if isinstance(x, int)] == [0, 1]
    assert operations[-1] == "close"


def test_duration_refused_before_hardware_callback():
    touched = []

    def action(*args):
        touched.append(args)
        yield from bps.null()

    re = RunEngine({})
    with pytest.raises(ValueError, match="estimated duration"):
        re(
            timelapse_plan(
                [0],
                3,
                exposure_ms=1000,
                maximum_duration_s=1,
                move_position=action,
                capture=action,
                open_shutter=action,
                close_shutter=action,
            )
        )
    assert touched == []


@pytest.mark.parametrize("fail", [False, True])
def test_plan_stores_ordered_frames_and_partial_sidecar(tmp_path, fail):
    path = tmp_path / "pixels"
    sink = TimeLapseFrameSink(str(path), np.zeros((2, 2), dtype=np.uint16), count=4)

    def noop(*args):
        yield from bps.null()

    def capture(point, *args):
        if fail and point.sequence == 2:
            raise RuntimeError("synthetic camera failure")
        sink.append(point, np.full((2, 2), point.sequence, dtype=np.uint16))
        yield from bps.null()

    plan = timelapse_plan(
        [0],
        2,
        z_values=[0, 5],
        exposure_ms=1,
        move_position=noop,
        capture=capture,
        open_shutter=noop,
        close_shutter=noop,
    )
    if fail:
        with pytest.raises(RuntimeError, match="synthetic camera failure"):
            RunEngine({})(plan)
        with pytest.raises(RuntimeError, match="incomplete"):
            sink.close()
    else:
        RunEngine({})(plan)
        sink.close()
    sidecar = json.loads((path / "timelapse-index.json").read_text())
    assert sidecar["complete"] is not fail
    assert sidecar["recorded"] == (2 if fail else 4)
    assert [f["sequence"] for f in sidecar["frames"]] == list(
        range(sidecar["recorded"])
    )
    assert all(
        f["host_receipt_ns"] >= f["host_start_ns"] > 0 for f in sidecar["frames"]
    )
    assert sidecar["median_start_interval_ms"] is not None
    assert (path / "timelapse.manifest.json").is_file()
