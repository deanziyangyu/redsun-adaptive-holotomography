"""Hardware-free time-lapse composition through the MCU MDA bridge."""

import json
from itertools import pairwise

import numpy as np
import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps
from test_unimmcore_software_shots import Owner, make_core

from redsun_aht.device.unified_mcu import AngularStack
from redsun_aht.plans.timelapse_mcu import timelapse_mcu_plan
from redsun_aht.storage.timelapse_sink import TimeLapseFrameSink


@pytest.mark.parametrize("bound", [True, False])
@pytest.mark.parametrize("sequence_mode", [False, True])
def test_timelapse_mcu_blocks_preserve_indices_and_store(
    tmp_path, bound, sequence_mode, monkeypatch
):
    from pymmcore_plus.mda import MDAEngine

    def forbidden(*args, **kwargs):
        raise AssertionError("the time-lapse path must not use MDAEngine")

    monkeypatch.setattr(MDAEngine, "exec_single_event", forbidden)
    core = make_core()

    class HoldOwner(Owner):
        def __init__(self):
            super().__init__()
            self.holds = []

        async def software_shots(self, *args, **kwargs):
            self.holds.append(kwargs["hold_illumination"])
            await super().software_shots(*args, **kwargs)

    owner = HoldOwner()
    path = tmp_path / "frames"
    sink = TimeLapseFrameSink(str(path), np.zeros((4, 4), dtype=np.uint16), count=16)
    operations, planes, events = [], [], []

    def action(value):
        operations.append(value)
        yield from bps.null()

    re = RunEngine({})
    re.subscribe(lambda name, doc: events.append((name, doc)))
    try:
        re(
            timelapse_mcu_plan(
                owner,
                core,
                sink,
                ["xy"],
                2,
                ["red"],
                [1] * 4,
                stack_for_block=lambda i: AngularStack(10 + i, 0, 18, 2, -180, 180),
                move_position=lambda value: action(("move", value)),
                configure_channel=lambda value: action(("channel", value)),
                open_shutter=lambda: action("open"),
                close_shutter=lambda: action("close"),
                shutter_gate="acquisition",
                camera_exclusive=True,
                illumination_ready=True,
                frame_period_s=0 if sequence_mode else 0.005,
                expected_exposure_ms=1.0,
                plane_sequence=sequence_mode,
                camera_internal_triggered=sequence_mode,
                bind_stage=bound,
                before_plane=None
                if bound
                else lambda plane, target: planes.append((plane, target)),
            )
        )
        sink.close()
        sidecar = json.loads((path / "timelapse-index.json").read_text())
        assert sidecar["complete"] and sidecar["recorded"] == 16
        starts = [frame["host_start_ns"] for frame in sidecar["frames"]]
        if sequence_mode:
            assert starts == [-1] * 16
            assert sidecar["median_start_interval_ms"] is None
            assert sidecar["median_receipt_interval_ms"] is not None
        else:
            assert all(b - a >= 5_000_000 for a, b in pairwise(starts))
        assert [(f["timepoint"], f["z"], f["pattern"]) for f in sidecar["frames"]] == [
            (t, z, s) for t in range(2) for z in range(2) for s in range(4)
        ]
        assert len([doc for kind, doc in events if kind == "event"]) == 16
        assert owner.requested == [0, 1, 0, 1]
        assert owner.holds == [True, False]
        assert planes == ([] if bound else [(0, 0), (1, 18)] * 2)
        assert operations[0] == "open" and operations[-1] == "close"
    finally:
        core.unloadAllDevices()


def test_exposure_mismatch_refused_before_callbacks(tmp_path):
    core = make_core()
    sink = TimeLapseFrameSink(
        str(tmp_path / "frames"), np.zeros((4, 4), dtype=np.uint16), count=4
    )
    operations = []

    def action(value):
        operations.append(value)
        yield from bps.null()

    try:
        with pytest.raises(ValueError, match="exposure readback mismatch"):
            RunEngine({})(
                timelapse_mcu_plan(
                    Owner(),
                    core,
                    sink,
                    [0],
                    1,
                    ["red"],
                    [1] * 4,
                    stack_for_block=lambda i: AngularStack(10 + i, 0, 0, 1, -180, 180),
                    move_position=action,
                    configure_channel=action,
                    open_shutter=lambda: action("open"),
                    close_shutter=lambda: action("close"),
                    shutter_gate="acquisition",
                    camera_exclusive=True,
                    illumination_ready=True,
                    expected_exposure_ms=2.0,
                )
            )
        assert operations == []
        with pytest.raises(RuntimeError, match="incomplete"):
            sink.close()
    finally:
        core.unloadAllDevices()


def test_timelapse_mcu_failure_retains_partial_store_and_closes_shutter(tmp_path):
    core = make_core()
    owner = Owner()
    path = tmp_path / "frames"
    sink = TimeLapseFrameSink(str(path), np.zeros((4, 4), dtype=np.uint16), count=8)
    append = sink.append
    operations = []

    def fail_after_two(point, image):
        if point.sequence == 2:
            raise RuntimeError("synthetic sink failure")
        return append(point, image)

    sink.append = fail_after_two

    def action(value):
        operations.append(value)
        yield from bps.null()

    try:
        with pytest.raises(RuntimeError, match="synthetic sink failure"):
            RunEngine({})(
                timelapse_mcu_plan(
                    owner,
                    core,
                    sink,
                    [0],
                    1,
                    ["red"],
                    [1] * 4,
                    stack_for_block=lambda i: AngularStack(10 + i, 0, 18, 2, -180, 180),
                    move_position=action,
                    configure_channel=action,
                    open_shutter=lambda: action("open"),
                    close_shutter=lambda: action("close"),
                    shutter_gate="acquisition",
                    camera_exclusive=True,
                    illumination_ready=True,
                )
            )
        assert owner.aborted and owner.requested == [0]
        assert operations[-1] == "close"
        with pytest.raises(RuntimeError, match="incomplete"):
            sink.close()
        sidecar = json.loads((path / "timelapse-index.json").read_text())
        assert not sidecar["complete"] and sidecar["recorded"] == 2
    finally:
        core.unloadAllDevices()


def test_dpct_channels_have_independent_patterns_and_exposure(tmp_path):
    core = make_core()

    class CoreWithExposure:
        exposure = 1.0

        def __getattr__(self, name):
            return getattr(core, name)

        def getExposure(self):
            return self.exposure

    class PatternOwner(Owner):
        def __init__(self):
            super().__init__()
            self.patterns = []

        async def software_shots(self, definition, per, patterns, snap, **kwargs):
            self.patterns.append(list(patterns))
            await super().software_shots(definition, per, patterns, snap, **kwargs)

    proxy, owner = CoreWithExposure(), PatternOwner()
    path = tmp_path / "frames"
    sink = TimeLapseFrameSink(str(path), np.zeros((4, 4), dtype=np.uint16), count=16)
    docs = []

    def action(*args):
        yield from bps.null()

    def configure(channel):
        proxy.exposure = float(channel)
        yield from bps.null()

    engine = RunEngine({})
    engine.subscribe(lambda kind, doc: docs.append((kind, doc)))
    try:
        engine(
            timelapse_mcu_plan(
                owner,
                proxy,
                sink,
                [0],
                1,
                [1, 2, 3, 4],
                [1] * 4,
                channel_patterns=[[i] * 4 for i in range(1, 5)],
                channel_exposures_ms=[1, 2, 3, 4],
                stack_for_block=lambda i: AngularStack(10 + i, 0, 0, 1, -180, 180),
                move_position=action,
                configure_channel=configure,
                open_shutter=action,
                close_shutter=action,
                shutter_gate="acquisition",
                camera_exclusive=True,
                illumination_ready=True,
                plane_sequence=True,
                camera_internal_triggered=True,
            )
        )
        sink.close()
        assert owner.patterns == [[1], [2], [3], [4]]
        sidecar = json.loads((path / "timelapse-index.json").read_text())
        manifest = json.loads((path / "timelapse.manifest.json").read_text())
        assert [frame["channel"] for frame in sidecar["frames"]] == [
            i for i in range(4) for _ in range(4)
        ]
        assert [frame["pattern_id"] for frame in manifest["frames"]] == [
            i for i in range(1, 5) for _ in range(4)
        ]
        assert all(frame["setting_revision"] is None for frame in manifest["frames"])
        start = next(doc for kind, doc in docs if kind == "start")
        assert start["channel_patterns"] == [[i] * 4 for i in range(1, 5)]
        assert start["channel_exposures_ms"] == [1, 2, 3, 4]
    finally:
        core.unloadAllDevices()


def test_time_lapse_per_snap_uses_native_illumination_gate(tmp_path):
    core, owner = make_core(), Owner()
    sink = TimeLapseFrameSink(
        str(tmp_path / "frames"), np.zeros((4, 4), dtype=np.uint16), count=4
    )
    gate_events = []

    class Gate:
        async def open(self):
            gate_events.append("open")

        async def close(self):
            gate_events.append("close")

    def action(*args):
        yield from bps.null()

    try:
        RunEngine({})(
            timelapse_mcu_plan(
                owner,
                core,
                sink,
                [0],
                1,
                ["dpct"],
                [1, 2, 3, 4],
                stack_for_block=lambda i: AngularStack(10 + i, 0, 0, 1, -180, 180),
                move_position=action,
                configure_channel=action,
                open_shutter=action,
                close_shutter=action,
                shutter_gate="per-snap",
                camera_exclusive=True,
                illumination_ready=True,
                frame_gate=Gate(),
            )
        )
        sink.close()
        assert gate_events == ["open", "close"] * 4
        assert owner.requested == [0, 1, 2, 3]
    finally:
        core.unloadAllDevices()


def test_channels_select_distinct_native_gates(tmp_path):
    core, owner = make_core(), Owner()
    sink = TimeLapseFrameSink(
        str(tmp_path / "frames"), np.zeros((4, 4), dtype=np.uint16), count=4
    )
    events = []

    class Gate:
        def __init__(self, name):
            self.name = name

        async def open(self):
            events.append((self.name, "open"))

        async def close(self):
            events.append((self.name, "close"))

    def action(*args):
        yield from bps.null()

    try:
        RunEngine({})(
            timelapse_mcu_plan(
                owner,
                core,
                sink,
                [0],
                1,
                ["laser", "mechanical"],
                [1, 1],
                stack_for_block=lambda i: AngularStack(10 + i, 0, 0, 1, -180, 180),
                move_position=action,
                configure_channel=action,
                open_shutter=action,
                close_shutter=action,
                shutter_gate="per-snap",
                camera_exclusive=True,
                illumination_ready=True,
                channel_gates=[Gate("laser"), Gate("mechanical")],
            )
        )
        sink.close()
        assert events == [
            (name, state)
            for name in ("laser", "laser", "mechanical", "mechanical")
            for state in ("open", "close")
        ]
    finally:
        core.unloadAllDevices()


def test_timepoint_start_floor_across_stack_blocks(tmp_path):
    core, owner = make_core(), Owner()
    path = tmp_path / "frames"
    sink = TimeLapseFrameSink(str(path), np.zeros((4, 4), dtype=np.uint16), count=8)

    def action(*args):
        yield from bps.null()

    try:
        RunEngine({})(
            timelapse_mcu_plan(
                owner,
                core,
                sink,
                [0],
                2,
                ["red"],
                [1] * 4,
                stack_for_block=lambda i: AngularStack(10 + i, 0, 0, 1, -180, 180),
                move_position=action,
                configure_channel=action,
                open_shutter=action,
                close_shutter=action,
                shutter_gate="acquisition",
                camera_exclusive=True,
                illumination_ready=True,
                plane_sequence=True,
                camera_internal_triggered=True,
                timepoint_period_s=0.15,
            )
        )
        sink.close()
        frames = json.loads((path / "timelapse-index.json").read_text())["frames"]
        first = {frame["timepoint_start_ns"] for frame in frames[:4]}
        second = {frame["timepoint_start_ns"] for frame in frames[4:]}
        assert len(first) == len(second) == 1
        assert next(iter(second)) - next(iter(first)) >= 150_000_000
    finally:
        core.unloadAllDevices()


def test_timepoint_schedule_refused_before_motion(tmp_path):
    core = make_core()
    sink = TimeLapseFrameSink(
        str(tmp_path / "frames"), np.zeros((4, 4), dtype=np.uint16), count=8
    )
    operations = []

    def action(*args):
        operations.append(args)
        yield from bps.null()

    try:
        with pytest.raises(ValueError, match="estimated duration"):
            RunEngine({})(
                timelapse_mcu_plan(
                    Owner(),
                    core,
                    sink,
                    [0],
                    2,
                    ["red"],
                    [1] * 4,
                    stack_for_block=lambda i: AngularStack(10 + i, 0, 0, 1, -180, 180),
                    move_position=action,
                    configure_channel=action,
                    open_shutter=action,
                    close_shutter=action,
                    shutter_gate="acquisition",
                    camera_exclusive=True,
                    illumination_ready=True,
                    timepoint_period_s=3,
                    maximum_duration_s=1,
                )
            )
        assert operations == []
        with pytest.raises(RuntimeError, match="incomplete"):
            sink.close()
    finally:
        core.unloadAllDevices()


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("position-time", [(0, 0), (0, 1), (1, 0), (1, 1)]),
        ("time-position", [(0, 0), (1, 0), (0, 1), (1, 1)]),
    ],
)
def test_position_time_axis_orders(tmp_path, order, expected):
    core, owner = make_core(), Owner()
    path = tmp_path / "frames"
    sink = TimeLapseFrameSink(str(path), np.zeros((4, 4), dtype=np.uint16), count=16)
    moves = []

    def move(position):
        moves.append(position)
        yield from bps.null()

    def action(*args):
        yield from bps.null()

    try:
        RunEngine({})(
            timelapse_mcu_plan(
                owner,
                core,
                sink,
                ["left", "right"],
                2,
                ["red"],
                [1] * 4,
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
                timepoint_period_s=0.15,
                axis_order=order,
            )
        )
        sink.close()
        frames = json.loads((path / "timelapse-index.json").read_text())["frames"]
        assert [
            (frames[i]["position"], frames[i]["timepoint"]) for i in range(0, 16, 4)
        ] == expected
        assert moves == (
            ["left", "right"]
            if order == "position-time"
            else ["left", "right", "left", "right"]
        )
        if order == "time-position":
            assert frames[0]["timepoint_start_ns"] == frames[4]["timepoint_start_ns"]
            assert frames[8]["timepoint_start_ns"] == frames[12]["timepoint_start_ns"]
            assert (
                frames[8]["timepoint_start_ns"] - frames[0]["timepoint_start_ns"]
                >= 150_000_000
            )
    finally:
        core.unloadAllDevices()


def test_abort_on_missed_timepoint_preserves_first_block(tmp_path):
    core, owner = make_core(), Owner()
    path = tmp_path / "frames"
    sink = TimeLapseFrameSink(str(path), np.zeros((4, 4), dtype=np.uint16), count=8)
    gates = []

    def action(*args):
        yield from bps.null()

    def slow_channel(*args):
        yield from bps.sleep(0.12)

    def gate(state):
        gates.append(state)
        yield from bps.null()

    try:
        with pytest.raises(TimeoutError, match="deadline missed"):
            RunEngine({})(
                timelapse_mcu_plan(
                    owner,
                    core,
                    sink,
                    [0],
                    2,
                    ["red"],
                    [1] * 4,
                    stack_for_block=lambda i: AngularStack(10 + i, 0, 0, 1, -180, 180),
                    move_position=action,
                    configure_channel=slow_channel,
                    open_shutter=lambda: gate("open"),
                    close_shutter=lambda: gate("close"),
                    shutter_gate="acquisition",
                    camera_exclusive=True,
                    illumination_ready=True,
                    plane_sequence=True,
                    camera_internal_triggered=True,
                    timepoint_period_s=0.05,
                    missed_timepoint_policy="abort",
                )
            )
        with pytest.raises(RuntimeError, match="incomplete"):
            sink.close()
        sidecar = json.loads((path / "timelapse-index.json").read_text())
        assert sidecar["recorded"] == 4 and not sidecar["complete"]
        assert owner.requested == [0] and gates == ["open", "close"]
    finally:
        core.unloadAllDevices()
