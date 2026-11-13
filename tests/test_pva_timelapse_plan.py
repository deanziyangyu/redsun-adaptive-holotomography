"""Hardware-free checks of service-owned time-lapse composition."""

import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps

from redsun_aht.device.pva_capture_evidence import CaptureEvidence
from redsun_aht.device.pva_internal_plane_capture import PlaneCapture
from redsun_aht.device.unified_mcu import AngularStack
from redsun_aht.plans.pva_timelapse import pva_timelapse_plan


class Owner:
    reservation_held = True

    def __init__(self):
        self.commands = []

    async def set_reserved_illumination(self, command):
        self.commands.append(command)


def make_plan(owner, actions, **kwargs):
    exposure_value = kwargs.pop("test_exposure", 1.0)
    positions = kwargs.pop("test_positions", ["x", "y"])
    timepoints = kwargs.pop("test_timepoints", 2)
    channels = kwargs.pop("test_channels", ["red"])

    class Exposure:
        async def get_value(self, *, cached):
            return exposure_value

    class Camera:
        exposure = Exposure()

    def action(name):
        actions.append(name)
        yield from bps.null()

    params = dict(
        stack_for_block=lambda i: AngularStack(10 + i, 0, 18, 2, -180, 180),
        store_for_plane=lambda block, plane: f"store-{block}-{plane}",
        move_position=lambda position: action(("move", position)),
        configure_channel=lambda channel: action(("channel", channel)),
        open_gate=lambda: action("open"),
        close_gate=lambda: action("close"),
        resources_exclusive=True,
        internal_trigger_configured=True,
        illumination_ready=True,
        hold_illumination=True,
    )
    params.update(kwargs)
    return pva_timelapse_plan(
        owner,
        Camera(),
        positions,
        timepoints,
        channels,
        [1] * len(channels),
        [1.0] * len(channels),
        4,
        **params,
    )


@pytest.mark.parametrize("order", ["position-time", "time-position"])
def test_blocks_publish_plane_events_and_blank(monkeypatch, order):
    calls, actions, docs = [], [], []

    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        calls.append((definition.run_id, kw["hold_illumination"]))
        for plane in range(definition.count):
            await on_plane(
                PlaneCapture(
                    plane,
                    plane * 18,
                    patterns[plane],
                    CaptureEvidence(store(plane), per, per),
                )
            )

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    owner = Owner()
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    engine(make_plan(owner, actions, axis_order=order))
    assert calls == [(10, True), (11, True), (12, True), (13, False)]
    events = [doc["data"] for name, doc in docs if name == "event"]
    assert len(events) == 8
    expected = (
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        if order == "position-time"
        else [(0, 0), (1, 0), (0, 1), (1, 1)]
    )
    assert [
        (e["timelapse_position"], e["timelapse_timepoint"]) for e in events[::2]
    ] == expected
    assert actions.index(("channel", "red")) < actions.index("open")
    assert actions[-1] == "close"
    assert owner.commands == ["00"]
    start = next(doc for name, doc in docs if name == "start")
    assert start["plane_stores"][2][1] == "store-2-1"


def test_failure_emits_completed_plane_and_cleans_up(monkeypatch):
    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        await on_plane(PlaneCapture(0, 0, 1, CaptureEvidence(store(0), per, per)))
        raise RuntimeError("camera failure")

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    owner, actions, docs = Owner(), [], []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    with pytest.raises(RuntimeError, match="camera failure"):
        engine(make_plan(owner, actions))
    assert len([1 for name, doc in docs if name == "event"]) == 1
    assert actions[-1] == "close" and owner.commands == ["00"]


def test_duplicate_store_rejected_before_callbacks():
    owner, actions = Owner(), []
    with pytest.raises(ValueError, match="unique plane stores"):
        RunEngine({})(
            make_plan(owner, actions, store_for_plane=lambda block, plane: "same")
        )
    assert actions == [] and owner.commands == []


def test_exposure_mismatch_prevents_capture_and_closes_gate(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("capture must not start")

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", forbidden
    )
    owner, actions = Owner(), []
    with pytest.raises(ValueError, match="exposure readback mismatch"):
        RunEngine({})(make_plan(owner, actions, test_exposure=2.0))
    assert actions[-1] == "close" and owner.commands == ["00"]


def test_channel_change_closes_gate_before_reconfiguration(monkeypatch):
    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        await on_plane(
            PlaneCapture(0, 0, patterns[0], CaptureEvidence(store(0), per, per))
        )

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    actions = []
    RunEngine({})(
        make_plan(
            Owner(), actions, test_positions=["x"], test_channels=["red", "laser"]
        )
    )
    channels_and_gates = [
        item
        for item in actions
        if item == "open"
        or item == "close"
        or (isinstance(item, tuple) and item[0] == "channel")
    ]
    assert channels_and_gates == [
        ("channel", "red"),
        "open",
        "close",
        ("channel", "laser"),
        "open",
        "close",
        ("channel", "red"),
        "open",
        "close",
        ("channel", "laser"),
        "open",
        "close",
    ]


def test_reviewed_revisit_targets_drive_pva_timelapse(monkeypatch):
    from redsun_aht.plans.reviewed_timelapse import (
        RevisitedTarget,
        reviewed_revisit_timelapse_chain,
    )

    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        await on_plane(
            PlaneCapture(0, 0, patterns[0], CaptureEvidence(store(0), per, per))
        )

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    targets = [RevisitedTarget("b", 20, 4), RevisitedTarget("a", 3, 7)]
    actions, docs = [], []

    def anchor():
        yield from bps.null()
        return (1, 2)

    def revisit():
        yield from bps.open_run(md={"purpose": "revisit"})
        yield from bps.close_run()
        return targets

    def timelapse(actual):
        yield from make_plan(
            Owner(),
            actions,
            test_positions=actual,
            position_ids=[target.target_id for target in actual],
            axis_order="time-position",
        )

    def restore(position):
        actions.append(("restore", position))
        yield from bps.null()

    engine = RunEngine({})
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    engine(
        reviewed_revisit_timelapse_chain(
            read_anchor=anchor,
            revisit=revisit,
            run_timelapse=timelapse,
            return_to_anchor=restore,
        )
    )
    start = next(
        doc
        for name, doc in docs
        if name == "start" and doc.get("plan_name") == "pva_timelapse"
    )
    assert start["position_ids"] == ("b", "a")
    assert [
        item[1].target_id
        for item in actions
        if isinstance(item, tuple) and item[0] == "move"
    ] == [
        "b",
        "a",
        "b",
        "a",
    ]
    assert actions[-1] == ("restore", (1.0, 2.0))


def test_missed_timepoint_can_abort_and_retain_first_block(monkeypatch):
    import asyncio

    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        await on_plane(
            PlaneCapture(0, 0, patterns[0], CaptureEvidence(store(0), per, per))
        )
        await asyncio.sleep(0.02)

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    owner, actions, docs = Owner(), [], []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    with pytest.raises(TimeoutError, match="timepoint deadline"):
        engine(
            make_plan(
                owner,
                actions,
                test_positions=["x"],
                timepoint_period_s=0.001,
                missed_timepoint_policy="abort",
            )
        )
    assert len([1 for name, doc in docs if name == "event"]) == 1
    assert actions[-1] == "close" and owner.commands == ["00"]


def test_uncertain_reservation_cannot_report_success(monkeypatch):
    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        await on_plane(
            PlaneCapture(0, 0, patterns[0], CaptureEvidence(store(0), per, per))
        )
        owner.reservation_held = False

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    owner, actions = Owner(), []
    with pytest.raises(RuntimeError, match="blanking not confirmed"):
        RunEngine({})(make_plan(owner, actions, test_positions=["x"]))
    assert actions[-1] == "close" and owner.commands == []


def test_center_exit_keeps_last_block_then_selects_center(monkeypatch):
    holds = []

    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        holds.append(kw["hold_illumination"])
        await on_plane(
            PlaneCapture(0, 0, patterns[0], CaptureEvidence(store(0), per, per))
        )

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    owner, actions = Owner(), []
    RunEngine({})(
        make_plan(owner, actions, test_positions=["x"], end_illumination="center")
    )
    assert (
        holds == [True, True]
        and owner.commands == ["01", "15"]
        and actions[-1] == "close"
    )


def test_center_exit_still_blanks_on_failure(monkeypatch):
    async def capture(*args, **kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    owner = Owner()
    with pytest.raises(RuntimeError, match="capture failed"):
        RunEngine({})(make_plan(owner, [], end_illumination="center"))
    assert owner.commands == ["00"]


def test_invalid_exit_illumination_is_rejected():
    with pytest.raises(ValueError, match="blank or center"):
        RunEngine({})(make_plan(Owner(), [], end_illumination="last"))


def test_partial_pattern_plane_emits_confirmed_shots_on_failure(monkeypatch):
    from redsun_aht.device.pva_pattern_shots import PatternShot

    async def capture(
        owner, camera, definition, patterns, store, on_shot, on_plane, **kw
    ):
        await on_shot(PatternShot(0, 0, 0, 1, 0, 1, 100))
        await on_shot(PatternShot(1, 0, 1, 2, 0, 2, 200))
        raise RuntimeError("camera failed during plane")

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_pattern_shots", capture
    )
    owner, actions, docs = Owner(), [], []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    with pytest.raises(RuntimeError, match="camera failed during plane"):
        engine(make_plan(owner, actions, channel_frame_patterns=[[1, 2, 3, 4]]))
    events = [doc["data"] for name, doc in docs if name == "event"]
    assert [(e["timelapse_frame"], e["timelapse_pattern"]) for e in events] == [
        (0, 1),
        (1, 2),
    ]
    assert actions[-1] == "close" and owner.commands == ["00"]


def test_journal_records_partial_pattern_failure_and_refuses_reuse(
    monkeypatch, tmp_path
):
    import json

    from redsun_aht.device.pva_pattern_shots import PatternShot

    async def capture(
        owner, camera, definition, patterns, store, on_shot, on_plane, **kw
    ):
        await on_shot(PatternShot(0, 0, 0, 1, 0, 1, 100))
        raise RuntimeError("injected capture failure")

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_pattern_shots", capture
    )
    path = tmp_path / "run.jsonl"
    owner, actions = Owner(), []
    with pytest.raises(RuntimeError, match="injected capture failure"):
        RunEngine({})(
            make_plan(
                owner,
                actions,
                journal_path=str(path),
                channel_frame_patterns=[[1, 2, 3, 4]],
            )
        )
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    start = next(x["data"] for x in entries if x["document"] == "start")
    event = next(x["data"] for x in entries if x["document"] == "event")
    stop = next(x["data"] for x in entries if x["document"] == "stop")
    assert start["plane_stores"][0][0] == "store-0-0"
    assert start["association"] == "host_order_inferred"
    assert event["data"]["timelapse_pattern"] == 1
    assert stop["exit_status"] == "fail"
    before = path.read_bytes()
    new_actions = []
    with pytest.raises(FileExistsError):
        RunEngine({})(make_plan(Owner(), new_actions, journal_path=str(path)))
    assert path.read_bytes() == before and new_actions == []


def test_journal_records_requested_stacks_and_channel_labels(monkeypatch, tmp_path):
    import json

    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        for plane in range(definition.count):
            await on_plane(
                PlaneCapture(
                    plane, plane * 18, 1, CaptureEvidence(store(plane), per, per)
                )
            )

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    path = tmp_path / "requested.jsonl"
    RunEngine({})(
        make_plan(
            Owner(),
            [],
            test_positions=["x"],
            journal_path=str(path),
            channel_labels=["red LED"],
        )
    )
    start = next(
        json.loads(line)["data"]
        for line in path.read_text().splitlines()
        if json.loads(line)["document"] == "start"
    )
    assert start["channel_labels"] == ["red LED"]
    assert start["stage_binding"] == "mcu"
    assert (
        start["requested_stacks"]
        == [
            {
                "first_tenths": 0,
                "step_tenths": 18,
                "planes": 2,
                "lower_tenths": -180,
                "upper_tenths": 180,
            }
        ]
        * 2
    )


def test_invalid_channel_labels_rejected_before_hardware():
    actions = []
    with pytest.raises(ValueError, match="channel labels"):
        RunEngine({})(make_plan(Owner(), actions, channel_labels=[]))
    assert actions == []


def test_deadlines_follow_previous_start_when_timepoints_run_late(monkeypatch):
    from types import SimpleNamespace

    clock = SimpleNamespace(now=1_000_000_000)
    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.time",
        SimpleNamespace(perf_counter_ns=lambda: clock.now),
    )

    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        await on_plane(PlaneCapture(0, 0, 1, CaptureEvidence(store(0), per, per)))
        clock.now += 1_500_000_000

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    docs = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    engine(
        make_plan(
            Owner(),
            [],
            test_positions=["x"],
            test_timepoints=3,
            timepoint_period_s=1,
            missed_timepoint_policy="continue",
        )
    )
    events = [doc["data"] for name, doc in docs if name == "event"]
    assert [e["timelapse_timepoint_lateness_ns"] for e in events] == [
        0,
        500_000_000,
        500_000_000,
    ]


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("position-time", [0, 500_000_000, 500_000_000] * 2),
        (
            "time-position",
            [0, 0, 500_000_000, 500_000_000, 500_000_000, 500_000_000],
        ),
    ],
)
def test_series_deadline_scope_follows_axis_order(monkeypatch, order, expected):
    from types import SimpleNamespace

    clock = SimpleNamespace(now=1_000_000_000)
    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.time",
        SimpleNamespace(perf_counter_ns=lambda: clock.now),
    )

    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        await on_plane(PlaneCapture(0, 0, 1, CaptureEvidence(store(0), per, per)))
        clock.now += 1_500_000_000 if order == "position-time" else 750_000_000

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    docs = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    engine(
        make_plan(
            Owner(),
            [],
            test_timepoints=3,
            axis_order=order,
            timepoint_period_s=1,
            missed_timepoint_policy="continue",
        )
    )
    events = [doc["data"] for name, doc in docs if name == "event"]
    assert [e["timelapse_timepoint_lateness_ns"] for e in events] == expected


def test_timepoint_floor_rechecks_after_early_wakeup(monkeypatch):
    from types import SimpleNamespace

    clock = SimpleNamespace(now=1_000_000_000, sleeps=0)
    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.time",
        SimpleNamespace(perf_counter_ns=lambda: clock.now),
    )

    class Stubs:
        def __getattr__(self, name):
            return getattr(bps, name)

        def sleep(self, seconds):
            clock.sleeps += 1
            advance = round(seconds * 1e9)
            clock.now += advance // 2 if clock.sleeps == 1 else advance
            yield from bps.null()

    monkeypatch.setattr("redsun_aht.plans.pva_timelapse.bps", Stubs())

    async def capture(owner, camera, definition, per, patterns, store, on_plane, **kw):
        await on_plane(PlaneCapture(0, 0, 1, CaptureEvidence(store(0), per, per)))

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_internal_planes", capture
    )
    docs = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    engine(make_plan(Owner(), [], test_positions=["x"], timepoint_period_s=1))
    starts = [
        doc["data"]["timelapse_timepoint_start_ns"]
        for name, doc in docs
        if name == "event"
    ]
    assert starts[1] - starts[0] >= 1_000_000_000
    assert clock.sleeps == 2


def test_auto_shutter_uses_per_shot_path_even_for_constant_pattern(monkeypatch):
    called = []

    async def capture(
        owner, camera, definition, patterns, store, on_shot, on_plane, **kw
    ):
        called.append((tuple(patterns), kw["open_shot_gate"], kw["close_shot_gate"]))
        await on_plane(PlaneCapture(0, 0, 1, CaptureEvidence(store(0), 4, 4)))

    async def gate():
        pass

    monkeypatch.setattr(
        "redsun_aht.plans.pva_timelapse.acquire_pva_pattern_shots", capture
    )
    actions = []
    RunEngine({})(
        make_plan(
            Owner(),
            actions,
            test_positions=["x"],
            auto_shutter=True,
            open_shot_gate=gate,
            close_shot_gate=gate,
        )
    )
    assert called == [((1, 1, 1, 1), gate, gate)] * 2
    assert "open" not in actions


def test_auto_shutter_requires_gate_callbacks_before_device_actions():
    actions = []
    with pytest.raises(ValueError, match="per-shot gate callbacks"):
        RunEngine({})(make_plan(Owner(), actions, auto_shutter=True))
    assert actions == []
