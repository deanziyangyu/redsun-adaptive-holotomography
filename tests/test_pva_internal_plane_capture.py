"""Internal-trigger service-plane orchestration with fake services."""

import pytest

from redsun_aht.device.pva_capture_evidence import CaptureEvidence
from redsun_aht.device.pva_internal_plane_capture import acquire_pva_internal_planes
from redsun_aht.device.unified_mcu import AngularStack


@pytest.mark.parametrize("per", [4, 16])
async def test_plane_stores_finish_before_next_shot(monkeypatch, per):
    events = []

    class FakeCapture:
        def __init__(self, camera, *, timeout_s):
            self.uri = ""

        async def prepare(self, uri, count, *, internal_trigger_configured):
            assert internal_trigger_configured and count == per
            self.uri = uri
            events.append(("start", uri))

        async def collect(self):
            events.append(("complete", self.uri))
            return CaptureEvidence(self.uri, per, per)

        async def close(self):
            events.append(("close", self.uri))

    class Owner:
        async def software_shots(self, definition, per_shot, patterns, snap, **kwargs):
            assert per_shot == 1
            for plane, pattern in enumerate(patterns):
                events.append(("shot", plane))
                await snap(plane, plane, pattern, plane * definition.step)

    async def on_plane(result):
        events.append(("publish", result.plane))
        assert result.quality == "host_order_inferred"

    monkeypatch.setattr(
        "redsun_aht.device.pva_internal_plane_capture.PvaCaptureEvidence", FakeCapture
    )
    await acquire_pva_internal_planes(
        Owner(),
        object(),
        AngularStack(7, 0, 18, 2, -180, 180),
        per,
        [1, 2],
        lambda plane: f"plane-{plane}",
        on_plane,
        resources_exclusive=True,
        internal_trigger_configured=True,
        illumination_ready=True,
    )
    assert events == [
        ("shot", 0),
        ("start", "plane-0"),
        ("complete", "plane-0"),
        ("close", "plane-0"),
        ("publish", 0),
        ("shot", 1),
        ("start", "plane-1"),
        ("complete", "plane-1"),
        ("close", "plane-1"),
        ("publish", 1),
    ]


async def test_duplicate_plane_store_rejected_before_owner():
    class Owner:
        async def software_shots(self, *args, **kwargs):
            raise AssertionError("owner must not be used")

    async def sink(*args):
        pass

    with pytest.raises(ValueError, match="unique plane stores"):
        await acquire_pva_internal_planes(
            Owner(),
            object(),
            AngularStack(7, 0, 18, 2, -180, 180),
            4,
            [1, 1],
            lambda plane: "same",
            sink,
            resources_exclusive=True,
            internal_trigger_configured=True,
            illumination_ready=True,
        )


async def test_failed_plane_stops_camera_and_prevents_next_shot(monkeypatch):
    events = []

    class FakeCapture:
        def __init__(self, camera, *, timeout_s):
            pass

        async def prepare(self, *args, **kwargs):
            events.append("start")

        async def collect(self):
            raise TimeoutError("missing frames")

        async def close(self):
            events.append("close")

    class Owner:
        async def software_shots(self, definition, per, patterns, snap, **kwargs):
            events.append("shot0")
            await snap(0, 0, 1, 0)
            events.append("shot1")

    async def sink(*args):
        events.append("publish")

    monkeypatch.setattr(
        "redsun_aht.device.pva_internal_plane_capture.PvaCaptureEvidence", FakeCapture
    )
    with pytest.raises(TimeoutError, match="missing frames"):
        await acquire_pva_internal_planes(
            Owner(),
            object(),
            AngularStack(7, 0, 18, 2, -180, 180),
            4,
            [1, 1],
            lambda plane: str(plane),
            sink,
            resources_exclusive=True,
            internal_trigger_configured=True,
            illumination_ready=True,
        )
    assert events == ["shot0", "start", "close"]


async def test_external_stage_moves_before_each_plane_capture(monkeypatch):
    events = []

    class FakeCapture:
        def __init__(self, camera, *, timeout_s):
            pass

        async def prepare(self, uri, count, *, internal_trigger_configured):
            events.append(("capture", uri))

        async def collect(self):
            return CaptureEvidence("unused", 4, 4)

        async def close(self):
            pass

    class Owner:
        async def software_shots(self, definition, per, patterns, snap, **kwargs):
            for plane in range(2):
                await snap(plane, plane, 1, plane * 18)

    async def move(plane, target):
        events.append(("move", plane, target))

    async def sink(result):
        pass

    monkeypatch.setattr(
        "redsun_aht.device.pva_internal_plane_capture.PvaCaptureEvidence", FakeCapture
    )
    await acquire_pva_internal_planes(
        Owner(),
        object(),
        AngularStack(7, 0, 18, 2, -180, 180),
        4,
        [1, 1],
        lambda plane: str(plane),
        sink,
        resources_exclusive=True,
        internal_trigger_configured=True,
        illumination_ready=True,
        bind_stage=False,
        before_plane=move,
    )
    assert events == [
        ("move", 0, 0),
        ("capture", "0"),
        ("move", 1, 18),
        ("capture", "1"),
    ]


async def test_plane_timing_reports_ordered_host_boundaries(monkeypatch):
    timing = []

    class FakeCapture:
        def __init__(self, camera, *, timeout_s):
            pass

        async def prepare(self, uri, count, *, internal_trigger_configured):
            pass

        async def collect(self):
            return CaptureEvidence("store", 4, 4)

        async def close(self):
            pass

    class Owner:
        async def software_shots(self, definition, per, patterns, snap, **kwargs):
            await snap(0, 0, 1, 0)

    async def sink(result):
        pass

    monkeypatch.setattr(
        "redsun_aht.device.pva_internal_plane_capture.PvaCaptureEvidence", FakeCapture
    )
    await acquire_pva_internal_planes(
        Owner(),
        object(),
        AngularStack(7, 0, 18, 1, -180, 180),
        4,
        [1],
        lambda plane: "store",
        sink,
        resources_exclusive=True,
        internal_trigger_configured=True,
        illumination_ready=True,
        on_timing=timing.append,
    )
    assert len(timing) == 1 and timing[0].run_id == 7
    item = timing[0]
    assert (
        item.ready_ns
        <= item.stage_ready_ns
        <= item.camera_started_ns
        <= item.capture_complete_ns
        <= item.camera_stopped_ns
    )
