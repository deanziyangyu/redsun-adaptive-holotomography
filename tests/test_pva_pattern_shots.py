"""Checks for per-shot gate timing with synthetic camera and MCU."""

import asyncio

import pytest

from redsun_aht.device.pva_capture_evidence import CaptureEvidence
from redsun_aht.device.pva_pattern_shots import acquire_pva_pattern_shots
from redsun_aht.device.unified_mcu import AngularStack


@pytest.mark.parametrize("fail_snap", [False, True])
def test_per_shot_gate_wraps_snap_and_closes_on_error(monkeypatch, fail_snap):
    actions = []

    class CameraCapture:
        def __init__(self, camera, *, timeout_s):
            self.count = 0

        async def prepare(self, uri, count, **kwargs):
            actions.append("prepare")

        async def snap(self):
            actions.append("snap")
            if fail_snap:
                raise RuntimeError("snap failed")
            self.count += 1
            return self.count

        async def collect(self):
            return CaptureEvidence("store", 2, self.count)

        async def close(self):
            actions.append("capture-close")

    class Owner:
        async def software_shots(self, definition, per, patterns, callback, **kwargs):
            for shot, pattern in enumerate(patterns):
                actions.append("shot")
                await callback(shot, 0, pattern, 0)

    async def open_gate():
        actions.append("open")

    async def close_gate():
        actions.append("close")

    async def shot(result):
        actions.append("event")

    async def plane(result):
        actions.append("plane")

    monkeypatch.setattr(
        "redsun_aht.device.pva_pattern_shots.PvaCaptureEvidence", CameraCapture
    )
    args = (
        Owner(),
        object(),
        AngularStack(1, 0, 1, 1, -10, 10),
        [1, 1],
        lambda _: "store",
        shot,
        plane,
    )
    kwargs = dict(
        resources_exclusive=True,
        internal_trigger_configured=True,
        illumination_ready=True,
        open_shot_gate=open_gate,
        close_shot_gate=close_gate,
    )
    if fail_snap:
        with pytest.raises(RuntimeError, match="snap failed"):
            asyncio.run(acquire_pva_pattern_shots(*args, **kwargs))
        assert actions == ["shot", "prepare", "open", "snap", "close", "capture-close"]
    else:
        asyncio.run(acquire_pva_pattern_shots(*args, **kwargs))
        assert actions == [
            "shot",
            "prepare",
            "open",
            "snap",
            "event",
            "close",
            "shot",
            "open",
            "snap",
            "event",
            "close",
            "capture-close",
            "plane",
        ]


async def test_software_trigger_sequence_stays_open_for_the_plane(monkeypatch):
    actions = []

    class CameraCapture:
        def __init__(self, camera, *, timeout_s):
            self.count = 0

        async def prepare(self, uri, count, **kwargs):
            assert kwargs == {
                "internal_trigger_configured": False,
                "software_trigger_configured": True,
                "software_snap": False,
            }
            actions.append("prepare")

        async def software_trigger(self):
            self.count += 1
            actions.append("trigger")
            return self.count

        async def collect(self):
            return CaptureEvidence("store", 4, self.count)

        async def close(self):
            actions.append("close")

    class Owner:
        async def software_shots(self, definition, per, patterns, callback, **kwargs):
            for shot, pattern in enumerate(patterns):
                await callback(shot, 0, pattern, 0)

    async def event(item):
        actions.append("event")

    monkeypatch.setattr(
        "redsun_aht.device.pva_pattern_shots.PvaCaptureEvidence", CameraCapture
    )
    await acquire_pva_pattern_shots(
        Owner(),
        object(),
        AngularStack(1, 0, 1, 1, -10, 10),
        [1, 2, 3, 4],
        lambda _: "store",
        event,
        event,
        resources_exclusive=True,
        internal_trigger_configured=False,
        software_trigger_configured=True,
        illumination_ready=True,
    )
    assert actions == [
        "prepare",
        "trigger",
        "event",
        "trigger",
        "event",
        "trigger",
        "event",
        "trigger",
        "event",
        "close",
        "event",
    ]
