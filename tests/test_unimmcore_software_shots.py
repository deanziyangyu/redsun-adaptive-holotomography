"""MDA software-shot bridge ordering and failure tests with synthetic devices."""

import numpy as np
import pytest
from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore

from redsun_aht.device.unified_mcu import AngularStack
from redsun_aht.device.unimmcore_software_shots import run_software_shot_mda


class Camera(CameraDevice):
    def get_exposure(self):
        return 1.0

    def set_exposure(self, value):
        pass

    def shape(self):
        return (4, 4)

    def dtype(self):
        return np.uint16

    def start_sequence(self, n, get_buffer):
        for _ in range(n or 0):
            image = get_buffer(self.shape(), self.dtype())
            image[:] = 42
            yield {}


class Owner:
    def __init__(self):
        self.requested = []
        self.aborted = False

    async def software_shots(self, definition, per, patterns, snap, **kwargs):
        try:
            for i, pattern in enumerate(patterns):
                self.requested.append(i)
                await snap(
                    i, i // per, pattern, definition.first + i // per * definition.step
                )
        except BaseException:
            self.aborted = True
            raise


def make_core():
    core = UniMMCore()
    core.loadPyDevice("camera", Camera())
    core.initializeDevice("camera")
    core.setCameraDevice("camera")
    core.setAutoShutter(False)
    return core


@pytest.mark.parametrize("per", [4, 16])
async def test_one_runner_matches_every_ready_to_one_frame(per):
    core = make_core()
    owner = Owner()
    received = []
    try:
        await run_software_shot_mda(
            owner,
            core,
            AngularStack(7, 0, 18, 2, -180, 180),
            per,
            [1] * (2 * per),
            lambda *frame: received.append(frame[:4]),
            camera_exclusive=True,
            illumination_ready=True,
        )
        assert owner.requested == list(range(2 * per))
        assert received == [(i, i // per, 1, 18 * (i // per)) for i in range(2 * per)]
    finally:
        core.unloadAllDevices()


@pytest.mark.parametrize("per", [4, 16])
@pytest.mark.parametrize("bound", [True, False])
async def test_open_shutter_coordinates_once_per_plane(per, bound):
    core = make_core()
    owner = Owner()
    received = []
    planes = []
    try:
        await run_software_shot_mda(
            owner,
            core,
            AngularStack(7, 0, 18, 2, -180, 180),
            per,
            [1] * per + [2] * per,
            lambda *frame: received.append(frame[:4]),
            camera_exclusive=True,
            illumination_ready=True,
            bind_stage=bound,
            before_plane=None
            if bound
            else lambda plane, target: planes.append((plane, target)),
            auto_shutter=False,
        )
        assert owner.requested == [0, 1]
        assert received == [
            (i, i // per, 1 + i // per, 18 * (i // per)) for i in range(2 * per)
        ]
        assert planes == ([] if bound else [(0, 0), (1, 18)])
    finally:
        core.unloadAllDevices()


async def test_open_shutter_failure_does_not_advance_plane():
    core = make_core()
    owner = Owner()

    def sink(shot, *args):
        if shot == 2:
            raise RuntimeError("sink failed")

    try:
        with pytest.raises(RuntimeError, match="sink failed"):
            await run_software_shot_mda(
                owner,
                core,
                AngularStack(7, 0, 18, 2, -180, 180),
                4,
                [1] * 8,
                sink,
                camera_exclusive=True,
                illumination_ready=True,
                auto_shutter=False,
            )
        assert owner.aborted and owner.requested == [0]
    finally:
        core.unloadAllDevices()


async def test_open_shutter_plane_timeout_does_not_advance(monkeypatch):
    import asyncio
    import time

    from pymmcore_plus.mda import MDAEngine

    original = MDAEngine.exec_single_event

    def slow_capture(self, event):
        time.sleep(0.15)
        yield from original(self, event)

    monkeypatch.setattr(MDAEngine, "exec_single_event", slow_capture)
    core = make_core()

    class TimeoutOwner(Owner):
        async def software_shots(self, definition, per, patterns, snap, **kwargs):
            async def timed_snap(*args):
                await asyncio.wait_for(snap(*args), kwargs["snap_timeout_s"])

            await super().software_shots(
                definition, per, patterns, timed_snap, **kwargs
            )

    owner = TimeoutOwner()
    received = []
    try:
        with pytest.raises(TimeoutError):
            await run_software_shot_mda(
                owner,
                core,
                AngularStack(7, 0, 18, 2, -180, 180),
                4,
                [1] * 8,
                lambda shot, *args: received.append(shot),
                camera_exclusive=True,
                illumination_ready=True,
                auto_shutter=False,
                timeout_s=0.05,
            )
        assert owner.aborted and owner.requested == [0]
        assert received == []
    finally:
        core.unloadAllDevices()


async def test_timeout_during_pacing_does_not_start_camera(monkeypatch):
    import asyncio
    import time

    from pymmcore_plus.mda import MDAEngine

    camera_calls = []
    original = MDAEngine.exec_single_event

    def record_capture(self, event):
        camera_calls.append(event)
        yield from original(self, event)

    monkeypatch.setattr(MDAEngine, "exec_single_event", record_capture)

    class TimeoutOwner(Owner):
        async def software_shots(self, definition, per, patterns, snap, **kwargs):
            async def timed_snap(*args):
                await asyncio.wait_for(snap(*args), kwargs["snap_timeout_s"])

            await super().software_shots(
                definition, per, patterns, timed_snap, **kwargs
            )

    core, owner = make_core(), TimeoutOwner()
    try:
        with pytest.raises(TimeoutError):
            await run_software_shot_mda(
                owner,
                core,
                AngularStack(7, 0, 18, 1, -180, 180),
                4,
                [1] * 4,
                lambda *args: None,
                camera_exclusive=True,
                illumination_ready=True,
                auto_shutter=False,
                timeout_s=0.05,
                frame_started=lambda *_: time.sleep(0.15),
            )
        assert camera_calls == [] and owner.aborted and owner.requested == [0]
    finally:
        core.unloadAllDevices()


async def test_open_shutter_rejects_pattern_changes_within_plane():
    with pytest.raises(ValueError, match="one pattern per plane"):
        await run_software_shot_mda(
            Owner(),
            None,
            AngularStack(7, 0, 18, 1, -180, 180),
            4,
            [1, 1, 2, 1],
            lambda *args: None,
            camera_exclusive=True,
            illumination_ready=True,
            auto_shutter=False,
        )


async def test_open_shutter_rejects_enabled_core_auto_shutter():
    core = make_core()
    try:
        core.setAutoShutter(True)
        with pytest.raises(ValueError, match="core auto-shutter must be disabled"):
            await run_software_shot_mda(
                Owner(),
                core,
                AngularStack(7, 0, 18, 1, -180, 180),
                4,
                [1] * 4,
                lambda *args: None,
                camera_exclusive=True,
                illumination_ready=True,
                auto_shutter=False,
            )
    finally:
        core.unloadAllDevices()


async def test_auto_shutter_requires_configured_device():
    core = make_core()
    try:
        core.setAutoShutter(True)
        with pytest.raises(ValueError, match="shutter device are required"):
            await run_software_shot_mda(
                Owner(),
                core,
                AngularStack(7, 0, 18, 1, -180, 180),
                4,
                [1] * 4,
                lambda *args: None,
                camera_exclusive=True,
                illumination_ready=True,
                auto_shutter=True,
            )
    finally:
        core.unloadAllDevices()


async def test_auto_shutter_uses_one_handshake_per_frame():
    core = make_core()
    owner = Owner()
    frames = []
    transitions = []

    class SimulatedShutterCore:
        def __getattr__(self, name):
            return getattr(core, name)

        def getAutoShutter(self):
            return True

        def getShutterDevice(self):
            return "fake_shutter"

        def setAutoShutter(self, enabled):
            assert enabled

        def setShutterOpen(self, opened):
            transitions.append(opened)

        def snapImage(self):
            transitions.append(True)
            try:
                core.snapImage()
            finally:
                transitions.append(False)

    try:
        await run_software_shot_mda(
            owner,
            SimulatedShutterCore(),
            AngularStack(7, 0, 18, 1, -180, 180),
            4,
            [1] * 4,
            lambda shot, *args: frames.append(shot),
            camera_exclusive=True,
            illumination_ready=True,
            auto_shutter=True,
        )
        assert owner.requested == frames == [0, 1, 2, 3]
        assert transitions.count(True) == 4
        assert transitions.count(False) >= 4
    finally:
        core.unloadAllDevices()


async def test_sink_failure_aborts_before_next_shot():
    core = make_core()
    owner = Owner()
    received = []

    def sink(shot, *args):
        received.append(shot)
        if shot == 3:
            raise RuntimeError("sink failed")

    try:
        with pytest.raises(RuntimeError, match="sink failed"):
            await run_software_shot_mda(
                owner,
                core,
                AngularStack(7, 0, 18, 2, -180, 180),
                4,
                [1] * 8,
                sink,
                camera_exclusive=True,
                illumination_ready=True,
            )
        assert owner.aborted and owner.requested == [0, 1, 2, 3]
    finally:
        core.unloadAllDevices()


async def test_missing_frame_aborts_before_next_shot(monkeypatch):
    from pymmcore_plus.mda import MDAEngine

    monkeypatch.setattr(MDAEngine, "exec_single_event", lambda self, event: iter(()))
    core = make_core()
    owner = Owner()
    try:
        with pytest.raises(RuntimeError, match="exactly one camera frame"):
            await run_software_shot_mda(
                owner,
                core,
                AngularStack(7, 0, 18, 2, -180, 180),
                4,
                [1] * 8,
                lambda *args: None,
                camera_exclusive=True,
                illumination_ready=True,
            )
        assert owner.aborted and owner.requested == [0]
    finally:
        core.unloadAllDevices()


async def test_owner_failure_terminates_waiting_runner():
    class FailedOwner:
        async def software_shots(self, *args, **kwargs):
            raise RuntimeError("controller fault")

    core = make_core()
    try:
        with pytest.raises(RuntimeError, match="controller fault"):
            await run_software_shot_mda(
                FailedOwner(),
                core,
                AngularStack(7, 0, 18, 2, -180, 180),
                4,
                [1] * 8,
                lambda *args: None,
                camera_exclusive=True,
                illumination_ready=True,
                timeout_s=1,
            )
    finally:
        core.unloadAllDevices()


async def test_sink_failure_does_not_interrupt_owner_abort():
    import asyncio

    class SlowAbortOwner(Owner):
        def __init__(self):
            super().__init__()
            self.abort_completed = False

        async def software_shots(self, definition, per, patterns, snap, **kwargs):
            try:
                await snap(0, 0, 1, 0)
            except BaseException:
                await asyncio.sleep(0.1)
                self.abort_completed = True
                raise

    core = make_core()
    owner = SlowAbortOwner()

    def sink(*args):
        raise RuntimeError("sink failed")

    try:
        with pytest.raises(RuntimeError, match="sink failed"):
            await run_software_shot_mda(
                owner,
                core,
                AngularStack(7, 0, 18, 2, -180, 180),
                4,
                [1] * 8,
                sink,
                camera_exclusive=True,
                illumination_ready=True,
            )
        assert owner.abort_completed
    finally:
        core.unloadAllDevices()


async def test_unbound_mode_runs_external_plane_hook_once_per_plane():
    core = make_core()
    owner = Owner()
    planes, frames = [], []
    try:
        await run_software_shot_mda(
            owner,
            core,
            AngularStack(7, 0, 18, 2, -180, 180),
            4,
            [1] * 8,
            lambda *frame: frames.append(frame[0]),
            camera_exclusive=True,
            illumination_ready=True,
            bind_stage=False,
            before_plane=lambda plane, target: planes.append((plane, target)),
        )
        assert planes == [(0, 0), (1, 18)] and frames == list(range(8))
    finally:
        core.unloadAllDevices()
