"""Failure handling for finite internally triggered camera sequences."""

import numpy as np
import pytest
from test_unimmcore_software_shots import Owner, make_core

from redsun_aht.device.software_shot_core import run_core_software_shots
from redsun_aht.device.unified_mcu import AngularStack


@pytest.mark.parametrize("failure", ["missing", "overflow"])
async def test_bad_sequence_aborts_before_second_plane(failure):
    class Camera:
        running = False
        stopped = False

        def getAutoShutter(self):
            return False

        def clearCircularBuffer(self):
            pass

        def startSequenceAcquisition(self, count, interval, stop_on_overflow):
            assert (count, interval, stop_on_overflow) == (4, 0, True)
            self.running = True

        def isBufferOverflowed(self):
            return failure == "overflow"

        def getRemainingImageCount(self):
            return 0

        def isSequenceRunning(self):
            if failure == "missing" and self.running:
                self.running = False
            return self.running

        def stopSequenceAcquisition(self):
            self.running = False
            self.stopped = True

    camera, owner, frames = Camera(), Owner(), []
    with pytest.raises(RuntimeError, match=r"overflow|ended before all frames"):
        await run_core_software_shots(
            owner,
            camera,
            AngularStack(7, 0, 18, 2, -180, 180),
            4,
            [1] * 8,
            lambda *args: frames.append(args),
            camera_exclusive=True,
            illumination_ready=True,
            auto_shutter=False,
            plane_sequence=True,
            camera_internal_triggered=True,
        )
    assert owner.aborted and owner.requested == [0] and frames == []
    if failure == "overflow":
        assert camera.stopped


async def test_sequence_requires_internal_trigger_assertion():
    class Camera:
        pass

    with pytest.raises(ValueError, match="internal trigger"):
        await run_core_software_shots(
            Owner(),
            Camera(),
            AngularStack(7, 0, 18, 1, -180, 180),
            4,
            [1] * 4,
            lambda *args: None,
            camera_exclusive=True,
            illumination_ready=True,
            auto_shutter=False,
            plane_sequence=True,
        )


async def test_native_frame_gate_can_control_laser_or_shutter_without_mm_shutter():
    core = make_core()
    actions, frames = [], []

    class Gate:
        is_open = False

        async def open(self):
            self.is_open = True
            actions.append("open")

        async def close(self):
            self.is_open = False
            actions.append("close")

    gate = Gate()

    class GatedCore:
        def __getattr__(self, name):
            return getattr(core, name)

        def snapImage(self):
            assert gate.is_open
            actions.append("snap")
            core.snapImage()

    owner = Owner()
    try:
        await run_core_software_shots(
            owner,
            GatedCore(),
            AngularStack(7, 0, 18, 1, -180, 180),
            4,
            [1] * 4,
            lambda shot, *args: frames.append(shot),
            camera_exclusive=True,
            illumination_ready=True,
            auto_shutter=True,
            frame_gate=gate,
        )
        assert frames == [0, 1, 2, 3]
        assert actions == ["open", "snap", "close"] * 4
        assert not gate.is_open
    finally:
        core.unloadAllDevices()


async def test_native_gate_closes_on_capture_failure():
    core = make_core()
    actions = []

    class Gate:
        async def open(self):
            actions.append("open")

        async def close(self):
            actions.append("close")

    owner = Owner()

    def sink(*args):
        raise RuntimeError("sink failed")

    try:
        with pytest.raises(RuntimeError, match="sink failed"):
            await run_core_software_shots(
                owner,
                core,
                AngularStack(7, 0, 18, 1, -180, 180),
                4,
                [1] * 4,
                sink,
                camera_exclusive=True,
                illumination_ready=True,
                auto_shutter=True,
                frame_gate=Gate(),
            )
        assert actions == ["open", "close"] and owner.aborted
    finally:
        core.unloadAllDevices()


async def test_finite_sequence_waits_for_natural_completion():
    class Camera:
        running = False
        remaining = 0
        idle_polls = 0
        explicit_stops = 0

        def getAutoShutter(self):
            return False

        def clearCircularBuffer(self):
            pass

        def startSequenceAcquisition(self, count, interval, overflow):
            self.running, self.remaining = True, count

        def isBufferOverflowed(self):
            return False

        def getRemainingImageCount(self):
            return self.remaining

        def popNextImage(self):
            self.remaining -= 1
            return np.zeros((2, 2), dtype=np.uint16)

        def isSequenceRunning(self):
            if self.running and not self.remaining:
                self.idle_polls += 1
                if self.idle_polls >= 3:
                    self.running = False
            return self.running

        def stopSequenceAcquisition(self):
            self.explicit_stops += 1
            self.running = False

    camera, owner, frames = Camera(), Owner(), []
    await run_core_software_shots(
        owner,
        camera,
        AngularStack(7, 0, 18, 1, -180, 180),
        4,
        [1] * 4,
        lambda shot, *args: frames.append(shot),
        camera_exclusive=True,
        illumination_ready=True,
        auto_shutter=False,
        plane_sequence=True,
        camera_internal_triggered=True,
    )
    assert frames == [0, 1, 2, 3]
    assert camera.idle_polls >= 3 and camera.explicit_stops == 0
