"""Finite triggered-camera adapter with a fake UniMMCore implementation."""

import asyncio
from collections import deque
from threading import Lock

import numpy as np
import pytest

from redsun_aht.device.automatic_shot_capture import acquire_automatic_shots
from redsun_aht.device.unified_mcu import AngularStack, AutomaticShotSummary
from redsun_aht.device.unimmcore_triggered_camera import UniMMCoreTriggeredCamera


class Core:
    def __init__(self):
        self.frames = deque()
        self.lock = Lock()
        self.running = False
        self.overflow = False
        self.stopped = False
        self.mode = "Edge Trigger"

    def getCameraDevice(self):
        return "cam"

    def getProperty(self, device, prop):
        return self.mode

    def isSequenceRunning(self):
        return self.running

    def clearCircularBuffer(self):
        self.frames.clear()

    def startSequenceAcquisition(self, count, interval, stop_on_overflow):
        self.running = True

    def startContinuousSequenceAcquisition(self, interval):
        self.running = True
        self.continuous = True

    def stopSequenceAcquisition(self):
        self.running = False
        self.stopped = True

    def isBufferOverflowed(self):
        return self.overflow

    def getRemainingImageCount(self):
        with self.lock:
            return len(self.frames)

    def popNextImageAndMD(self):
        with self.lock:
            return self.frames.popleft()

    def push(self, i):
        with self.lock:
            self.frames.append((np.full((2, 2), i, dtype=np.uint16), {"counter": i}))


class Owner:
    def __init__(self, core, count):
        self.core = core
        self.count = count
        self.started = 0

    async def automatic_shots(self, definition, per, mask, interval, **kwargs):
        assert self.core.running
        self.started += 1
        for i in range(1, self.count + 1):
            self.core.push(i)
            await asyncio.sleep(0.001)
        return AutomaticShotSummary(attempted_pulses=self.count)


def make(core):
    return UniMMCoreTriggeredCamera(
        core,
        device="cam",
        trigger_property="TriggerMode",
        trigger_value="Edge Trigger",
        counter_baseline=0,
        counter_from_metadata=lambda metadata: metadata["counter"],
    )


@pytest.mark.parametrize("per", [4, 16])
async def test_automatic_capture_from_fake_core(per):
    core = Core()
    camera = make(core)
    owner = Owner(core, 2 * per)
    result = await acquire_automatic_shots(
        owner,
        camera,
        AngularStack(7, 0, 18, 2, -180, 180),
        per,
        1,
        50,
        capture_exclusive=True,
        illumination_ready=True,
        timeout_s=2,
    )
    assert (
        result.status == "complete_order_inferred"
        and len(result.frames) == per * 2
        and core.stopped
    )
    assert [f.sequence for f in result.frames] == list(range(1, 2 * per + 1))
    assert not result.frames[0].data.flags.writeable


async def test_trigger_mismatch_does_not_start():
    core = Core()
    core.mode = "Internal Trigger"
    camera = make(core)
    owner = Owner(core, 8)
    with pytest.raises(RuntimeError, match="trigger property mismatch"):
        await acquire_automatic_shots(
            owner,
            camera,
            AngularStack(7, 0, 18, 2, -180, 180),
            4,
            1,
            50,
            capture_exclusive=True,
            illumination_ready=True,
            timeout_s=1,
        )
    assert owner.started == 0 and not core.running


async def test_overflow_rejects_frames_and_stops():
    core = Core()
    camera = make(core)
    await camera.arm(4)
    core.overflow = True
    try:
        with pytest.raises(RuntimeError, match="overflow"):
            await camera.collect(4)
    finally:
        await camera.close()
    assert core.stopped


async def test_bad_metadata_rejects_and_stops():
    core = Core()
    camera = make(core)
    await camera.arm(4)
    core.frames.append((np.zeros((2, 2)), {}))
    try:
        with pytest.raises(KeyError):
            await camera.collect(4)
    finally:
        await camera.close()
    assert core.stopped


async def test_cancelled_collection_closes_camera():
    core = Core()
    camera = make(core)
    await camera.arm(4)
    task = asyncio.create_task(camera.collect(4))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await camera.close()
    assert core.stopped


async def test_timeout_preserves_partial_frames_and_completed_mcu_count():
    from redsun_aht.device.automatic_shot_capture import AutomaticCaptureTimeout

    class PartialOwner:
        async def automatic_shots(self, definition, per, mask, interval, **kwargs):
            for i in range(1, 4):
                core.push(i)
                await asyncio.sleep(0.01)
            return AutomaticShotSummary(attempted_pulses=8)

    core = Core()
    camera = make(core)
    with pytest.raises(AutomaticCaptureTimeout) as failure:
        await acquire_automatic_shots(
            PartialOwner(),
            camera,
            AngularStack(7, 0, 18, 2, -180, 180),
            4,
            1,
            50,
            capture_exclusive=True,
            illumination_ready=True,
            timeout_s=0.2,
        )
    assert failure.value.attempted_pulses == 8
    assert [frame.sequence for frame in failure.value.frames] == [1, 2, 3]
    assert core.stopped


async def test_timeout_while_controller_active_has_unknown_attempt_count():
    from redsun_aht.device.automatic_shot_capture import AutomaticCaptureTimeout

    class SlowOwner:
        cancelled = False

        async def automatic_shots(self, *args, **kwargs):
            core.push(1)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    core = Core()
    camera = make(core)
    owner = SlowOwner()
    with pytest.raises(AutomaticCaptureTimeout) as failure:
        await acquire_automatic_shots(
            owner,
            camera,
            AngularStack(7, 0, 18, 2, -180, 180),
            4,
            1,
            50,
            capture_exclusive=True,
            illumination_ready=True,
            timeout_s=0.2,
        )
    assert failure.value.attempted_pulses is None
    assert [frame.sequence for frame in failure.value.frames] == [1]
    assert owner.cancelled and core.stopped


async def test_continuous_camera_checks_ready_and_extra_frames():
    core = Core()
    camera = UniMMCoreTriggeredCamera(
        core,
        device="cam",
        trigger_property="TriggerMode",
        trigger_value="Edge Trigger",
        counter_baseline=0,
        counter_from_metadata=lambda m: m["counter"],
        continuous=True,
    )
    await camera.arm(2)
    assert core.continuous and core.running
    await camera.confirm_ready()
    core.push(1)
    with pytest.raises(RuntimeError, match="before stack start"):
        await camera.confirm_ready()
    await camera.collect(1)
    core.push(2)
    with pytest.raises(RuntimeError, match="extra camera frames"):
        await camera.finish()
    await camera.close()
    assert core.stopped


async def test_required_trigger_properties_checked_before_acquisition():
    core = Core()
    camera = UniMMCoreTriggeredCamera(
        core,
        device="cam",
        trigger_property="TriggerMode",
        trigger_value="Edge Trigger",
        counter_baseline=0,
        counter_from_metadata=lambda m: m["counter"],
        continuous=True,
        required_properties={"Trigger Source": "Line0"},
    )
    with pytest.raises(RuntimeError, match="property mismatch"):
        await camera.arm(2)
    assert not core.running
    await camera.close()
