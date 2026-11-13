"""Synthetic camera/automatic-controller orchestration tests."""

import asyncio

import pytest

from redsun_aht.device.automatic_shot_capture import (
    CameraFrame,
    acquire_automatic_shots,
)
from redsun_aht.device.unified_mcu import AngularStack, AutomaticShotSummary


class Camera:
    def __init__(self, *, drop=None, fail=False):
        self.queue = asyncio.Queue(maxsize=1)
        self.armed = False
        self.closed = False
        self.drop = drop
        self.fail = fail

    async def arm(self, count):
        self.armed = True
        return 100

    async def collect(self, count):
        if self.fail:
            raise RuntimeError("camera failed")
        frames = []
        while True:
            frame = await self.queue.get()
            if frame is None:
                return tuple(frames)
            frames.append(frame)

    async def close(self):
        self.closed = True


class Owner:
    def __init__(self, camera):
        self.camera = camera
        self.started = 0
        self.cancelled = False

    async def automatic_shots(self, definition, per, pattern_mask, interval, **kwargs):
        assert self.camera.armed and kwargs["camera_armed"]
        self.started += 1
        try:
            for i in range(definition.count * per):
                if self.camera.drop != i:
                    await self.camera.queue.put(
                        CameraFrame(101 + i, 1000 + i, bytes([i % 256]))
                    )
            await self.camera.queue.put(None)
            return AutomaticShotSummary(attempted_pulses=definition.count * per)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def run(owner, camera, per):
    return await acquire_automatic_shots(
        owner,
        camera,
        AngularStack(7, 0, 18, 2, -180, 180),
        per,
        3,
        50,
        capture_exclusive=True,
        illumination_ready=True,
        timeout_s=1,
    )


@pytest.mark.parametrize("per", [4, 16])
async def test_concurrent_collection_and_order_inferred(per):
    camera = Camera()
    owner = Owner(camera)
    result = await run(owner, camera, per)
    assert (
        camera.closed
        and owner.started == 1
        and result.status == "complete_order_inferred"
    )
    assert len(result.associations) == per * 2
    assert [(a.plane, a.target_tenths, a.pattern) for a in result.associations] == [
        (i // per, 18 * (i // per), 1 + i % 2) for i in range(per * 2)
    ]


async def test_missing_camera_frame_remains_unassociated():
    camera = Camera(drop=1)
    result = await run(Owner(camera), camera, 4)
    assert (
        result.status == "count_or_sequence_mismatch"
        and len(result.frames) == 7
        and not result.associations
        and camera.closed
    )


async def test_collector_failure_cancels_owner():
    camera = Camera(fail=True)
    owner = Owner(camera)
    with pytest.raises(RuntimeError, match="camera failed"):
        await run(owner, camera, 4)
    assert owner.started == 1 and owner.cancelled and camera.closed


async def test_camera_arm_failure_never_starts():
    camera = Camera()
    owner = Owner(camera)

    async def failed(count):
        raise RuntimeError("arm failed")

    camera.arm = failed
    with pytest.raises(RuntimeError, match="arm failed"):
        await run(owner, camera, 4)
    assert owner.started == 0 and camera.closed


async def test_cancellation_cancels_controller_and_closes_camera():
    camera = Camera()
    owner = Owner(camera)
    started = asyncio.Event()

    async def waiting(definition, per, mask, interval, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            owner.cancelled = True
            raise

    owner.automatic_shots = waiting
    task = asyncio.create_task(run(owner, camera, 4))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert owner.cancelled and camera.closed
