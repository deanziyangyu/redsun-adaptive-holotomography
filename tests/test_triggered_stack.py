"""Headless orchestration tests using independent synthetic camera counters."""

import asyncio

import pytest

from redsun_aht.device.triggered_stack import acquire_stack
from redsun_aht.device.unified_mcu import AngularStack, StackProgress


class Camera:
    def __init__(self, *, drop=False, fail_arm=False, fail_close=False):
        self.armed = False
        self.closed = False
        self.frames = []
        self.drop = drop
        self.fail_arm = fail_arm
        self.fail_close = fail_close
        self.finished = asyncio.Event()

    async def arm(self, frame_count):
        if self.fail_arm:
            raise RuntimeError("camera not ready")
        self.armed = True
        return 100

    def edge(self, plane):
        assert self.armed
        if not (self.drop and plane == 1):
            self.frames.append(101 + len(self.frames))

    async def collect(self, frame_count):
        await self.finished.wait()
        return tuple(self.frames)

    async def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("cleanup failed")


class Client:
    def __init__(self, camera, *, fail=False):
        self.camera = camera
        self.fail = fail
        self.starts = 0
        self.aborts = 0

    async def execute(self, stack, *, ready):
        assert ready and self.camera.armed
        self.starts += 1
        for i in range(stack.count):
            self.camera.edge(i)
            if self.fail:
                raise RuntimeError("stack error")
        self.camera.finished.set()
        return StackProgress(stack.run_id, (0, 18, 36), complete=True)

    async def abort_stack(self, run_id):
        self.aborts += 1


async def run(client, camera):
    return await acquire_stack(
        client,
        camera,
        AngularStack(7, 0, 18, 3, -180, 180),
        capture_exclusive=True,
        illumination_ready=True,
        timeout_s=1,
    )


async def test_camera_arms_before_one_start_and_reconciles():
    camera = Camera()
    client = Client(camera)
    result = await run(client, camera)
    assert result.status == "complete_order_inferred"
    assert len(result.associations) == 3
    assert client.starts == 1 and client.aborts == 0 and camera.closed


async def test_missing_frame_is_unassociated():
    camera = Camera(drop=True)
    client = Client(camera)
    result = await run(client, camera)
    assert result.status == "camera_count_or_sequence_mismatch"
    assert not result.associations and camera.closed and client.starts == 1


async def test_not_ready_never_starts_and_failure_aborts_and_cleans_up():
    camera = Camera(fail_arm=True)
    client = Client(camera)
    with pytest.raises(RuntimeError, match="not ready"):
        await run(client, camera)
    assert client.starts == client.aborts == 0 and camera.closed
    camera = Camera(fail_close=True)
    client = Client(camera, fail=True)
    with pytest.raises(RuntimeError, match="stack error"):
        await run(client, camera)
    assert client.starts == client.aborts == 1 and camera.closed


async def test_cancellation_requests_abort_and_closes_camera():
    camera = Camera()
    client = Client(camera)
    started = asyncio.Event()

    async def wait_for_stack(stack, *, ready):
        started.set()
        await asyncio.Event().wait()

    client.execute = wait_for_stack
    task = asyncio.create_task(run(client, camera))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.aborts == 1 and camera.closed


async def test_camera_is_drained_during_stack_with_one_frame_buffer():
    class BufferedCamera(Camera):
        def __init__(self):
            super().__init__()
            self.queue = asyncio.Queue(maxsize=1)

        async def collect(self, frame_count):
            return tuple([await self.queue.get() for _ in range(frame_count)])

    class BufferedClient(Client):
        async def execute(self, stack, *, ready):
            assert ready and self.camera.armed
            self.starts += 1
            for i in range(stack.count):
                await self.camera.queue.put(101 + i)
            return StackProgress(stack.run_id, (0, 18, 36), complete=True)

    camera = BufferedCamera()
    client = BufferedClient(camera)
    result = await run(client, camera)
    assert result.status == "complete_order_inferred"
    assert client.starts == 1 and client.aborts == 0 and camera.closed


async def test_camera_failure_cancels_stack_and_requests_abort():
    camera = Camera()
    client = Client(camera)
    stack_cancelled = asyncio.Event()

    async def failing_collect(frame_count):
        await asyncio.sleep(0)
        raise RuntimeError("camera failed")

    async def wait_for_stack(stack, *, ready):
        try:
            await asyncio.Event().wait()
        finally:
            stack_cancelled.set()

    camera.collect = failing_collect
    client.execute = wait_for_stack
    with pytest.raises(RuntimeError, match="camera failed"):
        await run(client, camera)
    assert stack_cancelled.is_set() and client.aborts == 1 and camera.closed


async def test_timeout_requests_abort_before_stubborn_collector_finishes():
    camera = Camera()
    client = Client(camera)
    release = asyncio.Event()
    collector_started = asyncio.Event()
    abort_requested = asyncio.Event()

    async def stubborn_collect(frame_count):
        collector_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return ()

    async def stuck_stack(stack, *, ready):
        await asyncio.Event().wait()

    async def abort(run_id):
        abort_requested.set()

    camera.collect = stubborn_collect
    client.execute = stuck_stack
    client.abort_stack = abort
    task = asyncio.create_task(
        acquire_stack(
            client,
            camera,
            AngularStack(7, 0, 18, 3, -180, 180),
            capture_exclusive=True,
            illumination_ready=True,
            timeout_s=0.03,
        )
    )
    await asyncio.wait_for(collector_started.wait(), 1)
    await asyncio.wait_for(abort_requested.wait(), 0.5)
    assert not camera.closed
    release.set()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(task, 1)
    assert camera.closed


async def test_stubborn_collector_leaves_camera_release_unverified(caplog):
    camera = Camera()
    client = Client(camera)
    release = asyncio.Event()

    async def stubborn_collect(frame_count):
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return ()

    async def stuck_stack(stack, *, ready):
        await asyncio.Event().wait()

    camera.collect = stubborn_collect
    client.execute = stuck_stack
    try:
        with pytest.raises(TimeoutError):
            await acquire_stack(
                client,
                camera,
                AngularStack(7, 0, 18, 3, -180, 180),
                capture_exclusive=True,
                illumination_ready=True,
                timeout_s=0.03,
            )
        assert client.aborts == 1 and not camera.closed
        assert "camera release unverified" in caplog.text
    finally:
        release.set()
        await asyncio.sleep(0)


async def test_stubborn_abort_does_not_prevent_camera_cleanup(caplog):
    camera = Camera()
    client = Client(camera)
    release = asyncio.Event()
    abort_started = asyncio.Event()

    async def stuck_stack(stack, *, ready):
        await asyncio.Event().wait()

    async def stubborn_abort(run_id):
        abort_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    client.execute = stuck_stack
    client.abort_stack = stubborn_abort
    try:
        with pytest.raises(TimeoutError, match="capture timed out"):
            await asyncio.wait_for(
                acquire_stack(
                    client,
                    camera,
                    AngularStack(7, 0, 18, 3, -180, 180),
                    capture_exclusive=True,
                    illumination_ready=True,
                    timeout_s=0.03,
                ),
                timeout=3,
            )
        assert abort_started.is_set() and camera.closed
        assert "stop is unverified" in caplog.text
    finally:
        release.set()
        await asyncio.sleep(0)
