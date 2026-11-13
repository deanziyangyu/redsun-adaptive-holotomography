"""Experimental combined workflow with simulated services only."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from ophyd_async.core import get_mock_put

from redsun_aht.device.pva_capture_evidence import CaptureEvidence
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.unassociated_stack import acquire_unassociated_stack
from redsun_aht.device.unified_mcu import AngularStack, StackProgress


async def test_prepares_before_one_stack_and_returns_unassociated_evidence():
    camera = AsyncMock()
    client = AsyncMock()
    prepared = asyncio.Event()
    frames = asyncio.Event()
    stack = AngularStack(7, 0, 18, 3, -180, 180)

    async def prepare(*args, **kwargs):
        prepared.set()

    async def collect():
        await frames.wait()
        return CaptureEvidence("test.zarr", 3, 3)

    async def execute(definition, *, ready, require_reservation):
        assert prepared.is_set() and ready and require_reservation
        frames.set()
        return StackProgress(7, (0, 18, 36), complete=True)

    camera.prepare.side_effect = prepare
    camera.collect.side_effect = collect
    client.execute.side_effect = execute
    result = await acquire_unassociated_stack(
        client,
        camera,
        stack,
        "test.zarr",
        resources_exclusive=True,
        external_trigger_configured=True,
        illumination_ready=True,
        timeout_s=1,
    )
    assert result.quality == "unassociated"
    assert result.capture.stored_frames == 3 and result.controller.complete
    client.execute.assert_awaited_once()
    client.abort_stack.assert_not_awaited()
    camera.close.assert_awaited_once()
    client.reserve.assert_awaited_once()
    client.release_reservation.assert_awaited_once()


async def test_reservation_released_during_preparation_does_not_submit():
    client = PvaUnifiedMcu("MCU:", name="mcu")
    await client.connect(mock=True)
    camera = AsyncMock()

    async def release_during_prepare(*args, **kwargs):
        await client.release_reservation()

    async def wait_for_frames():
        await asyncio.Event().wait()

    camera.prepare.side_effect = release_during_prepare
    camera.collect.side_effect = wait_for_frames
    with pytest.raises(RuntimeError, match="reservation"):
        await acquire_unassociated_stack(
            client,
            camera,
            AngularStack(7, 0, 18, 3, -180, 180),
            "test.zarr",
            resources_exclusive=True,
            external_trigger_configured=True,
            illumination_ready=True,
            timeout_s=1,
        )
    get_mock_put(client.request).assert_not_called()
    camera.close.assert_awaited_once()


async def test_camera_failure_requests_abort_and_closes():
    camera = AsyncMock()
    client = AsyncMock()
    camera.collect.side_effect = RuntimeError("camera failed")

    async def stuck(*args, **kwargs):
        await asyncio.Event().wait()

    client.execute.side_effect = stuck
    with pytest.raises(RuntimeError, match="camera failed"):
        await acquire_unassociated_stack(
            client,
            camera,
            AngularStack(7, 0, 18, 3, -180, 180),
            "test.zarr",
            resources_exclusive=True,
            external_trigger_configured=True,
            illumination_ready=True,
            timeout_s=1,
        )
    client.abort_stack.assert_awaited_once_with(7)
    camera.close.assert_awaited_once()
    client.release_reservation.assert_not_awaited()


async def test_camera_preparation_failure_never_submits_stack():
    camera = AsyncMock()
    client = AsyncMock()
    camera.prepare.side_effect = RuntimeError("not ready")
    with pytest.raises(RuntimeError, match="not ready"):
        await acquire_unassociated_stack(
            client,
            camera,
            AngularStack(7, 0, 18, 3, -180, 180),
            "test.zarr",
            resources_exclusive=True,
            external_trigger_configured=True,
            illumination_ready=True,
            timeout_s=1,
        )
    client.execute.assert_not_awaited()
    camera.close.assert_awaited_once()
    client.release_reservation.assert_not_awaited()


async def test_missing_camera_frames_after_completed_stack_does_not_abort():
    camera, client = AsyncMock(), AsyncMock()
    client.execute.return_value = StackProgress(7, (0, 18, 36), complete=True)

    async def wait_for_frames():
        await asyncio.Event().wait()

    camera.collect.side_effect = wait_for_frames
    with pytest.raises(TimeoutError, match="stack capture timed out"):
        await acquire_unassociated_stack(
            client,
            camera,
            AngularStack(7, 0, 18, 3, -180, 180),
            "test.zarr",
            resources_exclusive=True,
            external_trigger_configured=True,
            illumination_ready=True,
            timeout_s=0.03,
        )
    client.abort_stack.assert_not_awaited()
    camera.close.assert_awaited_once()
    client.release_reservation.assert_not_awaited()


async def test_partial_camera_capture_is_not_successful_or_released():
    camera, client = AsyncMock(), AsyncMock()
    client.execute.return_value = StackProgress(7, (0, 18, 36), complete=True)
    camera.collect.return_value = CaptureEvidence("test.zarr", 3, 2)
    with pytest.raises(RuntimeError, match="incomplete"):
        await acquire_unassociated_stack(
            client,
            camera,
            AngularStack(7, 0, 18, 3, -180, 180),
            "test.zarr",
            resources_exclusive=True,
            external_trigger_configured=True,
            illumination_ready=True,
            timeout_s=1,
        )
    camera.close.assert_awaited_once()
    client.release_reservation.assert_not_awaited()


async def test_reservation_failure_does_not_touch_camera():
    camera, client = AsyncMock(), AsyncMock()
    client.reserve.side_effect = RuntimeError("already reserved")
    with pytest.raises(RuntimeError, match="already reserved"):
        await acquire_unassociated_stack(
            client,
            camera,
            AngularStack(7, 0, 18, 3, -180, 180),
            "test.zarr",
            resources_exclusive=True,
            external_trigger_configured=True,
            illumination_ready=True,
            timeout_s=1,
        )
    camera.prepare.assert_not_awaited()
    camera.close.assert_not_awaited()
