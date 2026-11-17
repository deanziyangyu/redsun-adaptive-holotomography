"""Headless Due-to-DCAM finite stack rehearsal."""

import asyncio
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from redsun_aht.device.unimmcore_triggered_camera import dcam_external_stack_camera
from redsun_aht.robotic.device.due_stack import capture_due_stack


class TestDueDcamCapture(unittest.IsolatedAsyncioTestCase):
    async def test_close_waiter_timeout_does_not_cancel_native_cleanup(self):
        stop_entered = Event()
        allow_stop = Event()
        core = Mock()
        core.getCameraDevice.return_value = "dcam"
        core.isSequenceRunning.side_effect = [False, True]
        core.isBufferOverflowed.return_value = False
        core.getRemainingImageCount.return_value = 0
        core.getImageWidth.return_value = 4
        core.getImageHeight.return_value = 4
        core.getBytesPerPixel.return_value = 2
        properties = {
            "TRIGGER SOURCE": "EXTERNAL", "TRIGGER ACTIVE": "EDGE",
            "TriggerPolarity": "NEGATIVE", "Trigger": "NORMAL",
        }
        core.getProperty.side_effect = lambda _, key: properties[key]

        def stop_sequence():
            stop_entered.set()
            if not allow_stop.wait(5):
                raise RuntimeError("Test did not release the native stop")

        core.stopSequenceAcquisition.side_effect = stop_sequence
        camera = dcam_external_stack_camera(
            core, device="dcam", polarity="NEGATIVE", owns_core=True,
        )
        await camera.arm(2)
        finish = asyncio.create_task(camera.finish())
        try:
            self.assertTrue(await asyncio.to_thread(stop_entered.wait, 1))
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(camera.close(), timeout=0.05)
            self.assertFalse(camera.released)
            core.unloadAllDevices.assert_not_called()
        finally:
            allow_stop.set()
            try:
                await finish
            finally:
                await asyncio.wait_for(camera.close(), timeout=1)
        self.assertTrue(camera.released)
        core.stopSequenceAcquisition.assert_called_once()
        core.unloadAllDevices.assert_called_once()

    async def test_polarity_change_after_arming_prevents_due_start(self):
        core = Mock()
        core.getCameraDevice.return_value = "dcam"
        core.isSequenceRunning.return_value = False
        core.getImageWidth.return_value = 4
        core.getImageHeight.return_value = 4
        core.getBytesPerPixel.return_value = 2
        properties = {
            "TRIGGER SOURCE": "EXTERNAL", "TRIGGER ACTIVE": "EDGE",
            "TriggerPolarity": "NEGATIVE", "Trigger": "NORMAL",
        }
        core.getProperty.side_effect = lambda _, key: properties[key]

        def start_sequence(*args):
            properties["TriggerPolarity"] = "POSITIVE"
            core.isSequenceRunning.return_value = True

        core.startSequenceAcquisition.side_effect = start_sequence
        camera = dcam_external_stack_camera(
            core, device="dcam", polarity="NEGATIVE", owns_core=True,
        )
        start = AsyncMock()
        mcu = SimpleNamespace(zdt_start=SimpleNamespace(set=start))
        with self.assertRaisesRegex(RuntimeError, "camera property mismatch: TriggerPolarity"):
            await capture_due_stack(
                mcu, camera, {"run_id": 7, "count": 2}, timeout_s=2,
                exclusive_camera=True, illumination_ready=True,
            )
        core.startSequenceAcquisition.assert_called_once_with(2, 0, True)
        core.stopSequenceAcquisition.assert_called_once()
        core.unloadAllDevices.assert_called_once()
        start.assert_not_awaited()

    async def test_live_property_mismatch_prevents_arming_and_releases_camera(self):
        core = Mock()
        core.getCameraDevice.return_value = "dcam"
        properties = {
            "TRIGGER SOURCE": "EXTERNAL", "TRIGGER ACTIVE": "EDGE",
            "TriggerPolarity": "POSITIVE", "Trigger": "NORMAL",
        }
        core.getProperty.side_effect = lambda _, key: properties[key]
        camera = dcam_external_stack_camera(
            core, device="dcam", polarity="NEGATIVE", owns_core=True,
        )
        start = AsyncMock()
        mcu = SimpleNamespace(zdt_start=SimpleNamespace(set=start))
        with self.assertRaisesRegex(RuntimeError, "camera property mismatch: TriggerPolarity"):
            await capture_due_stack(
                mcu, camera, {"run_id": 7, "count": 2}, timeout_s=2,
                exclusive_camera=True, illumination_ready=True,
            )
        core.startSequenceAcquisition.assert_not_called()
        core.unloadAllDevices.assert_called_once()
        start.assert_not_awaited()

    async def test_short_capture_aborts_without_waiting_for_due_completion(self):
        polling = asyncio.Event()
        poll_cancelled = asyncio.Event()

        async def collect(count):
            await polling.wait()
            return (SimpleNamespace(sequence=1),)

        async def status(*, cached):
            polling.set()
            try:
                await asyncio.Future()
            finally:
                poll_cancelled.set()

        camera = SimpleNamespace(
            arm=AsyncMock(return_value=0), confirm_ready=AsyncMock(),
            collect=collect, finish=AsyncMock(), close=AsyncMock(),
        )
        mcu = SimpleNamespace(
            zdt_start=SimpleNamespace(set=AsyncMock()),
            zdt_abort=SimpleNamespace(set=AsyncMock()),
            blank=SimpleNamespace(set=AsyncMock()),
            zdt_status_request=SimpleNamespace(set=AsyncMock()),
            zdt_status=SimpleNamespace(get_value=status),
        )
        with self.assertRaises(ExceptionGroup) as failure:
            await capture_due_stack(
                mcu, camera, {"run_id": 7, "count": 2}, timeout_s=2,
                exclusive_camera=True, illumination_ready=True,
            )
        self.assertIn("could not be associated", str(failure.exception.exceptions[0]))
        self.assertTrue(poll_cancelled.is_set())
        mcu.zdt_abort.set.assert_awaited_once_with(7)
        mcu.blank.set.assert_awaited_once_with(True)
        camera.finish.assert_not_awaited()
        camera.close.assert_awaited_once()

    async def test_operator_cancellation_aborts_submitted_run_and_releases_camera(self):
        submitted = asyncio.Event()
        collecting = asyncio.Event()
        cancelled = asyncio.Event()

        async def collect(count):
            collecting.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def start(request):
            submitted.set()

        camera = SimpleNamespace(
            arm=AsyncMock(return_value=0), confirm_ready=AsyncMock(),
            collect=collect, finish=AsyncMock(), close=AsyncMock(),
        )
        mcu = SimpleNamespace(
            zdt_start=SimpleNamespace(set=start),
            zdt_abort=SimpleNamespace(set=AsyncMock()),
            blank=SimpleNamespace(set=AsyncMock()),
            zdt_status_request=SimpleNamespace(set=AsyncMock()),
            zdt_status=SimpleNamespace(
                get_value=AsyncMock(return_value='{"run_id":7,"state":"active","edges":0}')
            ),
        )
        capture = asyncio.create_task(capture_due_stack(
            mcu, camera, {"run_id": 7, "count": 2}, timeout_s=2,
            exclusive_camera=True, illumination_ready=True,
        ))
        try:
            await asyncio.wait_for(collecting.wait(), timeout=1)
            self.assertTrue(submitted.is_set())
        finally:
            capture.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await capture
        self.assertTrue(cancelled.is_set())
        mcu.zdt_abort.set.assert_awaited_once_with(7)
        mcu.blank.set.assert_awaited_once_with(True)
        camera.finish.assert_not_awaited()
        camera.close.assert_awaited_once()

    async def test_terminal_firmware_fault_cancels_camera_collection(self):
        collecting = asyncio.Event()
        cancelled = asyncio.Event()

        async def collect(count):
            collecting.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def status(*, cached):
            await collecting.wait()
            return '{"run_id":7,"state":"fault_unverified","edges":0}'

        camera = SimpleNamespace(
            arm=AsyncMock(return_value=0), confirm_ready=AsyncMock(),
            collect=collect, finish=AsyncMock(), close=AsyncMock(),
        )
        mcu = SimpleNamespace(
            zdt_start=SimpleNamespace(set=AsyncMock()),
            zdt_abort=SimpleNamespace(set=AsyncMock()),
            blank=SimpleNamespace(set=AsyncMock()),
            zdt_status_request=SimpleNamespace(set=AsyncMock()),
            zdt_status=SimpleNamespace(get_value=status),
        )
        with self.assertRaises(ExceptionGroup) as failure:
            await capture_due_stack(
                mcu, camera, {"run_id": 7, "count": 2}, timeout_s=2,
                exclusive_camera=True, illumination_ready=True,
            )
        self.assertIn("fault_unverified", str(failure.exception.exceptions[0]))
        self.assertTrue(cancelled.is_set())
        mcu.zdt_abort.set.assert_awaited_once_with(7)
        mcu.blank.set.assert_awaited_once_with(True)
        camera.finish.assert_not_awaited()
        camera.close.assert_awaited_once()

    async def test_contiguous_frames_and_gap_cleanup(self):
        for indices in ((0, 1), (0, 2)):
            with self.subTest(indices=indices):
                core = Mock()
                core.getCameraDevice.return_value = "dcam"
                core.getImageWidth.return_value = 1
                core.getImageHeight.return_value = 1
                core.getBytesPerPixel.return_value = 2
                properties = {
                    "TRIGGER SOURCE": "EXTERNAL", "TRIGGER ACTIVE": "EDGE",
                    "TriggerPolarity": "POSITIVE", "Trigger": "NORMAL",
                }
                core.getProperty.side_effect = lambda _, key: properties[key]
                core.isSequenceRunning.side_effect = [False, True, False]
                core.isBufferOverflowed.return_value = False
                core.getRemainingImageCount.side_effect = [0, 2, 1, 0]
                frames = []
                for index in indices:
                    metadata = Mock()
                    metadata.GetSingleTag.return_value.GetValue.return_value = str(index)
                    frames.append(([[index]], metadata))
                core.popNextImageAndMD.side_effect = frames
                camera = dcam_external_stack_camera(core, device="dcam", polarity="POSITIVE")

                async def start(_):
                    core.startSequenceAcquisition.assert_called_once_with(2, 0, True)
                    self.assertEqual(core.getRemainingImageCount.call_count, 1)

                abort = AsyncMock()
                blank = AsyncMock()
                mcu = SimpleNamespace(
                    zdt_start=SimpleNamespace(set=AsyncMock(side_effect=start)),
                    zdt_abort=SimpleNamespace(set=abort),
                    blank=SimpleNamespace(set=blank),
                    zdt_status_request=SimpleNamespace(set=AsyncMock()),
                    zdt_status=SimpleNamespace(
                        get_value=AsyncMock(return_value='{"run_id":7,"state":"complete","edges":2}')
                    ),
                )
                capture = capture_due_stack(
                    mcu, camera, {"run_id": 7, "count": 2}, timeout_s=2,
                    exclusive_camera=True, illumination_ready=True,
                )
                if indices == (0, 1):
                    status, frames = await capture
                    self.assertEqual(status.edges, 2)
                    self.assertEqual([frame.sequence for frame in frames], [1, 2])
                    abort.assert_not_awaited()
                    blank.assert_not_awaited()
                else:
                    with self.assertRaises(ExceptionGroup) as failure:
                        await capture
                    self.assertIn("could not be associated", str(failure.exception.exceptions[0]))
                    self.assertIn("Check due_z_stack_status", failure.exception.__notes__[0])
                    abort.assert_awaited_once_with(7)
                    blank.assert_awaited_once_with(True)
                mcu.zdt_status.get_value.assert_awaited_once_with(cached=False)
                core.stopSequenceAcquisition.assert_not_called()

    async def test_oversize_stack_rejected_before_camera_or_due_starts(self):
        core = Mock()
        core.getCameraDevice.return_value = "dcam"
        properties = {
            "TRIGGER SOURCE": "EXTERNAL", "TRIGGER ACTIVE": "EDGE",
            "TriggerPolarity": "POSITIVE", "Trigger": "NORMAL",
        }
        core.getProperty.side_effect = lambda _, key: properties[key]
        core.isSequenceRunning.return_value = False
        core.getImageWidth.return_value = 4096
        core.getImageHeight.return_value = 4096
        core.getBytesPerPixel.return_value = 2
        camera = dcam_external_stack_camera(core, device="dcam", polarity="POSITIVE")
        start = AsyncMock()
        mcu = SimpleNamespace(zdt_start=SimpleNamespace(set=start))
        with self.assertRaisesRegex(RuntimeError, "exceeds the triggered camera frame budget"):
            await capture_due_stack(
                mcu, camera, {"run_id": 7, "count": 5}, timeout_s=2,
                exclusive_camera=True, illumination_ready=True,
            )
        core.startSequenceAcquisition.assert_not_called()
        start.assert_not_awaited()

    async def test_invalid_plane_count_releases_owned_camera_without_arming(self):
        core = Mock()
        camera = dcam_external_stack_camera(
            core, device="dcam", polarity="POSITIVE", owns_core=True,
        )
        start = AsyncMock()
        mcu = SimpleNamespace(zdt_start=SimpleNamespace(set=start))
        with self.assertRaisesRegex(ValueError, "Invalid Due stack plane count"):
            await capture_due_stack(
                mcu, camera, {"run_id": 7, "count": 33}, timeout_s=2,
                exclusive_camera=True, illumination_ready=True,
            )
        core.startSequenceAcquisition.assert_not_called()
        core.unloadAllDevices.assert_called_once()
        start.assert_not_awaited()
