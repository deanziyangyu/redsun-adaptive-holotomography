"""Headless OpenCV replay through the real Redsun-managed PVA service."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from redsun import Session
from redsun.aio import run_coro
from redsun_aht.configurations.arh_pva import arh_direct_pva
from redsun_aht.robotic.services.workcell_webcam import WorkcellWebcamController

try:
    import cv2
except ImportError:
    cv2 = None


async def wait_for_state(device, state):
    async with asyncio.timeout(15):
        while True:
            status = json.loads(await device.status.get_value(cached=False))
            if status["state"] == "error":
                raise RuntimeError(status)
            if status["state"] == state:
                return status
            await asyncio.sleep(0.05)


@unittest.skipIf(cv2 is None, "requires the camera-webcam extra")
class TestWorkcellWebcam(unittest.TestCase):
    def test_pva_start_stop_and_decode(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "input.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
            try:
                self.assertTrue(writer.isOpened())
                for index in range(20):
                    writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
            finally:
                writer.release()

            plugin = {"plugin_name": "redsun-aht", "plugin_id": "workcell-webcam"}
            config = {
                "session": "webcam-rehearsal", "mock": False, "strict": True,
                "storage": {"base_dir": str(path)},
                "services": {"webcam_service": {
                    **plugin, "prefix": "ARH-WEBCAM:",
                    "args": {"replay-file": str(source), "output-dir": str(path / "recordings"),
                             "fps": 5, "duration": 3},
                }},
                "devices": {"webcam": {**plugin, "service": "webcam_service"}},
            }

            async def record(device):
                await device.snap.set(True, timeout=10)
                still = json.loads(await device.status.get_value(cached=False))
                await device.snap.set(True, timeout=10)
                second_still = json.loads(await device.status.get_value(cached=False))
                await device.start.set(True)
                complete = await wait_for_state(device, "complete")
                await device.start.set(True)
                await wait_for_state(device, "recording")
                async with asyncio.timeout(5):
                    while not list((path / "recordings").glob("*.partial.avi")):
                        await asyncio.sleep(0.05)
                await device.finish.set(True)
                return still, second_still, complete, await wait_for_state(device, "complete")

            with arh_direct_pva():
                session = Session.from_config(config)
                try:
                    session.build()
                    device = session.devices["webcam"]
                    self.assertIn("ARH-WEBCAM:Start", device.start.source)
                    self.assertIn("ARH-WEBCAM:Finish", device.finish.source)
                    self.assertIn("ARH-WEBCAM:Status", device.status.source)
                    still, second_still, full, stopped = run_coro(record(device))
                finally:
                    session.shutdown()
            self.assertEqual(full["frames"], 15)
            self.assertTrue(0 < stopped["frames"] < 15)
            self.assertNotEqual(full["path"], stopped["path"])
            self.assertNotEqual(still["path"], second_still["path"])
            for result in (still, second_still):
                self.assertEqual((result["state"], result["frames"]), ("complete", 1))
                self.assertEqual(cv2.imread(result["path"]).shape, (48, 64, 3))
            self.assertEqual(len(list((path / "recordings").iterdir())), 4)
            output = cv2.VideoCapture(full["path"])
            actual = []
            try:
                for _ in range(16):
                    ok, image = output.read()
                    if not ok:
                        break
                    actual.append(round(image.mean()))
            finally:
                output.release()
            self.assertEqual(actual, list(range(0, 150, 10)))


class TestWebcamWorkerCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_finish_reaps_blocked_worker_and_discards_partial(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            controller = WorkcellWebcamController(0, 0, path, fps=5, duration=30)
            spawn = asyncio.create_subprocess_exec
            processes = []

            async def blocked_worker(*args, **kwargs):
                partial = Path(args[6]).with_suffix(".partial.avi")
                child = await spawn(
                    sys.executable, "-c",
                    "from pathlib import Path; import sys, time; "
                    "Path(sys.argv[1]).touch(); time.sleep(30)",
                    str(partial), **kwargs,
                )
                processes.append(child)
                return child

            with patch(
                "redsun_aht.robotic.services.workcell_webcam.asyncio.create_subprocess_exec",
                side_effect=blocked_worker,
            ):
                await controller.start_recording()
                try:
                    async with asyncio.timeout(5):
                        while not list(path.glob("*.partial.avi")):
                            await asyncio.sleep(0.05)
                    await asyncio.wait_for(controller.stop_recording(), 5)
                    self.assertIsNotNone(processes[0].returncode)
                    self.assertEqual(list(path.iterdir()), [])
                    self.assertEqual(json.loads(controller.status.get())["state"], "error")
                finally:
                    await controller.disconnect()
