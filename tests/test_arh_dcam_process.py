"""Exercise external DCAM worker cleanup without opening a camera."""

import asyncio
import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from redsun_aht.device.dcam_process import DcamProcess


class TestDcamProcess(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_collection_rejects_queued_command(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        camera._process = await asyncio.create_subprocess_exec(
            sys.executable, "-B", "-c", "import sys, time; sys.stdin.readline(); time.sleep(30)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        reading = asyncio.Event()

        async def wait_for_response():
            reading.set()
            await asyncio.Future()

        collection = None
        ready = None
        try:
            with patch.object(camera, "_read_response", side_effect=wait_for_response) as read:
                collection = asyncio.create_task(camera.collect(2))
                await asyncio.wait_for(reading.wait(), timeout=3)
                ready = asyncio.create_task(camera.confirm_ready())
                await asyncio.sleep(0)
                collection.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await collection
                with self.assertRaisesRegex(RuntimeError, "exchange was interrupted"):
                    await asyncio.wait_for(ready, timeout=3)
                read.assert_awaited_once()
        finally:
            pending = [task for task in (collection, ready) if task is not None]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.wait_for(camera.close(), timeout=5)
        self.assertTrue(camera.released)

    async def test_close_interrupts_collection_without_competing_for_pipe(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        camera._process = await asyncio.create_subprocess_exec(
            sys.executable, "-B", "-c", "import sys, time; sys.stdin.readline(); time.sleep(30)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        collection = asyncio.create_task(camera.collect(2))
        try:
            await asyncio.sleep(0)
            await asyncio.wait_for(camera.close(), timeout=5)
            with self.assertRaisesRegex(RuntimeError, "disconnected during collect"):
                await collection
        finally:
            if not collection.done():
                collection.cancel()
                await asyncio.gather(collection, return_exceptions=True)
            await asyncio.wait_for(camera.close(), timeout=5)
        self.assertTrue(camera.released)

    async def test_close_during_worker_launch_prevents_arm(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        launch_started = asyncio.Event()
        finish_launch = asyncio.Event()
        spawn = asyncio.create_subprocess_exec

        async def delayed_spawn(*args, **kwargs):
            launch_started.set()
            await finish_launch.wait()
            return await spawn(
                sys.executable, "-B", "-c",
                "import pickle, struct, sys; "
                "assert sys.stdin.readline().strip() == 'close 0'; "
                "data = pickle.dumps((True, None)); "
                "sys.stdout.buffer.write(struct.pack('!I', len(data)) + data)",
                **kwargs,
            )

        with patch("redsun_aht.device.dcam_process.asyncio.create_subprocess_exec", delayed_spawn):
            arm = asyncio.create_task(camera.arm(2))
            try:
                await asyncio.wait_for(launch_started.wait(), timeout=2)
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(camera.close(), timeout=0.05)
                self.assertFalse(camera.released)
            finally:
                finish_launch.set()
                try:
                    with self.assertRaisesRegex(RuntimeError, "closed during worker startup"):
                        await asyncio.wait_for(arm, timeout=5)
                finally:
                    await asyncio.wait_for(camera.close(), timeout=5)
        self.assertTrue(camera.released)

    async def test_collect_rejects_frames_exceeding_total_byte_budget(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        camera._process = await asyncio.create_subprocess_exec(
            sys.executable, "-B", "-c", """
import sys
import numpy as np
from redsun_aht.device.automatic_shot_capture import CameraFrame
from redsun_aht.device.dcam_process import _write_response

sys.stdin.readline()
wire = sys.stdout.buffer
_write_response(wire, (True, 2))
for sequence in (1, 2):
    _write_response(wire, CameraFrame(sequence, 100, np.zeros((4, 4), dtype=np.uint16)))
wire.flush()
""",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        try:
            with patch("redsun_aht.device.dcam_process.DCAM_STACK_MAX_BYTES", 48):
                with self.assertRaisesRegex(RuntimeError, "exceeded its frame budget"):
                    await asyncio.wait_for(camera.collect(2), timeout=5)
        finally:
            await asyncio.wait_for(camera.close(), timeout=3)
        self.assertTrue(camera.released)

    async def test_worker_disconnect_identifies_interrupted_command(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        camera._process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(7)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "disconnected during collect"):
                await asyncio.wait_for(camera.collect(2), timeout=5)
        finally:
            await asyncio.wait_for(camera.close(), timeout=3)
        self.assertTrue(camera.released)

    async def test_collect_preserves_frame_order_and_pixels_over_pipe(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        camera._process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", """
import sys
import numpy as np
from redsun_aht.device.automatic_shot_capture import CameraFrame
from redsun_aht.device.dcam_process import _write_response

wire = sys.stdout.buffer
assert sys.stdin.readline().strip() == "collect 2"
_write_response(wire, (True, 2))
for sequence in (1, 2):
    pixels = np.full((4, 4), sequence, dtype=np.uint16)
    _write_response(wire, CameraFrame(sequence, sequence * 100, pixels))
wire.flush()
assert sys.stdin.readline().strip() == "close 0"
_write_response(wire, (True, None))
wire.flush()
""",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        try:
            frames = await asyncio.wait_for(camera.collect(2), timeout=5)
            self.assertEqual([frame.sequence for frame in frames], [1, 2])
            self.assertEqual([frame.receipt_ns for frame in frames], [100, 200])
            for frame in frames:
                self.assertEqual(frame.data.shape, (4, 4))
                self.assertEqual(frame.data.dtype.name, "uint16")
                self.assertTrue((frame.data == frame.sequence).all())
                self.assertFalse(frame.data.flags.writeable)
        finally:
            await asyncio.wait_for(camera.close(), timeout=3)
        self.assertTrue(camera.released)

    async def test_worker_rejects_changed_config_before_loading_camera(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "external.cfg"
            config.write_bytes(b"changed")
            camera = DcamProcess(
                config, "dcam", "NEGATIVE", None,
                config_digest=hashlib.sha256(b"validated").hexdigest(),
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "config changed after validation"):
                    await asyncio.wait_for(camera.arm(2), timeout=5)
            finally:
                await asyncio.wait_for(camera.close(), timeout=3)
            self.assertTrue(camera.released)

    async def test_worker_failure_before_close_is_reported(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        worker = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "raise SystemExit(7)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        camera._process = worker
        await worker.wait()
        with self.assertRaisesRegex(RuntimeError, "code 7 before close"):
            await camera.close()
        self.assertTrue(camera.released)

    async def test_already_closed_command_pipe_can_still_reap_worker(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        worker = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        camera._process = worker
        worker.stdin.close()
        try:
            await asyncio.wait_for(camera.close(), timeout=5)
            self.assertIsNotNone(worker.returncode)
            self.assertTrue(camera.released)
        finally:
            if worker.returncode is None:
                worker.kill()
                await worker.wait()

    async def test_acknowledged_close_with_worker_failure_is_reported(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        worker = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "import pickle, struct, sys; sys.stdin.readline(); "
            "data = pickle.dumps((True, None)); "
            "sys.stdout.buffer.write(struct.pack('!I', len(data)) + data); sys.exit(7)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        )
        camera._process = worker
        with self.assertRaisesRegex(RuntimeError, "code 7 after acknowledging close"):
            await asyncio.wait_for(camera.close(), timeout=5)
        self.assertTrue(camera.released)

    async def test_close_timeout_does_not_cancel_worker_reap(self):
        camera = DcamProcess("unused", "dcam", "NEGATIVE", None)
        worker = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        camera._process = worker
        try:
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(camera.close(), timeout=0.05)
            self.assertFalse(camera.released)

            with self.assertRaisesRegex(RuntimeError, "did not close cleanly"):
                await asyncio.wait_for(camera.close(), timeout=5)
            self.assertIsNotNone(worker.returncode)
            self.assertTrue(camera.released)
        finally:
            if worker.returncode is None:
                try:
                    worker.kill()
                except ProcessLookupError:
                    pass
                await worker.wait()
