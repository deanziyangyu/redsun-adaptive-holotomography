"""Subprocess lifecycle test for idle Redsun-managed fake services."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid

import pytest
from ophyd_async.core import wait_for_value
from p4p.client.asyncio import Context
from redsun.path_provider import SessionPathProvider

from redsun_aht.device.mmcore.camera import MMCamera


@pytest.fixture(autouse=True)
def quiet_p4p_debug(caplog):
    """Avoid malformed third-party debug formatting during subscriptions."""
    caplog.set_level(logging.INFO, logger="p4p.client.raw")


@pytest.mark.parametrize("service", ["coarse", "fast", "camera"])
async def test_managed_stdin_close_exits_cleanly(service):
    prefix = "LIFE" + uuid.uuid4().hex[:10].upper()
    env = os.environ.copy()
    env["REDSUN_SERVICE_NAME"] = "fake-" + service
    env["REDSUN_SERVICE_PREFIX"] = prefix
    process = subprocess.Popen(
        [sys.executable, __file__, "--serve", service, prefix],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    context = Context("pva")
    try:
        await asyncio.wait_for(context.get(prefix + ":PVI"), 10)
        assert process.stdin is not None
        process.stdin.close()
        code = await asyncio.wait_for(asyncio.to_thread(process.wait), 10)
        assert code == 0
    finally:
        context.close()
        if process.poll() is None:
            process.kill()
            await asyncio.to_thread(process.wait)


@pytest.mark.parametrize("received", [0, 1])
async def test_managed_camera_shutdown_while_waiting_for_frames(tmp_path, received):
    prefix = "LIFE" + uuid.uuid4().hex[:10].upper()
    env = os.environ.copy()
    env["REDSUN_SERVICE_NAME"] = "fake-waiting-camera"
    env["REDSUN_SERVICE_PREFIX"] = prefix
    process = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "--serve",
            "camera_one" if received else "camera",
            prefix,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    camera = MMCamera(
        prefix + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="shutdown"),
        name="camera",
    )
    store = tmp_path / "waiting.zarr"
    try:
        await camera.connect(timeout=15)
        await camera.file_path.set(str(store))
        await camera.num_capture.set(3)
        await camera.capture.set(True)
        await camera.acquire.set(True)
        await wait_for_value(camera.sequence_started, True, timeout=5)
        if received:
            await wait_for_value(camera.frame_count, 1, timeout=5)
        assert await camera.captured.get_value(cached=False) == 0
        assert process.stdin is not None
        process.stdin.close()
        assert await asyncio.wait_for(asyncio.to_thread(process.wait), 10) == 0
        if received:
            manifest = json.loads((store / "cam.manifest.json").read_text())
            assert manifest["frame_count"] == 1
            assert manifest["planned_frames"] == 3
        else:
            assert not (store / "cam.manifest.json").exists()
            assert not (store / "cam" / "c").exists()
    finally:
        if process.poll() is None:
            process.kill()
            await asyncio.to_thread(process.wait)


if __name__ == "__main__":
    from fastcs.transports.epics.pva.transport import EpicsPVATransport
    from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
    from redsun.services import serve_controller
    from test_esp300_readonly import Port
    from test_unified_mcu import FakePort

    from redsun_aht.device.esp300_readonly import Esp300ReadOnly
    from redsun_aht.device.unified_mcu import UnifiedMcu
    from redsun_aht.services.esp300_readonly import Esp300ReadOnlyController
    from redsun_aht.services.mmcore_camera import MMCameraController
    from redsun_aht.services.unified_mcu import UnifiedMcuController

    kind, prefix = sys.argv[2:]
    if kind == "coarse":
        controller = Esp300ReadOnlyController(Esp300ReadOnly(Port()))
    elif kind == "fast":
        controller = UnifiedMcuController(UnifiedMcu(FakePort(repeat=True)))
    else:

        class PartialCamera(FakeCameraCore):
            def startContinuousSequenceAcquisition(self, interval=0):
                super().startContinuousSequenceAcquisition(interval)
                if kind == "camera_one":
                    self.produce(1)

        controller = MMCameraController(
            core_of(PartialCamera()),
            LABEL,
            DATA_KEY,
            defer_initial_frame=True,
            buffered_capture=True,
        )
    asyncio.run(serve_controller(controller, EpicsPVATransport(), prefix))
