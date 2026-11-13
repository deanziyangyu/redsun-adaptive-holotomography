"""Quantized camera exposure readback over a fake PVA service."""

import asyncio
import json
import uuid

from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from redsun.path_provider import SessionPathProvider
from redsun.services import serve_controller

from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.services.mmcore_camera import MMCameraController


async def test_pva_reports_actual_quantized_exposure(tmp_path):
    class QuantizedCore(FakeCameraCore):
        def setExposure(self, exposure):
            super().setExposure(exposure + 0.002)

    core = QuantizedCore()
    controller = MMCameraController(
        core_of(core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    prefix = "EX" + uuid.uuid4().hex[:10].upper()
    server = asyncio.create_task(
        serve_controller(controller, EpicsPVATransport(), prefix)
    )
    camera = MMCamera(
        prefix + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="fake"),
        name="camera",
    )
    try:
        await asyncio.wait_for(camera.connect(), 10)
        await camera.exposure.set(5.0)
        change = json.loads(await camera.last_setting_change.get_value(cached=False))
        assert core.getExposure() == 5.002
        assert float(change["readback"]) == 5.002
        assert controller.exposure.get() == 5.002
        assert await camera.exposure.get_value(cached=False) == 5.002
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
