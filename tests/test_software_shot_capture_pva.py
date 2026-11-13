"""Software-shot owner plus real loopback PVA fake camera; no hardware."""

import asyncio
import uuid
from itertools import pairwise

import numpy as np
import pytest
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from redsun.services import serve_controller
from test_unified_mcu import ShotFakePort

from redsun_aht.device.software_shot_capture import acquire_software_shots
from redsun_aht.device.software_snap import SoftwareSnapCamera
from redsun_aht.device.unified_mcu import AngularStack, UnifiedMcu
from redsun_aht.services.mmcore_camera import MMCameraController


@pytest.mark.parametrize("fail_at", [None, 3])
async def test_shots_with_pva_software_snap_and_partial_sink(fail_at):
    prefix = "SC" + uuid.uuid4().hex[:10].upper()
    core = FakeCameraCore()
    server = asyncio.create_task(
        serve_controller(
            MMCameraController(
                core_of(core), LABEL, DATA_KEY, defer_initial_frame=True
            ),
            EpicsPVATransport(),
            prefix,
        )
    )
    camera = SoftwareSnapCamera(prefix + ":", name="snap")
    port = ShotFakePort()
    owner = UnifiedMcu(port)
    frames = []

    async def sink(metadata, image):
        frames.append((metadata, image.copy()))
        if metadata.shot == fail_at:
            raise RuntimeError("injected sink failure")

    async def acquire():
        await acquire_software_shots(
            owner,
            camera,
            AngularStack(9, 0, 18, 2, -180, 180),
            4,
            [1] * 8,
            sink,
            camera_exclusive=True,
            illumination_ready=True,
        )

    try:
        await asyncio.wait_for(camera.connect(), 10)
        await owner.identify()
        if fail_at is None:
            await acquire()
            assert len(frames) == 8 and core.snapped == 8
            assert [m.camera_count for m, _ in frames] == list(range(1, 9))
            assert [m.target_tenths for m, _ in frames] == [0] * 4 + [18] * 4
            assert all(np.all(image == i + 1) for i, (_, image) in enumerate(frames))
            assert all(
                a[0].host_receipt_perf_counter_ns < b[0].host_receipt_perf_counter_ns
                for a, b in pairwise(frames)
            )
            assert port.commands[-1] == "SHOTFIN 9 7" and not owner.uncertain
        else:
            with pytest.raises(RuntimeError, match="sink failure"):
                await acquire()
            assert len(frames) == 4 and core.snapped == 4
            assert port.commands[-2:] == ["SHOT 9 3 1", "SHOTABORT 9"]
            assert owner.uncertain
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
