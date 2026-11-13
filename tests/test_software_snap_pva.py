"""Software-snap transport rehearsal with a fake camera only."""

import asyncio
import uuid

import numpy as np
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from redsun.services import serve_controller

from redsun_aht.device.software_snap import SoftwareSnapCamera
from redsun_aht.services.mmcore_camera import MMCameraController


async def test_one_shot_snap_over_pva():
    prefix = "SN" + uuid.uuid4().hex[:10].upper()
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
    try:
        await asyncio.wait_for(camera.connect(), 10)
        count, first = await camera.snap()
        assert count == 1 and core.snapped == 1 and np.all(first == 1)
        count, second = await camera.snap()
        assert count == 2 and core.snapped == 2 and np.all(second == 2)
        assert np.all(first == 1)
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_service_stores_multiple_software_snaps_over_pva(tmp_path, caplog):
    import json
    import logging

    import zarr
    from redsun.path_provider import SessionPathProvider

    from redsun_aht.device.mmcore.camera import MMCamera
    from redsun_aht.device.pva_capture_evidence import PvaCaptureEvidence

    caplog.set_level(logging.INFO, logger="p4p.client.raw")
    prefix = "SC" + uuid.uuid4().hex[:10].upper()
    core = FakeCameraCore()
    server = asyncio.create_task(
        serve_controller(
            MMCameraController(
                core_of(core),
                LABEL,
                DATA_KEY,
                defer_initial_frame=True,
                buffered_capture=True,
                finite_capture=True,
            ),
            EpicsPVATransport(),
            prefix,
        )
    )
    camera = MMCamera(
        prefix + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="snap"),
        name="camera",
    )
    try:
        await asyncio.wait_for(camera.connect(), 10)
        capture = PvaCaptureEvidence(camera)
        uri = str(tmp_path / "snap-store")
        try:
            await capture.prepare(
                uri, 4, internal_trigger_configured=True, software_snap=True
            )
            assert [await capture.snap() for _ in range(4)] == [1, 2, 3, 4]
            evidence = await capture.collect()
            assert evidence.stored_frames == 4
        finally:
            await capture.close()
        frames = zarr.open_array(str(tmp_path / "snap-store" / DATA_KEY), mode="r")
        assert [int(frames[i][0, 0]) for i in range(4)] == [1, 2, 3, 4]
        timing = json.loads(
            (tmp_path / "snap-store" / f"{DATA_KEY}.timing.json").read_text()
        )
        assert timing["written_count"] == 4 and core.snapped == 4
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
