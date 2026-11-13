"""Loopback PVA combined stack with fake serial and software-generated frames."""

import asyncio
import json
import logging
import uuid

import numpy as np
import pytest
import zarr
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from redsun.path_provider import SessionPathProvider
from redsun.services import serve_controller
from test_unified_mcu import FakePort

from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.device.pva_capture_evidence import PvaCaptureEvidence
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.unassociated_stack import acquire_unassociated_stack
from redsun_aht.device.unified_mcu import AngularStack, UnifiedMcu
from redsun_aht.services import mmcore_camera
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.services.unified_mcu import UnifiedMcuController


@pytest.mark.parametrize(
    ("drop_frame", "buffered", "write_failure", "sidecar_failure"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (True, True, False, False),
        (False, True, True, False),
        (False, True, False, True),
    ],
)
async def test_combined_pva_with_synthetic_event_driven_frames(
    tmp_path, drop_frame, buffered, write_failure, sidecar_failure, caplog, monkeypatch
):
    # p4p's debug-only raw subscription message has malformed %-arguments;
    # another suite enables DEBUG globally. Keep this unrelated logger quiet.
    caplog.set_level(logging.INFO, logger="p4p.client.raw")
    camera_core = FakeCameraCore()
    attempts = []
    if write_failure:
        factory = mmcore_camera.FrameStore

        class FailingStore:
            def __init__(self, *args):
                self.wrapped = factory(*args)

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def append(self, frame):
                attempts.append(1)
                if len(attempts) == 2:
                    raise RuntimeError("injected PVA store failure")
                return self.wrapped.append(frame)

            def append_batch(self, frames):
                original = self.wrapped._array

                class PartialArray:
                    def __getattr__(self, name):
                        return getattr(original, name)

                    def append(self, values, *, axis):
                        attempts.extend((1, 1))
                        original.append(values[:1], axis=axis)
                        raise RuntimeError("injected PVA store failure")

                self.wrapped._array = PartialArray()
                return self.wrapped.append_batch(frames)

        monkeypatch.setattr(mmcore_camera, "FrameStore", FailingStore)

    class EventDrivenPort(FakePort):
        def emit(self, line):
            super().emit(line)
            if line.startswith("ZSTACK EDGE ") and not (
                drop_frame and line.startswith("ZSTACK EDGE 1 ")
            ):
                camera_core.produce(1)

    port = EventDrivenPort()
    camera_controller = MMCameraController(
        core_of(camera_core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=buffered,
    )
    if sidecar_failure:

        def fail_sidecar():
            raise RuntimeError("injected timing sidecar failure")

        monkeypatch.setattr(camera_controller, "_write_timing_sidecar", fail_sidecar)
    mcu_controller = UnifiedMcuController(UnifiedMcu(port))
    suffix = uuid.uuid4().hex[:10].upper()
    camera_prefix, mcu_prefix = "UC" + suffix, "UM" + suffix
    servers = [
        asyncio.create_task(
            serve_controller(camera_controller, EpicsPVATransport(), camera_prefix)
        ),
        asyncio.create_task(
            serve_controller(mcu_controller, EpicsPVATransport(), mcu_prefix)
        ),
    ]
    camera = MMCamera(
        camera_prefix + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="fake"),
        name="camera",
    )
    client = PvaUnifiedMcu(mcu_prefix + ":", name="mcu")
    store = tmp_path / "capture.zarr"
    try:
        await asyncio.wait_for(asyncio.gather(camera.connect(), client.connect()), 10)

        async def run():
            return await acquire_unassociated_stack(
                client,
                PvaCaptureEvidence(camera, timeout_s=10),
                AngularStack(7, 0, 18, 3, -180, 180),
                str(store),
                resources_exclusive=True,
                external_trigger_configured=True,
                illumination_ready=True,
                timeout_s=1 if drop_frame else 10,
            )

        if write_failure or sidecar_failure:
            with pytest.raises(RuntimeError, match="camera service fault"):
                await run()
            assert await client.reservation_status.get_value(cached=False) == "held"
            error = await camera.last_error.get_value(cached=False)
            assert "injected" in error
            if write_failure:
                assert await camera.captured.get_value(cached=False) == 0
                assert len(attempts) == 2
            else:
                assert await camera.captured.get_value(cached=False) == 3
                assert "timing sidecar" in error
        elif drop_frame:
            with pytest.raises(TimeoutError, match="stack capture timed out"):
                await run()
            assert await client.reservation_status.get_value(cached=False) == "held"
        else:
            result = await run()
            assert result.controller.targets_tenths == (0, 18, 36)
            assert result.capture.stored_frames == 3
            assert result.quality == "unassociated"
            assert await client.reservation_status.get_value(cached=False) == "free"
        assert port.commands.count("START 7") == 1
        image = zarr.open_array(str(store / DATA_KEY), mode="r")
        expected_count = 1 if write_failure else 2 if drop_frame else 3
        assert image.shape == (expected_count, 4, 6)
        for frame in image:
            assert np.all(frame == frame[0, 0]) and 1 <= frame[0, 0] <= 3
        if write_failure and buffered:
            assert not (store / f"{DATA_KEY}.manifest.json").exists()
        if not write_failure and not sidecar_failure:
            timing = json.loads((store / f"{DATA_KEY}.timing.json").read_text())
            assert timing["written_count"] == expected_count
            assert len(timing["records"]) == expected_count
            assert timing["camera_frame_counter"] is None
        assert camera_core.snapped == 0
    finally:
        for server in servers:
            server.cancel()
        await asyncio.gather(*servers, return_exceptions=True)
