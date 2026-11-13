"""Gated host-paced shots with one camera sequence over PVA."""

import asyncio
import json
import uuid

import pytest
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from redsun.path_provider import SessionPathProvider
from redsun.services import serve_controller
from test_unified_mcu import HoldStagePort

from redsun_aht.device.pva_gated_shots import acquire_pva_gated_shots
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.spinnaker_camera import SpinnakerCamera
from redsun_aht.device.unified_mcu import AngularStack, UnifiedMcu
from redsun_aht.services.spinnaker_camera import SpinnakerCameraController
from redsun_aht.services.unified_mcu import UnifiedMcuController


@pytest.mark.parametrize(
    ("drop", "presenter"),
    [(False, False), (True, False), (False, True)],
    ids=["complete", "missing", "presenter"],
)
async def test_gated_shots_arm_once_and_wait_for_each_frame(tmp_path, drop, presenter):
    class CameraSource(FakeCameraCore):
        starts = 0

        def startContinuousSequenceAcquisition(self, interval=0):
            self.starts += 1
            super().startContinuousSequenceAcquisition(interval)

        def popNextImageAndMD(self):
            frame = self.popNextImage()
            return frame, {"frame_id": str(self.popped - 1), "timestamp_ns": "42"}

        def close(self):
            pass

    source = CameraSource(pixel_type="16bit")

    class Port(HoldStagePort):
        async def write(self, data):
            if data.decode().startswith("SHOT ") and not (
                drop and data.decode().startswith("SHOT 7 2 ")
            ):
                source.produce()
            return await super().write(data)

    port = Port()
    controller = SpinnakerCameraController(
        core_of(source),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
    )
    suffix = uuid.uuid4().hex[:10].upper()
    cp, mp = "GC" + suffix, "GM" + suffix
    servers = [
        asyncio.create_task(serve_controller(controller, EpicsPVATransport(), cp)),
        asyncio.create_task(
            serve_controller(
                UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), mp
            )
        ),
    ]
    camera = SpinnakerCamera(
        cp + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="gated"),
        name="camera",
    )
    owner = PvaUnifiedMcu(mp + ":", name="mcu")
    try:
        await asyncio.wait_for(asyncio.gather(camera.connect(), owner.connect()), 10)
        if presenter:
            from bluesky import RunEngine
            from bluesky import plan_stubs as bps

            from redsun_aht.presenter.pva_timelapse import (
                PvaTimeLapseBindings,
                PvaTimeLapsePresenter,
            )

            def run_plan():
                def noop(*args):
                    yield from bps.null()

                bindings = PvaTimeLapseBindings(
                    owner,
                    camera,
                    lambda _: None,
                    lambda *_: str(tmp_path / "presenter"),
                    noop,
                    noop,
                    noop,
                    noop,
                )
                plan = PvaTimeLapsePresenter("test", bindings=bindings).gated_stack(
                    "blue",
                    [1, 2, 3, 4],
                    first=0,
                    step=18,
                    planes=2,
                    lower=-180,
                    upper=180,
                    resources_exclusive=True,
                    external_trigger_configured=True,
                    illumination_ready=True,
                    bind_stage=False,
                )
                docs = []
                engine = RunEngine({}, context_managers=[])
                engine.subscribe(lambda name, doc: docs.append((name, doc)))
                engine(plan)
                return docs

            docs = await asyncio.wait_for(asyncio.to_thread(run_plan), 30)
            events = [doc["data"] for name, doc in docs if name == "event"]
            assert [event["pattern"] for event in events if "pattern" in event] == [
                1,
                2,
                3,
                4,
                1,
                2,
                3,
                4,
            ]
            assert "c1" in port.commands and "SHOTBIND 0" in port.commands
            assert not owner.reservation_held
            assert source.starts == 1
            return
        await owner.reserve()
        call = acquire_pva_gated_shots(
            owner,
            camera,
            AngularStack(7, 0, 18, 2, -180, 180),
            [1, 2, 3, 4],
            str(tmp_path / "stack"),
            resources_exclusive=True,
            external_trigger_configured=True,
            automatic_pulses_enabled=True,
            illumination_ready=True,
            bind_stage=False,
            timeout_s=3 if drop else 20,
        )
        if drop:
            with pytest.raises(TimeoutError):
                await call
            assert not any(x.startswith("SHOT 7 3 ") for x in port.commands)
        else:
            result = await call
            assert result.evidence.stored_frames == 8
            assert result.requested_frames == tuple(
                (i // 4, 18 * (i // 4), i % 4 + 1) for i in range(8)
            )
            timing = json.loads((tmp_path / "stack" / "cam.timing.json").read_text())
            assert [x["camera_frame_counter"] for x in timing["records"]] == list(
                range(8)
            )
        assert source.starts == 1
    finally:
        for server in servers:
            server.cancel()
        await asyncio.gather(*servers, return_exceptions=True)
