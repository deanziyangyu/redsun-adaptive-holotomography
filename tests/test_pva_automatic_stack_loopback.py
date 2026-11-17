"""One-command automatic stacks through fake PVA camera and MCU services."""

import asyncio
import json
import logging
import uuid

import pytest
import zarr
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from redsun.engine import RunEngine
from redsun.path_provider import SessionPathProvider
from redsun.services import serve_controller
from test_unified_mcu import AutoShotFakePort

from redsun_aht.device.esp300_stage import Esp300Stage
from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.device.pva_automatic_stack import acquire_pva_automatic_stack
from redsun_aht.device.pva_rotation import PvaRotationMotor
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.spinnaker_camera import SpinnakerCamera
from redsun_aht.device.unified_mcu import AngularStack, UnifiedMcu
from redsun_aht.presenter.holospim import HolospimPresenter
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.services.spinnaker_camera import SpinnakerCameraController
from redsun_aht.services.unified_mcu import UnifiedMcuController


@pytest.mark.parametrize("case", ["complete", "dropped", "extra"])
async def test_automatic_continuous_full_stack(tmp_path, caplog, case):
    caplog.set_level(logging.INFO, logger="p4p.client.raw")

    class SequenceCamera(FakeCameraCore):
        def isBufferOverflowed(self):
            return False

    fake_camera = SequenceCamera()

    class Port(AutoShotFakePort):
        async def write(self, data):
            if data.decode().startswith("SHOTAUTO "):
                fake_camera.produce({"complete": 48, "dropped": 47, "extra": 49}[case])
            return await super().write(data)

    port = Port()
    camera_service = MMCameraController(
        core_of(fake_camera),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
    )
    mcu_service = UnifiedMcuController(UnifiedMcu(port))
    suffix = uuid.uuid4().hex[:10].upper()
    cp, mp = "AC" + suffix, "AM" + suffix
    servers = [
        asyncio.create_task(serve_controller(camera_service, EpicsPVATransport(), cp)),
        asyncio.create_task(serve_controller(mcu_service, EpicsPVATransport(), mp)),
    ]
    camera = MMCamera(
        cp + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="fake"),
        name="camera",
    )
    owner = PvaUnifiedMcu(mp + ":", name="mcu")
    try:
        await asyncio.wait_for(asyncio.gather(camera.connect(), owner.connect()), 10)
        await owner.reserve()
        kwargs = dict(
            resources_exclusive=True,
            external_trigger_configured=True,
            continuous_capture_configured=True,
            illumination_ready=True,
            timeout_s=2 if case == "dropped" else 15,
        )
        call = acquire_pva_automatic_stack(
            owner,
            camera,
            AngularStack(9, 0, 18, 3, -180, 180),
            16,
            1,
            50,
            str(tmp_path / "stack.zarr"),
            **kwargs,
        )
        if case == "dropped":
            with pytest.raises(TimeoutError):
                await call
        elif case == "extra":
            with pytest.raises(Exception, match=r"count|extra|overflow"):
                await call
        else:
            result = await call
            assert (
                result.evidence.stored_frames
                == result.camera_count_delta
                == result.attempted_pulses
                == 48
            )
            assert len(result.requested_frames) == 48
            assert result.requested_frames[16] == (1, 18, 1)
            assert (
                zarr.open_array(
                    str(tmp_path / "stack.zarr" / DATA_KEY), mode="r"
                ).shape[0]
                == 48
            )
        assert len([x for x in port.commands if x.startswith("SHOTAUTO ")]) == 1
        assert not any(x.startswith("SHOT ") for x in port.commands)
    finally:
        for server in servers:
            server.cancel()
        await asyncio.gather(*servers, return_exceptions=True)


@pytest.mark.parametrize("case", ["complete", "extra", "presenter", "wrong-trigger"])
async def test_spinnaker_service_supports_the_same_automatic_stack(tmp_path, case):
    class NativeCamera(FakeCameraCore):
        def __init__(self):
            super().__init__(pixel_type="16bit")

        def isPropertyReadOnly(self, label, name):
            return not name.startswith("Trigger") and super().isPropertyReadOnly(
                label, name
            )

        def isBufferOverflowed(self):
            return False

        def popNextImageAndMD(self):
            frame = self.popNextImage()
            return frame, {"frame_id": str(self.popped - 1), "timestamp_ns": "42"}

        def close(self):
            pass

    fake_camera = NativeCamera()
    fake_camera.properties.update({
        "TriggerMode": "Off" if case == "wrong-trigger" else "On",
        "TriggerSelector": "FrameStart",
        "TriggerSource": "Line0",
        "TriggerActivation": "FallingEdge",
    })

    class Port(AutoShotFakePort):
        async def write(self, data):
            if data.decode().startswith("SHOTAUTO "):
                fake_camera.produce({"presenter": 2, "extra": 49}.get(case, 48))
            return await super().write(data)

    port = Port()
    camera_service = SpinnakerCameraController(
        core_of(fake_camera),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
    )
    mcu_service = UnifiedMcuController(UnifiedMcu(port))
    suffix = uuid.uuid4().hex[:10].upper()
    cp, mp = "SC" + suffix, "SM" + suffix
    servers = [
        asyncio.create_task(serve_controller(camera_service, EpicsPVATransport(), cp)),
        asyncio.create_task(serve_controller(mcu_service, EpicsPVATransport(), mp)),
    ]
    camera = SpinnakerCamera(
        cp + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="fake"),
        name="camera",
    )
    owner = PvaUnifiedMcu(mp + ":", name="mcu")
    try:
        await asyncio.wait_for(asyncio.gather(camera.connect(), owner.connect()), 10)
        if case in ("presenter", "wrong-trigger"):
            presenter = HolospimPresenter(
                "test", x28_bounds_tenths=(-36, 36), enable_x28_triggered_stack=True,
            )
            plan = presenter.x28_triggered_stack(
                owner, camera, Esp300Stage("UNUSED:", name="survey"),
                PvaRotationMotor("UNUSED:", name="rotation"),
                0, 18, 2, "green", 5, confirm_setup=True,
            )
            events = []
            future = RunEngine({})(plan, {"event": lambda _, doc: events.append(doc)})
            if case == "wrong-trigger":
                with pytest.raises(RuntimeError, match="TriggerMode=On"):
                    await asyncio.wrap_future(future)
                assert port.commands == ["identity", "00"]
            else:
                await asyncio.wait_for(asyncio.wrap_future(future), 20)
                record = events[0]["data"]
                assert record == {
                    "attempted_pulses": 2, "stored_frames": 2,
                    "camera_frame_count_delta": 2,
                }
                assert port.commands[-2:] == ["00", "c0"]
                assert not owner.reservation_held
            return
        await owner.reserve()
        call = acquire_pva_automatic_stack(
            owner,
            camera,
            AngularStack(9, 0, 18, 3, -180, 180),
            16,
            1,
            50,
            str(tmp_path / "spin.zarr"),
            resources_exclusive=True,
            external_trigger_configured=True,
            continuous_capture_configured=True,
            illumination_ready=True,
            timeout_s=15,
        )
        if case == "extra":
            with pytest.raises(Exception, match=r"extra|count|overflow"):
                await call
            return
        result = await call
        assert result.evidence.stored_frames == result.attempted_pulses == 48
        assert len([x for x in port.commands if x.startswith("SHOTAUTO ")]) == 1
        assert not any(x.startswith("SHOT ") for x in port.commands)
        timing = json.loads((tmp_path / "spin.zarr" / "cam.timing.json").read_text())
        assert timing["records"][0]["camera_metadata_source"] == "PySpin.GetNextImage"
        assert [r["camera_frame_counter"] for r in timing["records"]] == list(range(48))
    finally:
        for server in servers:
            server.cancel()
        await asyncio.gather(*servers, return_exceptions=True)
