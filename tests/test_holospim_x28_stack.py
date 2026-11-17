"""HoloSPIM desktop stack through loopback camera and MCU services."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastcs.control_system import FastCS
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, InternalTriggerCameraCore, core_of
from redsun.engine import RunEngine
from redsun.path_provider import SessionPathProvider
from test_unified_mcu import HoldStagePort

from redsun_aht.device.esp300_stage import Esp300Stage
from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.device.pva_rotation import PvaRotationMotor
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.spinnaker_camera import SpinnakerCamera
from redsun_aht.device.unified_mcu import UnifiedMcu
from redsun_aht.presenter.holospim import HolospimPresenter
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.services.spinnaker_camera import SpinnakerCameraController
from redsun_aht.services.unified_mcu import UnifiedMcuController
from redsun_aht.storage.writer import timing_sidecar_path


@pytest.mark.parametrize("native_flir", (False, True), ids=("pvcam", "flir"))
@pytest.mark.parametrize("bad_ready", (False, True))
async def test_desktop_stack_uses_session_devices_and_handles_failure(
    tmp_path, bad_ready, native_flir
):
    suffix = uuid4().hex[:10].upper()
    cp, mp = "XC" + suffix, "XM" + suffix

    class FakeSpinnakerCore(InternalTriggerCameraCore):
        closed = False

        def snap_with_metadata(self):
            frame = self.snap()
            return frame, {"frame_id": str(self.snapped), "timestamp_ns": "42"}

        def close(self):
            self.closed = True

    if native_flir:
        core = FakeSpinnakerCore(pixel_type="16bit")
        controller_type, client_type, camera_name = (
            SpinnakerCameraController,
            SpinnakerCamera,
            "camera",
        )
    else:
        core = InternalTriggerCameraCore()
        controller_type, client_type, camera_name = (
            MMCameraController,
            MMCamera,
            "pvcam",
        )
    port = HoldStagePort()
    port.bad_ready = bad_ready
    server_specs = (
        (
            controller_type(core_of(core), LABEL, DATA_KEY, defer_initial_frame=True),
            cp,
        ),
        (UnifiedMcuController(UnifiedMcu(port)), mp),
    )
    # One loopback IOC avoids two servers competing for the default PVA port.
    for controller, prefix in server_specs:
        controller.set_path([prefix])
    control_system = FastCS(
        [controller for controller, _ in server_specs], [EpicsPVATransport()]
    )
    server = asyncio.create_task(control_system.serve(interactive=False))
    camera = client_type(
        cp + ":", path_provider=SessionPathProvider(base_dir=tmp_path), name=camera_name
    )
    mcu = PvaUnifiedMcu(mp + ":", name="mcu")
    survey = Esp300Stage("UNUSED:", name="survey")
    rotation = PvaRotationMotor(mp + ":", name="rotation")
    engine = RunEngine({})
    locks = []
    engine.sig_locks_changed.connect(locks.append)
    events = []

    def collect(name, doc):
        if name == "event":
            events.append(doc["data"])

    try:
        await asyncio.wait_for(asyncio.gather(camera.connect(), mcu.connect()), 10)
        presenter = HolospimPresenter("holospim", x28_bounds_tenths=(-180, 180))
        if native_flir and not bad_ready:
            await camera.properties["TriggerMode"].set("On")
            before = list(port.commands)
            rejected = presenter.x28_software_stack(
                mcu, camera, survey, rotation, 0, 18, 2, "red", 2, 2, True
            )
            with pytest.raises(RuntimeError, match="internal trigger mode"):
                await asyncio.wrap_future(engine(rejected))
            assert port.commands == before and core.snapped == 0
            await camera.properties["TriggerMode"].set("Off")
            locks.clear()
        plan = presenter.x28_software_stack(
            mcu, camera, survey, rotation, 0, 18, 2, "red", 2, 2, True
        )
        result = asyncio.wrap_future(engine(plan, collect))
        if bad_ready:
            with pytest.raises(Exception, match=r"shot|Shot|Server error"):
                await asyncio.wait_for(result, 20)
            assert any(command.startswith("SHOTABORT ") for command in port.commands)
            assert await mcu.reservation_status.get_value() == "held"
            assert not events
        else:
            await asyncio.wait_for(result, 20)
            assert core.snapped == 4
            assert [event["target_tenths"] for event in events] == [0, 18]
            assert all(event["stored_frames"] == 2 for event in events)
            assert all(Path(event["store_uri"]).is_dir() for event in events)
            if native_flir:
                records = [
                    record
                    for event in events
                    for record in json.loads(
                        timing_sidecar_path(event["store_uri"], DATA_KEY).read_text(
                            "utf-8"
                        )
                    )["records"]
                ]
                assert [record["camera_frame_counter"] for record in records] == [
                    1,
                    2,
                    3,
                    4,
                ]
            assert port.commands[-1] == "00"
            assert await mcu.reservation_status.get_value() == "free"
        assert port.commands.index("c0") < port.commands.index("53")
        assert locks == [
            frozenset({"mcu", camera_name, "survey", "rotation"}),
            frozenset(),
        ]
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
    if native_flir:
        assert core.closed
