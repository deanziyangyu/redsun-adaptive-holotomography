"""Headless stepped stage against the unified service and fake serial."""

import asyncio
import uuid

import numpy as np
from bluesky import RunEngine
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore
from pymmcore_plus.mda import MDAEngine, MDARunner
from redsun.services import serve_controller
from test_unified_mcu import FakePort
from useq import MDAEvent

from redsun_aht.device.pva_stepped_axis import PvaSteppedAxis
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.pva_x_state import PvaXState
from redsun_aht.device.software_snap import SoftwareSnapCamera
from redsun_aht.device.software_stepped_plan import software_stepped_plan
from redsun_aht.device.unified_mcu import UnifiedMcu
from redsun_aht.device.unimmcore_stepped_stage import SteppedPvaStage
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.services.unified_mcu import UnifiedMcuController
from redsun_aht.storage import read_manifest


async def test_stepped_stage_over_fake_pva_owner(tmp_path):
    class Port(FakePort):
        def __init__(self):
            super().__init__(repeat=True)
            self.position = 0
            self.target = 0

        async def write(self, data):
            command = data.decode().strip()
            if command == "identity":
                self.commands.append(command)
                self.emit("ARH-UNIFIED-STACK/0.18")
                return len(data)
            if command.startswith("COMMIT "):
                self.target = int(command.split()[2])
            if command == "zasyncxstate":
                self.commands.append(command)
                frame = bytearray(37)
                frame[:4] = bytes([1, 0x43, 37, 12])
                frame[14] = int(self.position < 0)
                frame[15:19] = abs(self.position).to_bytes(4, "big")
                frame[22] = int(self.position < 0)
                frame[23:27] = abs(self.position).to_bytes(4, "big")
                frame[34:37] = bytes([3, 3, 0x6B])
                self.emit("ZA pending")
                self.emit("ZA state_hex=" + frame.hex())
                return len(data)
            if command.startswith("START "):
                self.commands.append(command)
                self.position = self.target
                self.emit("ZSTACK started")
                self.emit(f"ZSTACK EDGE 0 {self.target}")
                self.emit("ZSTACK complete edges=1")
                return len(data)
            return await super().write(data)

    prefix = "STEP" + uuid.uuid4().hex[:10].upper()
    port = Port()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    camera_prefix = "SC" + uuid.uuid4().hex[:10].upper()
    camera_core = FakeCameraCore()
    camera_server = asyncio.create_task(
        serve_controller(
            MMCameraController(
                core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
            ),
            EpicsPVATransport(),
            camera_prefix,
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    state = PvaXState(prefix + ":", name="state")
    camera = SoftwareSnapCamera(camera_prefix + ":", name="camera")
    try:
        await asyncio.wait_for(
            asyncio.gather(client.connect(), state.connect(), camera.connect()), 10
        )
        await client.reserve()
        stage = SteppedPvaStage(
            client,
            state,
            asyncio.get_running_loop(),
            first_run_id=50,
            um_per_rev=1000,
            origin_um=0,
            lower_tenths=-180,
            upper_tenths=180,
            trigger_mask=2,
            resources_exclusive=True,
            ready=True,
        )
        for position in (0, 5, 10):
            await asyncio.to_thread(stage.set_position_um, position)
        assert await asyncio.to_thread(stage.get_position_um) == 10
        assert [c for c in port.commands if c.startswith("START ")] == ["START 50"] * 3

        class Camera(CameraDevice):
            def __init__(self):
                super().__init__()
                self.exposure = 1.0
                self.frames = 0

            def get_exposure(self):
                return self.exposure

            def set_exposure(self, value):
                self.exposure = value

            def shape(self):
                return (4, 4)

            def dtype(self):
                return np.uint16

            def start_sequence(self, n, get_buffer):
                for _ in range(n or 0):
                    self.frames += 1
                    image = get_buffer(self.shape(), self.dtype())
                    image[:] = self.frames
                    yield {"frame": self.frames}

        def run_mda():
            core = UniMMCore()
            core.loadPyDevice("camera", Camera())
            core.loadPyDevice("stage", stage)
            core.initializeDevice("camera")
            core.initializeDevice("stage")
            core.setCameraDevice("camera")
            core.setFocusDevice("stage")
            core.setAutoShutter(False)
            runner = MDARunner()
            runner.set_engine(
                MDAEngine(
                    core, use_hardware_sequencing=False, restore_initial_state=False
                )
            )
            frames = []
            runner.events.frameReady.connect(
                lambda image, event, metadata: frames.append(
                    (dict(event.index), int(image[0, 0]))
                )
            )
            try:
                runner.run(
                    [
                        MDAEvent(index={"z": i}, z_pos=z)
                        for i, z in enumerate((0, 5, 10))
                    ]
                )
            finally:
                core.unloadAllDevices()
            return frames

        frames = await asyncio.wait_for(asyncio.to_thread(run_mda), 20)
        assert frames == [({"z": 0}, 1), ({"z": 1}, 2), ({"z": 2}, 3)]
        assert [c for c in port.commands if c.startswith("START ")] == ["START 50"] * 6

        redsun_axis = PvaSteppedAxis(
            client,
            state,
            asyncio.get_running_loop(),
            first_run_id=stage.run_id,
            um_per_rev=1000,
            origin_um=0,
            lower_tenths=-180,
            upper_tenths=180,
            trigger_mask=2,
            resources_exclusive=True,
            ready=True,
        )

        def run_redsun():
            engine = RunEngine({}, context_managers=[])
            events = []
            engine.subscribe(
                lambda name, doc: (
                    events.append(doc["data"]) if name == "event" else None
                )
            )
            engine(
                software_stepped_plan(
                    redsun_axis,
                    camera,
                    (0, 5, 10),
                    reservation_held=True,
                    camera_exclusive=True,
                    store_uri=str(tmp_path / "software.zarr"),
                )
            )
            return events

        events = await asyncio.wait_for(asyncio.to_thread(run_redsun), 20)
        assert [event["camera_frame_count"] for event in events] == [1, 2, 3]
        assert [event["stored_index"] for event in events] == [0, 1, 2]
        manifest = read_manifest(
            str(tmp_path / "software.zarr"), data_key="software_snap"
        )
        assert manifest.is_complete and manifest.frame_count == 3
        assert [event["controller_position_um"] for event in events] == [0, 5, 10]
        assert [c for c in port.commands if c.startswith("START ")] == ["START 50"] * 9
        await client.release_reservation()
    finally:
        server.cancel()
        camera_server.cancel()
        await asyncio.gather(server, camera_server, return_exceptions=True)
