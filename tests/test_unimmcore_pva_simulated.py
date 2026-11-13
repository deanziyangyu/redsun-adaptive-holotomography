"""UniMMCore MDA to shared PVA stack service with fake serial and camera."""

import asyncio
import logging
import uuid

import pytest
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from pymmcore_plus.experimental.unicore import UniMMCore
from pymmcore_plus.mda import MDARunner
from redsun.services import serve_controller
from test_unified_mcu import FakePort
from test_unimmcore_deferred_mda import TriggerCamera
from useq import MDAEvent

from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.stack_frame_quarantine import FrameQuarantine
from redsun_aht.device.unified_mcu import AngularStack, UnifiedMcu
from redsun_aht.device.unimmcore_pva_stack import DeferredPvaEngine, DeferredPvaStage
from redsun_aht.services.unified_mcu import UnifiedMcuController


@pytest.fixture(autouse=True)
def quiet_p4p_debug(caplog):
    """Avoid malformed third-party debug formatting during subscriptions."""
    caplog.set_level(logging.INFO, logger="p4p.client.raw")


@pytest.mark.parametrize("drop", [None, 1])
@pytest.mark.parametrize("completion_delay", [0, 0.25])
@pytest.mark.parametrize("trigger_mask", [1, 6])
async def test_unimmcore_runner_uses_shared_stack_pva_service(
    drop, completion_delay, trigger_mask
):
    loop = asyncio.get_running_loop()
    camera = TriggerCamera(drop=drop)

    class TriggeringPort(FakePort):
        def emit(self, line):
            if line.startswith("ZSTACK complete") and completion_delay:
                loop.call_later(completion_delay, super().emit, line)
            else:
                super().emit(line)

        async def write(self, data):
            if data == b"identity\n" and trigger_mask != 1:
                self.commands.append("identity")
                self.emit("ARH-UNIFIED-STACK/0.18")
                return len(data)
            count = await super().write(data)
            if data == b"START 7\n":
                for _ in range(3):
                    camera.trigger()
                camera.pending.put(None)
            return count

    port = TriggeringPort(repeat=True)
    prefix = "UM" + uuid.uuid4().hex[:12].upper()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    stack = AngularStack(7, 0, 18, 3, -180, 180, trigger_mask=trigger_mask)
    quarantine = FrameQuarantine(max_frames=3, max_bytes=1024)
    stages = []

    def run_mda():
        core = UniMMCore()

        stage = DeferredPvaStage(
            client,
            loop,
            run_id=7,
            um_per_rev=1000,
            origin_um=0,
            lower_tenths=-180,
            upper_tenths=180,
            ready=True,
            trigger_mask=trigger_mask,
        )
        stages.append(stage)
        core.loadPyDevice("sim-camera", camera)
        core.loadPyDevice("sim-z", stage)
        core.initializeDevice("sim-camera")
        core.initializeDevice("sim-z")
        core.setCameraDevice("sim-camera")
        core.setFocusDevice("sim-z")
        core.setAutoShutter(False)
        engine = DeferredPvaEngine(core, stage)
        engine.restore_initial_state = False
        runner = MDARunner()
        runner.set_engine(engine)
        runner.events.frameReady.connect(
            lambda image, event, metadata: quarantine.append(int(image[0, 0]), image)
        )
        events = [MDAEvent(index={"z": i}, z_pos=z) for i, z in enumerate((0, 5, 10))]
        try:
            runner.run(events)
        finally:
            core.unloadAllDevices()

    try:
        await asyncio.wait_for(client.connect(), 10)
        await client.reserve()
        await asyncio.wait_for(asyncio.to_thread(run_mda), 20)
        assert len(stages) == 1 and stages[0].future is not None
        progress = await asyncio.wait_for(asyncio.wrap_future(stages[0].future), 10)
        assert progress.complete
        evidence = quarantine.finish(
            stack, progress, camera_baseline=0, capture_exclusive=True
        )
        assert port.commands.count("START 7") == 1
        assert port.commands.count("TRIGMASK 7 6") == int(trigger_mask == 6)
        assert not any(command.startswith("ABORT") for command in port.commands)
        if drop is None:
            assert evidence.status == "complete_order_inferred"
            assert len(evidence.associations) == 3
        else:
            assert evidence.status == "camera_count_or_sequence_mismatch"
            assert len(evidence.frames) == 2 and not evidence.associations
        await client.release_reservation()
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
