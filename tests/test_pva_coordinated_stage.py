"""End-to-end synthetic PVA coordination; no physical devices."""

import asyncio
import uuid

import pytest
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.services import serve_controller

from redsun_aht.device.motion_coordinator import AxisBinding, CompositeMotionCoordinator
from redsun_aht.operator_input.xinput import BUTTON_A, DPAD_UP, ControllerState
from redsun_aht.services.simulated_stack import SimulatedStackController
from redsun_aht.simulation.mcu_stack import Stack
from redsun_aht.simulation.pva_coordinated_stage import PvaSyntheticSurveyStage
from redsun_aht.simulation.pva_jog import SimulatedPvaJog
from redsun_aht.simulation.pva_stack import PvaSimulatedStack


async def test_synthetic_controller_and_stack_share_coordinator():
    prefix = "SC" + uuid.uuid4().hex[:12].upper()
    service = SimulatedStackController()
    server = asyncio.create_task(serve_controller(service, EpicsPVATransport(), prefix))
    client = PvaSimulatedStack(prefix + ":", name="synthetic")
    try:
        await asyncio.wait_for(client.connect(), 10)
        stage = PvaSyntheticSurveyStage(client)
        coord = CompositeMotionCoordinator(
            {
                name: AxisBinding(stage, name, 1)
                for name in ("survey_x", "survey_y", "coarse_z")
            }
        )
        bridge = SimulatedPvaJog(coord)
        await bridge.update(ControllerState(0, 0, 0))
        await bridge.update(ControllerState(BUTTON_A | DPAD_UP, 32767, -32768))
        assert (await stage.read_position()).positions == {
            "survey_x": 1,
            "survey_y": -1,
            "coarse_z": 1,
        }
        await client.reserve_service()
        token = coord.reserve()
        rival = PvaSimulatedStack(prefix + ":", name="rival")
        await rival.connect()
        with pytest.raises(Exception, match="reservation"):
            await rival.jog_by({"survey_x": 1})
        with pytest.raises(Exception, match="reservation"):
            await rival.execute(Stack("rival", (0,), 1))
        stack = Stack("shared", (0, 0.25, 0.5), 1)
        task = asyncio.create_task(
            coord.run_reserved(token, lambda: stage.execute_stack(stack))
        )
        for _ in range(100):
            if service.stage.owner is not None:
                break
            await asyncio.sleep(0.005)
        assert service.stage.owner is not None
        with pytest.raises(RuntimeError, match=r"busy|reservation"):
            await coord.jog_by({"survey_x": 1})
        evidence = await task
        assert len(evidence.edges) == 3 and service.stage.positions["fast_z"] == 0.5
        coord.release(token)
        await client.release_service()
        await coord.jog_by({"survey_x": 0.25})
        assert (await stage.read_position()).positions["survey_x"] == 1.25
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
