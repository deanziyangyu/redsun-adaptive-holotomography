"""Loopback PVA rehearsal for simulated XInput survey jogging."""

import asyncio
import json
import uuid

import pytest
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.services import serve_controller

from redsun_aht.operator_input.xinput import BUTTON_A, ControllerState
from redsun_aht.services.simulated_stack import SimulatedStackController
from redsun_aht.simulation.pva_jog import SimulatedPvaJog
from redsun_aht.simulation.pva_session import PvaStackSession, SimulatedJogPresenter
from redsun_aht.simulation.pva_stack import PvaSimulatedStack


async def test_jog_over_loopback_and_stack_conflict():
    prefix = "SJ" + uuid.uuid4().hex[:12].upper()
    service = SimulatedStackController()
    server = asyncio.create_task(serve_controller(service, EpicsPVATransport(), prefix))
    device = PvaSimulatedStack(prefix + ":", name="sim")
    bridge = SimulatedPvaJog(device)
    presenter = SimulatedJogPresenter("manual", joggers={"sim": device})
    neutral = ControllerState(0, 0, 0)
    held = ControllerState(BUTTON_A, 32767, -32768)
    try:
        await asyncio.wait_for(device.connect(), 10)
        assert (await presenter.positions())["fast_z"] == 0
        await bridge.update(neutral)
        await bridge.update(held)
        assert service.stage.positions["survey_x"] == 1
        assert service.stage.positions["survey_y"] == -1
        assert (await presenter.positions())["survey_x"] == 1
        await presenter.jog_by({"survey_x": 0.25})
        assert (await presenter.positions())["survey_x"] == 1.25
        with pytest.raises(ValueError, match="invalid"):
            await presenter.jog_by({"survey_x": 0.1})
        task = asyncio.ensure_future(
            device.request.set(
                json.dumps(
                    {"run_id": "pva-jog", "targets_um": [0, 0.25, 0.5], "pattern": 1}
                )
            )
        )
        for _ in range(100):
            if service.stage.owner is not None:
                break
            await asyncio.sleep(0.005)
        assert service.stage.owner is not None
        with pytest.raises(Exception, match="reserved"):
            await bridge.update(held)
        with pytest.raises(Exception, match="reserved"):
            await presenter.jog_by({"survey_x": 0.25})
        await task
        assert (await presenter.positions())["fast_z"] == 0.5
        assert service.stage.positions["survey_x"] == 1.25
        await bridge.update(held)
        assert service.stage.positions["survey_x"] == 1.25
        await bridge.update(neutral)
        await bridge.update(held)
        assert service.stage.positions["survey_x"] == 2.25
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_headless_session_builds_jog_presenter():
    server = asyncio.create_task(
        serve_controller(SimulatedStackController(), EpicsPVATransport(), "SIMSTACK")
    )

    def build_and_check():
        session = PvaStackSession().build()
        try:
            assert session.jog_presenter._one() is session.stack
        finally:
            session.shutdown()

    try:
        await asyncio.wait_for(asyncio.to_thread(build_and_check), 20)
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_presenter_runs_one_controller_against_shared_service():
    prefix = "SJ" + uuid.uuid4().hex[:12].upper()
    service = SimulatedStackController()
    server = asyncio.create_task(serve_controller(service, EpicsPVATransport(), prefix))
    device = PvaSimulatedStack(prefix + ":", name="sim")
    presenter = SimulatedJogPresenter("manual", joggers={"sim": device})
    stop = asyncio.Event()

    class Poller:
        state = ControllerState(0, 0, 0)

        def read(self):
            return self.state

    poller = Poller()
    task = None
    try:
        await asyncio.wait_for(device.connect(), 10)
        task = asyncio.create_task(
            presenter.run_controller(poller, stop, period_s=0.01)
        )
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError, match="already running"):
            await presenter.run_controller(poller, stop, period_s=0.01)
        poller.state = ControllerState(BUTTON_A, 32767, -32768)
        for _ in range(100):
            if service.stage.positions["survey_x"]:
                break
            await asyncio.sleep(0.01)
        poller.state = None
        await asyncio.sleep(0.05)
        stop.set()
        await task
        assert service.stage.positions["survey_x"] > 0
        assert service.stage.positions["survey_y"] < 0
        assert (await presenter.positions())["survey_x"] == service.stage.positions[
            "survey_x"
        ]
        before = service.stage.positions["survey_x"]
        await presenter.jog_by({"survey_x": 0.25})
        assert (await presenter.positions())["survey_x"] == before + 0.25
    finally:
        stop.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
