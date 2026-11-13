"""Loopback Redsun session with fake coarse and fake MCU serial endpoints."""

import asyncio
import json

from bluesky import RunEngine
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.services import serve_controller
from test_esp300_readonly import Port as EspPort
from test_unified_mcu import FakePort

from redsun_aht.device.controller_evidence_session import ControllerEvidenceSession
from redsun_aht.device.esp300_readonly import Esp300ReadOnly
from redsun_aht.device.unified_mcu import UnifiedMcu
from redsun_aht.services.esp300_readonly import Esp300ReadOnlyController
from redsun_aht.services.unified_mcu import UnifiedMcuController


async def test_two_services_connect_into_one_headless_session():
    coarse_port = EspPort()
    fast_port = FakePort(repeat=True)
    servers = [
        asyncio.create_task(
            serve_controller(
                Esp300ReadOnlyController(Esp300ReadOnly(coarse_port)),
                EpicsPVATransport(),
                "COARSE",
            )
        ),
        asyncio.create_task(
            serve_controller(
                UnifiedMcuController(UnifiedMcu(fast_port)),
                EpicsPVATransport(),
                "FASTZ",
            )
        ),
    ]

    def run_session():
        session = ControllerEvidenceSession().build()
        try:
            plans = session.presenter.plan_map()
            assert "controller_stack_report" in plans
            docs = []
            engine = RunEngine({}, context_managers=[])
            engine.subscribe(lambda name, doc: docs.append((name, doc)))
            engine(
                plans["controller_stack_report"]["plan"](
                    7, 0, 18, 3, -180, 180, ready=True
                )
            )
            return [doc["data"] for name, doc in docs if name == "event"]
        finally:
            session.shutdown()

    try:
        events = await asyncio.wait_for(asyncio.to_thread(run_session), 30)
        assert len(events) == 2
        assert (
            json.loads(events[0]["coarse_snapshot_json"])["snapshot"]["axes"][0][
                "actual_um"
            ]
            == 2
        )
        assert json.loads(events[1]["fast_stack_json"])["targets_tenths"] == [0, 18, 36]
        assert fast_port.commands.count("START 7") == 1
    finally:
        for server in servers:
            server.cancel()
        await asyncio.gather(*servers, return_exceptions=True)
