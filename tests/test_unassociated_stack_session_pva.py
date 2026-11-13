"""Headless Redsun session with two loopback PVA fake services."""

import asyncio
import json
import logging

from bluesky import RunEngine
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from redsun.services import serve_controller
from test_unified_mcu import FakePort

from redsun_aht.device.unassociated_stack_session import UnassociatedStackSession
from redsun_aht.device.unified_mcu import UnifiedMcu
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.services.unified_mcu import UnifiedMcuController


async def test_session_runs_one_synthetic_stack_over_pva(tmp_path, caplog):
    # p4p's debug subscription message has incompatible formatting arguments.
    caplog.set_level(logging.INFO, logger="p4p.client.raw")
    camera_core = FakeCameraCore()

    class EventPort(FakePort):
        def emit(self, line):
            super().emit(line)
            if line.startswith("ZSTACK EDGE "):
                camera_core.produce(1)

    port = EventPort()
    servers = [
        asyncio.create_task(
            serve_controller(
                MMCameraController(
                    core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
                ),
                EpicsPVATransport(),
                "CAMERA",
            )
        ),
        asyncio.create_task(
            serve_controller(
                UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), "FASTZ"
            )
        ),
    ]

    def run_session():
        session = UnassociatedStackSession().build()
        try:
            plan = session.presenter.plan_map()["unassociated_mcu_stack"]["plan"]
            documents = []
            engine = RunEngine({}, context_managers=[])
            engine.subscribe(lambda name, doc: documents.append((name, doc)))
            engine(
                plan(
                    7,
                    0,
                    18,
                    3,
                    -180,
                    180,
                    str(tmp_path / "session.zarr"),
                    resources_exclusive=True,
                    external_trigger_configured=True,
                    illumination_ready=True,
                )
            )
            return [doc["data"] for name, doc in documents if name == "event"]
        finally:
            session.shutdown()

    try:
        events = await asyncio.wait_for(asyncio.to_thread(run_session), 30)
        assert len(events) == 1
        assert json.loads(events[0]["controller_report_json"])["targets_tenths"] == [
            0,
            18,
            36,
        ]
        assert (
            json.loads(events[0]["camera_capture_json"])["quality"]
            == "unassociated_service_count"
        )
        assert port.commands.count("START 7") == 1
    finally:
        for server in servers:
            server.cancel()
        await asyncio.gather(*servers, return_exceptions=True)
