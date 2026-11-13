"""Loopback PVA test for the read-only ESP300 service with fake serial."""

import asyncio
import json
import uuid

from fastcs.transports.epics.pva.transport import EpicsPVATransport
from p4p.client.asyncio import Context
from redsun.services import serve_controller
from test_esp300_readonly import Port

from redsun_aht.device.esp300_readonly import Esp300ReadOnly
from redsun_aht.device.pva_esp300_readonly import PvaEsp300ReadOnly
from redsun_aht.services.esp300_readonly import Esp300ReadOnlyController


async def test_loopback_snapshot_and_refresh():
    prefix = "ER" + uuid.uuid4().hex[:12].upper()
    port = Port()
    server = asyncio.create_task(
        serve_controller(
            Esp300ReadOnlyController(Esp300ReadOnly(port)), EpicsPVATransport(), prefix
        )
    )
    context = Context("pva")
    try:
        await asyncio.wait_for(context.get(prefix + ":PVI"), 10)
        first = json.loads(await asyncio.wait_for(context.get(prefix + ":Snapshot"), 5))
        await asyncio.wait_for(context.put(prefix + ":Refresh", True), 5)
        second = json.loads(
            await asyncio.wait_for(context.get(prefix + ":Snapshot"), 5)
        )
        assert second["start_ns"] >= first["end_ns"]
        assert [axis["actual_um"] for axis in second["axes"]] == [2, 2, 2]
        assert len(port.commands) == 50 and all(
            s.endswith("?\r") for s in port.commands
        )
        client = PvaEsp300ReadOnly(prefix + ":", name="esp")
        await asyncio.wait_for(client.connect(), 10)
        reading = await asyncio.wait_for(client.read_observation(), 10)
        assert [axis.actual_um for axis in reading.snapshot.axes] == [2, 2, 2]
        assert len(port.commands) == 75
    finally:
        context.close()
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
