"""Exercise the registered positioner protocol on fake ESP300 serial."""

import asyncio
import math
from threading import Event
from uuid import uuid4

import pytest
from bluesky import plan_stubs as bps
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from ophyd_async.core import wait_for_value
from redsun.engine import RunEngine
from redsun.services import serve_controller

from redsun_aht.device.esp300_stage import Esp300Stage
from redsun_aht.device.esp300_z_serial import Esp300SurveySerial
from redsun_aht.services.esp300_stage import Esp300StageIO, build_controller


class SurveyPort:
    def __init__(self):
        self.positions = [0.0, 0.0, 0.0]
        self.writes = []
        self.power = ["1", "1", "1"]

    def write(self, payload):
        command = payload.decode("ascii").rstrip("\r")
        self.writes.append(command)
        if command[1:3] == "PA":
            self.positions[int(command[0]) - 1] = float(command[3:])
        return len(payload)

    def flush(self):
        pass

    def readline(self, size=-1):
        command = self.writes[-1].removesuffix("?")
        if command == "VE":
            return b"ESP300 fixture\r\n"
        axis = int(command[0])
        position = str(self.positions[axis - 1])
        response = {
            "ID": str(axis),
            "SN": "2",
            "SU": "0.0001",
            "SL": "0",
            "SR": "12.5",
            "TP": position,
            "DP": position,
            "MD": "1",
            "MO": self.power[axis - 1],
        }[command[1:]]
        return (response + "\r\n").encode("ascii")


async def test_xyz_positioner_publishes_bounds_and_routes_moves_through_one_owner():
    port = SurveyPort()
    owner = Esp300SurveySerial(port, (("1", 0, 10), ("2", 0, 20), ("3", 0, 30)))
    io = Esp300StageIO(owner)
    prefix = "ESP" + uuid4().hex[:10].upper()
    server = asyncio.create_task(
        serve_controller(
            build_controller(io, provenance="simulated"), EpicsPVATransport(), prefix
        )
    )
    stage = Esp300Stage(prefix + ":", name="survey")
    try:
        await asyncio.wait_for(stage.connect(), 10)
        assert {axis: spec.upper for axis, spec in stage.axes.items()} == {
            "x": 10,
            "y": 20,
            "z": 30,
        }
        with pytest.raises(ValueError, match="limits"):
            await stage.move({"x": 5, "y": 99})
        assert all(command.endswith("?") for command in port.writes)
        reading = await stage.move({"x": 5, "y": 6, "z": 7})
        assert {axis.name: axis.value for axis in reading.axes} == {
            "x": 5,
            "y": 6,
            "z": 7,
        }
        for target in (8, 0):
            await asyncio.wait_for(asyncio.wrap_future(
                RunEngine({})(bps.abs_set(stage.axis["z"], target, wait=True))
            ), 10)
        assert (await stage.axis["z"].locate())["readback"] == 0
        assert [command for command in port.writes if not command.endswith("?")] == [
            "1PA0.005",
            "2PA0.006",
            "3PA0.007",
            "3PA0.008",
            "3PA0",
        ]
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
        await io.drain()


@pytest.mark.parametrize("cleanup", [False, True])
async def test_pva_stop_interrupts_a_pending_z_put_and_blocks_reuse(cleanup):
    entered, release = Event(), Event()

    class InterruptiblePort(SurveyPort):
        def readline(self, size=-1):
            if self.writes[-1] == "3MD?" and "3PA0.005" in self.writes:
                entered.set()
                assert release.wait(3)
            return super().readline(size)

    port = InterruptiblePort()
    owner = Esp300SurveySerial(port, (("1", 0, 10), ("2", 0, 20), ("3", 0, 30)))
    io = Esp300StageIO(owner)
    prefix = "ESP" + uuid4().hex[:10].upper()
    server = asyncio.create_task(
        serve_controller(
            build_controller(io, provenance="simulated"), EpicsPVATransport(), prefix
        )
    )
    stage = Esp300Stage(prefix + ":", name="survey")
    try:
        await asyncio.wait_for(stage.connect(), 10)
        move = asyncio.create_task(stage.move({"z": 5}))
        assert await asyncio.to_thread(entered.wait, 3)
        stop = asyncio.create_task(
            stage.axis["z"].stop(success=True) if cleanup else stage.stop()
        )
        assert await asyncio.to_thread(owner._uncertain.wait, 3)
        release.set()
        await asyncio.wait_for(stop, 3)
        with pytest.raises(Exception, match=r"stop requested|uncertain"):
            await asyncio.wait_for(move, 3)
        await wait_for_value(stage.axis["z"].position, math.isnan, timeout=3)
        assert (await stage.read_position()).quality == "unavailable"
        with pytest.raises(Exception, match="uncertain"):
            await asyncio.wait_for(stage.move({"x": 5}), 3)
        assert [command for command in port.writes if not command.endswith("?")] == [
            "3PA0.005",
            "3ST",
        ]
    finally:
        release.set()
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
        await io.drain()


def test_unpowered_axis_refuses_move_before_writing_position():
    port = SurveyPort()
    port.power[0] = "0"
    owner = Esp300SurveySerial(port, (("1", 0, 10), ("2", 0, 20), ("3", 0, 30)))
    with pytest.raises(RuntimeError, match="motor power is off"):
        owner.move_um(1, 5)
    assert all(command.endswith("?") for command in port.writes)
