"""In-process composite motion arbitration with synthetic stage backends."""

import asyncio
import time

import pytest

from redsun_aht.contracts import AxisReading, AxisSpec, QualityLabel, StageReading
from redsun_aht.device.motion_coordinator import AxisBinding, CompositeMotionCoordinator
from redsun_aht.operator_input.xinput import BUTTON_A, DPAD_UP, ControllerState
from redsun_aht.simulation.pva_jog import SimulatedPvaJog


class FakeStage:
    def __init__(self, *names):
        self.axes = {n: AxisSpec(n, "um", -10, 10, 0.25) for n in names}
        self.positions = {n: 0.0 for n in names}
        self.moves = []
        self.stop_count = 0
        self.gate = None
        self.fail = False

    async def read_position(self):
        return StageReading(
            tuple(AxisReading(n, p, p, False) for n, p in self.positions.items()),
            time.monotonic_ns(),
            QualityLabel.SIMULATED,
        )

    async def move(self, targets):
        self.moves.append(dict(targets))
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise OSError("unknown motion outcome")
        self.positions.update(targets)
        return await self.read_position()

    async def stop(self):
        self.stop_count += 1


def setup():
    xy, z = FakeStage("x", "y"), FakeStage("z")
    coord = CompositeMotionCoordinator(
        {
            "survey_x": AxisBinding(xy, "x", 1),
            "survey_y": AxisBinding(xy, "y", 1),
            "coarse_z": AxisBinding(z, "z", 1),
        }
    )
    return coord, xy, z


async def test_xinput_and_acquisition_share_coordinator():
    coord, xy, z = setup()
    bridge = SimulatedPvaJog(coord)
    await bridge.update(ControllerState(0, 0, 0))
    await bridge.update(ControllerState(BUTTON_A | DPAD_UP, 32767, -32768))
    assert xy.positions == {"x": 1, "y": -1} and z.positions == {"z": 1}
    token = coord.reserve()
    with pytest.raises(RuntimeError, match="reservation"):
        await coord.jog_by({"survey_x": 1})
    await coord.move({"survey_x": 2, "coarse_z": 2}, token=token)
    coord.release(token)
    assert xy.positions["x"] == 2 and z.positions["z"] == 2


async def test_all_targets_checked_before_any_stage_moves():
    coord, xy, z = setup()
    with pytest.raises(ValueError, match="outside"):
        await coord.move({"survey_x": 1, "coarse_z": 11})
    assert xy.moves == z.moves == [] and not coord.uncertain
    with pytest.raises(ValueError, match="oversized"):
        await coord.jog_by({"survey_x": 2})
    assert xy.moves == []


async def test_failed_or_cancelled_move_quarantines():
    coord, xy, z = setup()
    z.fail = True
    with pytest.raises(OSError, match="unknown"):
        await coord.move({"survey_x": 1, "coarse_z": 1})
    assert coord.uncertain and xy.positions["x"] == 1
    with pytest.raises(RuntimeError, match="uncertain"):
        coord.reserve()


async def test_stop_during_move_reaches_all_and_quarantines():
    coord, xy, z = setup()
    xy.gate = asyncio.Event()
    task = asyncio.create_task(coord.move({"survey_x": 1, "coarse_z": 1}))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="busy"):
        await coord.jog_by({"coarse_z": 1})
    await coord.stop()
    xy.gate.set()
    with pytest.raises(RuntimeError, match="uncertain"):
        await task
    assert coord.uncertain and xy.stop_count == z.stop_count == 1
    assert z.moves == []


async def test_stop_during_jog_read_prevents_move():
    coord, xy, z = setup()
    started, resume = asyncio.Event(), asyncio.Event()
    original = xy.read_position

    async def delayed_read():
        started.set()
        await resume.wait()
        return await original()

    xy.read_position = delayed_read
    task = asyncio.create_task(coord.jog_by({"survey_x": 1}))
    await started.wait()
    await coord.stop()
    resume.set()
    with pytest.raises(RuntimeError, match="uncertain"):
        await task
    assert xy.moves == z.moves == [] and coord.uncertain
