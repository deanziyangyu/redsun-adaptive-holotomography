"""Fake-input and simulated-service jog tests; no physical motion."""

import asyncio
import json

import pytest
from ophyd_async.core import get_mock_put

from redsun_aht.operator_input.xinput import (
    BUTTON_A,
    DPAD_DOWN,
    DPAD_UP,
    ControllerState,
)
from redsun_aht.services.simulated_stack import SimulatedStackController
from redsun_aht.simulation.pva_jog import SimulatedPvaJog
from redsun_aht.simulation.pva_stack import PvaSimulatedStack


async def test_server_jog_is_bounded_atomic_and_reservation_aware():
    server = SimulatedStackController()
    await server.jog.put(
        json.dumps({"survey_x": 1, "survey_y": -0.5, "coarse_z": 0.25})
    )
    assert server.stage.positions["survey_x"] == 1
    assert server.stage.positions["survey_y"] == -0.5
    for invalid in (
        {"survey_x": 1.25},
        {"survey_x": True},
        {"fast_z": 1},
        {"survey_y": 0.1},
    ):
        before = dict(server.stage.positions)
        with pytest.raises(ValueError):
            await server.jog.put(json.dumps(invalid))
        assert server.stage.positions == before
    server.stage.reserve("stack")
    try:
        with pytest.raises(RuntimeError, match="reserved"):
            await server.jog.put(json.dumps({"survey_x": 0.25}))
    finally:
        server.stage.release("stack")


async def test_input_rearm_disconnect_and_no_queued_writes():
    device = PvaSimulatedStack("SIM:", name="sim")
    await device.connect(mock=True)
    bridge = SimulatedPvaJog(device)
    held = ControllerState(BUTTON_A | DPAD_UP, 32767, -32768)
    await bridge.update(held)
    get_mock_put(device.jog).assert_not_called()
    await bridge.update(ControllerState(0, 0, 0))
    await bridge.update(held)
    assert json.loads(get_mock_put(device.jog).call_args.args[0]) == {
        "survey_x": 1,
        "survey_y": -1,
        "coarse_z": 1,
    }
    await bridge.update(None)
    await bridge.update(held)
    assert get_mock_put(device.jog).call_count == 1
    await bridge.update(ControllerState(0, 0, 0))
    entered = asyncio.Event()

    async def slow(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    get_mock_put(device.jog).side_effect = slow
    task = asyncio.create_task(bridge.update(held))
    await asyncio.wait_for(entered.wait(), 1)
    with pytest.raises(RuntimeError, match="pending"):
        await bridge.update(held)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await bridge.update(held)
    assert get_mock_put(device.jog).call_count == 2


async def test_rearm_requires_neutral_axes_after_disconnect():
    device = PvaSimulatedStack("SIM:", name="sim")
    await device.connect(mock=True)
    bridge = SimulatedPvaJog(device)
    await bridge.update(None)
    for released in (
        ControllerState(0, 32767, 0),
        ControllerState(DPAD_UP, 0, 0),
        ControllerState(DPAD_UP | DPAD_DOWN, 0, 0),
    ):
        await bridge.update(released)
        await bridge.update(ControllerState(BUTTON_A, 32767, 0))
    get_mock_put(device.jog).assert_not_called()
    await bridge.update(ControllerState(0, 0, 0))
    await bridge.update(ControllerState(BUTTON_A, 32767, 0))
    assert get_mock_put(device.jog).call_count == 1


async def test_poll_loop_samples_disconnect_while_write_is_pending():
    device = PvaSimulatedStack("SIM:", name="sim")
    await device.connect(mock=True)
    bridge = SimulatedPvaJog(device)
    entered = asyncio.Event()
    samples = []

    class Poller:
        def read(self):
            samples.append(1)
            if len(samples) >= 3:
                return None
            return (
                ControllerState(0, 0, 0)
                if len(samples) == 1
                else ControllerState(BUTTON_A, 32767, 0)
            )

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    get_mock_put(device.jog).side_effect = blocked
    task = asyncio.create_task(bridge.run(Poller(), asyncio.Event(), period_s=0.01))
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.sleep(0.04)
    assert len(samples) >= 3 and get_mock_put(device.jog).call_count == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await bridge.update(ControllerState(BUTTON_A, 32767, 0))
    assert get_mock_put(device.jog).call_count == 1


async def test_disconnect_during_write_requires_rearm_after_completion():
    device = PvaSimulatedStack("SIM:", name="sim")
    await device.connect(mock=True)
    bridge = SimulatedPvaJog(device)
    entered = asyncio.Event()
    release_write = asyncio.Event()
    stop = asyncio.Event()

    class Poller:
        state = ControllerState(0, 0, 0)

        def read(self):
            return self.state

    poller = Poller()

    async def slow(*args, **kwargs):
        entered.set()
        await release_write.wait()

    get_mock_put(device.jog).side_effect = slow
    task = asyncio.create_task(bridge.run(poller, stop, period_s=0.01))
    try:
        await asyncio.sleep(0.03)
        poller.state = ControllerState(BUTTON_A, 32767, 0)
        await asyncio.wait_for(entered.wait(), 1)
        poller.state = None
        await asyncio.sleep(0.03)
        poller.state = ControllerState(BUTTON_A, 32767, 0)
        release_write.set()
        await asyncio.sleep(0.05)
        assert get_mock_put(device.jog).call_count == 1
        poller.state = ControllerState(0, 0, 0)
        await asyncio.sleep(0.03)
        poller.state = ControllerState(BUTTON_A, 32767, 0)
        await asyncio.sleep(0.03)
        assert get_mock_put(device.jog).call_count > 1
    finally:
        stop.set()
        await task
