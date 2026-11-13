"""Fake-only host-stepped UniMMCore stage contract tests."""

import asyncio
from dataclasses import replace

import pytest

from redsun_aht.device.pva_stepped_axis import PvaSteppedAxis
from redsun_aht.device.unified_mcu import StackProgress, XControllerState
from redsun_aht.device.unimmcore_stepped_stage import SteppedPvaStage


class FakeState:
    def __init__(self):
        self.value = XControllerState(0, 0, 0, 0, 3, 3, "fake")

    async def read_state(self):
        return self.value


class FakeClient:
    def __init__(self, state):
        self.state = state
        self.calls = []
        self.fail = False
        self.aborts = []
        self.reservation_held = True

    async def execute(self, stack, *, ready, require_reservation=False):
        self.calls.append(stack)
        if self.fail:
            raise RuntimeError("injected uncertain result")
        self.state.value = replace(
            self.state.value, target_tenths=stack.first, position_tenths=stack.first
        )
        return StackProgress(stack.run_id, (stack.first,), complete=True)

    async def abort_stack(self, run_id):
        self.aborts.append(run_id)


def stage_for(loop, cls=SteppedPvaStage):
    state = FakeState()
    client = FakeClient(state)
    stage = cls(
        client,
        state,
        loop,
        first_run_id=40,
        um_per_rev=1000,
        origin_um=0,
        lower_tenths=-180,
        upper_tenths=180,
        trigger_mask=2,
        resources_exclusive=True,
        ready=True,
    )
    return stage, client, state


async def test_each_move_is_one_bounded_submission_and_fresh_readback():
    stage, client, _ = stage_for(asyncio.get_running_loop())
    await asyncio.to_thread(stage.set_position_um, 5)
    await asyncio.to_thread(stage.set_position_um, 10)
    assert [(c.run_id, c.first, c.count, c.trigger_mask) for c in client.calls] == [
        (40, 18, 1, 2),
        (40, 36, 1, 2),
    ]
    assert await asyncio.to_thread(stage.get_position_um) == 10
    for value in (0.1, 100, float("nan")):
        with pytest.raises(ValueError):
            await asyncio.to_thread(stage.set_position_um, value)
    assert len(client.calls) == 2


async def test_failure_latches_uncertainty_without_replay():
    stage, client, _ = stage_for(asyncio.get_running_loop())
    client.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        await asyncio.to_thread(stage.set_position_um, 5)
    client.fail = False
    with pytest.raises(RuntimeError, match="uncertain"):
        await asyncio.to_thread(stage.set_position_um, 5)
    assert len(client.calls) == 1


async def test_shared_axis_works_without_unimmcore_adapter():
    stage, client, _ = stage_for(asyncio.get_running_loop(), PvaSteppedAxis)
    await asyncio.to_thread(stage.set_position_um, 5)
    assert client.calls[0].first == 18
    assert await asyncio.to_thread(stage.get_position_um) == 5


async def test_unreserved_axis_refuses_before_submission():
    stage, client, _ = stage_for(asyncio.get_running_loop(), PvaSteppedAxis)
    client.reservation_held = False
    with pytest.raises(RuntimeError, match="reservation"):
        await asyncio.to_thread(stage.set_position_um, 5)
    assert client.calls == [] and stage.run_id == 40


async def test_mismatched_state_quarantines_stage():
    stage, client, state = stage_for(asyncio.get_running_loop())
    original = client.execute

    async def bad_state(*args, **kwargs):
        report = await original(*args, **kwargs)
        state.value = replace(state.value, position_tenths=0)
        return report

    client.execute = bad_state
    with pytest.raises(RuntimeError, match="does not confirm"):
        await asyncio.to_thread(stage.set_position_um, 5)
    with pytest.raises(RuntimeError, match="uncertain"):
        await asyncio.to_thread(stage.get_position_um)


async def test_timeout_requests_one_abort_and_blocks_reuse():
    stage, client, _ = stage_for(asyncio.get_running_loop())
    stage.timeout_s = 0.05

    async def blocked(stack, *, ready, require_reservation=False):
        client.calls.append(stack)
        await asyncio.Event().wait()

    client.execute = blocked
    with pytest.raises(TimeoutError, match="uncertain"):
        await asyncio.to_thread(stage.set_position_um, 5)
    assert stage.abort_future is not None
    await asyncio.wrap_future(stage.abort_future)
    assert client.aborts == [40] and len(client.calls) == 1
    with pytest.raises(RuntimeError, match="uncertain"):
        await asyncio.to_thread(stage.set_position_um, 5)


async def test_actual_mda_runner_uses_separate_software_steps(tmp_path):
    import numpy as np
    from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore
    from pymmcore_plus.mda import MDAEngine, MDARunner
    from useq import MDAEvent

    from redsun_aht.storage import read_manifest
    from redsun_aht.storage.unimmcore_step_sink import UniMMCoreStepSink

    stage, client, _ = stage_for(asyncio.get_running_loop())

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

    def run():
        core = UniMMCore()
        camera = Camera()
        core.loadPyDevice("camera", camera)
        core.loadPyDevice("stage", stage)
        core.initializeDevice("camera")
        core.initializeDevice("stage")
        core.setCameraDevice("camera")
        core.setFocusDevice("stage")
        core.setAutoShutter(False)
        engine = MDAEngine(
            core, use_hardware_sequencing=False, restore_initial_state=False
        )
        runner = MDARunner()
        runner.set_engine(engine)
        frames = []
        sink = UniMMCoreStepSink(
            str(tmp_path / "mda.zarr"), np.zeros((4, 4), dtype=np.uint16), count=3
        )

        def record(image, event, metadata):
            sink.append(image, dict(event.index))
            frames.append((dict(event.index), int(image[0, 0])))

        runner.events.frameReady.connect(record)
        try:
            runner.run(
                [MDAEvent(index={"z": i}, z_pos=z) for i, z in enumerate((0, 5, 10))]
            )
        finally:
            try:
                sink.close()
            finally:
                core.unloadAllDevices()
        return frames

    frames = await asyncio.wait_for(asyncio.to_thread(run), 15)
    assert [c.first for c in client.calls] == [0, 18, 36]
    assert [c.run_id for c in client.calls] == [40, 40, 40]
    assert frames == [({"z": 0}, 1), ({"z": 1}, 2), ({"z": 2}, 3)]
    assert read_manifest(
        str(tmp_path / "mda.zarr"), data_key="software_snap"
    ).is_complete
