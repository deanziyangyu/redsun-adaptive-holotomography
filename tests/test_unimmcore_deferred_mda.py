"""Full MDA runner rehearsal with Python-only synthetic camera and stage."""

from queue import Queue

import numpy as np
import pytest
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore
from pymmcore_plus.mda import MDARunner
from useq import MDAEvent

from redsun_aht.device.stack_frame_quarantine import FrameQuarantine
from redsun_aht.device.unified_mcu import AngularStack, StackProgress
from redsun_aht.simulation.mcu_stack import Axis, SimulatedMcu, Stack
from redsun_aht.simulation.unimmcore_deferred import (
    DeferredSimulatedEngine,
    DeferredSimulatedStage,
)


class TriggerCamera(CameraDevice):
    def __init__(self, drop=None):
        super().__init__()
        self.pending = Queue()
        self.exposure = 1.0
        self.drop = drop
        self.trigger_count = 0
        self.frame_count = 0
        self.accepted_triggers = []

    def trigger(self):
        trigger = self.trigger_count
        self.trigger_count += 1
        if trigger != self.drop:
            self.accepted_triggers.append(trigger)
            self.frame_count += 1
            self.pending.put(self.frame_count)

    def get_exposure(self):
        return self.exposure

    def set_exposure(self, exposure):
        self.exposure = exposure

    def shape(self):
        return (4, 4)

    def dtype(self):
        return np.uint16

    def start_sequence(self, n, get_buffer):
        for _ in range(n):
            number = self.pending.get(timeout=3)
            if number is None:
                return
            data = get_buffer(self.shape(), self.dtype())
            data[:] = number
            yield {"synthetic_frame": str(number)}


@pytest.mark.parametrize("presequenced", [True, False])
@pytest.mark.parametrize("drop", [None, 1])
def test_full_runner_with_deferred_synthetic_sequence(presequenced, drop):
    core = UniMMCore()
    camera = TriggerCamera(drop=drop)
    releases = []
    mcu = SimulatedMcu(Axis("fast_z", -10.0, 10.0, 1.0))

    def trigger_frames(sequence):
        releases.append(sequence)
        mcu.reserve("mda")
        try:
            mcu.commit("mda", Stack("synthetic-mda", sequence, 1))
            mcu.arm("mda", camera_ready=True, referenced=True)
            mcu.start(
                "mda",
                "synthetic-mda",
                on_edge=lambda edge: camera.trigger(),
            )
            camera.pending.put(None)
        finally:
            mcu.release("mda")

    stage = DeferredSimulatedStage(trigger_frames)
    core.loadPyDevice("sim-camera", camera)
    core.loadPyDevice("sim-z", stage)
    core.initializeDevice("sim-camera")
    core.initializeDevice("sim-z")
    core.setCameraDevice("sim-camera")
    core.setFocusDevice("sim-z")
    core.setAutoShutter(False)
    events = tuple(MDAEvent(index={"z": i}, z_pos=z) for i, z in enumerate((0, 1, 2)))
    sequence = SequencedEvent(events=events, z_sequence=(0, 1, 2))
    engine = DeferredSimulatedEngine(core, stage)
    engine.restore_initial_state = False
    runner = MDARunner()
    runner.set_engine(engine)
    frames = []
    quarantine = FrameQuarantine(max_frames=3, max_bytes=1024)

    def receive(image, event, metadata):
        number = int(image[0, 0])  # Synthetic camera-local counter, not a plane.
        frames.append((number, dict(event.index)))  # Diagnostic, untrusted index.
        quarantine.append(number, image)

    runner.events.frameReady.connect(receive)
    try:
        runner.run([sequence] if presequenced else events)
        assert releases == [(0.0, 1.0, 2.0)]
        assert mcu.start_count == 1
        assert [edge.target_um for edge in mcu.edges] == [0.0, 1.0, 2.0]
        expected = [(1, {"z": 0}), (2, {"z": 1})]
        if drop is None:
            expected.append((3, {"z": 2}))
        assert frames == expected
        reconciliation = quarantine.finish(
            AngularStack(7, 0, 1, 3, -10, 10),
            StackProgress(7, (0, 1, 2), complete=True),
            camera_baseline=0,
            capture_exclusive=True,
        )
        if drop is None:
            assert reconciliation.status == "complete_order_inferred"
        else:
            # The second delivered frame came from trigger 2 but was assigned
            # MDA index 1; order alone cannot locate the missed trigger.
            assert camera.accepted_triggers == [0, 2]
            assert frames[1][1] == {"z": 1}
            assert reconciliation.status == "camera_count_or_sequence_mismatch"
            assert not reconciliation.associations
    finally:
        core.unloadAllDevices()
