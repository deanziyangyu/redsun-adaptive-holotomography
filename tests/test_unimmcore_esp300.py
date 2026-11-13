"""Fake-transport tests for the optional ESP300 UniMMCore focus stage."""

import pytest

from redsun_aht.device.unimmcore_esp300 import Esp300UniStage


class FakeTransport:
    def __init__(self):
        self.position = 0.0
        self.moves = []
        self.fail = False
        self.stops = 0

    def move_and_settle_mm(self, value):
        self.moves.append(value)
        if self.fail:
            raise TimeoutError("move uncertain")
        self.position = value

    def position_mm(self):
        return self.position

    def stop(self):
        self.stops += 1


def make_stage(transport):
    return Esp300UniStage(transport, lower_um=0, upper_um=10, resources_exclusive=True)


def test_bounds_conversion_and_uncertain_motion():
    transport = FakeTransport()
    stage = make_stage(transport)
    stage.set_position_um(5)
    assert transport.moves == [0.005] and stage.get_position_um() == 5
    for invalid in (-1, 11, float("nan")):
        with pytest.raises(ValueError):
            stage.set_position_um(invalid)
    transport.fail = True
    with pytest.raises(TimeoutError):
        stage.set_position_um(10)
    with pytest.raises(RuntimeError, match="uncertain"):
        stage.set_position_um(0)
    assert transport.moves == [0.005, 0.01]


def test_stop_is_a_request_and_inhibits_reuse():
    transport = FakeTransport()
    stage = make_stage(transport)
    stage.stop()
    assert transport.stops == 1
    with pytest.raises(RuntimeError, match="uncertain"):
        stage.get_position_um()


def test_registered_unimmcore_dispatches_focus_moves():
    from pymmcore_plus.experimental.unicore import UniMMCore

    transport = FakeTransport()
    stage = make_stage(transport)
    core = UniMMCore()
    core.loadPyDevice("esp-z", stage)
    try:
        core.initializeDevice("esp-z")
        core.setFocusDevice("esp-z")
        core.setPosition(5.0)
        assert core.getPosition() == 5.0
        assert transport.moves == [0.005]
    finally:
        core.unloadAllDevices()


def test_software_mda_uses_registered_esp300_stage():
    import numpy as np
    from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore
    from pymmcore_plus.mda import MDAEngine, MDARunner
    from useq import MDAEvent

    class FakeCamera(CameraDevice):
        def get_exposure(self):
            return 1.0

        def set_exposure(self, value):
            pass

        def shape(self):
            return (4, 4)

        def dtype(self):
            return np.uint16

        def start_sequence(self, n, get_buffer):
            for _ in range(n or 0):
                frame = get_buffer(self.shape(), self.dtype())
                frame[:] = 1
                yield {}

    transport = FakeTransport()
    core = UniMMCore()
    core.loadPyDevice("camera", FakeCamera())
    core.loadPyDevice("esp-z", make_stage(transport))
    try:
        core.initializeDevice("camera")
        core.initializeDevice("esp-z")
        core.setCameraDevice("camera")
        core.setFocusDevice("esp-z")
        core.setAutoShutter(False)
        engine = MDAEngine(
            core, use_hardware_sequencing=False, restore_initial_state=False
        )
        runner = MDARunner()
        runner.set_engine(engine)
        frames = []
        runner.events.frameReady.connect(
            lambda image, event, metadata: frames.append(event.index)
        )
        runner.run(
            [MDAEvent(index={"z": i}, z_pos=z) for i, z in enumerate((0, 5, 10))]
        )
        assert transport.moves == [0, 0.005, 0.01]
        assert [index["z"] for index in frames] == [0, 1, 2]
    finally:
        core.unloadAllDevices()
