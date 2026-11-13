"""UniMMCore Python-stage and custom engine sequencing rehearsal."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pymmcore_plus.experimental.unicore import UniMMCore

from redsun_aht.simulation.unimmcore_deferred import (
    DeferredSimulatedEngine,
    DeferredSimulatedStage,
)


def test_registered_stage_releases_only_after_camera_start():
    calls = []
    core = UniMMCore()
    stage = DeferredSimulatedStage(lambda seq: calls.append(("release", seq)))
    core.loadPyDevice("fake-z", stage)
    core.initializeDevice("fake-z")
    try:
        assert core.isStageSequenceable("fake-z")
        camera_core = MagicMock()
        camera_core.getAutoShutter.return_value = False
        camera_core.getFocusDevice.return_value = "fake-z"
        camera_core.isSequenceRunning.return_value = False
        camera_core.getNumberOfCameraChannels.return_value = 1
        camera_core.loadStageSequence.side_effect = core.loadStageSequence
        camera_core.startStageSequence.side_effect = core.startStageSequence
        camera_core.stopStageSequence.side_effect = core.stopStageSequence
        camera_core.startSequenceAcquisition.side_effect = lambda *_: calls.append(
            "camera-start"
        )
        engine = DeferredSimulatedEngine(camera_core, stage)
        engine._set_event_channel = lambda _event: None
        engine._exec_single_camera_sequence = lambda *_: iter(())
        event = SimpleNamespace(
            exposure_sequence=(),
            x_sequence=(),
            y_sequence=(),
            z_sequence=(0.0, 1.0),
            slm_sequence=(),
            property_sequences={},
            properties={},
            slm_image=None,
            roi=None,
            x_pos=None,
            y_pos=None,
            z_pos=None,
            exposure=None,
            metadata={},
            events=(object(), object()),
        )
        engine.setup_sequenced_event(event)
        assert stage.armed and not calls and stage.position == 0
        list(engine.exec_sequenced_event(event))
        assert calls == ["camera-start", ("release", (0.0, 1.0))]
        assert stage.position == 1.0
        with pytest.raises(RuntimeError, match="not armed"):
            stage.release_after_camera_start()
    finally:
        core.unloadAllDevices()


def test_camera_start_failure_never_releases_stage():
    calls = []
    stage = DeferredSimulatedStage(lambda seq: calls.append(seq))
    stage.send_sequence((0.0, 1.0))
    stage.start_sequence()
    core = MagicMock()
    core.getAutoShutter.return_value = False
    core.getNumberOfCameraChannels.return_value = 1
    core.startSequenceAcquisition.side_effect = RuntimeError("camera failed")
    engine = DeferredSimulatedEngine(core, stage)
    event = SimpleNamespace(
        events=(1, 2), metadata={}, slm_image=None, z_sequence=(0, 1)
    )
    with pytest.raises(RuntimeError, match="camera failed"):
        list(engine.exec_sequenced_event(event))
    assert calls == [] and stage.armed
    stage.stop_sequence()
    assert not stage.armed
