"""Check installed MDA sequence ordering without hardware."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from pymmcore_plus.mda import MDAEngine


def test_stock_engine_starts_stage_before_camera_sequence():
    calls = []
    core = MagicMock()
    core.getAutoShutter.return_value = False
    core.getFocusDevice.return_value = "fake-z"
    core.isSequenceRunning.return_value = False
    core.getNumberOfCameraChannels.return_value = 1
    core.stopStageSequence.side_effect = lambda *_: calls.append("stop-stage")
    core.loadStageSequence.side_effect = lambda *_: calls.append("load-stage")
    core.startStageSequence.side_effect = lambda *_: calls.append("start-stage")
    core.startSequenceAcquisition.side_effect = lambda *_: calls.append("start-camera")
    engine = MDAEngine(core)
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
    assert calls == ["stop-stage", "load-stage", "start-stage"]
    list(engine.exec_sequenced_event(event))
    assert calls == ["stop-stage", "load-stage", "start-stage", "start-camera"]
