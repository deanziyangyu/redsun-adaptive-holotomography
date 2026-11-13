"""Validation of the experimental UniMMCore sequence-only adapter."""

import asyncio

import pytest
from ophyd_async.core import get_mock_put

from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.unimmcore_pva_stack import DeferredPvaStage


async def test_sequence_conversion_and_rejection_clears_previous_definition():
    stage = DeferredPvaStage(
        None,
        asyncio.get_running_loop(),
        run_id=7,
        um_per_rev=1000,
        origin_um=0,
        lower_tenths=-180,
        upper_tenths=180,
        ready=True,
    )
    stage.send_sequence((0, 5, 10))
    assert stage.definition.first == 0 and stage.definition.step == 18
    for invalid in ((), (0, 0.1), (0, 5, 15), (0, 100), (0, float("nan"))):
        with pytest.raises((ValueError, OverflowError)):
            stage.send_sequence(invalid)
        assert stage.definition is None
        with pytest.raises(RuntimeError, match="not available"):
            stage.start_sequence()
        stage.send_sequence((0, 5, 10))
    stage.start_sequence()
    with pytest.raises(RuntimeError, match="already attempted"):
        stage.send_sequence((0, 5))
    stage.stop_sequence()
    with pytest.raises(NotImplementedError):
        stage.get_position_um()
    with pytest.raises(NotImplementedError):
        stage.set_position_um(0)


async def test_readiness_must_be_explicit():
    with pytest.raises(ValueError, match="readiness"):
        DeferredPvaStage(
            None,
            asyncio.get_running_loop(),
            run_id=7,
            um_per_rev=1000,
            origin_um=0,
            lower_tenths=-180,
            upper_tenths=180,
            ready=False,
        )


async def test_unreserved_sequence_cannot_be_submitted():
    client = PvaUnifiedMcu("MCU:", name="mcu")
    await client.connect(mock=True)
    stage = DeferredPvaStage(
        client,
        asyncio.get_running_loop(),
        run_id=7,
        um_per_rev=1000,
        origin_um=0,
        lower_tenths=-180,
        upper_tenths=180,
        ready=True,
    )
    stage.send_sequence((0, 5, 10))
    stage.start_sequence()
    with pytest.raises(RuntimeError, match="reservation"):
        stage.release_after_camera_start()
    get_mock_put(client.request).assert_not_called()
    assert stage.future is None


async def test_trigger_mask_is_validated_and_preserved():
    for mask in (0, 8, True):
        with pytest.raises(ValueError, match="trigger mask"):
            DeferredPvaStage(
                None,
                asyncio.get_running_loop(),
                run_id=7,
                um_per_rev=1000,
                origin_um=0,
                lower_tenths=-180,
                upper_tenths=180,
                ready=True,
                trigger_mask=mask,
            )
    stage = DeferredPvaStage(
        None,
        asyncio.get_running_loop(),
        run_id=7,
        um_per_rev=1000,
        origin_um=0,
        lower_tenths=-180,
        upper_tenths=180,
        ready=True,
        trigger_mask=6,
    )
    stage.send_sequence((0, 5, 10))
    assert stage.definition.trigger_mask == 6


def test_engine_cancel_requests_abort_without_waiting(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from pymmcore_plus.mda import MDAEngine

    from redsun_aht.device.unimmcore_pva_stack import DeferredPvaEngine

    signals = []

    def frames(self, event):
        signals.append((yield "frame"))

    monkeypatch.setattr(MDAEngine, "exec_sequenced_event", frames)
    core = MagicMock()
    stage = SimpleNamespace(future=Future(), stop_sequence=MagicMock())
    engine = DeferredPvaEngine(core, stage, completion_timeout_s=0.01)
    gen = engine.exec_sequenced_event(SimpleNamespace(z_sequence=(0, 1)))
    assert next(gen) == "frame"
    with pytest.raises(StopIteration):
        gen.send("cancel")
    assert signals == ["cancel"]
    stage.stop_sequence.assert_called_once()


def test_engine_completion_timeout_requests_abort(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from pymmcore_plus.mda import MDAEngine

    from redsun_aht.device.unimmcore_pva_stack import DeferredPvaEngine

    def frames(self, event):
        yield "frame"

    monkeypatch.setattr(MDAEngine, "exec_sequenced_event", frames)
    core = MagicMock()
    stage = SimpleNamespace(future=Future(), stop_sequence=MagicMock())
    engine = DeferredPvaEngine(core, stage, completion_timeout_s=0.01)
    gen = engine.exec_sequenced_event(SimpleNamespace(z_sequence=(0, 1)))
    assert next(gen) == "frame"
    with pytest.raises(TimeoutError):
        next(gen)
    stage.stop_sequence.assert_called_once()
