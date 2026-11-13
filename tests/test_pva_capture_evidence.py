"""Camera capture adapter tests; all signals and frames are simulated."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from redsun_aht.device.pva_capture_evidence import PvaCaptureEvidence


def camera():
    def signal(value=None):
        return SimpleNamespace(get_value=AsyncMock(return_value=value), set=AsyncMock())

    return SimpleNamespace(
        acquire=signal(False),
        capture=signal(False),
        sequence_started=signal(False),
        file_path=signal(),
        num_capture=signal(),
        record_frame_timing=signal(),
        captured=signal(0),
        state=signal("acquiring"),
        last_error=signal(""),
    )


async def test_prepare_then_collect_returns_only_unassociated_counts(monkeypatch):
    device = camera()
    wait = AsyncMock()
    monkeypatch.setattr("redsun_aht.device.pva_capture_evidence.wait_for_value", wait)
    adapter = PvaCaptureEvidence(device)
    await adapter.prepare("test.zarr", 3, external_trigger_configured=True)
    device.captured.get_value.return_value = 3
    evidence = await adapter.collect()
    assert (
        evidence.stored_frames == 3 and evidence.quality == "unassociated_service_count"
    )
    assert evidence.store_uri == "test.zarr"
    assert wait.await_args_list[0].args == (device.sequence_started, True)
    assert wait.await_args_list[1].args == (device.capture, False)
    await adapter.close()
    assert device.acquire.set.await_args_list[-1].args == (False,)
    assert device.capture.set.await_args_list[-1].args == (False,)


async def test_rejects_busy_and_unasserted_trigger_without_commands():
    device = camera()
    adapter = PvaCaptureEvidence(device)
    with pytest.raises(ValueError, match="configuration"):
        await adapter.prepare("test.zarr", 3, external_trigger_configured=False)
    device.acquire.get_value.return_value = True
    with pytest.raises(RuntimeError, match="not idle"):
        await adapter.prepare("test.zarr", 3, external_trigger_configured=True)
    device.capture.set.assert_not_awaited()
    with pytest.raises(RuntimeError, match="one-use"):
        await adapter.prepare("test.zarr", 3, external_trigger_configured=True)


async def test_failed_start_is_cleaned_up_even_if_stop_fails(monkeypatch):
    device = camera()
    monkeypatch.setattr(
        "redsun_aht.device.pva_capture_evidence.wait_for_value",
        AsyncMock(side_effect=TimeoutError()),
    )
    adapter = PvaCaptureEvidence(device)
    with pytest.raises(TimeoutError):
        await adapter.prepare("test.zarr", 3, external_trigger_configured=True)
    device.acquire.set.side_effect = RuntimeError("stop uncertain")
    with pytest.raises(RuntimeError, match="stop uncertain"):
        await adapter.close()
    assert device.capture.set.await_args_list[-1].args == (False,)


async def test_internal_trigger_requires_one_explicit_mode(monkeypatch):
    device = camera()
    monkeypatch.setattr(
        "redsun_aht.device.pva_capture_evidence.wait_for_value", AsyncMock()
    )
    with pytest.raises(ValueError, match="exactly one trigger"):
        await PvaCaptureEvidence(device).prepare(
            "test.zarr",
            4,
            external_trigger_configured=True,
            internal_trigger_configured=True,
        )
    adapter = PvaCaptureEvidence(device)
    await adapter.prepare("test.zarr", 4, internal_trigger_configured=True)
    await adapter.close()
    assert device.acquire.set.await_args_list[-1].args == (False,)
