"""Headless four-pattern snapshots and durable pixel evidence."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bluesky import RunEngine
from bluesky.utils import RunEngineInterrupted
from test_dpct_plan import FakeCamera, FakeIllumination

from redsun_aht.contracts import (
    CameraReadout,
    PlanAbortReason,
    PlanStatus,
    TriggerMode,
)
from redsun_aht.contracts.assets import PlaneReceipt
from redsun_aht.plans.dpct_software import software_dpct_plan
from redsun_aht.storage import RefusedStore, assemble_receipt
from redsun_aht.storage.writer import manifest_path


class SoftwareCamera(FakeCamera):
    """Return a distinct YX frame for every software-side read."""

    def __init__(self, store: Path, name: str = "dhm") -> None:
        super().__init__(name)
        self.trigger_modes = (TriggerMode.FREE_RUN, TriggerMode.SOFTWARE)
        self.store = store
        self.reads = 0
        self.capture_calls = 0
        self.fail_at: int | None = None
        self.block_capture = False

    def dpct_store_uri(self, run_id: str) -> str:
        """Name this run's pixel store."""
        return str(self.store)

    async def capture_frame(self) -> CameraReadout:
        """Supply one new frame after a pattern command."""
        self.capture_calls += 1
        if self.block_capture and self.capture_calls > 1:
            await asyncio.Event().wait()
        self.reads += 1
        if self.reads == self.fail_at:
            raise TimeoutError("camera stopped providing frames")
        return CameraReadout(
            data=np.full((2, 3), self.reads, dtype=np.uint16),
            sequence=self.reads,
            setting_revision=17,
            exposure_s=0.01,
            captured_monotonic_ns=time.monotonic_ns(),
        )


def test_software_dpct_writes_four_pixel_frames_and_manifest(tmp_path: Path) -> None:
    """One selected pattern has one persisted frame and matching Event identity."""
    camera = SoftwareCamera(tmp_path / "dpct.zarr")
    led = FakeIllumination()
    documents: list[tuple[str, Any]] = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))  # type: ignore[no-untyped-call]
    outcomes: list[Any] = []

    def run() -> Any:
        outcomes.append(
            (
                yield from software_dpct_plan(
                    [camera], None, led, run_id="software-1", settle_s=0
                )
            )
        )

    engine(run())
    assert outcomes[0].status is PlanStatus.COMPLETE
    assert camera.reads == 4
    assert camera.trigger_calls == 0
    assert len([name for name, _ in documents if name == "event"]) == 4
    start = next(doc for name, doc in documents if name == "start")
    assert start["frame_trigger"] == "software-fresh-frame-after-pattern"
    assert start["dpct_producer_sequence_available"] is False
    assert start["dpct_setting_revision_available"] is False

    manifest = json.loads(manifest_path(str(camera.store), camera.name).read_text())
    assert manifest["frame_count"] == 4
    assert manifest["planned_frames"] == 4
    assert manifest["setting_timeline"] == [{"first_frame": 0, "revision": None}]
    assert [entry["pattern_id"] for entry in manifest["frames"]] == [1, 2, 3, 4]
    assert all(entry["setting_revision"] is None for entry in manifest["frames"])
    assert manifest["detector_id"] == camera.name
    receipt = assemble_receipt(str(camera.store))
    assert isinstance(receipt, PlaneReceipt)
    assert receipt.frame_count == 4
    assert led.log[-2:] == ["led:00", "led:unstage"]


def test_software_dpct_closes_a_short_store_on_camera_failure(tmp_path: Path) -> None:
    """A failed third frame leaves a readable two-frame partial manifest."""
    camera = SoftwareCamera(tmp_path / "partial.zarr")
    camera.fail_at = 3
    led = FakeIllumination()
    engine = RunEngine({})
    outcomes: list[Any] = []
    stops: list[Any] = []
    engine.subscribe(  # type: ignore[no-untyped-call]
        lambda name, doc: stops.append(doc) if name == "stop" else None
    )

    def run() -> Any:
        outcomes.append(
            (
                yield from software_dpct_plan(
                    [camera], None, led, run_id="software-partial", settle_s=0
                )
            )
        )

    engine(run())
    assert outcomes[0].status is PlanStatus.ABORTED
    assert outcomes[0].reason is PlanAbortReason.CAMERA_TIMEOUT
    assert len(outcomes[0].produced) == 2
    assert stops[0]["exit_status"] == "abort"
    manifest = json.loads(manifest_path(str(camera.store), camera.name).read_text())
    assert manifest["frame_count"] == 2
    assert manifest["planned_frames"] == 4
    assert isinstance(assemble_receipt(str(camera.store)), RefusedStore)
    assert led.log[-2:] == ["led:00", "led:unstage"]


def test_software_dpct_keeps_multi_detector_stores_and_event_identity_separate(
    tmp_path: Path,
) -> None:
    first = SoftwareCamera(tmp_path / "first.zarr", "first")
    second = SoftwareCamera(tmp_path / "second.zarr", "second")
    led = FakeIllumination()
    documents: list[tuple[str, Any]] = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))  # type: ignore[no-untyped-call]

    outcomes: list[Any] = []

    def run() -> Any:
        outcomes.append(
            (
                yield from software_dpct_plan(
                    [first, second], None, led, run_id="software-multi", settle_s=0
                )
            )
        )

    engine(run())

    outcome = outcomes[0]
    assert outcome.status is PlanStatus.COMPLETE
    events = [doc for name, doc in documents if name == "event"]
    assert len(events) == 8
    assert [event["data"]["dpct-detector-id"] for event in events] == [
        "first",
        "second",
        "first",
        "second",
        "first",
        "second",
        "first",
        "second",
    ]
    assert [event["data"]["dpct-sequence"] for event in events] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert all(event["data"]["dpct-producer-sequence"] == -1 for event in events)
    for camera in (first, second):
        manifest = json.loads(manifest_path(str(camera.store), camera.name).read_text())
        assert manifest["frame_count"] == 4
        assert manifest["detector_id"] == camera.name
        assert manifest["setting_timeline"] == [{"first_frame": 0, "revision": None}]
        assert all(entry["setting_revision"] is None for entry in manifest["frames"])
        assert all(
            entry["identity"]["producer_sequence"] is None
            for entry in manifest["frames"]
        )


def test_software_dpct_refuses_two_detectors_sharing_a_store(tmp_path: Path) -> None:
    shared = tmp_path / "shared.zarr"
    first = SoftwareCamera(shared, "first")
    second = SoftwareCamera(shared, "second")
    engine = RunEngine({})

    with pytest.raises(ValueError, match="shares a pixel store"):
        engine(
            software_dpct_plan([first, second], None, FakeIllumination(), run_id="same")
        )


def test_software_dpct_abort_closes_partial_store_and_runs_cleanup(
    tmp_path: Path,
) -> None:
    camera = SoftwareCamera(tmp_path / "cancelled.zarr")
    camera.block_capture = True
    led = FakeIllumination()
    engine = RunEngine({})
    first_pattern = threading.Event()
    original_pattern = led.pattern

    async def observe_pattern(pattern_id: int) -> Any:
        receipt = await original_pattern(pattern_id)
        first_pattern.set()
        return receipt

    led.pattern = observe_pattern  # type: ignore[method-assign]

    def interrupt_when_waiting() -> None:
        if first_pattern.wait(timeout=5):
            time.sleep(0.1)
            engine.abort()  # type: ignore[no-untyped-call]

    interrupter = threading.Thread(target=interrupt_when_waiting)
    interrupter.start()
    try:
        with pytest.raises(RunEngineInterrupted):
            engine(
                software_dpct_plan(
                    [camera], None, led, run_id="software-cancel", settle_s=0
                )
            )
    finally:
        interrupter.join(timeout=5)

    assert not interrupter.is_alive()
    assert led.log[-2:] == ["led:00", "led:unstage"]
    manifest = json.loads(manifest_path(str(camera.store), camera.name).read_text())
    assert manifest["frame_count"] == 1
    assert manifest["planned_frames"] == 4
