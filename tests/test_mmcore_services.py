"""Tests for the Micro-Manager camera and stage services, over in-process fakes.

Nothing here loads Micro-Manager, a device adapter or a network transport: the
fakes answer the same ``CMMCorePlus`` calls the services make, so what is
qualified is the service logic, the published attributes and the records, not
a camera or a stage.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import zarr
from fastcs.attributes import AttributeIORef, AttrRW
from fastcs.datatypes import String
from mmcore_fakes import (
    DATA_KEY,
    FRAME_SHAPE,
    LABEL,
    FakeCameraCore,
    FakeStageCore,
    MemoryFrameStore,
    core_of,
)

from redsun_aht.contracts.assets import PlaneReceipt
from redsun_aht.contracts.frames import RunIdentity
from redsun_aht.services import mmcore_camera
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.services.mmcore_stage import (
    DEFAULT_PRECISION_UM,
    MOTION_PROPERTIES,
    MMAxisController,
    MMStageController,
    MotionProfileIO,
    StageIO,
    parse_axis_limits,
)
from redsun_aht.storage import assemble_receipt, read_manifest
from redsun_aht.storage.writer import (
    RunFrameStore,
    StoreManifest,
    frame_store_factory,
    manifest_path,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path

    from numpy.typing import NDArray

#: Seconds a test waits for a state the service reaches in its own thread.
TIMEOUT = 5.0

#: The bounds the stage fixtures enforce, in micrometres.
LIMITS = {"x": (-10.0, 10.0), "y": (-10.0, 10.0)}


async def until(
    predicate: Callable[[], bool], camera: MMCameraController, timeout: float = TIMEOUT
) -> None:
    """Publish ticks until *predicate* holds, or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await camera.publish_frame()
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out after {timeout} s")


@pytest.fixture
def camera_core() -> FakeCameraCore:
    """Return a fake camera the test can produce frames on."""
    return FakeCameraCore()


@pytest.fixture
async def camera(
    camera_core: FakeCameraCore, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[MMCameraController, None]:
    """Return a camera controller on the fake, brought up as FastCS does."""
    monkeypatch.setattr(mmcore_camera, "FrameStore", MemoryFrameStore)
    controller = MMCameraController(core_of(camera_core), LABEL, DATA_KEY)
    await controller.initialise()
    controller.post_initialise()
    yield controller
    await controller.disconnect()


@pytest.fixture
def stage_core() -> FakeStageCore:
    """Return a fake stage the test can inspect after a move or a stop."""
    return FakeStageCore()


@pytest.fixture
def stage(stage_core: FakeStageCore) -> MMStageController:
    """Return a stage controller on the fake, wired as FastCS wires it."""
    controller = MMStageController(
        core_of(stage_core),
        "XY",
        ("x", "y"),
        io=StageIO(core_of(stage_core), "XY", limits=LIMITS),
    )
    controller.post_initialise()
    return controller


def axis_controller(controller: MMStageController, axis: str) -> MMAxisController:
    """Return the sub-controller serving *axis*, as ``stage/axis/<axis>``."""
    stage = controller.sub_controllers["stage"]
    axes = stage.sub_controllers["axis"]
    return cast("MMAxisController", axes.sub_controllers[axis])


async def test_a_camera_setting_is_applied_and_read_back(
    camera: MMCameraController, camera_core: FakeCameraCore
) -> None:
    """Write exposure, ROI and pixel dtype, and read what the camera took."""
    await camera.exposure.put(25.0)
    assert camera_core.exposure == 25.0
    assert camera.exposure.get() == 25.0

    await camera.roi.put("1,1,2,2")
    assert camera_core.roi == (1, 1, 2, 2)
    assert camera.roi.get() == "1,1,2,2"

    await camera.pixel_dtype.put("uint16")
    assert camera_core.properties["PixelType"] == "16bit"
    assert camera.pixel_dtype.get().name == "uint16"
    # the buffer follows the pixel type, so a viewer is never handed the
    # previous dtype cast into the new one
    assert camera.buffer.datatype.array_dtype == np.uint16


async def test_a_rejected_edit_leaves_the_previous_value(
    camera: MMCameraController, camera_core: FakeCameraCore
) -> None:
    """Refuse a malformed ROI and a dtype the camera does not offer."""
    await camera.roi.put("1,1,2,2")

    # the region is validated by the settled contract, not by a private parser
    with pytest.raises(ValueError, match="roi_invalid"):
        await camera.roi.put("1,2,3")

    assert camera_core.roi == (1, 1, 2, 2)
    assert camera.roi.get() == "1,1,2,2"

    await camera.pixel_dtype.put("uint8")

    with pytest.raises(ValueError, match="Failed to cast uint32"):
        await camera.pixel_dtype.put("uint32")

    assert camera_core.properties["PixelType"] == "8bit"
    assert camera.pixel_dtype.get().name == "uint8"
    assert camera.buffer.datatype.array_dtype == np.uint8


async def test_a_roi_wider_than_the_sensor_is_refused_after_uncropping(
    camera: MMCameraController, camera_core: FakeCameraCore
) -> None:
    """Uncrop before a growing ROI, refuse it, and stay usable afterwards."""
    await camera.roi.put("1,1,2,2")

    with pytest.raises(RuntimeError, match="must be equal or smaller"):
        await camera.roi.put("0,0,100,100")

    # the adapter uncrops first, so the refused write leaves the camera on the
    # whole sensor rather than half-way between the two readouts
    assert camera_core.roi == (0, 0, 6, 4)

    await camera.roi.put("0,0,2,2")

    assert camera.roi.get() == "0,0,2,2"


async def test_the_setting_change_record_names_the_old_and_the_new_value(
    camera: MMCameraController,
) -> None:
    """Record what was asked for, what stood before it and what came back."""
    await camera.exposure.put(20.0)

    record = json.loads(camera.last_setting_change.get())

    assert record["schema"] == "redsun-camera-setting-change-v1"
    assert record["setting"] == "exposure"
    assert record["previous"] == "100.0"
    assert record["requested"] == "20.0"
    assert record["readback"] == "20.0"
    assert record["outcome"] == "ok"
    assert record["error"] is None
    assert record["setting_change_id"]
    assert record["frame_latency_status"] == "pending"


async def test_a_refused_edit_is_recorded_as_an_error(
    camera: MMCameraController,
) -> None:
    """Record the previous value of a write the camera rejected."""
    await camera.roi.put("1,1,2,2")

    with pytest.raises(RuntimeError, match="must be equal or smaller"):
        await camera.roi.put("0,0,100,100")

    record = json.loads(camera.last_setting_change.get())

    assert record["setting"] == "roi"
    assert record["previous"] == "1,1,2,2"
    assert record["requested"] == "0,0,100,100"
    assert record["outcome"] == "error"
    assert record["frame_latency_status"] == "not_applicable"
    assert "must be equal or smaller" in record["error"]


async def test_the_next_frame_completes_the_setting_latency_record(
    camera: MMCameraController,
) -> None:
    """Join a successful readback to the next frame the service grabs."""
    await camera.exposure.put(20.0)
    pending = json.loads(camera.last_setting_change.get())
    expected_frame = camera._grabbed + 1

    camera.grab_once()
    await camera.publish_frame()

    completed = json.loads(camera.last_setting_change.get())
    batch = json.loads(camera.setting_change_latency.get())
    measurement = batch["measurements"][0]

    assert completed["setting_change_id"] == pending["setting_change_id"]
    assert completed["frame_latency_status"] == "measured"
    assert completed["readback_to_next_frame_arrival_ms"] >= 0
    assert measurement["setting_change_id"] == pending["setting_change_id"]
    assert measurement["next_frame_id"] == expected_frame
    assert measurement["next_frame_id_semantics"] == "camera_service_grab_ordinal"


async def test_two_cameras_keep_independent_settings() -> None:
    """Serve two cameras whose exposure and pixel type never follow each other."""
    first_core = FakeCameraCore()
    second_core = FakeCameraCore()
    first = MMCameraController(core_of(first_core), LABEL, DATA_KEY)
    second = MMCameraController(core_of(second_core), LABEL, DATA_KEY)
    await first.initialise()
    first.post_initialise()
    await second.initialise()
    second.post_initialise()
    try:
        await first.exposure.put(25.0)
        await second.exposure.put(50.0)
        await first.pixel_dtype.put("uint16")

        assert (first.exposure.get(), second.exposure.get()) == (25.0, 50.0)
        assert (first_core.exposure, second_core.exposure) == (25.0, 50.0)
        assert first_core.properties["PixelType"] == "16bit"
        assert second_core.properties["PixelType"] == "8bit"
        assert second.pixel_dtype.get().name == "uint8"
        assert first.buffer.datatype.array_dtype == np.uint16
        assert second.buffer.datatype.array_dtype == np.uint8
    finally:
        await first.disconnect()
        await second.disconnect()


async def test_a_camera_property_is_published_and_its_write_recorded(
    camera: MMCameraController, camera_core: FakeCameraCore
) -> None:
    """Publish the writable properties, and record a change to one of them."""
    properties = camera.sub_controllers["properties"]
    binning = getattr(properties, "Binning")  # noqa: B009 - named for the PV

    await binning.put("2")

    assert camera_core.properties["Binning"] == "2"
    assert binning.get() == "2"
    record = json.loads(camera.last_setting_change.get())
    assert record["setting"] == "Binning"
    assert record["previous"] == "1x1"
    assert record["readback"] == "2"
    assert record["outcome"] == "ok"


async def test_the_frame_count_follows_the_frames_and_not_the_clock(
    camera: MMCameraController, camera_core: FakeCameraCore
) -> None:
    """Advance ``FrameCount`` only when the camera grabbed another frame."""
    await camera.publish_frame()
    published = camera.frame_count.get()
    assert published == 1

    await camera.publish_frame()
    assert camera.frame_count.get() == published

    camera_core.produce(1)
    camera._start_sequence()
    assert camera.grab_once() is not None
    await camera.publish_frame()

    assert camera.frame_count.get() == published + 1
    assert np.array_equal(camera.buffer.get(), np.full(FRAME_SHAPE, 1, dtype=np.uint8))


async def test_a_capture_writes_the_frames_it_is_asked_for(
    camera: MMCameraController,
    camera_core: FakeCameraCore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open a store, write the frames the capture asked for, and close it."""
    stores: list[MemoryFrameStore] = []

    def open_store(uri: str, data_key: str, frame: NDArray[Any]) -> MemoryFrameStore:
        store = MemoryFrameStore(uri, data_key, frame)
        stores.append(store)
        return store

    monkeypatch.setattr(mmcore_camera, "FrameStore", open_store)
    await camera.file_path.put(str(tmp_path / "capture.zarr"))
    await camera.num_capture.put(2)
    camera._start_sequence()
    await camera.capture.put(True)
    # The active capture keeps its original count even if the control changes.
    await camera.num_capture.put(1)
    for _ in range(2):
        camera_core.produce(1)
        camera.grab_once()
    await camera.capture.wait_for_value(False, timeout=TIMEOUT)

    assert len(stores) == 1
    assert stores[0].frames_written == 2
    assert [int(frame[0, 0]) for frame in stores[0].frames] == [1, 2]
    assert stores[0].closed
    assert camera.captured.get() == 2
    assert camera.capture.get() is False


async def test_deferred_camera_prepares_capture_without_exposure(
    camera_core: FakeCameraCore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An externally triggered capture can be prepared before its first frame."""
    stores: list[MemoryFrameStore] = []

    def open_store(uri: str, data_key: str, frame: NDArray[Any]) -> MemoryFrameStore:
        store = MemoryFrameStore(uri, data_key, frame)
        stores.append(store)
        return store

    monkeypatch.setattr(mmcore_camera, "FrameStore", open_store)
    camera = MMCameraController(
        core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    try:
        await camera.initialise()
        camera.post_initialise()
        assert camera_core.snapped == 0 and camera._latest is None
        await camera.file_path.put(str(tmp_path / "capture.zarr"))
        await camera.num_capture.put(2)
        await camera.capture.put(True)
        assert stores[0].frames_written == 0
        camera._start_sequence()
        for _ in range(2):
            camera_core.produce(1)
            camera.grab_once()
        await camera.capture.wait_for_value(False, timeout=TIMEOUT)
        assert [int(frame[0, 0]) for frame in stores[0].frames] == [1, 2]
        assert camera_core.snapped == 0 and camera.captured.get() == 2
    finally:
        await camera.disconnect()


async def test_buffered_capture_reports_stored_frames_only_after_close(
    camera_core: FakeCameraCore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores: list[MemoryFrameStore] = []

    def open_store(uri: str, data_key: str, frame: NDArray[Any]) -> MemoryFrameStore:
        store = MemoryFrameStore(uri, data_key, frame)
        stores.append(store)
        return store

    monkeypatch.setattr(mmcore_camera, "FrameStore", open_store)
    camera = MMCameraController(
        core_of(camera_core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
    )
    try:
        await camera.initialise()
        camera.post_initialise()
        await camera.file_path.put(str(tmp_path / "buffered.zarr"))
        await camera.num_capture.put(2)
        await camera.capture.put(True)
        camera._start_sequence()
        camera_core.produce(1)
        camera.grab_once()
        await camera.publish_frame()
        assert camera.buffered_frames.get() == 1
        assert camera.captured.get() == stores[0].frames_written == 0
        camera_core.produce(1)
        camera.grab_once()
        await camera.capture.wait_for_value(False, timeout=TIMEOUT)
        assert camera.captured.get() == 2
        assert stores[0].closed
        assert [int(frame[0, 0]) for frame in stores[0].frames] == [1, 2]
    finally:
        await camera.disconnect()


async def test_buffered_capture_rejects_unbounded_and_oversized(
    camera_core: FakeCameraCore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    camera = MMCameraController(
        core_of(camera_core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
        buffer_limit_mib=64,
    )
    try:
        await camera.initialise()
        camera.post_initialise()
        await camera.file_path.put(str(tmp_path / "buffered.zarr"))
        for count in (0, 2049):
            await camera.num_capture.put(count)
            with pytest.raises(ValueError, match=r"1\.\.2048"):
                await camera._on_capture(True)
        await camera.num_capture.put(9)
        monkeypatch.setattr(
            camera, "_layout_template", lambda: np.empty((2048, 2048), dtype=np.uint16)
        )
        with pytest.raises(ValueError, match="64 MiB"):
            await camera._on_capture(True)
    finally:
        await camera.disconnect()


async def test_buffered_capture_flush_failure_is_reported(
    camera_core: FakeCameraCore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingStore(MemoryFrameStore):
        def append(self, frame: NDArray[Any]) -> int:
            if self.frames_written == 1:
                raise RuntimeError("injected write failure")
            return super().append(frame)

    stores: list[FailingStore] = []

    def open_store(uri: str, data_key: str, frame: NDArray[Any]) -> FailingStore:
        store = FailingStore(uri, data_key, frame)
        stores.append(store)
        return store

    monkeypatch.setattr(mmcore_camera, "FrameStore", open_store)
    camera = MMCameraController(
        core_of(camera_core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
    )
    try:
        await camera.initialise()
        camera.post_initialise()
        await camera.file_path.put(str(tmp_path / "failure.zarr"))
        await camera.num_capture.put(2)
        await camera.capture.put(True)
        camera._start_sequence()
        for _ in range(2):
            camera_core.produce(1)
            camera.grab_once()
        await camera.capture.wait_for_value(False, timeout=TIMEOUT)
        assert "injected write failure" in camera.last_error.get()
        assert camera.captured.get() == 0
        assert stores[0].frames_written == 1 and stores[0].closed
    finally:
        await camera.disconnect()


async def test_deferred_software_snap_is_one_frame_and_refuses_busy(
    camera_core: FakeCameraCore,
) -> None:
    camera = MMCameraController(
        core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    try:
        await camera.initialise()
        camera.post_initialise()
        assert camera_core.snapped == 0
        await camera.snap_request.put(True)
        assert camera_core.snapped == 1 and camera.frame_count.get() == 1
        assert np.all(camera.buffer.get() == 1)
        await camera._on_acquire(True)
        with pytest.raises(RuntimeError, match="idle"):
            await camera.snap_request.put(True)
        assert camera_core.snapped == 1
    finally:
        await camera.disconnect()


async def test_deferred_snap_blocks_overlapping_capture_and_acquisition(
    camera_core: FakeCameraCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    camera = MMCameraController(
        core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    entered, release = threading.Event(), threading.Event()
    snap = camera_core.snap

    def blocked_snap():
        entered.set()
        assert release.wait(5)
        return snap()

    monkeypatch.setattr(camera_core, "snap", blocked_snap)
    try:
        await camera.initialise()
        camera.post_initialise()
        task = asyncio.create_task(camera.snap_request.put(True))
        assert await asyncio.to_thread(entered.wait, 2)
        with pytest.raises(RuntimeError, match="in progress"):
            await camera._on_acquire(True)
        with pytest.raises(RuntimeError, match="in progress"):
            await camera._on_capture(True)
        with pytest.raises(RuntimeError, match="busy"):
            await camera.snap_request.put(True)
        release.set()
        await task
        assert camera_core.snapped == 1
    finally:
        release.set()
        await camera.disconnect()


async def test_cancelled_snap_waits_for_inflight_camera_call(
    camera_core: FakeCameraCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    camera = MMCameraController(
        core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    entered, release = threading.Event(), threading.Event()
    snap = camera_core.snap

    def blocked_snap():
        entered.set()
        assert release.wait(5)
        return snap()

    monkeypatch.setattr(camera_core, "snap", blocked_snap)
    try:
        await camera.initialise()
        camera.post_initialise()
        task = asyncio.create_task(camera.snap_request.put(True))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        with pytest.raises(RuntimeError, match="in progress"):
            await camera._on_acquire(True)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert camera_core.snapped == 1
    finally:
        release.set()
        await camera.disconnect()


async def test_deferred_camera_rejects_unknown_pixel_layout(
    camera_core: FakeCameraCore,
) -> None:
    camera_core.properties["PixelType"] = "RGB32"
    camera = MMCameraController(
        core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    with pytest.raises(ValueError, match="supported monochrome layout"):
        await camera.initialise()
    assert camera_core.snapped == 0


async def test_deferred_camera_accepts_explicit_mono8_pixel_format(
    camera_core: FakeCameraCore,
) -> None:
    del camera_core.properties["PixelType"]
    camera_core.properties["Pixel Format"] = "Mono8"
    camera = MMCameraController(
        core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    await camera.initialise()
    try:
        assert camera_core.snapped == 0
        assert camera.buffer.datatype.array_dtype == np.dtype("uint8")
    finally:
        await camera.disconnect()


async def test_deferred_camera_refuses_sequence_fallback(
    camera_core: FakeCameraCore,
) -> None:
    camera_core.sequences = False
    camera = MMCameraController(
        core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    await camera.initialise()
    try:
        with pytest.raises(RuntimeError, match="required sequence"):
            camera._start_sequence()
        assert camera_core.snapped == 0
    finally:
        await camera.disconnect()


async def test_sequence_status_tracks_start_refusal_and_stop(
    camera: MMCameraController, camera_core: FakeCameraCore
) -> None:
    await camera._publish_state()
    assert not camera.sequence_started.get()
    camera_core.sequences = False
    camera._start_sequence()
    await camera._publish_state()
    assert not camera.sequence_started.get()
    camera_core.sequences = True
    camera._start_sequence()
    await camera._publish_state()
    assert camera.sequence_started.get()
    camera._stop_sequence()
    await camera._publish_state()
    assert not camera.sequence_started.get()


async def test_deferred_settings_do_not_snap_and_refuse_active_capture(
    camera_core: FakeCameraCore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mmcore_camera, "FrameStore", MemoryFrameStore)
    camera = MMCameraController(
        core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    await camera.initialise()
    camera.post_initialise()
    try:
        binning = getattr(camera.sub_controllers["properties"], "Binning")  # noqa: B009
        await binning.put("2")
        await camera.pixel_dtype.put("uint16")
        assert camera_core.snapped == 0
        await camera.file_path.put(str(tmp_path / "capture.zarr"))
        await camera.capture.put(True)
        assert camera._store.frame_shape == (2, 3)
        with pytest.raises(RuntimeError, match="idle capture"):
            await binning.put("1")
        with pytest.raises(RuntimeError, match="idle capture"):
            await camera.exposure.put(20)
        assert camera_core.snapped == 0
        await camera.capture.put(False)
    finally:
        await camera.disconnect()


async def test_a_faulted_camera_reports_the_cause_and_restarts(
    camera: MMCameraController, camera_core: FakeCameraCore
) -> None:
    """Report a fault, then forget it once acquisition is asked for again."""
    await camera.acquire.put(True)
    camera_core.fault = RuntimeError("the camera was unplugged")

    await until(lambda: camera.state.get() == "faulted", camera)

    assert "the camera was unplugged" in camera.last_error.get()

    camera_core.fault = None
    await camera.reconnect()
    await camera.publish_frame()

    assert camera.state.get() == "acquiring"


async def test_disconnecting_stops_the_thread_and_closes_the_store(
    camera: MMCameraController, camera_core: FakeCameraCore, tmp_path: Path
) -> None:
    """Release the grabbing thread and the open store after a fault."""
    await camera.file_path.put(str(tmp_path / "faulted.zarr"))
    await camera.acquire.put(True)
    camera._latest = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    await camera.capture.put(True)
    camera_core.fault = RuntimeError("the camera stopped answering")
    await until(lambda: camera.state.get() == "faulted", camera)

    await camera.disconnect()

    assert camera._store is None
    assert camera._grab_task is None
    assert not camera._grabbing.is_set()
    assert camera.wait_until_idle(TIMEOUT)


async def test_an_out_of_range_target_is_refused_and_the_position_kept(
    stage: MMStageController, stage_core: FakeStageCore
) -> None:
    """Refuse a move outside the published limits, and keep where the stage is."""
    axis = axis_controller(stage, "x")

    with pytest.raises(ValueError, match="outside the axis limits"):
        await axis.position.put(100.0)

    assert stage_core.position == (100.0, 200.0)
    assert np.array_equal(axis.limits.get(), np.array([LIMITS["x"][0], LIMITS["x"][1]]))


async def test_an_applied_move_reads_back_where_the_stage_went(
    stage: MMStageController, stage_core: FakeStageCore
) -> None:
    """Move inside the limits, and read back the new position."""
    axis = axis_controller(stage, "x")

    await axis.position.put(5.0)

    assert stage_core.position == (5.0, 200.0)
    assert axis.position.get() == 5.0


async def test_a_stop_request_reaches_the_device(
    stage: MMStageController, stage_core: FakeStageCore
) -> None:
    """Stop the stage through the service, stage wide and per axis."""
    group = stage.sub_controllers["stage"]
    await group.stop_request.put(True)

    assert stage_core.stop_calls == ["XY"]
    assert group.stop_request.get() is False

    axis = axis_controller(stage, "y")
    await axis.stop_request.put(True)

    assert stage_core.stop_calls == ["XY", "XY"]
    assert axis.stop_request.get() is False


async def test_disconnecting_unloads_the_device_even_when_stop_fails(
    stage_core: FakeStageCore,
) -> None:
    """Release the MMCore devices even when the stage refuses to stop."""
    stage_core.fail_on_stop = True
    controller = MMStageController(
        core_of(stage_core),
        "XY",
        ("x", "y"),
        io=StageIO(core_of(stage_core), "XY", limits=LIMITS),
    )
    controller.post_initialise()

    with pytest.raises(RuntimeError, match="refused to stop"):
        await controller.disconnect()

    assert stage_core.stop_calls == ["XY"]
    assert stage_core.unloaded


async def test_a_motion_profile_is_read_and_refused_without_a_ceiling(
    stage_core: FakeStageCore,
) -> None:
    """Read the adapter's own motion settings, and refuse a write with no ceiling."""
    io = StageIO(core_of(stage_core), "XY", limits=LIMITS)
    profile = MotionProfileIO(core_of(stage_core), "XY", io, MOTION_PROPERTIES, None)
    profile_attr: AttrRW[str] = AttrRW(String(), io_ref=AttributeIORef())

    await profile.update(profile_attr)
    payload = json.loads(profile_attr.get())
    assert payload["schema"] == "redsun.mmcore-stage-motion-profile/v1"
    assert payload["properties"] == {
        "Speed": "400",
        "StartSpeed": "100",
        "Acceleration": "4",
    }
    assert payload["writes_enabled"] is False
    assert payload["unsupported"] == []

    with pytest.raises(RuntimeError, match="writes are disabled"):
        await profile.send(
            profile_attr, json.dumps({"property": "Speed", "value": 100})
        )

    assert stage_core.properties["Speed"] == "400"


async def test_a_motion_profile_write_is_applied_and_ceiling_checked(
    stage_core: FakeStageCore,
) -> None:
    """Write one adapter setting, and refuse one above the configured ceiling."""
    io = StageIO(core_of(stage_core), "XY", limits=LIMITS)
    profile = MotionProfileIO(core_of(stage_core), "XY", io, MOTION_PROPERTIES, 500)
    profile_attr: AttrRW[str] = AttrRW(String(), io_ref=AttributeIORef())

    await profile.send(profile_attr, json.dumps({"property": "Speed", "value": 100}))
    assert stage_core.properties["Speed"] == "100"
    assert json.loads(profile_attr.get())["properties"]["Speed"] == "100"

    with pytest.raises(ValueError, match="exceeds the configured"):
        await profile.send(
            profile_attr, json.dumps({"property": "Speed", "value": 900})
        )

    assert stage_core.properties["Speed"] == "100"

    with pytest.raises(ValueError, match="Acceleration must be"):
        await profile.send(
            profile_attr, json.dumps({"property": "Acceleration", "value": 900})
        )


def test_axis_limits_are_read_from_the_command_line() -> None:
    """Read per-axis bounds, and refuse an entry that names no moved axis."""
    assert parse_axis_limits("x=-1000:1000,y=-500:500", ("x", "y")) == {
        "x": (-1000.0, 1000.0),
        "y": (-500.0, 500.0),
    }
    assert parse_axis_limits("", ("x", "y")) == {}

    with pytest.raises(ValueError, match="does not move"):
        parse_axis_limits("z=0:1", ("x", "y"))
    with pytest.raises(ValueError, match="must be finite and increasing"):
        parse_axis_limits("x=10:10", ("x",))
    with pytest.raises(ValueError, match="expected axis=low:high"):
        parse_axis_limits("x=10", ("x",))


async def test_the_default_store_writes_a_readable_array(
    camera: MMCameraController,
    camera_core: FakeCameraCore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record a bounded capture with the default store, and open what it wrote."""
    monkeypatch.setattr(mmcore_camera, "FrameStore", frame_store_factory)
    store_path = tmp_path / "default.zarr"
    await camera.file_path.put(str(store_path))
    data_key = camera.data_key.get()
    await camera.exposure.put(20.0)
    await camera.num_capture.put(2)
    camera._start_sequence()
    await camera.capture.put(True)
    await camera.file_path.put(str(tmp_path / "changed.zarr"))
    await camera.data_key.put("changed-key")
    camera_core.produce(1)
    camera.grab_once()
    await camera.exposure.put(30.0)
    camera_core.produce(1)
    camera.grab_once()
    await camera.capture.wait_for_value(False, timeout=TIMEOUT)

    array = zarr.open_array(str(store_path / data_key), mode="r")
    frames = np.asarray(array[:])

    assert array.shape == (2, *FRAME_SHAPE)
    assert array.dtype == np.uint8
    assert [int(frame[0, 0]) for frame in frames] == [1, 2]
    assert manifest_path(str(store_path), data_key).exists()
    manifest = read_manifest(str(store_path), data_key=data_key)
    assert isinstance(manifest, StoreManifest)
    assert manifest.planned_frames == 2 and manifest.is_complete
    timeline = [(mark.first_frame, mark.revision) for mark in manifest.setting_timeline]
    assert timeline == [
        (0, 1),
        (1, 2),
    ]
    assert not (tmp_path / "changed.zarr").exists()
    receipt = assemble_receipt(
        str(store_path),
        data_key=data_key,
        run=RunIdentity(
            run_id="camera-capture",
            plan_name="live",
            purpose="service capture",
            scan_id=1,
        ),
        detector_id="camera-1",
        plane_ordinal=0,
        expected_frames=2,
    )
    assert isinstance(receipt, PlaneReceipt)
    assert receipt.frame_count == 2


async def test_a_capture_that_records_no_frame_leaves_no_store(
    camera: MMCameraController,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove the placeholder a capture opens before its first frame arrives."""
    monkeypatch.setattr(mmcore_camera, "FrameStore", frame_store_factory)
    store_path = tmp_path / "empty.zarr"
    await camera.file_path.put(str(store_path))
    await camera.capture.put(True)

    await camera.capture.put(False)

    assert not store_path.exists()


async def test_a_timed_capture_with_no_frame_leaves_no_timing_sidecar(
    camera: MMCameraController,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled timed capture with no frames leaves no sidecar or store."""
    monkeypatch.setattr(mmcore_camera, "FrameStore", frame_store_factory)
    store_path = tmp_path / "empty-timed.zarr"
    await camera.file_path.put(str(store_path))
    await camera.num_capture.put(1)
    await camera.record_frame_timing.put(True)
    await camera.capture.put(True)
    await camera.capture.put(False)

    assert not store_path.exists()
    assert not (store_path / f"{camera.data_key.get()}.timing.json").exists()


async def test_empty_plane_cleanup_keeps_other_plane_data(
    camera: MMCameraController,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a zero-frame plane must not remove a sibling plane."""
    monkeypatch.setattr(mmcore_camera, "FrameStore", frame_store_factory)
    store_path = tmp_path / "multi-plane.zarr"
    store_path.mkdir()
    sibling = RunFrameStore(str(store_path), "plane-1", np.zeros(FRAME_SHAPE, np.uint8))
    sibling.append(np.ones(FRAME_SHAPE, np.uint8))
    sibling.close()
    await camera.file_path.put(str(store_path))
    await camera.data_key.put("plane-0")
    await camera.capture.put(True)
    await camera.capture.put(False)
    assert (store_path / "plane-1").is_dir()
    assert manifest_path(str(store_path), "plane-1").exists()


async def test_a_bounded_capture_writes_its_frame_timing_sidecar(
    camera: MMCameraController,
    camera_core: FakeCameraCore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record one service-arrival stamp per written frame, with its MMCore tags."""
    monkeypatch.setattr(mmcore_camera, "FrameStore", MemoryFrameStore)
    store_path = tmp_path / "timed.zarr"
    store_path.mkdir()
    await camera.file_path.put(str(store_path))
    await camera.num_capture.put(2)
    await camera.record_frame_timing.put(True)
    await camera.capture.put(True)
    camera._start_sequence()
    for _ in range(2):
        camera_core.produce(1)
        camera.grab_once()
    await camera.capture.wait_for_value(False, timeout=TIMEOUT)

    sidecar = json.loads((store_path / f"{DATA_KEY}.timing.json").read_text("utf-8"))

    assert sidecar["schema"] == "redsun-camera-frame-timing-v1"
    assert sidecar["timestamp_semantics"] == "service_arrival_after_core_frame_read"
    assert [record["written_index"] for record in sidecar["records"]] == [0, 1]
    assert sidecar["written_count"] == 2
    assert sidecar["records"][0]["mmcore_metadata"] == {
        "ImageNumber": "0",
        "ElapsedTime-ms": "20",
    }


async def test_position_telemetry_publishes_sampled_coordinates(
    stage_core: FakeStageCore,
) -> None:
    """Sample the paired XY readbacks into one bounded JSON batch."""
    controller = MMStageController(
        core_of(stage_core),
        "XY",
        ("x", "y"),
        io=StageIO(core_of(stage_core), "XY", limits=LIMITS),
        telemetry_period_ms=20,
    )
    controller.post_initialise()
    telemetry = controller._telemetry_io
    assert telemetry is not None
    telemetry.start_run(
        "telemetry-run",
        source="test stage",
        timestamp_semantics="host monotonic bracket",
    )
    task = asyncio.create_task(controller._sample_positions())
    await asyncio.sleep(0.06)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    published = _StringAttribute()
    await telemetry.update(cast("Any", published))

    payload = json.loads(published.value)
    assert payload["schema"] == "redsun.mmcore-stage-telemetry/v1"
    assert payload["run_id"] == "telemetry-run"
    assert payload["overflow"] is False
    assert {record["axis"] for record in payload["records"]} == {"x", "y"}
    assert {
        axis: position
        for record in payload["records"]
        for axis, position in [(record["axis"], record["position_um"])]
    } == {"x": 100.0, "y": 200.0}
    assert all(
        record["read_call_finished_ns"] >= record["read_call_started_ns"]
        for record in payload["records"]
    )


class _StringAttribute:
    """A stand-in for the attribute a telemetry poll publishes into."""

    value = ""

    async def update(self, value: str) -> None:
        self.value = value


async def test_every_accepted_setting_change_moves_the_revision_on(
    camera: MMCameraController,
) -> None:
    """Count the writes the camera accepted, and name the revision in the record."""
    assert camera.settings_revision.get() == 0

    await camera.exposure.put(20.0)

    assert camera.settings_revision.get() == 1
    record = json.loads(camera.last_setting_change.get())
    assert record["settings_revision"] == 1
    assert record["outcome"] == "ok"

    await camera.roi.put("0,0,2,2")

    assert camera.settings_revision.get() == 2
    assert json.loads(camera.last_setting_change.get())["settings_revision"] == 2


async def test_a_refused_setting_change_leaves_the_revision_alone(
    camera: MMCameraController,
) -> None:
    """Never count a write the camera refused."""
    await camera.exposure.put(20.0)
    accepted = camera.settings_revision.get()

    with pytest.raises(ValueError, match="roi_invalid"):
        await camera.roi.put("1,2,3")

    assert camera.settings_revision.get() == accepted
    record = json.loads(camera.last_setting_change.get())
    assert record["outcome"] == "error"
    assert record["settings_revision"] == accepted


async def test_the_exposure_range_is_what_the_adapter_declares() -> None:
    """Publish the adapter's limits, and an empty range when it declares none."""
    first_core = FakeCameraCore(exposure_limits=(0.1, 1000.0))
    second_core = FakeCameraCore()
    first = MMCameraController(core_of(first_core), LABEL, DATA_KEY)
    second = MMCameraController(core_of(second_core), LABEL, DATA_KEY)
    await first.initialise()
    first.post_initialise()
    await second.initialise()
    second.post_initialise()
    try:
        assert np.array_equal(first.exposure_range.get(), np.array([0.1, 1000.0]))
        # the empty interval is the service saying it does not know the bound
        assert np.array_equal(second.exposure_range.get(), np.array([0.0, 0.0]))
    finally:
        await first.disconnect()
        await second.disconnect()


async def test_the_binning_factors_and_pixel_dtypes_are_published(
    camera: MMCameraController,
) -> None:
    """Publish the factors and dtypes a plan validates a request against."""
    assert camera.binning_factors.get() == "1,2"
    # native first, so the description's first dtype is the camera's own
    assert camera.pixel_dtypes.get() == "uint8,uint16"


async def test_an_axis_precision_comes_from_the_adapter_or_the_default(
    stage_core: FakeStageCore,
) -> None:
    """Publish the adapter's step size when it names one, the default otherwise."""
    undeclared = MMStageController(
        core_of(stage_core),
        "XY",
        ("x", "y"),
        io=StageIO(core_of(stage_core), "XY", limits=LIMITS),
    )
    undeclared.post_initialise()
    axis = axis_controller(undeclared, "x")
    assert axis.precision.get() == pytest.approx(DEFAULT_PRECISION_UM)

    stage_core.properties["StepSizeUm"] = "0.05"
    declared = MMStageController(
        core_of(stage_core),
        "XY",
        ("x", "y"),
        io=StageIO(core_of(stage_core), "XY", limits=LIMITS),
    )
    declared.post_initialise()

    assert axis_controller(declared, "y").precision.get() == pytest.approx(0.05)


def test_readback_provenance_fails_closed_for_unknown_and_demo() -> None:
    """A known demo adapter cannot be configured as a measured stage."""
    from redsun_aht.services.mmcore_stage import _readback_provenance

    assert _readback_provenance("DemoCamera", "auto") == "simulated"
    assert _readback_provenance("other", "auto") == "unknown"
    assert _readback_provenance("other", "measured") == "measured"
    with pytest.raises(ValueError, match="cannot provide measured"):
        _readback_provenance("DemoCamera", "measured")


@pytest.mark.parametrize("fault", ["overflow", "extra"])
def test_finite_sequence_rejects_overflow_and_extra_frames(fault):
    class FiniteCore(FakeCameraCore):
        def isSequenceRunning(self):
            return False

        def isBufferOverflowed(self):
            return fault == "overflow"

    core = FiniteCore()
    if fault == "extra":
        core.produce(1)
    camera = MMCameraController(
        core_of(core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
        finite_capture=True,
    )
    camera._sequencing = True
    camera._sequence_is_finite = True
    camera._capture_full = True
    with pytest.raises(RuntimeError, match="overflow or extra"):
        camera._stop_sequence()
    assert not camera._sequencing


def test_interrupted_finite_sequence_is_explicitly_stopped():
    class FiniteCore(FakeCameraCore):
        stops = 0

        def isSequenceRunning(self):
            return self.sequencing

        def stopSequenceAcquisition(self):
            self.stops += 1
            super().stopSequenceAcquisition()

    core = FiniteCore()
    core.sequencing = True
    camera = MMCameraController(
        core_of(core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
        finite_capture=True,
    )
    camera._sequencing = True
    camera._sequence_is_finite = True
    camera._capture_full = False
    camera._stop_sequence()
    assert core.stops == 1 and not core.sequencing


def test_finite_sequence_early_end_reports_incomplete():
    class ShortCore(FakeCameraCore):
        def isSequenceRunning(self):
            return False

        def isBufferOverflowed(self):
            return False

        def clearCircularBuffer(self):
            pass

        def startSequenceAcquisition(self, count, interval, stop_on_overflow):
            self.produce(count - 1)

    class Store:
        def append(self, frame):
            pass

    core = ShortCore()
    camera = MMCameraController(
        core_of(core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
        finite_capture=True,
    )
    camera._store = Store()
    camera._capture_wanted = 4
    camera._capture_full = False
    camera._grabbing.set()
    camera._grab_loop()
    assert camera._accepted == 3
    assert "ended before all frames" in camera._error
    assert camera._stopped.is_set()


def test_finite_capable_service_keeps_live_view_continuous():
    core = FakeCameraCore()
    camera = MMCameraController(
        core_of(core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
        finite_capture=True,
    )
    camera._start_sequence()
    assert core.sequencing and not camera._sequence_is_finite
    camera._stop_sequence()
    assert not core.sequencing


def test_camera_cli_exposes_finite_capture_only_with_prerequisites(monkeypatch):
    observed = {}
    monkeypatch.setattr(mmcore_camera, "identity", lambda: None)
    monkeypatch.setattr(mmcore_camera, "configure_logging", lambda: None)
    monkeypatch.setattr(
        mmcore_camera,
        "build_controller",
        lambda *a, **kw: observed.update(kw) or object(),
    )
    monkeypatch.setattr(mmcore_camera, "serve_controller", lambda *a: object())
    monkeypatch.setattr(mmcore_camera, "run", lambda serving: 0)
    with pytest.raises(SystemExit):
        mmcore_camera.main(["--finite-capture"])
    with pytest.raises(SystemExit):
        mmcore_camera.main(["--finite-capture", "--buffered-capture"])
    assert (
        mmcore_camera.main(
            ["--finite-capture", "--buffered-capture", "--defer-initial-frame"]
        )
        == 0
    )
    assert (
        observed["finite_capture"]
        and observed["buffered_capture"]
        and observed["defer_initial_frame"]
    )


async def test_snap_waits_for_concurrent_frame_publication(camera_core, monkeypatch):
    camera = MMCameraController(
        core_of(camera_core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = camera.follow_layout
    first = True

    async def delayed(frame):
        nonlocal first
        if first:
            first = False
            entered.set()
            await release.wait()
        await original(frame)

    await camera.initialise()
    camera.post_initialise()
    monkeypatch.setattr(camera, "follow_layout", delayed)
    try:
        await asyncio.to_thread(camera.grab_once)
        publishing = asyncio.create_task(camera.publish_frame())
        await asyncio.wait_for(entered.wait(), 2)
        snapping = asyncio.create_task(camera.snap_request.put(True))
        for _ in range(200):
            if camera_core.snapped == 2:
                break
            await asyncio.sleep(0.01)
        assert camera_core.snapped == 2
        assert not snapping.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(publishing, snapping), 2)
        assert camera.frame_count.get() == 2
    finally:
        release.set()
        await camera.disconnect()
