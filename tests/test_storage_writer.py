"""The finite-run frame store: order, layout, manifest, refusal and timing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from redsun_aht.contracts.frames import RunIdentity
from redsun_aht.storage import writer
from redsun_aht.storage.writer import (
    MANIFEST_SCHEMA,
    TIMING_CLOCK,
    TIMING_SCHEMA,
    TIMING_SEMANTICS,
    FrameStamp,
    RunFrameStore,
    StoreManifest,
    StoreRefusal,
    StoreRefused,
    TimingSidecar,
    frame_checksum,
    frame_store_factory,
    frames_checksum,
    manifest_path,
    timing_sidecar_path,
)

RUN = RunIdentity(
    run_id="run-20261103-a",
    plan_name="dpct",
    purpose="four-pattern",
    scan_id=7,
)
SHAPE = (8, 6)
DATA_KEY = "dhm"


def _frame(value: int, dtype: Any = np.uint16) -> NDArray[Any]:
    """Return a frame filled with *value*, so append order is visible."""
    return np.full(SHAPE, value, dtype=dtype)


def _open(
    path: Path,
    *,
    planned_frames: int | None = 4,
    record_timing: bool = False,
    run: RunIdentity | None = RUN,
    detector_id: str | None = "dhm",
    sequence_offset: int = 0,
    plane_start: int = 0,
) -> RunFrameStore:
    """Open a store at *path* with the test's run identity."""
    return RunFrameStore(
        str(path),
        DATA_KEY,
        _frame(0),
        run=run,
        detector_id=detector_id,
        planned_frames=planned_frames,
        record_timing=record_timing,
        sequence_offset=sequence_offset,
        plane_start=plane_start,
    )


def test_a_finite_run_round_trips_with_exact_shape_dtype_and_checksums(
    tmp_path: Path,
) -> None:
    """A four-frame run lands as a Zarr v3 array a reader opens without help."""
    store = _open(tmp_path / "run.zarr")
    indices = [store.append(_frame(value)) for value in (1, 2, 3, 4)]
    store.close()

    manifest = store.manifest
    assert manifest is not None
    assert indices == [0, 1, 2, 3]
    assert store.frames_written == 4
    assert store.closed

    array = zarr.open_array(str(tmp_path / "run.zarr" / DATA_KEY), mode="r")
    frames = np.asarray(array[:])

    assert array.metadata.zarr_format == 3
    assert array.shape == (4, *SHAPE)
    assert array.dtype == np.uint16
    assert array.chunks == (1, *SHAPE)
    assert [int(frame[0, 0]) for frame in frames] == [1, 2, 3, 4]

    assert manifest.schema == MANIFEST_SCHEMA
    assert manifest.frame_count == 4
    assert manifest.shape == SHAPE
    assert manifest.dtype == "uint16"
    assert manifest.chunks == (1, *SHAPE)
    assert manifest.planned_frames == 4
    assert manifest.is_complete
    assert [entry.checksum for entry in manifest.frames] == [
        frame_checksum(frame) for frame in frames
    ]
    assert manifest.array_checksum == frames_checksum(frames)
    assert manifest.array_uri.endswith(f"run.zarr/{DATA_KEY}")


def test_frames_are_appended_in_order_with_their_identity(tmp_path: Path) -> None:
    """Each frame keeps its run, detector, sequence, plane and pattern."""
    store = _open(tmp_path / "run.zarr", sequence_offset=4, plane_start=1)
    for index, pattern_id in enumerate((11, 12, 13, 14)):
        store.append(
            _frame(index + 1),
            stamp=FrameStamp(pattern_id=pattern_id, scan_index=index),
        )
    store.close()

    manifest = store.manifest
    assert manifest is not None
    assert [entry.index for entry in manifest.frames] == [0, 1, 2, 3]
    identities = [entry.identity for entry in manifest.frames]
    assert all(identity is not None for identity in identities)
    assert [identity.sequence for identity in identities if identity] == [4, 5, 6, 7]
    # every frame of one store belongs to the same plane, which is what a
    # plane receipt admits together
    assert [identity.plane_ordinal for identity in identities if identity] == [
        1,
        1,
        1,
        1,
    ]
    assert [identity.pattern_id for identity in identities if identity] == [
        11,
        12,
        13,
        14,
    ]
    assert [identity.scan_index for identity in identities if identity] == [0, 1, 2, 3]
    assert manifest.run == RUN
    assert manifest.detector_id == "dhm"
    assert manifest.plane_start == 1


def test_capture_planned_count_is_fixed_before_frames(tmp_path: Path) -> None:
    frame = np.zeros((2, 2), dtype=np.uint8)
    store = RunFrameStore(str(tmp_path / "planned.zarr"), "cam", frame)
    store.set_planned_frames(3)
    store.append(frame)
    with pytest.raises(RuntimeError, match="already fixed"):
        store.set_planned_frames(4)
    store.close()
    assert store.manifest is not None
    assert store.manifest.planned_frames == 3
    assert not store.manifest.is_complete


def test_a_second_open_on_a_non_empty_store_is_refused(tmp_path: Path) -> None:
    """A store with frames and no manifest is never adopted for writing."""
    first = _open(tmp_path / "run.zarr")
    first.append(_frame(1))
    first.append(_frame(2))

    with pytest.raises(StoreRefused) as refused:
        _open(tmp_path / "run.zarr")

    assert refused.value.reason is StoreRefusal.STORE_NOT_EMPTY
    assert refused.value.evidence["frames_held"] == 2
    first.close()


def test_a_closed_store_refuses_a_second_open(tmp_path: Path) -> None:
    """A manifest is the store's statement that its run is finished."""
    store = _open(tmp_path / "run.zarr")
    store.append(_frame(1))
    store.close()

    with pytest.raises(StoreRefused) as refused:
        _open(tmp_path / "run.zarr")

    assert refused.value.reason is StoreRefusal.STORE_ALREADY_CLOSED
    assert (tmp_path / "run.zarr" / f"{DATA_KEY}.manifest.json").is_file()


def test_an_interrupted_open_is_adopted_when_it_holds_no_frame(
    tmp_path: Path,
) -> None:
    """An empty array can be reclaimed after its stale lock is cleared."""
    _open(tmp_path / "run.zarr")
    (tmp_path / "run.zarr" / f".{DATA_KEY}.writer.lock").unlink()
    second = _open(tmp_path / "run.zarr")
    second.append(_frame(9))
    second.close()

    manifest = second.manifest
    assert manifest is not None
    assert manifest.frame_count == 1
    array = zarr.open_array(str(tmp_path / "run.zarr" / DATA_KEY), mode="r")
    assert int(np.asarray(array[0])[0, 0]) == 9


def test_a_second_live_writer_cannot_adopt_the_first_empty_array(
    tmp_path: Path,
) -> None:
    """An empty array still has one active writer until that writer closes."""
    first = _open(tmp_path / "run.zarr")
    with pytest.raises(StoreRefused) as refused:
        _open(tmp_path / "run.zarr")
    assert refused.value.reason is StoreRefusal.STORE_NOT_EMPTY
    first.append(_frame(1))
    first.close()
    assert not (tmp_path / "run.zarr" / f".{DATA_KEY}.writer.lock").exists()


def test_a_foreign_directory_is_refused_rather_than_truncated(
    tmp_path: Path,
) -> None:
    """Something else living at the data key is reported, never overwritten."""
    occupied = tmp_path / "run.zarr" / DATA_KEY
    occupied.mkdir(parents=True)
    (occupied / "notes.txt").write_text("not a store", encoding="utf-8")

    with pytest.raises(StoreRefused) as refused:
        _open(tmp_path / "run.zarr")

    assert refused.value.reason is StoreRefusal.STORE_NOT_EMPTY
    assert (occupied / "notes.txt").read_text(encoding="utf-8") == "not a store"


def test_a_planned_run_refuses_a_frame_beyond_its_extent(tmp_path: Path) -> None:
    """A finite run accepts exactly the frames it planned for."""
    store = _open(tmp_path / "run.zarr", planned_frames=2)
    store.append(_frame(1))
    store.append(_frame(2))

    with pytest.raises(StoreRefused) as refused:
        store.append(_frame(3))

    assert refused.value.reason is StoreRefusal.PLANNED_FRAMES_EXCEEDED
    assert refused.value.evidence["planned_frames"] == 2
    store.close()
    manifest = store.manifest
    assert manifest is not None
    assert manifest.frame_count == 2


def test_the_manifest_and_the_timing_sidecar_round_trip_through_json(
    tmp_path: Path,
) -> None:
    """Both documents survive a write and a read, and keep the service's schema."""
    store = _open(tmp_path / "run.zarr", record_timing=True)
    for index in range(2):
        store.append(
            _frame(index + 1),
            arrival_ns=1_000 + index,
            mmcore_metadata={"ImageNumber": str(index), "ElapsedTime-ms": "20"},
        )
    store.close()

    manifest = store.manifest
    assert manifest is not None
    manifest_text = manifest_path(str(tmp_path / "run.zarr"), DATA_KEY).read_text(
        encoding="utf-8"
    )
    assert StoreManifest.from_json(manifest_text) == manifest
    assert StoreManifest.from_payload(manifest.as_payload()) == manifest
    assert manifest.timing_sidecar_uri is not None
    assert manifest.timing_sidecar_uri.endswith(f"{DATA_KEY}.timing.json")

    sidecar_text = timing_sidecar_path(str(tmp_path / "run.zarr"), DATA_KEY).read_text(
        encoding="utf-8"
    )
    sidecar_payload = json.loads(sidecar_text)
    assert set(sidecar_payload) == {
        "schema",
        "timestamp_semantics",
        "clock",
        "frame_metadata_source",
        "frame_metadata_semantics",
        "store_uri",
        "data_key",
        "written_count",
        "records",
        "camera_frame_counter",
    }
    assert sidecar_payload["schema"] == TIMING_SCHEMA
    assert sidecar_payload["clock"] == TIMING_CLOCK
    assert sidecar_payload["timestamp_semantics"] == TIMING_SEMANTICS
    assert sidecar_payload["written_count"] == 2
    assert [record["written_index"] for record in sidecar_payload["records"]] == [0, 1]
    assert set(sidecar_payload["records"][0]) == {
        "written_index",
        "frame_id",
        "service_monotonic_ns",
        "camera_frame_counter",
        "mmcore_metadata_source",
        "mmcore_metadata",
    }
    assert sidecar_payload["records"][1]["mmcore_metadata"] == {
        "ImageNumber": "1",
        "ElapsedTime-ms": "20",
    }
    assert sidecar_payload["records"][1]["service_monotonic_ns"] == 1_001

    sidecar = TimingSidecar.from_json(sidecar_text)
    assert sidecar == TimingSidecar.from_payload(sidecar.as_payload())
    assert sidecar.as_json() == sidecar_text
    assert sidecar.records[0].written_index == 0
    assert sidecar.records[0].mmcore_metadata_source == (
        "CMMCorePlus.popNextImageAndMD"
    )


def test_the_setting_revision_timeline_follows_the_camera(tmp_path: Path) -> None:
    """A revision change between frames is recorded at the frame it affects."""
    store = _open(tmp_path / "run.zarr")
    store.append(_frame(1))
    store.append(_frame(2))
    store.set_setting_revision(4)
    store.set_setting_revision(4)
    store.append(_frame(3))
    store.close()

    manifest = store.manifest
    assert manifest is not None
    assert [entry.setting_revision for entry in manifest.frames] == [0, 0, 4]
    assert [mark.as_payload() for mark in manifest.setting_timeline] == [
        {"first_frame": 0, "revision": 0},
        {"first_frame": 2, "revision": 4},
    ]
    assert manifest.setting_revision == 4


def test_a_frame_stamp_revision_updates_the_manifest_timeline(tmp_path: Path) -> None:
    """Per-frame revisions and the timeline must describe the same change."""
    store = _open(tmp_path / "run.zarr")
    store.append(_frame(1), stamp=FrameStamp(setting_revision=5))
    store.append(_frame(2))
    store.close()
    manifest = store.manifest
    assert manifest is not None
    assert [entry.setting_revision for entry in manifest.frames] == [5, 5]
    assert [mark.as_payload() for mark in manifest.setting_timeline] == [
        {"first_frame": 0, "revision": 5}
    ]


def test_unknown_setting_revision_stays_unknown_until_a_revision_is_known(
    tmp_path: Path,
) -> None:
    """An unknown frame provenance is not replaced with a plausible revision."""
    store = RunFrameStore(
        str(tmp_path / "unknown.zarr"), "camera", _frame(0), setting_revision=None
    )
    store.append(_frame(1))
    store.set_setting_revision(5)
    store.append(_frame(2))
    store.close()

    manifest = store.manifest
    assert manifest is not None
    assert [entry.setting_revision for entry in manifest.frames] == [None, 5]
    assert [mark.as_payload() for mark in manifest.setting_timeline] == [
        {"first_frame": 0, "revision": None},
        {"first_frame": 1, "revision": 5},
    ]


def test_a_frame_of_another_layout_or_after_close_is_refused(tmp_path: Path) -> None:
    """The array's shape and dtype are fixed when the store opens."""
    store = _open(tmp_path / "run.zarr")

    with pytest.raises(StoreRefused) as shape:
        store.append(np.zeros((8, 7), dtype=np.uint16))
    assert shape.value.reason is StoreRefusal.FRAME_SHAPE_MISMATCH

    with pytest.raises(StoreRefused) as dtype:
        store.append(np.zeros(SHAPE, dtype=np.uint8))
    assert dtype.value.reason is StoreRefusal.FRAME_DTYPE_MISMATCH

    store.close()
    with pytest.raises(StoreRefused) as closed:
        store.append(_frame(1))
    assert closed.value.reason is StoreRefusal.STORE_CLOSED


def test_the_camera_service_factory_writes_a_readable_store(tmp_path: Path) -> None:
    """The three-argument factory is the seam the camera service calls."""
    store = frame_store_factory(str(tmp_path / "capture.zarr"), DATA_KEY, _frame(0))
    store.append(_frame(4))
    store.close()

    array = zarr.open_array(str(tmp_path / "capture.zarr" / DATA_KEY), mode="r")
    assert array.shape == (1, *SHAPE)
    manifest = StoreManifest.from_json(
        manifest_path(str(tmp_path / "capture.zarr"), DATA_KEY).read_text("utf-8")
    )
    assert manifest.frame_count == 1
    assert manifest.run is None
    assert manifest.frames[0].identity is None


def test_the_run_identity_is_locked_after_the_first_frame(tmp_path: Path) -> None:
    """A store names the run of every frame it holds, or of none of them."""
    store = _open(tmp_path / "run.zarr")
    store.append(_frame(1))

    with pytest.raises(StoreRefused) as refused:
        store.bind(RUN, "dhm")

    assert refused.value.reason is StoreRefusal.IDENTITY_LOCKED
    store.close()


def test_a_factory_store_binds_its_run_before_the_first_frame(
    tmp_path: Path,
) -> None:
    """A capture that learns its run late still records every frame's identity."""
    store = frame_store_factory(str(tmp_path / "capture.zarr"), DATA_KEY, _frame(0))
    assert isinstance(store, RunFrameStore)
    store.bind(RUN, "dhm", sequence_offset=2, plane_start=1)
    store.append(_frame(1))
    store.close()

    manifest = store.manifest
    assert manifest is not None
    assert manifest.run == RUN
    assert manifest.detector_id == "dhm"
    identity = manifest.frames[0].identity
    assert identity is not None
    assert identity.sequence == 2
    assert identity.plane_ordinal == 1


def test_timing_offered_to_an_untimed_store_is_refused(tmp_path: Path) -> None:
    """A store records arrival stamps only when it was opened to."""
    store = _open(tmp_path / "run.zarr")

    with pytest.raises(StoreRefused) as refused:
        store.append(_frame(1), arrival_ns=1)

    assert refused.value.reason is StoreRefusal.TIMING_NOT_RECORDED
    store.close()


def test_the_timing_series_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture cannot grow its timing evidence without limit."""
    monkeypatch.setattr(writer, "TIMING_RECORD_LIMIT", 2)
    store = _open(tmp_path / "run.zarr", record_timing=True, planned_frames=None)
    store.append(_frame(1))
    store.append(_frame(2))

    with pytest.raises(StoreRefused) as refused:
        store.append(_frame(3))

    assert refused.value.reason is StoreRefusal.TIMING_CAPACITY_EXCEEDED
    store.close()


def test_a_revision_set_before_the_first_frame_replaces_the_initial_mark(
    tmp_path: Path,
) -> None:
    """A capture that learns the camera's revision first still closes cleanly."""
    store = _open(tmp_path / "run.zarr")
    store.set_setting_revision(3)
    store.set_setting_revision(5)
    store.append(_frame(1))
    store.close()

    manifest = store.manifest
    assert manifest is not None
    assert [entry.setting_revision for entry in manifest.frames] == [5]
    assert [mark.as_payload() for mark in manifest.setting_timeline] == [
        {"first_frame": 0, "revision": 5}
    ]


def test_bulk_append_matches_sequential_identity_and_setting_metadata(tmp_path):
    values = [_frame(i) for i in range(4)]
    sequential = _open(tmp_path / "seq", sequence_offset=4, plane_start=7)
    bulk = _open(tmp_path / "bulk", sequence_offset=4, plane_start=7)
    for store in (sequential, bulk):
        store.set_setting_revision(9)
    for frame in values:
        sequential.append(frame)
    assert bulk.append_batch(values) == 0
    sequential.close()
    bulk.close()
    a, b = sequential.manifest, bulk.manifest
    assert a is not None and b is not None
    assert [x.as_payload() for x in a.frames] == [x.as_payload() for x in b.frames]
    assert a.setting_timeline == b.setting_timeline
    assert a.array_checksum == b.array_checksum == frames_checksum(values)
    assert a.is_complete and b.is_complete


def test_bulk_prevalidation_keeps_store_unchanged(tmp_path):
    store = _open(tmp_path / "bulk", planned_frames=2)
    with pytest.raises(StoreRefused) as failure:
        store.append_batch([_frame(1), _frame(2), _frame(3)])
    assert failure.value.reason == StoreRefusal.PLANNED_FRAMES_EXCEEDED
    with pytest.raises(StoreRefused) as failure:
        store.append_batch([_frame(1), _frame(2, dtype=np.uint8)])
    assert failure.value.reason == StoreRefusal.FRAME_DTYPE_MISMATCH
    assert store.frames_written == 0
    assert store.append_batch([_frame(1), _frame(2)]) == 0
    store.close()
    assert store.manifest is not None and store.manifest.is_complete
