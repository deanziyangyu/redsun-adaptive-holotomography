"""Receipts for closed stores: assets, partial runs, altered bytes, refusals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from redsun_aht.contracts.assets import AssetRef, AssetRefusal, PlaneReceipt
from redsun_aht.contracts.frames import RunIdentity
from redsun_aht.storage import (
    STORE_MIMETYPE,
    TIMING_MIMETYPE,
    ClosedStore,
    RefusedStore,
    assemble_receipt,
    build_asset,
    open_closed_store,
    read_manifest,
    read_timing,
    refusal_from_exception,
    timing_asset,
)
from redsun_aht.storage.writer import (
    FrameStamp,
    RunFrameStore,
    StoreRefusal,
    StoreRefused,
    frames_checksum,
    manifest_path,
    timing_sidecar_path,
)

RUN = RunIdentity(
    run_id="run-20261103-b",
    plan_name="dpct",
    purpose="four-pattern",
    scan_id=8,
)
OTHER_RUN = RunIdentity(
    run_id="run-20261103-c",
    plan_name="dpct",
    purpose="another attempt",
    scan_id=9,
)
SHAPE = (8, 6)
DATA_KEY = "dhm"
STORE_NAME = "run.zarr"


def _frame(value: int) -> NDArray[Any]:
    """Return a frame filled with *value*, so a checksum identifies its bytes."""
    return np.full(SHAPE, value, dtype=np.uint16)


def _write(
    tmp_path: Path,
    *,
    frames: int = 4,
    planned: int | None = 4,
    timing: bool = False,
    run: RunIdentity | None = RUN,
    detector_id: str | None = "dhm",
    plane_start: int = 0,
) -> RunFrameStore:
    """Record a closed store with *frames* frames that count from one."""
    store = RunFrameStore(
        str(tmp_path / STORE_NAME),
        DATA_KEY,
        _frame(0),
        run=run,
        detector_id=detector_id,
        planned_frames=planned,
        record_timing=timing,
        plane_start=plane_start,
    )
    for value in range(1, frames + 1):
        store.append(_frame(value), stamp=FrameStamp(pattern_id=10 + value))
    store.close()
    return store


def test_a_closed_store_becomes_a_plane_receipt(tmp_path: Path) -> None:
    """A complete plane becomes a content-addressed receipt, from disk alone."""
    store = _write(tmp_path)

    receipt = assemble_receipt(str(tmp_path / STORE_NAME))

    assert isinstance(receipt, PlaneReceipt)
    assert receipt.run == RUN
    assert receipt.detector == "dhm"
    assert receipt.plane_ordinal == 0
    assert receipt.frame_count == 4
    assert receipt.asset.shape == (4, *SHAPE)
    assert receipt.asset.dtype == "uint16"
    assert receipt.asset.mimetype == STORE_MIMETYPE
    assert receipt.asset.checksum == frames_checksum(
        np.asarray(zarr.open_array(str(tmp_path / STORE_NAME / DATA_KEY), mode="r")[:])
    )
    assert len(receipt.fingerprint) == 64
    assert receipt.as_json() == PlaneReceipt.from_json(receipt.as_json()).as_json()

    manifest = store.manifest
    assert manifest is not None
    for entry in manifest.frames:
        assert entry.identity is not None
        receipt.admits(entry.identity)

    asset = build_asset(str(tmp_path / STORE_NAME))
    assert isinstance(asset, AssetRef)
    assert asset.checksum == receipt.asset.checksum
    assert asset.shape == receipt.asset.shape


def test_a_receipt_from_an_unbound_store_needs_its_run_and_detector(
    tmp_path: Path,
) -> None:
    """A capture store names no run, so the caller supplies the run it made."""
    _write(tmp_path, run=None, detector_id=None)

    refused = assemble_receipt(str(tmp_path / STORE_NAME))

    assert isinstance(refused, RefusedStore)
    assert refused.reason is AssetRefusal.WRONG_RUN
    assert refused.condition == "unbound_store"

    receipt = assemble_receipt(str(tmp_path / STORE_NAME), run=RUN, detector_id="dhm")
    assert isinstance(receipt, PlaneReceipt)
    assert receipt.run == RUN
    assert receipt.plane_ordinal == 0


def test_a_partial_run_refuses_a_complete_receipt(tmp_path: Path) -> None:
    """A short run reports the ordinals that never arrived, not a bare failure."""
    _write(tmp_path, frames=3, planned=4)

    refused = assemble_receipt(str(tmp_path / STORE_NAME))

    assert isinstance(refused, RefusedStore)
    assert refused.reason is AssetRefusal.MISSING_FRAMES
    assert refused.condition == "partial_run"
    assert refused.missing_ordinals == (3,)
    assert refused.is_partial_run
    assert refused.evidence["requested_frames"] == 4
    assert refused.evidence["stored_frames"] == 3
    assert refused.evidence["uri"].endswith(f"{STORE_NAME}/{DATA_KEY}")
    assert "partial_run" in refused.message


def test_an_expectation_beyond_the_store_names_what_never_arrived(
    tmp_path: Path,
) -> None:
    """The caller's request wins over the store's own planned count."""
    _write(tmp_path, frames=2, planned=None)

    refused = assemble_receipt(str(tmp_path / STORE_NAME), expected_frames=4)

    assert isinstance(refused, RefusedStore)
    assert refused.reason is AssetRefusal.MISSING_FRAMES
    assert refused.condition == "partial_run"
    assert refused.missing_ordinals == (2, 3)


def test_a_checksum_mismatch_is_refused(tmp_path: Path) -> None:
    """A frame altered after it was recorded cannot be receipted."""
    _write(tmp_path)
    array = zarr.open_array(str(tmp_path / STORE_NAME / DATA_KEY), mode="r+")
    altered = np.asarray(array[1]).copy()
    altered[0, 0] = altered[0, 0] + 1
    array[1] = altered

    refused = assemble_receipt(str(tmp_path / STORE_NAME))

    assert isinstance(refused, RefusedStore)
    assert refused.reason is AssetRefusal.CHECKSUM_MISMATCH
    assert refused.condition == "frame_checksum"
    assert refused.missing_ordinals == (1,)
    assert refused.evidence["changed_ordinals"] == [1]


def test_lost_frames_are_named_when_the_array_is_short(tmp_path: Path) -> None:
    """A manifest that declares more frames than exist names the absent ones."""
    _write(tmp_path, frames=3)
    path = manifest_path(str(tmp_path / STORE_NAME), DATA_KEY)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["frames"].append({**payload["frames"][-1], "index": 3})
    payload["frame_count"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")

    refused = assemble_receipt(str(tmp_path / STORE_NAME))

    assert isinstance(refused, RefusedStore)
    assert refused.reason is AssetRefusal.MISSING_FRAMES
    assert refused.condition == "missing_frames"
    assert refused.missing_ordinals == (3,)
    assert refused.is_partial_run


def test_a_shape_mismatch_still_means_the_array_shape(tmp_path: Path) -> None:
    """A reshaped array is a shape refusal, not a missing-frames one."""
    _write(tmp_path)
    path = manifest_path(str(tmp_path / STORE_NAME), DATA_KEY)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["shape"] = [5, 5]
    payload["chunks"] = [1, 5, 5]
    path.write_text(json.dumps(payload), encoding="utf-8")

    refused = assemble_receipt(str(tmp_path / STORE_NAME))

    assert isinstance(refused, RefusedStore)
    assert refused.reason is AssetRefusal.SHAPE_MISMATCH
    assert refused.condition == "array_shape"
    assert not refused.is_partial_run


def test_a_changed_array_chunk_layout_is_refused(tmp_path: Path) -> None:
    """The reader verifies actual Zarr chunking, not only manifest metadata."""
    _write(tmp_path)
    array_metadata = tmp_path / STORE_NAME / DATA_KEY / "zarr.json"
    payload = json.loads(array_metadata.read_text(encoding="utf-8"))
    payload["chunk_grid"]["configuration"]["chunk_shape"] = [2, *SHAPE]
    array_metadata.write_text(json.dumps(payload), encoding="utf-8")

    refused = open_closed_store(str(tmp_path / STORE_NAME))

    assert isinstance(refused, RefusedStore)
    assert refused.reason is StoreRefusal.STORE_LAYOUT_MISMATCH
    assert refused.condition == "array_chunks"


def test_a_store_with_frames_but_no_manifest_is_refused(tmp_path: Path) -> None:
    """An open or interrupted capture is not a receiptable plane."""
    store = RunFrameStore(
        str(tmp_path / STORE_NAME),
        DATA_KEY,
        _frame(0),
        run=RUN,
        detector_id="dhm",
    )
    store.append(_frame(1))
    store.append(_frame(2))

    refused = assemble_receipt(str(tmp_path / STORE_NAME))

    assert isinstance(refused, RefusedStore)
    assert refused.reason is StoreRefusal.STORE_UNCOMMITTED
    assert refused.condition == "uncommitted_store"
    assert refused.evidence["frames_on_disk"] == 2
    store.close()


def test_a_manifest_from_another_schema_is_refused(tmp_path: Path) -> None:
    """A store of another schema or array format is refused, not reinterpreted."""
    _write(tmp_path)
    path = manifest_path(str(tmp_path / STORE_NAME), DATA_KEY)
    payload = json.loads(path.read_text(encoding="utf-8"))

    payload["schema"] = "redsun-dpct-zarr-v1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    mixed = assemble_receipt(str(tmp_path / STORE_NAME))
    assert isinstance(mixed, RefusedStore)
    assert mixed.reason is AssetRefusal.CONTRACT_VERSION_MISMATCH
    assert mixed.condition == "mixed_schema"
    assert mixed.evidence["found"] == "redsun-dpct-zarr-v1"

    payload["schema"] = "redsun-aht-frame-store-manifest-v1"
    payload["array_format"] = "zarr-v2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    legacy = assemble_receipt(str(tmp_path / STORE_NAME))
    assert isinstance(legacy, RefusedStore)
    assert legacy.reason is AssetRefusal.CONTRACT_VERSION_MISMATCH
    assert legacy.condition == "array_format"


def test_v1_manifest_with_integer_revisions_remains_readable(tmp_path: Path) -> None:
    """The v2 reader still accepts closed v1 stores with measured revisions."""
    _write(tmp_path)
    path = manifest_path(str(tmp_path / STORE_NAME), DATA_KEY)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "redsun-aht-frame-store-manifest-v1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    receipt = assemble_receipt(str(tmp_path / STORE_NAME))

    assert isinstance(receipt, PlaneReceipt)
    assert receipt.frame_count == 4


def test_a_non_empty_store_is_reported_as_a_refusal_not_an_exception(
    tmp_path: Path,
) -> None:
    """The create-only refusal becomes the contract's own member, with evidence."""
    store = RunFrameStore(str(tmp_path / STORE_NAME), DATA_KEY, _frame(0))
    store.append(_frame(1))

    with pytest.raises(StoreRefused) as error:
        RunFrameStore(str(tmp_path / STORE_NAME), DATA_KEY, _frame(0))

    refused = refusal_from_exception(error.value)

    assert refused.reason is AssetRefusal.STORE_NOT_EMPTY
    assert refused.condition == "store_not_empty"
    assert refused.evidence["frames_held"] == 1
    assert refused.evidence["plane_count"] == 1
    assert refused.evidence["uri"].endswith(STORE_NAME)
    store.close()


def test_a_reader_loads_the_store_without_the_writer_object(tmp_path: Path) -> None:
    """A closed store is readable from its manifest and array alone."""
    _write(tmp_path)

    opened = open_closed_store(str(tmp_path / STORE_NAME))

    assert isinstance(opened, ClosedStore)
    assert opened.frame_count == 4
    assert opened.frame_shape == SHAPE
    assert opened.dtype == "uint16"
    assert opened.data_key == DATA_KEY
    assert opened.frames().shape == (4, *SHAPE)
    assert int(opened.frame(2)[0, 0]) == 3
    assert len(opened.checksums) == 4
    assert opened.checksums[0] == opened.manifest.frames[0].checksum
    assert opened.array_uri.endswith(f"{STORE_NAME}/{DATA_KEY}")

    manifest = read_manifest(str(tmp_path / STORE_NAME))
    assert not isinstance(manifest, RefusedStore)
    assert manifest == opened.manifest


def test_the_timing_sidecar_is_an_asset_and_reads_back(tmp_path: Path) -> None:
    """The sidecar is addressable by content, and its stamps join the frames."""
    _write(tmp_path, timing=True)

    asset = timing_asset(str(tmp_path / STORE_NAME))

    assert isinstance(asset, AssetRef)
    assert asset.mimetype == TIMING_MIMETYPE
    assert asset.shape == (1,)
    assert asset.uri.endswith(".timing.json")
    sidecar_path = timing_sidecar_path(str(tmp_path / STORE_NAME), DATA_KEY)
    assert asset.checksum == hashlib.sha256(sidecar_path.read_bytes()).hexdigest()

    sidecar = read_timing(str(tmp_path / STORE_NAME))
    assert not isinstance(sidecar, RefusedStore)
    assert sidecar.written_count == 4
    assert [record.written_index for record in sidecar.records] == [0, 1, 2, 3]

    absent = read_timing(str(tmp_path / "other.zarr"))
    assert isinstance(absent, RefusedStore)
    assert absent.condition == "no_timing_sidecar"


def test_two_timed_planes_keep_separate_sidecars(tmp_path: Path) -> None:
    """Closing a second plane must not replace the first plane's timing."""
    uri = str(tmp_path / STORE_NAME)
    for key, arrival in (("first", 101), ("second", 202)):
        store = RunFrameStore(uri, key, _frame(0), record_timing=True)
        store.append(_frame(1), arrival_ns=arrival)
        store.close()
    first = read_timing(uri, data_key="first")
    second = read_timing(uri, data_key="second")
    assert not isinstance(first, RefusedStore)
    assert not isinstance(second, RefusedStore)
    assert first.records[0].service_monotonic_ns == 101
    assert second.records[0].service_monotonic_ns == 202
    ambiguous = read_timing(uri)
    assert isinstance(ambiguous, RefusedStore)
    assert ambiguous.condition == "several_timing_sidecars"


def test_a_manifest_cannot_redirect_a_named_plane_to_another_array(
    tmp_path: Path,
) -> None:
    """The manifest filename and embedded data key must identify one plane."""
    uri = str(tmp_path / STORE_NAME)
    for key in ("first", "second"):
        store = RunFrameStore(
            uri, key, _frame(0), run=RUN, detector_id=key, planned_frames=1
        )
        store.append(_frame(1))
        store.close()
    path = manifest_path(uri, "first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["data_key"] = "second"
    path.write_text(json.dumps(payload), encoding="utf-8")
    refused = assemble_receipt(
        uri, data_key="first", run=RUN, detector_id="first", expected_frames=1
    )
    assert isinstance(refused, RefusedStore)
    assert refused.condition == "manifest_key_mismatch"


def test_a_receipt_refuses_another_run_detector_or_plane(tmp_path: Path) -> None:
    """A bound store is checked against what the caller says, not overridden."""
    _write(tmp_path, plane_start=0)

    wrong_run = assemble_receipt(str(tmp_path / STORE_NAME), run=OTHER_RUN)
    assert isinstance(wrong_run, RefusedStore)
    assert wrong_run.reason is AssetRefusal.WRONG_RUN
    assert wrong_run.condition == "wrong_run"

    wrong_detector = assemble_receipt(
        str(tmp_path / STORE_NAME), detector_id="camera-2"
    )
    assert isinstance(wrong_detector, RefusedStore)
    assert wrong_detector.reason is AssetRefusal.WRONG_DETECTOR

    wrong_plane = assemble_receipt(str(tmp_path / STORE_NAME), plane_ordinal=3)
    assert isinstance(wrong_plane, RefusedStore)
    assert wrong_plane.reason is AssetRefusal.WRONG_PLANE


def test_a_store_that_recorded_no_frame_is_refused(tmp_path: Path) -> None:
    """A capture that stopped before its first frame has no plane to receipt."""
    _write(tmp_path, frames=0, planned=4)

    refused = assemble_receipt(str(tmp_path / STORE_NAME))

    assert isinstance(refused, RefusedStore)
    assert refused.reason is AssetRefusal.ZERO_FRAME_COUNT
    assert refused.condition == "zero_frames"
