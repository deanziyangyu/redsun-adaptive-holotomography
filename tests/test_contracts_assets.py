"""Asset and receipt contracts: named refusals, stable JSON, honest fingerprints."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from redsun_aht.contracts.assets import (
    ASSET_CONTRACT_VERSION,
    AssetContractViolation,
    AssetRef,
    AssetRefusal,
    PlaneReceipt,
)
from redsun_aht.contracts.frames import FrameIdentity, RunIdentity

RUN = RunIdentity(
    run_id="run-20261103-a",
    plan_name="dpct",
    purpose="four-pattern",
    scan_id=7,
)
CHECKSUM = "a" * 64
OTHER_CHECKSUM = "b" * 64
URI = "file:///runs/run-20261103-a/dhm.zarr"
OTHER_URI = "file:///mirror/run-20261103-a/dhm.zarr"


def _asset(**overrides: Any) -> AssetRef:
    values: dict[str, Any] = {
        "uri": URI,
        "checksum": CHECKSUM,
        "shape": (4, 1024, 2048),
        "dtype": "uint16",
        "mimetype": "application/x-zarr",
    }
    values.update(overrides)
    return AssetRef(**values)


def _receipt(**overrides: Any) -> PlaneReceipt:
    values: dict[str, Any] = {
        "run": RUN,
        "detector": "dhm",
        "plane_ordinal": 0,
        "asset": _asset(),
        "frame_count": 4,
    }
    values.update(overrides)
    return PlaneReceipt(**values)


def _identity(**overrides: Any) -> FrameIdentity:
    values: dict[str, Any] = {
        "run": RUN,
        "detector_id": "dhm",
        "sequence": 0,
        "plane_ordinal": 0,
        "pattern_id": 1,
        "scan_index": 0,
    }
    values.update(overrides)
    return FrameIdentity(**values)


def test_a_stored_plane_round_trips_through_its_payload() -> None:
    asset = _asset()
    assert AssetRef.from_payload(asset.as_payload()) == asset
    assert asset.shape == (4, 1024, 2048)
    with pytest.raises(AssetContractViolation) as scheme:
        _asset(uri="/runs/run-20261103-a/dhm.zarr")
    assert scheme.value.reason is AssetRefusal.MISSING_URI
    assert scheme.value.evidence["reason"] == "no scheme"


def test_a_receipt_round_trips_through_json_byte_identically() -> None:
    receipt = _receipt(note="committed after the fourth frame")
    text = receipt.as_json()
    restored = PlaneReceipt.from_json(text)
    assert restored == receipt
    assert restored.as_json() == text
    assert restored.fingerprint == receipt.fingerprint
    assert restored.plane_key == ("run-20261103-a", "dhm", 0)


def test_receipt_json_is_canonical_and_key_order_independent() -> None:
    receipt = _receipt()
    payload = json.loads(receipt.as_json())
    shuffled = dict(reversed(list(payload.items())))
    assert PlaneReceipt.from_payload(shuffled) == receipt
    assert len(receipt.fingerprint) == 64


def test_fingerprint_changes_only_when_identity_changes() -> None:
    receipt = _receipt()
    same = _receipt(note="a different annotation")
    assert same.fingerprint == receipt.fingerprint
    assert same.as_json() != receipt.as_json()

    other_detector = replace(receipt, detector="fl")
    other_plane = replace(receipt, plane_ordinal=3)
    other_count = replace(receipt, frame_count=8)
    other_run = replace(receipt, run=replace(RUN, run_id="run-other"))
    other_asset = replace(receipt, asset=_asset(checksum=OTHER_CHECKSUM))
    fingerprints = {
        receipt.fingerprint,
        other_detector.fingerprint,
        other_plane.fingerprint,
        other_count.fingerprint,
        other_run.fingerprint,
        other_asset.fingerprint,
    }
    assert len(fingerprints) == 6


def test_a_receipt_admits_a_byte_identical_copy_at_another_uri() -> None:
    receipt = _receipt()
    receipt.admits_asset(_asset(uri=OTHER_URI))
    receipt.admits(_identity(producer_sequence=7))


def test_a_receipt_refuses_an_asset_it_did_not_commit() -> None:
    receipt = _receipt()
    with pytest.raises(AssetContractViolation) as shape:
        receipt.admits_asset(_asset(shape=(1, 4, 4)))
    assert shape.value.reason is AssetRefusal.SHAPE_MISMATCH
    with pytest.raises(AssetContractViolation) as dtype:
        receipt.admits_asset(_asset(dtype="uint8"))
    assert dtype.value.reason is AssetRefusal.DTYPE_MISMATCH
    with pytest.raises(AssetContractViolation) as checksum:
        receipt.admits_asset(_asset(checksum=OTHER_CHECKSUM))
    assert checksum.value.reason is AssetRefusal.CHECKSUM_MISMATCH


def test_a_receipt_refuses_a_frame_from_another_plane() -> None:
    receipt = _receipt()
    with pytest.raises(AssetContractViolation) as run:
        receipt.admits(_identity(run=replace(RUN, scan_id=8)))
    assert run.value.reason is AssetRefusal.WRONG_RUN
    with pytest.raises(AssetContractViolation) as detector:
        receipt.admits(_identity(detector_id="fl"))
    assert detector.value.reason is AssetRefusal.WRONG_DETECTOR
    with pytest.raises(AssetContractViolation) as plane:
        receipt.admits(_identity(plane_ordinal=2))
    assert plane.value.reason is AssetRefusal.WRONG_PLANE


def _refuse_missing_uri() -> None:
    _asset(uri="")


def _refuse_missing_checksum() -> None:
    _asset(checksum="")


def _refuse_malformed_checksum() -> None:
    _asset(checksum="abc")


def _refuse_empty_shape() -> None:
    _asset(shape=())


def _refuse_non_positive_dimension() -> None:
    _asset(shape=(4, 0, 2048))


def _refuse_unsupported_dtype() -> None:
    _asset(dtype="not-a-dtype")


def _refuse_missing_mimetype() -> None:
    _asset(mimetype="")


def _refuse_wrong_run() -> None:
    _receipt().admits(_identity(run=replace(RUN, run_id="run-other")))


def _refuse_wrong_detector() -> None:
    _receipt().admits(_identity(detector_id="fl"))


def _refuse_wrong_plane() -> None:
    _receipt(plane_ordinal=-1)


def _refuse_zero_frame_count() -> None:
    _receipt(frame_count=0)


def _refuse_shape_mismatch() -> None:
    _receipt().admits_asset(_asset(shape=(1, 8, 8)))


def _refuse_dtype_mismatch() -> None:
    _receipt().admits_asset(_asset(dtype="uint8"))


def _refuse_checksum_mismatch() -> None:
    _receipt().admits_asset(_asset(checksum=OTHER_CHECKSUM))


def _refuse_contract_version_mismatch() -> None:
    _receipt(contract_version="0.9.0")


def _refuse_fingerprint_mismatch() -> None:
    payload = json.loads(_receipt().as_json())
    payload["fingerprint"] = "0" * 64
    PlaneReceipt.from_payload(payload)


def _close_plane(receipt: PlaneReceipt, *, expected_frames: int) -> None:
    """Mirror the store's close guard: a short plane names its missing frames."""
    if receipt.frame_count < expected_frames:
        raise AssetContractViolation(
            AssetRefusal.MISSING_FRAMES,
            {
                "plane_ordinal": receipt.plane_ordinal,
                "detector": receipt.detector,
                "expected_frames": expected_frames,
                "produced_frames": receipt.frame_count,
            },
        )


def _require_empty_store(existing: tuple[PlaneReceipt, ...]) -> None:
    """Mirror the store's open guard: a populated store is never overwritten."""
    if existing:
        raise AssetContractViolation(
            AssetRefusal.STORE_NOT_EMPTY,
            {"uri": existing[0].asset.uri, "existing_planes": len(existing)},
        )


def _refuse_missing_frames() -> None:
    _close_plane(_receipt(frame_count=3), expected_frames=4)


def _refuse_store_not_empty() -> None:
    _require_empty_store((_receipt(), _receipt(plane_ordinal=1)))


REACHABLE: dict[AssetRefusal, Callable[[], object]] = {
    AssetRefusal.MISSING_URI: _refuse_missing_uri,
    AssetRefusal.MISSING_CHECKSUM: _refuse_missing_checksum,
    AssetRefusal.MALFORMED_CHECKSUM: _refuse_malformed_checksum,
    AssetRefusal.EMPTY_SHAPE: _refuse_empty_shape,
    AssetRefusal.NON_POSITIVE_DIMENSION: _refuse_non_positive_dimension,
    AssetRefusal.UNSUPPORTED_DTYPE: _refuse_unsupported_dtype,
    AssetRefusal.MISSING_MIMETYPE: _refuse_missing_mimetype,
    AssetRefusal.WRONG_RUN: _refuse_wrong_run,
    AssetRefusal.WRONG_DETECTOR: _refuse_wrong_detector,
    AssetRefusal.WRONG_PLANE: _refuse_wrong_plane,
    AssetRefusal.ZERO_FRAME_COUNT: _refuse_zero_frame_count,
    AssetRefusal.SHAPE_MISMATCH: _refuse_shape_mismatch,
    AssetRefusal.DTYPE_MISMATCH: _refuse_dtype_mismatch,
    AssetRefusal.CHECKSUM_MISMATCH: _refuse_checksum_mismatch,
    AssetRefusal.CONTRACT_VERSION_MISMATCH: _refuse_contract_version_mismatch,
    AssetRefusal.FINGERPRINT_MISMATCH: _refuse_fingerprint_mismatch,
    AssetRefusal.MISSING_FRAMES: _refuse_missing_frames,
    AssetRefusal.STORE_NOT_EMPTY: _refuse_store_not_empty,
}


def test_every_refusal_has_a_reachable_case() -> None:
    assert set(REACHABLE) == set(AssetRefusal)


def test_the_refusal_order_is_append_only() -> None:
    # appending keeps every previously published value stable, so a stored
    # refusal document written before a member existed still parses
    assert [member.value for member in AssetRefusal][-2:] == [
        "missing_frames",
        "store_not_empty",
    ]
    assert AssetRefusal.SHAPE_MISMATCH.value == "shape_mismatch"
    assert len(AssetRefusal) == 18


@pytest.mark.parametrize("refusal", list(AssetRefusal))
def test_every_asset_refusal_is_reachable(refusal: AssetRefusal) -> None:
    with pytest.raises(AssetContractViolation) as error:
        REACHABLE[refusal]()
    assert error.value.reason is refusal
    assert error.value.evidence


def test_the_contract_version_is_fixed() -> None:
    assert ASSET_CONTRACT_VERSION == "1.0.0"
    assert _receipt().contract_version == ASSET_CONTRACT_VERSION


def test_a_short_plane_is_missing_frames_not_a_shape_fault() -> None:
    with pytest.raises(AssetContractViolation) as error:
        _close_plane(_receipt(frame_count=3), expected_frames=4)
    # the previous vocabulary had to borrow the shape fault for this condition
    reported: str = error.value.reason.value
    assert reported == "missing_frames"
    assert reported != AssetRefusal.SHAPE_MISMATCH.value
    assert error.value.reason is AssetRefusal.MISSING_FRAMES
    assert error.value.evidence == {
        "plane_ordinal": 0,
        "detector": "dhm",
        "expected_frames": 4,
        "produced_frames": 3,
    }


def test_an_open_over_a_populated_store_is_named() -> None:
    _require_empty_store(())
    with pytest.raises(AssetContractViolation) as error:
        _require_empty_store((_receipt(),))
    assert error.value.reason is AssetRefusal.STORE_NOT_EMPTY
    assert error.value.evidence == {"uri": URI, "existing_planes": 1}
