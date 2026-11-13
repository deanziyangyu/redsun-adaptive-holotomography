"""Frame identity and quality labels: byte-bounded runs and no upgrades."""

from __future__ import annotations

import copy
import json
from dataclasses import fields, replace
from typing import Any

import pytest

from redsun_aht.contracts.frames import (
    MAX_RUN_ID_BYTES,
    ExposureWindow,
    FrameIdentity,
    FrameRecord,
    QualityLabel,
    RunIdentity,
    SettingChange,
)

RUN = RunIdentity(
    run_id="run-20261103-a",
    plan_name="dpct",
    purpose="four-pattern",
    scan_id=7,
)
CHECKSUM = "a" * 64
URI = "file:///runs/run-20261103-a/dhm.zarr"


def _run(**overrides: Any) -> RunIdentity:
    values: dict[str, Any] = {
        "run_id": RUN.run_id,
        "plan_name": RUN.plan_name,
        "purpose": RUN.purpose,
        "scan_id": RUN.scan_id,
    }
    values.update(overrides)
    return RunIdentity(**values)


def _identity(**overrides: Any) -> FrameIdentity:
    values: dict[str, Any] = {
        "run": RUN,
        "detector_id": "dhm",
        "sequence": 0,
        "plane_ordinal": 0,
        "pattern_id": 1,
        "scan_index": 0,
        "producer_sequence": None,
    }
    values.update(overrides)
    return FrameIdentity(**values)


def _measured_window() -> ExposureWindow:
    return ExposureWindow(
        duration_s=0.1,
        start_ns=1_000_000_000,
        end_ns=1_100_000_000,
        quality=QualityLabel.MEASURED,
        source="camera frame timestamps",
    )


def _simulated_window() -> ExposureWindow:
    return ExposureWindow(
        duration_s=0.1,
        quality=QualityLabel.SIMULATED,
        source="declared exposure; no clock was read",
    )


def _record(**overrides: Any) -> FrameRecord:
    values: dict[str, Any] = {
        "identity": _identity(),
        "shape": (1024, 2048),
        "dtype": "uint16",
        "checksum": CHECKSUM,
        "uri": URI,
        "exposure": _simulated_window(),
        "setting_revision": 3,
        "quality": QualityLabel.SIMULATED,
    }
    values.update(overrides)
    return FrameRecord(**values)


def test_run_id_is_limited_to_32_utf8_bytes() -> None:
    assert MAX_RUN_ID_BYTES == 32
    _run(run_id="a" * 32)
    with pytest.raises(ValueError, match="32 UTF-8 bytes"):
        _run(run_id="a" * 33)
    # a multi-byte character costs as many bytes as its encoding
    _run(run_id="é" * 16)
    with pytest.raises(ValueError, match="32 UTF-8 bytes"):
        _run(run_id="é" * 17)
    _run(run_id="€" * 10)
    with pytest.raises(ValueError, match="32 UTF-8 bytes"):
        _run(run_id="€" * 11)


def test_run_identity_refuses_a_token_the_store_cannot_key_on() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _run(run_id="")
    with pytest.raises(ValueError, match="whitespace"):
        _run(run_id="two words")
    with pytest.raises(ValueError, match="plan name"):
        _run(plan_name="")
    with pytest.raises(ValueError, match="purpose"):
        _run(purpose="  ")
    with pytest.raises(ValueError, match="scan id"):
        _run(scan_id=-1)


def test_run_identity_round_trips_through_its_payload() -> None:
    assert RunIdentity.from_payload(RUN.as_payload()) == RUN


def test_frame_identity_has_no_bluesky_seq_num() -> None:
    assert "seq_num" not in {field.name for field in fields(FrameIdentity)}
    # a restarted run reassigns Event seq_num; the acquisition identity does not
    # move, because it is built from the plan's own frame ordinal
    frames = [_identity(sequence=0), _identity(sequence=0)]
    events = [
        {"seq_num": 1, "data": {"dhm": "frame-0"}},
        {"seq_num": 9, "data": {"dhm": "frame-0"}},
    ]
    assert events[0]["seq_num"] != events[1]["seq_num"]
    assert frames[0] == frames[1]


def test_identity_key_ignores_the_producer_readout_sequence() -> None:
    requested = _identity(sequence=2, producer_sequence=None)
    produced = _identity(sequence=2, producer_sequence=41)
    assert requested != produced
    assert requested.identity_key == produced.identity_key
    assert requested.as_payload()["sequence"] == 2


def test_frame_identity_round_trips_through_its_payload() -> None:
    identity = _identity(producer_sequence=41, plane_ordinal=2, pattern_id=3)
    assert FrameIdentity.from_payload(identity.as_payload()) == identity


def test_frame_identity_refuses_a_negative_ordinal() -> None:
    for name in ("sequence", "plane_ordinal", "pattern_id", "scan_index"):
        with pytest.raises(ValueError, match="non-negative"):
            _identity(**{name: -1})


def test_quality_label_carries_the_four_required_members() -> None:
    assert {member.value for member in QualityLabel} >= {
        "measured",
        "simulated",
        "derived",
        "unavailable",
    }


def test_a_measured_record_needs_a_measured_exposure_window() -> None:
    with pytest.raises(ValueError, match="measured exposure window"):
        _record(quality=QualityLabel.MEASURED)
    measured = _record(quality=QualityLabel.MEASURED, exposure=_measured_window())
    assert measured.exposure.is_measured is True


def test_a_quality_label_cannot_be_upgraded_by_copying() -> None:
    simulated = _record()
    with pytest.raises(ValueError, match="measured exposure window"):
        replace(simulated, quality=QualityLabel.MEASURED)
    # a plain copy, shallow or deep, keeps the label it was given
    assert replace(simulated).quality is QualityLabel.SIMULATED
    assert copy.copy(simulated).quality is QualityLabel.SIMULATED
    assert copy.deepcopy(simulated).quality is QualityLabel.SIMULATED
    assert replace(simulated) == simulated


def test_a_quality_label_cannot_be_upgraded_by_serialising() -> None:
    simulated = _record()
    payload = simulated.as_payload()
    assert payload["quality"] == "simulated"
    assert FrameRecord.from_payload(payload) == simulated

    upgraded = dict(payload)
    upgraded["quality"] = "measured"
    with pytest.raises(ValueError, match="measured exposure window"):
        FrameRecord.from_payload(upgraded)

    # a JSON round trip preserves both the label and the evidence behind it
    text = json.dumps(payload, sort_keys=True)
    restored = FrameRecord.from_payload(json.loads(text))
    assert restored.quality is QualityLabel.SIMULATED
    assert restored.exposure.quality is QualityLabel.SIMULATED
    assert restored.as_payload() == payload


def test_a_measured_record_survives_serialisation_unchanged() -> None:
    measured = _record(quality=QualityLabel.MEASURED, exposure=_measured_window())
    payload = measured.as_payload()
    restored = FrameRecord.from_payload(json.loads(json.dumps(payload)))
    assert restored == measured
    assert restored.quality is QualityLabel.MEASURED


def test_frame_record_refuses_malformed_evidence() -> None:
    with pytest.raises(ValueError, match="checksum"):
        _record(checksum="A" * 64)
    with pytest.raises(ValueError, match="checksum"):
        _record(checksum="abc")
    with pytest.raises(ValueError, match="positive"):
        _record(shape=(0, 8))
    with pytest.raises(ValueError, match="two integers"):
        _record(shape=(8, 8, 1))
    with pytest.raises(ValueError, match="scheme-qualified"):
        _record(uri="/runs/run-1/zarr")
    with pytest.raises(ValueError, match="dtype"):
        _record(dtype="not-a-dtype")
    with pytest.raises(ValueError, match="setting revision"):
        _record(setting_revision=-1)


def test_exposure_window_refuses_evidence_that_contradicts_its_label() -> None:
    with pytest.raises(ValueError, match="measured start and end"):
        ExposureWindow(duration_s=0.1, quality=QualityLabel.MEASURED)
    with pytest.raises(ValueError, match="both known or both absent"):
        ExposureWindow(duration_s=0.1, start_ns=10, quality=QualityLabel.DERIVED)
    with pytest.raises(ValueError, match="follow its start"):
        ExposureWindow(
            duration_s=0.1,
            start_ns=10,
            end_ns=10,
            quality=QualityLabel.DERIVED,
        )
    with pytest.raises(ValueError, match="disagrees"):
        ExposureWindow(
            duration_s=0.5,
            start_ns=0,
            end_ns=100_000_000,
            quality=QualityLabel.MEASURED,
        )
    with pytest.raises(ValueError, match="positive"):
        ExposureWindow(duration_s=0.0, quality=QualityLabel.SIMULATED)


def test_declared_duration_without_endpoints_is_not_a_measured_window() -> None:
    declared = _simulated_window()
    assert declared.is_measured is False
    assert declared.duration_s == 0.1
    assert ExposureWindow.from_payload(declared.as_payload()) == declared


def test_setting_change_records_the_readback_it_actually_got() -> None:
    applied = SettingChange(
        name="exposure",
        value=0.05,
        revision=4,
        applied_monotonic_ns=1_234,
        readback=0.05,
    )
    assert applied.confirmed is True
    assert applied.clamped is False
    clamped = SettingChange(
        name="exposure",
        value=0.0001,
        revision=5,
        applied_monotonic_ns=1_235,
        readback=0.001,
    )
    assert clamped.clamped is True
    unread = SettingChange(
        name="roi", value="0,0,512,512", revision=6, applied_monotonic_ns=1_236
    )
    assert unread.confirmed is False
    assert SettingChange.from_payload(clamped.as_payload()) == clamped


def test_setting_change_refuses_values_that_cannot_be_recorded() -> None:
    with pytest.raises(ValueError, match="boolean, number or text"):
        SettingChange(
            name="exposure",
            value=[0.1],  # type: ignore[arg-type]
            revision=1,
            applied_monotonic_ns=1,
        )
    with pytest.raises(ValueError, match="revision"):
        SettingChange(name="exposure", value=0.1, revision=-1, applied_monotonic_ns=1)
    with pytest.raises(ValueError, match="name"):
        SettingChange(name="", value=0.1, revision=1, applied_monotonic_ns=1)
