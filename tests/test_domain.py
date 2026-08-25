from __future__ import annotations

from typing import Any, cast

import pytest

from redsun_aht.domain import (
    BufferUsage,
    DetectorCapabilities,
    FrameBufferDescriptor,
    PoseSample,
    RunEvent,
    RunEventKind,
)


def test_detector_capabilities_reject_invalid_exposure_range() -> None:
    with pytest.raises(ValueError, match="ordered"):
        DetectorCapabilities(frozenset(), (2.0, 1.0), frozenset(), False, False)


def test_pose_requires_explicit_transform_and_joint_units() -> None:
    with pytest.raises(ValueError, match="4x4"):
        PoseSample(0, "measured", (), (), "stage", (), "mm", (), "mock")

    with pytest.raises(ValueError, match="explicit unit"):
        PoseSample(
            0,
            "measured",
            (1.0,),
            (),
            "stage",
            tuple(float(index) for index in range(16)),
            "mm",
            (),
            "mock",
        )


def test_run_event_recursively_copies_and_freezes_detail() -> None:
    source: dict[str, Any] = {"nested": {"values": [1, 2]}}
    event = RunEvent("run-1", 0, RunEventKind.STARTED, 1, source)

    source["nested"]["values"].append(3)

    nested = cast(dict[str, Any], event.detail["nested"])
    assert nested["values"] == (1, 2)
    with pytest.raises(TypeError):
        cast(Any, event.detail)["new"] = "value"


def test_frame_buffer_descriptor_rejects_invalid_transport_metadata() -> None:
    with pytest.raises(ValueError, match="positive"):
        FrameBufferDescriptor(
            service_id="camera",
            run_id="run",
            frame_id="frame",
            sequence=0,
            generation=0,
            shared_memory_name="memory",
            slot=0,
            shape=(0, 2),
            dtype="uint16",
            byte_order="=",
            timestamp_ns=0,
            checksum="checksum",
            usage=BufferUsage.PREVIEW,
            lease_timeout_ns=1,
        )
