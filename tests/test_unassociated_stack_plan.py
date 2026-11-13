"""Headless Bluesky documents for the experimental combined workflow."""

import json
from unittest.mock import AsyncMock

import pytest
from bluesky import RunEngine

from redsun_aht.device.pva_capture_evidence import CaptureEvidence
from redsun_aht.device.unassociated_stack_plan import unassociated_stack_plan
from redsun_aht.device.unified_mcu import AngularStack, StackProgress


def test_plan_emits_separate_unassociated_evidence():
    client, camera = AsyncMock(), AsyncMock()
    client.execute.return_value = StackProgress(7, (0, 18, 36), complete=True)
    camera.collect.return_value = CaptureEvidence("test.zarr", 3, 3)
    documents = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))
    engine(
        unassociated_stack_plan(
            client,
            camera,
            AngularStack(7, 0, 18, 3, -180, 180),
            "test.zarr",
            resources_exclusive=True,
            external_trigger_configured=True,
            illumination_ready=True,
            timeout_s=2,
        )
    )
    start = next(doc for name, doc in documents if name == "start")
    event = next(doc for name, doc in documents if name == "event")
    assert start["association"] == "unassociated" and start["store_uri"] == "test.zarr"
    assert json.loads(event["data"]["controller_report_json"])["run_id"] == 7
    assert (
        json.loads(event["data"]["camera_capture_json"])["quality"]
        == "unassociated_service_count"
    )
    assert (
        next(doc for name, doc in documents if name == "stop")["exit_status"]
        == "success"
    )


def test_failed_capture_records_unavailable_evidence_and_failed_run():
    client, camera = AsyncMock(), AsyncMock()
    camera.prepare.side_effect = RuntimeError("not ready")
    camera.snapshot.side_effect = RuntimeError("capture not attempted")
    client.read_progress.side_effect = RuntimeError("no report")
    documents = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))
    with pytest.raises(RuntimeError, match="not ready"):
        engine(
            unassociated_stack_plan(
                client,
                camera,
                AngularStack(7, 0, 18, 3, -180, 180),
                "test.zarr",
                resources_exclusive=True,
                external_trigger_configured=True,
                illumination_ready=True,
                timeout_s=2,
            )
        )
    event = next(doc for name, doc in documents if name == "event")
    failure = json.loads(event["data"]["failure_evidence_json"])
    assert failure["controller"] is None and failure["camera"] is None
    assert set(failure["read_errors"]) == {"controller", "camera"}
    assert failure["quality"] == "unassociated_stop_unverified"
    assert (
        next(doc for name, doc in documents if name == "stop")["exit_status"] == "fail"
    )
    client.execute.assert_not_awaited()


def test_failed_run_records_available_partial_reports():
    client, camera = AsyncMock(), AsyncMock()
    client.execute.return_value = StackProgress(7, (0, 18, 36), complete=True)
    client.read_progress.return_value = StackProgress(7, (0, 18, 36), complete=True)
    camera.collect.side_effect = TimeoutError("frames missing")
    camera.snapshot.return_value = CaptureEvidence("test.zarr", 3, 2)
    documents = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))
    with pytest.raises(TimeoutError, match="frames missing"):
        engine(
            unassociated_stack_plan(
                client,
                camera,
                AngularStack(7, 0, 18, 3, -180, 180),
                "test.zarr",
                resources_exclusive=True,
                external_trigger_configured=True,
                illumination_ready=True,
                timeout_s=2,
            )
        )
    failure = json.loads(
        next(doc for name, doc in documents if name == "event")["data"][
            "failure_evidence_json"
        ]
    )
    assert failure["controller"]["targets_tenths"] == [0, 18, 36]
    assert failure["camera"]["stored_frames"] == 2
    assert failure["quality"] == "unassociated_stop_unverified"
