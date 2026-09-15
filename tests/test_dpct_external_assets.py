from __future__ import annotations

import hashlib
import multiprocessing
import os
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr
from bluesky.run_engine import RunEngineResult
from event_model import DocumentRouter
from event_model.documents import Datum, Event, EventDescriptor, Resource, RunStart
from redsun.engine import RunEngine

from redsun_aht.acquisition import dpct_plan
from redsun_aht.configurations import build_camera_service_dpct, build_dpct_simulation
from redsun_aht.device import (
    DpctDetectorGroupDevice,
    DpctPatternDevice,
    DpctStageDevice,
)
from redsun_aht.motion import ScanPoint


class AssetDocuments(DocumentRouter):
    def __init__(self) -> None:
        super().__init__()
        self.start_doc: RunStart | None = None
        self.descriptors: list[EventDescriptor] = []
        self.events: list[Event] = []
        self.resources: list[Resource] = []
        self.datums: list[Datum] = []

    def start(self, doc: RunStart) -> None:
        self.start_doc = doc

    def descriptor(self, doc: EventDescriptor) -> None:
        self.descriptors.append(doc)

    def event(self, doc: Event) -> Event:
        self.events.append(doc)
        return doc

    def resource(self, doc: Resource) -> None:
        self.resources.append(doc)

    def datum(self, doc: Datum) -> Datum:
        self.datums.append(doc)
        return doc


def test_redsun_run_preserves_service_owned_multidimensional_assets(
    tmp_path: Path,
) -> None:
    scan_points = (
        ScanPoint(17, (("z", 0.1),)),
        ScanPoint(3, (("z", 0.2),)),
    )
    application = build_dpct_simulation(
        "external-assets",
        output_root=tmp_path / "run",
        pattern_ids=(10, 11),
        scan_points=scan_points,
    )
    documents = AssetDocuments()
    engine = RunEngine({})
    engine.subscribe(documents)  # type: ignore[no-untyped-call]
    application.container.build()
    try:
        stage = cast(DpctStageDevice, application.container.devices["stage"])
        illumination = cast(
            DpctPatternDevice, application.container.devices["illumination"]
        )
        detectors = cast(
            DpctDetectorGroupDevice, application.container.devices["detectors"]
        )
        result = cast(
            RunEngineResult,
            engine(
                dpct_plan(stage, illumination, detectors, application.recipe)
            ).result(timeout=30),
        )
        assert result.plan_result is not None
    finally:
        application.container.shutdown()

    assert documents.start_doc is not None
    assert len(documents.resources) == 2
    assert len(documents.datums) == 8
    assert len(documents.events) == 4

    resources = {resource["uid"]: resource for resource in documents.resources}
    for resource in resources.values():
        assert resource["run_start"] == documents.start_doc["uid"]
        assert resource["spec"] == "AHT_OME_ZARR_DPCT_V1"
        assert resource["resource_kwargs"]["dimension_names"] == [
            "pattern",
            "scan",
            "y",
            "x",
        ]
        assert resource["resource_kwargs"]["scan_indices"] == [17, 3]

    placements = {
        (
            cast(int, datum["datum_kwargs"]["pattern_id"]),
            cast(int, datum["datum_kwargs"]["scan_index"]),
            tuple(cast(list[int], datum["datum_kwargs"]["index"])),
        )
        for datum in documents.datums
    }
    assert placements == {
        (10, 17, (0, 0)),
        (11, 17, (1, 0)),
        (10, 3, (0, 1)),
        (11, 3, (1, 1)),
    }

    image_keys = {
        key
        for descriptor in documents.descriptors
        for key, data_key in descriptor["data_keys"].items()
        if data_key.get("external") == "FILESTORE:"
    }
    assert image_keys == {"detectors_fluorescence_image", "detectors_dhm_image"}
    datum_ids = {datum["datum_id"] for datum in documents.datums}
    for event in documents.events:
        assert event["data"]["stage_scan_index"] in {17, 3}
        assert event["data"]["illumination_pattern_id"] in {10, 11}
        assert {event["data"][key] for key in image_keys} <= datum_ids

    for datum in documents.datums:
        resource = resources[datum["resource"]]
        asset = Path(resource["root"]) / resource["resource_path"]
        group = zarr.open_group(asset, mode="r")
        kwargs = datum["datum_kwargs"]
        pattern_ordinal, scan_ordinal = cast(list[int], kwargs["index"])
        stored_array = cast(Any, group[cast(str, kwargs["array_path"])])
        array = np.asarray(stored_array[pattern_ordinal, scan_ordinal, :, :])
        assert (
            hashlib.sha256(array.tobytes(order="C")).hexdigest() == kwargs["checksum"]
        )


def _run_service_asset_case(
    root: str, result_queue: Any, camera_ioc_args: tuple[str, ...] = ()
) -> None:
    """Run with a fresh Channel Access context in a spawned process."""
    application = build_camera_service_dpct(
        "epics-service-assets",
        output_root=Path(root),
        camera_ioc_args=camera_ioc_args,
        scan_points=(ScanPoint(9, (("z", 0.25),)),),
        pattern_ids=(10, 11),
    )
    documents = AssetDocuments()
    engine = RunEngine({})
    engine.subscribe(documents)  # type: ignore[no-untyped-call]
    application.container.build()
    service = cast(Any, application.container.services["camera_ioc"])
    try:
        try:
            assert service.running
            stage = cast(DpctStageDevice, application.container.devices["stage"])
            illumination = cast(
                DpctPatternDevice, application.container.devices["illumination"]
            )
            detectors = cast(
                DpctDetectorGroupDevice, application.container.devices["detectors"]
            )
            assert detectors.control is not None
            result = cast(
                RunEngineResult,
                engine(
                    dpct_plan(stage, illumination, detectors, application.recipe)
                ).result(timeout=30),
            )
            assert result.plan_result is not None
        finally:
            application.container.shutdown()

        result_queue.put(
            {
                "stopped": not service.running,
                "resources": len(documents.resources),
                "scan_indices": [
                    datum["datum_kwargs"]["scan_index"] for datum in documents.datums
                ],
                "event_has_image": all(
                    "detectors_dhm_image" in event["data"] for event in documents.events
                ),
            }
        )
    except BaseException:
        result_queue.put({"error": traceback.format_exc()})
        raise


def test_redsun_service_composes_ophyd_async_camera_and_external_assets(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_run_service_asset_case,
        args=(str(tmp_path / "run"), result_queue),
    )
    process.start()
    process.join(timeout=45)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise TimeoutError("RedSun service external-asset subprocess did not exit")
    result = result_queue.get(timeout=5)
    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    assert result == {
        "stopped": True,
        "resources": 1,
        "scan_indices": [9, 9],
        "event_has_image": True,
    }


@pytest.mark.hardware
@pytest.mark.skipif(
    os.environ.get("AHT_TEST_REDSUN_FLIR_ASSETS") != "1",
    reason="set AHT_TEST_REDSUN_FLIR_ASSETS=1 for one FLIR external-asset run",
)
def test_redsun_service_writes_one_physical_flir_external_asset(
    tmp_path: Path,
) -> None:
    required = {
        "AHT_TEST_MMCORE_PATH": "--mm-path",
        "AHT_TEST_REAL_MMCORE_CONFIG": "--mm-config",
        "AHT_TEST_REAL_ADAPTER": "--adapter",
        "AHT_TEST_REAL_DEVICE_NAME": "--device-name",
        "AHT_TEST_REAL_CAMERA_LABEL": "--camera-label",
        "AHT_TEST_REAL_SERIAL_PROPERTY": "--serial-property",
        "AHT_TEST_REAL_EXPECTED_SERIAL": "--expected-serial",
    }
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(f"missing FLIR hardware variables: {', '.join(missing)}")
    arguments = ["--backend", "mmcore", "--camera-property", "Exposure=2"]
    for variable, option in required.items():
        arguments.extend((option, os.environ[variable]))

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_run_service_asset_case,
        args=(str(tmp_path / "flir-run"), result_queue, tuple(arguments)),
    )
    process.start()
    process.join(timeout=90)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise TimeoutError("RedSun FLIR external-asset subprocess did not exit")
    result = result_queue.get(timeout=5)
    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    assert result == {
        "stopped": True,
        "resources": 1,
        "scan_indices": [9, 9],
        "event_has_image": True,
    }
