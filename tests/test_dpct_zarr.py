from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr

from redsun_aht.configurations import build_dpct_simulation
from redsun_aht.storage import (
    DpctAcquisitionManifest,
    DpctZarrReplay,
    FrameCommit,
)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"run_id": ""},
        {"shot_index": -1},
        {"shape": ()},
        {"exposure_started_ns": 2, "exposure_ended_ns": 1},
        {"pattern_id": 0x10000},
        {"checksum": "not-a-checksum"},
        {"byte_order": "?"},
        {"pose_joint_units": ()},
    ],
)
def test_frame_commit_rejects_invalid_provenance(overrides: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "run",
        "shot_index": 0,
        "scan_index": 0,
        "pattern_id": 1,
        "detector_id": "camera",
        "frame_id": "frame",
        "frame_sequence": 0,
        "configuration_revision": "revision",
        "array_path": "detectors/camera/0",
        "array_index": (0, 0),
        "shape": (8, 8),
        "dtype": "uint16",
        "byte_order": "<" if np.little_endian else ">",
        "checksum": "0" * 64,
        "source_checksum": "0" * 64,
        "exposure_started_ns": 1,
        "exposure_ended_ns": 2,
        "pose_monotonic_ns": 1,
        "pose_joints": (0.0,),
        "pose_joint_units": ("mm",),
        "pose_coordinate_frame": "sample",
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        FrameCommit(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"ome_zarr_version": "0.4"},
        {"run_id": ""},
        {"created_ns": -1},
        {"scan_count": 0},
        {"frame_commit_count": 0},
        {"pattern_ids": (1, 1)},
        {"detector_arrays": {"": "detectors/camera/0"}},
        {
            "detector_arrays": {
                "camera-a": "detectors/shared/0",
                "camera-b": "detectors/shared/0",
            },
            "frame_commit_count": 2,
        },
        {"frame_commit_count": 2},
        {"commit_journal_checksum": "invalid"},
    ],
)
def test_acquisition_manifest_rejects_invalid_completion(
    overrides: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "run",
        "created_ns": 1,
        "completed_ns": 2,
        "ome_zarr_version": "0.5",
        "zarr_format": 3,
        "detector_arrays": {"camera": "detectors/camera/0"},
        "scan_count": 1,
        "pattern_ids": (1,),
        "frame_commit_count": 1,
        "commit_journal_checksum": "0" * 64,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        DpctAcquisitionManifest(**values)


def test_dpct_simulation_persists_and_replays_verified_czyx(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run-1"
    simulation = build_dpct_simulation("run-1", output_root=output)

    result = asyncio.run(simulation.run())
    replay = DpctZarrReplay(output)
    manifest = replay.verify()
    fluorescence = replay.read_detector("fluorescence")
    dhm = replay.read_detector("dhm")

    assert result.storage_uri == output.resolve().as_uri()
    assert manifest.run_id == "run-1"
    assert manifest.ome_zarr_version == "0.5"
    assert manifest.zarr_format == 3
    assert manifest.frame_commit_count == 8
    assert manifest.pattern_ids == (10, 11)
    assert fluorescence.shape == (2, 2, 8, 8)
    assert dhm.shape == (2, 2, 8, 8)
    fluorescence_identity = sum(b"fluorescence") % 256
    assert fluorescence[:, :, 0, 0].tolist() == [
        [fluorescence_identity, fluorescence_identity + 2],
        [fluorescence_identity + 1, fluorescence_identity + 3],
    ]
    assert not np.array_equal(fluorescence, dhm)

    group = zarr.open_group(output / "data.ome.zarr", mode="r")
    assert group.attrs["ome"] == {"version": "0.5"}
    aht = cast(dict[str, Any], group.attrs["aht"])
    assert aht["status"] == "complete"
    detector_group = cast(Any, group["detectors/fluorescence"])
    ome = cast(dict[str, Any], detector_group.attrs["ome"])
    assert ome["version"] == "0.5"
    assert ome["multiscales"][0]["axes"] == [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    fluorescence_array = cast(Any, group["detectors/fluorescence/0"])
    assert fluorescence_array.metadata.dimension_names == (
        "c",
        "z",
        "y",
        "x",
    )
    run_manifest = json.loads(
        (output / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["experiment"]["pattern_ids"] == [10, 11]


def test_dpct_replay_detects_array_tampering(tmp_path: Path) -> None:
    output = tmp_path / "run-tampered"
    simulation = build_dpct_simulation("run-tampered", output_root=output)
    asyncio.run(simulation.run())
    group = zarr.open_group(output / "data.ome.zarr", mode="a")
    array = cast(Any, group["detectors/fluorescence/0"])
    array[0, 0, 0, 0] = int(array[0, 0, 0, 0]) + 1

    with pytest.raises(ValueError, match="checksum mismatch"):
        DpctZarrReplay(output).verify()


def test_dpct_failure_leaves_inspectable_incomplete_bundle(tmp_path: Path) -> None:
    output = tmp_path / "run-failed"
    simulation = build_dpct_simulation("run-failed", output_root=output)
    simulation.container.build()
    detector_group = simulation.container.devices["detectors"]
    first_detector = cast(Any, detector_group).detectors[0]
    first_detector._run_id = "wrong-run"

    with pytest.raises(RuntimeError, match="run identity"):
        asyncio.run(simulation.run())

    assert (output / "run-manifest.json").is_file()
    assert not (output / "acquisition-manifest.json").exists()
    group = zarr.open_group(output / "data.ome.zarr", mode="r")
    aht = cast(dict[str, Any], group.attrs["aht"])
    assert aht["status"] == "failed"
    assert aht["failureType"] == "RuntimeError"

    with pytest.raises(ValueError, match="invalid DPCT acquisition manifest"):
        DpctZarrReplay(output).verify()


def test_dpct_store_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "existing.txt").write_text("owned", encoding="utf-8")
    simulation = build_dpct_simulation("occupied", output_root=output)

    with pytest.raises(FileExistsError, match="not empty"):
        asyncio.run(simulation.run())

    assert (output / "existing.txt").read_text(encoding="utf-8") == "owned"


def test_dpct_replay_rejects_unknown_detector(tmp_path: Path) -> None:
    output = tmp_path / "run-unknown"
    asyncio.run(build_dpct_simulation("run-unknown", output_root=output).run())

    with pytest.raises(KeyError, match="unknown detector"):
        DpctZarrReplay(output).read_detector("not-present")


def test_dpct_replay_reports_corrupt_commit_line(tmp_path: Path) -> None:
    output = tmp_path / "run-corrupt-journal"
    asyncio.run(build_dpct_simulation("run-corrupt-journal", output_root=output).run())
    commit_path = output / "frame-commits.jsonl"
    commit_path.write_text("{}\n", encoding="utf-8")
    manifest_path = output / "acquisition-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["commit_journal_checksum"] = hashlib.sha256(
        commit_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frame-commit journal at line 1"):
        DpctZarrReplay(output).verify()
