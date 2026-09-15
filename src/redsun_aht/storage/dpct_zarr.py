"""Crash-inspectable OME-Zarr v0.5 persistence for DPCT acquisitions."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import quote

import numpy as np
import zarr

from redsun_aht.storage.manifest import RunManifest, RunManifestStore

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import numpy.typing as npt

    from redsun_aht.acquisition import DpctRecipe, DpctRunResult, DpctShotRecord

from redsun_aht.acquisition.dpct import ExternalAssetInfo

OME_ZARR_VERSION = "0.5"
ZARR_FORMAT: Literal[3] = 3
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "created_ns",
        "completed_ns",
        "ome_zarr_version",
        "zarr_format",
        "detector_arrays",
        "scan_count",
        "pattern_ids",
        "frame_commit_count",
        "commit_journal_checksum",
    }
)
_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "shot_index",
        "scan_index",
        "pattern_id",
        "detector_id",
        "frame_id",
        "frame_sequence",
        "configuration_revision",
        "array_path",
        "array_index",
        "shape",
        "dtype",
        "byte_order",
        "checksum",
        "source_checksum",
        "exposure_started_ns",
        "exposure_ended_ns",
        "pose_monotonic_ns",
        "pose_joints",
        "pose_joint_units",
        "pose_coordinate_frame",
    }
)


@dataclass(frozen=True, slots=True)
class FrameCommit:
    """One durable detector-frame commit and its complete provenance."""

    schema_version: int
    run_id: str
    shot_index: int
    scan_index: int
    pattern_id: int
    detector_id: str
    frame_id: str
    frame_sequence: int
    configuration_revision: str
    array_path: str
    array_index: tuple[int, int]
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    checksum: str
    source_checksum: str
    exposure_started_ns: int
    exposure_ended_ns: int
    pose_monotonic_ns: int
    pose_joints: tuple[float, ...]
    pose_joint_units: tuple[str, ...]
    pose_coordinate_frame: str

    def __post_init__(self) -> None:
        """Reject ambiguous or structurally invalid commit records."""
        if self.schema_version != 1:
            raise ValueError("only frame-commit schema version 1 is supported")
        if not all(
            (
                self.run_id,
                self.detector_id,
                self.frame_id,
                self.configuration_revision,
                self.array_path,
                self.dtype,
                self.checksum,
                self.source_checksum,
                self.pose_coordinate_frame,
            )
        ):
            raise ValueError("frame commit string identities cannot be empty")
        if any(
            value < 0
            for value in (
                self.shot_index,
                self.scan_index,
                self.pattern_id,
                self.frame_sequence,
                *self.array_index,
                self.exposure_started_ns,
                self.exposure_ended_ns,
                self.pose_monotonic_ns,
            )
        ):
            raise ValueError("frame commit indices and timestamps must be non-negative")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("frame commit shape must contain positive dimensions")
        if self.byte_order not in {"=", "<", ">", "|"}:
            raise ValueError("frame commit byte order is invalid")
        if self.exposure_ended_ns < self.exposure_started_ns:
            raise ValueError("frame commit exposure timestamps are not ordered")
        if not 0 <= self.pattern_id <= 0xFFFF:
            raise ValueError("frame commit pattern ID must fit unsigned 16-bit")
        if not _is_sha256(self.checksum) or not _is_sha256(self.source_checksum):
            raise ValueError("frame commit checksums must be SHA-256 hex digests")
        if len(self.pose_joints) != len(self.pose_joint_units):
            raise ValueError("frame commit pose joints require explicit units")

    def as_json(self) -> dict[str, object]:
        """Return a standard JSON-compatible payload."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "shot_index": self.shot_index,
            "scan_index": self.scan_index,
            "pattern_id": self.pattern_id,
            "detector_id": self.detector_id,
            "frame_id": self.frame_id,
            "frame_sequence": self.frame_sequence,
            "configuration_revision": self.configuration_revision,
            "array_path": self.array_path,
            "array_index": list(self.array_index),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_order": self.byte_order,
            "checksum": self.checksum,
            "source_checksum": self.source_checksum,
            "exposure_started_ns": self.exposure_started_ns,
            "exposure_ended_ns": self.exposure_ended_ns,
            "pose_monotonic_ns": self.pose_monotonic_ns,
            "pose_joints": list(self.pose_joints),
            "pose_joint_units": list(self.pose_joint_units),
            "pose_coordinate_frame": self.pose_coordinate_frame,
        }


@dataclass(frozen=True, slots=True)
class DpctAcquisitionManifest:
    """Immutable completion marker for one fully verified DPCT bundle."""

    schema_version: int
    run_id: str
    created_ns: int
    completed_ns: int
    ome_zarr_version: str
    zarr_format: int
    detector_arrays: Mapping[str, str]
    scan_count: int
    pattern_ids: tuple[int, ...]
    frame_commit_count: int
    commit_journal_checksum: str

    def __post_init__(self) -> None:
        """Freeze mappings and validate supported storage schema versions."""
        if self.schema_version != 1:
            raise ValueError("only DPCT acquisition schema version 1 is supported")
        if self.ome_zarr_version != OME_ZARR_VERSION or self.zarr_format != 3:
            raise ValueError("unsupported OME-Zarr storage version")
        if not self.run_id or not self.commit_journal_checksum:
            raise ValueError("acquisition identity and journal checksum are required")
        if self.created_ns < 0 or self.completed_ns < self.created_ns:
            raise ValueError("acquisition timestamps are not ordered")
        if self.scan_count <= 0 or not self.pattern_ids:
            raise ValueError("acquisition dimensions must be positive")
        if self.frame_commit_count <= 0 or not self.detector_arrays:
            raise ValueError("completed acquisition must contain detector frames")
        if len(set(self.pattern_ids)) != len(self.pattern_ids) or any(
            not 0 <= pattern_id <= 0xFFFF for pattern_id in self.pattern_ids
        ):
            raise ValueError("manifest pattern IDs must be unique unsigned 16-bit")
        if any(
            not detector_id or not path
            for detector_id, path in self.detector_arrays.items()
        ):
            raise ValueError("manifest detector identities and paths cannot be empty")
        if len(set(self.detector_arrays.values())) != len(self.detector_arrays):
            raise ValueError("manifest detector array paths must be unique")
        expected_count = (
            len(self.detector_arrays) * self.scan_count * len(self.pattern_ids)
        )
        if self.frame_commit_count != expected_count:
            raise ValueError("manifest frame count does not cover all coordinates")
        if not _is_sha256(self.commit_journal_checksum):
            raise ValueError("manifest journal checksum must be a SHA-256 hex digest")
        object.__setattr__(
            self, "detector_arrays", MappingProxyType(dict(self.detector_arrays))
        )

    def as_json(self) -> dict[str, object]:
        """Return a standard JSON-compatible payload."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_ns": self.created_ns,
            "completed_ns": self.completed_ns,
            "ome_zarr_version": self.ome_zarr_version,
            "zarr_format": self.zarr_format,
            "detector_arrays": dict(self.detector_arrays),
            "scan_count": self.scan_count,
            "pattern_ids": list(self.pattern_ids),
            "frame_commit_count": self.frame_commit_count,
            "commit_journal_checksum": self.commit_journal_checksum,
        }


class DpctZarrStore:
    """Write detector-separated CZYX arrays with append-only frame commits."""

    def __init__(self, root: Path, run_manifest: RunManifest) -> None:
        self.root = root
        self.run_manifest = run_manifest
        self.zarr_path = root / "data.ome.zarr"
        self.commit_path = root / "frame-commits.jsonl"
        self.manifest_path = root / "acquisition-manifest.json"
        self._created_ns = time.time_ns()
        self._group: Any | None = None
        self._recipe: DpctRecipe | None = None
        self._detector_paths: dict[str, str] = {}
        self._arrays: dict[str, Any] = {}
        self._scan_ordinals: dict[int, int] = {}
        self._pattern_ordinals: dict[int, int] = {}
        self._committed: set[tuple[str, int, int]] = set()
        self._complete = False
        self._lock = RLock()

    @property
    def uri(self) -> str:
        """Return the absolute run-bundle URI."""
        return self.root.resolve().as_uri()

    def external_asset_info(self, detector_id: str) -> ExternalAssetInfo:
        """Describe one detector array without routing frames through RedSun."""
        with self._lock:
            if detector_id not in self._detector_paths:
                raise KeyError(f"DPCT detector is not declared: {detector_id}")
            array_path = self._detector_paths[detector_id]
            array = self._arrays.get(detector_id)
            if array is None:
                raise RuntimeError(
                    "DPCT external asset is unavailable before its first frame"
                )
            recipe = self._recipe
            if recipe is None:  # pragma: no cover - array implies begun store
                raise RuntimeError("DPCT Zarr store has not begun")
            return ExternalAssetInfo(
                spec="AHT_OME_ZARR_DPCT_V1",
                root=str(self.root.resolve()),
                resource_path=self.zarr_path.name,
                resource_kwargs={
                    "schema_version": 1,
                    "array_path": array_path,
                    "dimension_names": ["pattern", "scan", "y", "x"],
                    "pattern_ids": list(recipe.pattern_ids),
                    "scan_indices": [point.index for point in recipe.scan_points],
                    "shape": list(array.shape),
                    "chunks": list(array.chunks),
                    "dtype": str(array.dtype),
                },
                path_semantics="windows" if os.name == "nt" else "posix",
            )

    def begin(self, recipe: DpctRecipe, detector_ids: tuple[str, ...]) -> None:
        """Create metadata and coordinate arrays for a fresh incomplete run."""
        with self._lock:
            self._begin(recipe, detector_ids)

    def _begin(self, recipe: DpctRecipe, detector_ids: tuple[str, ...]) -> None:
        if self._group is not None:
            raise RuntimeError("DPCT Zarr store has already begun")
        if recipe.run_id != self.run_manifest.run_id:
            raise ValueError("recipe and run manifest identities do not match")
        if not detector_ids or len(set(detector_ids)) != len(detector_ids):
            raise ValueError("detector identities must be non-empty and unique")
        if self.root.exists() and not self.root.is_dir():
            raise FileExistsError(f"DPCT output bundle is not a directory: {self.root}")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"DPCT output bundle is not empty: {self.root}")
        scan_indices = tuple(point.index for point in recipe.scan_points)
        if len(set(scan_indices)) != len(scan_indices):
            raise ValueError("DPCT scan indices must be unique")
        axis_names = tuple(name for name, _ in recipe.scan_points[0].positions)
        if not axis_names or any(
            tuple(name for name, _ in point.positions) != axis_names
            for point in recipe.scan_points
        ):
            raise ValueError("DPCT scan points must share ordered axes")

        self.root.mkdir(parents=True, exist_ok=True)
        RunManifestStore(self.root / "run-manifest.json").write(self.run_manifest)

        self._recipe = recipe
        self._scan_ordinals = {
            scan_index: ordinal for ordinal, scan_index in enumerate(scan_indices)
        }
        self._pattern_ordinals = {
            pattern_id: ordinal for ordinal, pattern_id in enumerate(recipe.pattern_ids)
        }
        self._detector_paths = {
            detector_id: f"detectors/{quote(detector_id, safe='')}/0"
            for detector_id in detector_ids
        }
        group = zarr.open_group(self.zarr_path, mode="w", zarr_format=ZARR_FORMAT)
        self._group = group
        group.attrs.update(
            {
                "ome": {"version": OME_ZARR_VERSION},
                "aht": {
                    "schemaVersion": 1,
                    "runId": recipe.run_id,
                    "status": "writing",
                    "logicalAxes": ["z", "c", "y", "x"],
                    "storedAxes": ["c", "z", "y", "x"],
                    "detectorArrays": self._detector_paths,
                },
            }
        )
        coordinates = group.create_group("coordinates")
        coordinates.attrs.update({"axisNames": list(axis_names)})
        coordinates.create_array(
            "scan_index",
            data=np.asarray(scan_indices, dtype=np.int64),
            chunks=(len(scan_indices),),
            dimension_names=("z",),
        )
        coordinates.create_array(
            "pattern_id",
            data=np.asarray(recipe.pattern_ids, dtype=np.uint16),
            chunks=(len(recipe.pattern_ids),),
            dimension_names=("c",),
        )
        requested = np.asarray(
            [[value for _, value in point.positions] for point in recipe.scan_points],
            dtype=np.float64,
        )
        coordinates.create_array(
            "requested_position",
            data=requested,
            chunks=(1, requested.shape[1]),
            dimension_names=("z", "joint"),
        )
        pose_shape = (len(scan_indices), len(recipe.pattern_ids), requested.shape[1])
        coordinates.create_array(
            "measured_position",
            shape=pose_shape,
            chunks=(1, 1, requested.shape[1]),
            dtype="float64",
            fill_value=np.nan,
            dimension_names=("z", "c", "joint"),
        )
        coordinates.create_array(
            "pose_uncertainty",
            shape=pose_shape,
            chunks=(1, 1, requested.shape[1]),
            dtype="float64",
            fill_value=np.nan,
            dimension_names=("z", "c", "joint"),
        )
        coordinates.create_array(
            "pose_monotonic_ns",
            shape=(len(scan_indices), len(recipe.pattern_ids)),
            chunks=(1, 1),
            dtype="int64",
            fill_value=0,
            dimension_names=("z", "c"),
        )

    def write_shot(
        self,
        shot: DpctShotRecord,
        arrays: tuple[npt.NDArray[Any], ...],
    ) -> None:
        """Write, read back, checksum, journal, and then expose one shot."""
        with self._lock:
            self._write_shot(shot, arrays)

    def _write_shot(
        self,
        shot: DpctShotRecord,
        arrays: tuple[npt.NDArray[Any], ...],
    ) -> None:
        group, recipe = self._require_writing()
        if len(shot.frames) != len(arrays):
            raise ValueError("shot frame and payload counts do not match")
        scan_ordinal = self._scan_ordinals.get(shot.scan_index)
        pattern_ordinal = self._pattern_ordinals.get(shot.pattern_id)
        if scan_ordinal is None or pattern_ordinal is None:
            raise ValueError("shot coordinates are not declared by the recipe")
        if shot.pose.coordinate_frame == "" or len(shot.pose.joints) != len(
            recipe.scan_points[0].positions
        ):
            raise ValueError("shot pose does not match declared scan axes")

        coordinates = group["coordinates"]
        coordinates["measured_position"][scan_ordinal, pattern_ordinal, :] = (
            shot.pose.joints
        )
        coordinates["pose_uncertainty"][scan_ordinal, pattern_ordinal, :] = (
            shot.pose.uncertainty
        )
        coordinates["pose_monotonic_ns"][scan_ordinal, pattern_ordinal] = (
            shot.pose.monotonic_ns
        )
        coordinates.attrs.update(
            {
                "jointUnits": list(shot.pose.joint_units),
                "poseCoordinateFrame": shot.pose.coordinate_frame,
                "poseSource": shot.pose.source,
            }
        )

        for frame, raw_array in zip(shot.frames, arrays, strict=True):
            key = (frame.detector_id, shot.scan_index, shot.pattern_id)
            if key in self._committed:
                raise ValueError(f"duplicate DPCT frame commit: {key!r}")
            payload = np.asarray(raw_array)
            if payload.ndim != 2:
                raise ValueError("initial DPCT OME-Zarr layout requires 2-D frames")
            source_checksum = hashlib.sha256(payload.tobytes(order="C")).hexdigest()
            if source_checksum != frame.array.checksum:
                raise ValueError("source payload checksum does not match frame")
            array = self._array_for(frame.detector_id, payload)
            array[pattern_ordinal, scan_ordinal, :, :] = payload
            stored = np.asarray(array[pattern_ordinal, scan_ordinal, :, :])
            stored_checksum = hashlib.sha256(stored.tobytes(order="C")).hexdigest()
            if stored_checksum != source_checksum:
                raise OSError("OME-Zarr frame readback checksum mismatch")
            commit = FrameCommit(
                schema_version=1,
                run_id=frame.run_id,
                shot_index=shot.shot_index,
                scan_index=shot.scan_index,
                pattern_id=shot.pattern_id,
                detector_id=frame.detector_id,
                frame_id=frame.frame_id,
                frame_sequence=frame.sequence,
                configuration_revision=frame.configuration_revision,
                array_path=self._detector_paths[frame.detector_id],
                array_index=(pattern_ordinal, scan_ordinal),
                shape=tuple(payload.shape),
                dtype=str(payload.dtype),
                byte_order=_canonical_byte_order(payload.dtype),
                checksum=stored_checksum,
                source_checksum=frame.array.checksum,
                exposure_started_ns=frame.exposure_started_ns,
                exposure_ended_ns=frame.exposure_ended_ns,
                pose_monotonic_ns=shot.pose.monotonic_ns,
                pose_joints=shot.pose.joints,
                pose_joint_units=shot.pose.joint_units,
                pose_coordinate_frame=shot.pose.coordinate_frame,
            )
            self._append_commit(commit)
            self._committed.add(key)

    def complete(self, result: DpctRunResult) -> None:
        """Verify complete coverage and atomically publish the final manifest."""
        with self._lock:
            self._complete_run(result)

    def _complete_run(self, result: DpctRunResult) -> None:
        group, recipe = self._require_writing()
        if result.run_id != recipe.run_id:
            raise ValueError("result and storage run identities do not match")
        expected = {
            (detector_id, point.index, pattern_id)
            for detector_id in result.detector_ids
            for point in recipe.scan_points
            for pattern_id in recipe.pattern_ids
        }
        if self._committed != expected:
            missing = expected.difference(self._committed)
            raise RuntimeError(
                f"DPCT storage cannot complete; missing {len(missing)} frames"
            )
        aht_metadata = cast("dict[str, object]", dict(group.attrs["aht"]))
        aht_metadata["status"] = "complete"
        group.attrs["aht"] = aht_metadata
        journal_checksum = hashlib.sha256(self.commit_path.read_bytes()).hexdigest()
        manifest = DpctAcquisitionManifest(
            schema_version=1,
            run_id=result.run_id,
            created_ns=self._created_ns,
            completed_ns=max(time.time_ns(), self._created_ns),
            ome_zarr_version=OME_ZARR_VERSION,
            zarr_format=ZARR_FORMAT,
            detector_arrays=self._detector_paths,
            scan_count=len(recipe.scan_points),
            pattern_ids=recipe.pattern_ids,
            frame_commit_count=len(self._committed),
            commit_journal_checksum=journal_checksum,
        )
        _write_json_atomic(self.manifest_path, manifest.as_json())
        self._complete = True

    def fail(self, error: BaseException) -> None:
        """Record an incomplete status while deliberately omitting completion."""
        with self._lock:
            self._fail(error)

    def _fail(self, error: BaseException) -> None:
        if self._group is None or self._complete:
            return
        aht_metadata = cast("dict[str, object]", dict(self._group.attrs["aht"]))
        aht_metadata.update(
            {"status": "failed", "failureType": type(error).__qualname__}
        )
        self._group.attrs["aht"] = aht_metadata

    def _array_for(self, detector_id: str, payload: npt.NDArray[Any]) -> Any:
        group, recipe = self._require_writing()
        array = self._arrays.get(detector_id)
        if array is not None:
            if tuple(array.shape[2:]) != payload.shape or np.dtype(
                array.dtype
            ) != np.dtype(payload.dtype):
                raise ValueError("detector frame layout changed during the run")
            return array
        array_path = self._detector_paths.get(detector_id)
        if array_path is None:
            raise ValueError(f"undeclared detector identity: {detector_id}")
        detector_group_path = array_path.rsplit("/", 1)[0]
        detector_group = group.create_group(detector_group_path)
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]
        detector_group.attrs["ome"] = {
            "version": OME_ZARR_VERSION,
            "multiscales": [
                {
                    "name": detector_id,
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]}
                            ],
                        }
                    ],
                }
            ],
        }
        detector_group.attrs["aht"] = {
            "detectorId": detector_id,
            "patternIds": list(recipe.pattern_ids),
            "scanIndices": [point.index for point in recipe.scan_points],
        }
        array = detector_group.create_array(
            "0",
            shape=(
                len(recipe.pattern_ids),
                len(recipe.scan_points),
                *payload.shape,
            ),
            chunks=(1, 1, *payload.shape),
            dtype=payload.dtype,
            dimension_names=("c", "z", "y", "x"),
        )
        self._arrays[detector_id] = array
        return array

    def _append_commit(self, commit: FrameCommit) -> None:
        line = json.dumps(commit.as_json(), separators=(",", ":"), sort_keys=True)
        with self.commit_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _require_writing(self) -> tuple[Any, DpctRecipe]:
        if self._group is None or self._recipe is None:
            raise RuntimeError("DPCT Zarr store has not begun")
        if self._complete:
            raise RuntimeError("DPCT Zarr store is already complete")
        return self._group, self._recipe


class DpctZarrReplay:
    """Verify and read one completed detector-separated DPCT bundle."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.zarr_path = root / "data.ome.zarr"
        self.commit_path = root / "frame-commits.jsonl"
        self.manifest_path = root / "acquisition-manifest.json"

    def verify(self) -> DpctAcquisitionManifest:
        """Verify manifests, journal identity, array coverage, and frame hashes."""
        manifest = self._read_manifest()
        run_manifest = RunManifestStore(self.root / "run-manifest.json").read()
        if run_manifest.run_id != manifest.run_id:
            raise ValueError("run and acquisition manifest identities differ")
        journal_checksum = hashlib.sha256(self.commit_path.read_bytes()).hexdigest()
        if journal_checksum != manifest.commit_journal_checksum:
            raise ValueError("frame-commit journal checksum mismatch")
        commits = self._read_commits()
        if len(commits) != manifest.frame_commit_count:
            raise ValueError("frame-commit count does not match manifest")
        group = zarr.open_group(self.zarr_path, mode="r")
        aht_metadata = cast("Mapping[str, object]", group.attrs["aht"])
        if (
            aht_metadata.get("status") != "complete"
            or aht_metadata.get("runId") != manifest.run_id
        ):
            raise ValueError("OME-Zarr hierarchy is not a completed matching run")
        if aht_metadata.get("detectorArrays") != dict(manifest.detector_arrays):
            raise ValueError("OME-Zarr detector mapping does not match manifest")
        scan_array = cast("Any", group["coordinates/scan_index"])
        pattern_array = cast("Any", group["coordinates/pattern_id"])
        scan_indices = tuple(int(value) for value in np.asarray(scan_array[:]).tolist())
        pattern_ids = tuple(
            int(value) for value in np.asarray(pattern_array[:]).tolist()
        )
        if (
            len(scan_indices) != manifest.scan_count
            or len(set(scan_indices)) != len(scan_indices)
            or pattern_ids != manifest.pattern_ids
        ):
            raise ValueError("OME-Zarr coordinates do not match manifest")
        scan_ordinals = {
            scan_index: ordinal for ordinal, scan_index in enumerate(scan_indices)
        }
        pattern_ordinals = {
            pattern_id: ordinal for ordinal, pattern_id in enumerate(pattern_ids)
        }
        observed: set[tuple[str, int, int]] = set()
        for commit in commits:
            if commit.run_id != manifest.run_id:
                raise ValueError("frame commit belongs to another run")
            key = (commit.detector_id, commit.scan_index, commit.pattern_id)
            if key in observed:
                raise ValueError("duplicate frame commit")
            observed.add(key)
            if (
                manifest.detector_arrays.get(commit.detector_id) != commit.array_path
                or scan_ordinals.get(commit.scan_index) is None
                or pattern_ordinals.get(commit.pattern_id) is None
                or commit.array_index
                != (
                    pattern_ordinals[commit.pattern_id],
                    scan_ordinals[commit.scan_index],
                )
            ):
                raise ValueError("frame commit coordinates do not match manifest")
            array = cast("Any", group[commit.array_path])
            stored = np.asarray(array[*commit.array_index, :, :])
            checksum = hashlib.sha256(stored.tobytes(order="C")).hexdigest()
            if checksum != commit.checksum or commit.checksum != commit.source_checksum:
                raise ValueError("stored DPCT frame checksum mismatch")
            if (
                tuple(stored.shape) != commit.shape
                or str(stored.dtype) != commit.dtype
                or _canonical_byte_order(stored.dtype) != commit.byte_order
            ):
                raise ValueError("stored DPCT frame layout does not match commit")
        expected_count = (
            len(manifest.detector_arrays)
            * manifest.scan_count
            * len(manifest.pattern_ids)
        )
        if len(observed) != expected_count:
            raise ValueError("completed DPCT bundle does not cover every frame")
        return manifest

    def read_detector(self, detector_id: str) -> npt.NDArray[Any]:
        """Return a verified detector's complete CZYX array copy."""
        manifest = self.verify()
        try:
            array_path = manifest.detector_arrays[detector_id]
        except KeyError as error:
            raise KeyError(f"unknown detector: {detector_id}") from error
        group = zarr.open_group(self.zarr_path, mode="r")
        array = cast("Any", group[array_path])
        return np.asarray(array[:])

    def _read_manifest(self) -> DpctAcquisitionManifest:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
                raise TypeError("manifest fields differ")
            payload["pattern_ids"] = tuple(payload["pattern_ids"])
            return DpctAcquisitionManifest(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid DPCT acquisition manifest") from error

    def _read_commits(self) -> tuple[FrameCommit, ...]:
        commits: list[FrameCommit] = []
        try:
            with self.commit_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        payload = json.loads(line)
                        if (
                            not isinstance(payload, dict)
                            or set(payload) != _COMMIT_FIELDS
                        ):
                            raise TypeError("commit fields differ")
                        payload["array_index"] = tuple(payload["array_index"])
                        payload["shape"] = tuple(payload["shape"])
                        payload["pose_joints"] = tuple(payload["pose_joints"])
                        payload["pose_joint_units"] = tuple(payload["pose_joint_units"])
                        commits.append(FrameCommit(**payload))
                    except (
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as error:
                        raise ValueError(
                            f"invalid frame-commit journal at line {line_number}"
                        ) from error
        except OSError as error:
            raise ValueError("invalid frame-commit journal") from error
        return tuple(commits)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_byte_order(dtype: np.dtype[Any]) -> str:
    if dtype.itemsize == 1:
        return "|"
    if dtype.byteorder == "=":
        return "<" if np.little_endian else ">"
    return dtype.byteorder


__all__ = [
    "OME_ZARR_VERSION",
    "ZARR_FORMAT",
    "DpctAcquisitionManifest",
    "DpctZarrReplay",
    "DpctZarrStore",
    "FrameCommit",
]
