"""Verified OME-Zarr persistence for one detached camera-service frame."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

import numpy as np
import zarr

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_MANIFEST_NAME = "camera-capture-manifest.json"
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "service_id",
        "frame_id",
        "sequence",
        "timestamp_ns",
        "array_path",
        "shape",
        "dtype",
        "byte_order",
        "source_checksum",
        "array_checksum",
        "created_ns",
    }
)


@dataclass(frozen=True, slots=True)
class CameraCaptureManifest:
    """Immutable, single-frame capture record distinct from a DPCT recipe."""

    schema_version: int
    run_id: str
    service_id: str
    frame_id: str
    sequence: int
    timestamp_ns: int
    array_path: str
    shape: tuple[int, int, int, int]
    dtype: str
    byte_order: str
    source_checksum: str
    array_checksum: str
    created_ns: int

    def __post_init__(self) -> None:
        """Validate the immutable capture identity and CZYX layout."""
        if self.schema_version != 1:
            raise ValueError("only camera-capture schema version 1 is supported")
        if not all(
            (
                self.run_id,
                self.service_id,
                self.frame_id,
                self.array_path,
                self.dtype,
                self.source_checksum,
                self.array_checksum,
            )
        ):
            raise ValueError("camera-capture identities cannot be empty")
        if self.sequence < 0 or self.timestamp_ns < 0 or self.created_ns < 0:
            raise ValueError(
                "camera-capture counters and timestamps must be non-negative"
            )
        if self.shape[:2] != (1, 1) or any(value <= 0 for value in self.shape):
            raise ValueError("camera-capture arrays must use one C and one Z plane")
        if self.byte_order not in {"=", "<", ">", "|"}:
            raise ValueError("camera-capture byte order is invalid")
        if not _is_sha256(self.source_checksum) or not _is_sha256(self.array_checksum):
            raise ValueError("camera-capture checksums must be SHA-256 hex")

    def as_json(self) -> dict[str, object]:
        """Return a JSON-compatible immutable manifest record."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "service_id": self.service_id,
            "frame_id": self.frame_id,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "array_path": self.array_path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_order": self.byte_order,
            "source_checksum": self.source_checksum,
            "array_checksum": self.array_checksum,
            "created_ns": self.created_ns,
        }


class CameraCaptureZarrStore:
    """Persist one copied camera frame without claiming DPCT acquisition semantics."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.zarr_path = root / "data.ome.zarr"
        self.manifest_path = root / _MANIFEST_NAME

    def write(
        self,
        *,
        run_id: str,
        service_id: str,
        frame_id: str,
        sequence: int,
        timestamp_ns: int,
        array: np.ndarray[Any, Any],
    ) -> CameraCaptureManifest:
        """Write, read back, checksum, and return one normalized CZYX frame."""
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"camera-capture output is not empty: {self.root}")
        payload = np.asarray(array)
        if payload.ndim != 2 or payload.size == 0:
            raise ValueError("camera-capture input must be one non-empty 2-D frame")
        if not all((run_id, service_id, frame_id)):
            raise ValueError("camera-capture run, service, and frame IDs are required")
        source_checksum = hashlib.sha256(payload.tobytes(order="C")).hexdigest()
        c_zyx = payload[np.newaxis, np.newaxis, :, :]
        array_path = f"detectors/{quote(service_id, safe='')}/0"
        self.root.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(self.zarr_path, mode="w", zarr_format=3)
        group.attrs["ahtCameraCapture"] = {
            "schemaVersion": 1,
            "runId": run_id,
            "serviceId": service_id,
            "frameId": frame_id,
            "status": "writing",
        }
        detector = group.create_group(array_path.rsplit("/", 1)[0])
        detector.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "name": service_id,
                    "axes": [
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [{"path": "0"}],
                }
            ],
        }
        stored_array = detector.create_array(
            "0",
            data=c_zyx,
            chunks=c_zyx.shape,
            dimension_names=("c", "z", "y", "x"),
        )
        stored = np.asarray(stored_array[:])
        array_checksum = hashlib.sha256(stored.tobytes(order="C")).hexdigest()
        if stored.shape != c_zyx.shape or not np.array_equal(stored, c_zyx):
            raise OSError("camera-capture OME-Zarr readback differs from source")
        group.attrs["ahtCameraCapture"] = {
            "schemaVersion": 1,
            "runId": run_id,
            "serviceId": service_id,
            "frameId": frame_id,
            "status": "complete",
        }
        if stored.ndim != 4:  # pragma: no cover - construction invariant
            raise OSError("camera-capture storage lost its CZYX axes")
        manifest = CameraCaptureManifest(
            schema_version=1,
            run_id=run_id,
            service_id=service_id,
            frame_id=frame_id,
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            array_path=array_path,
            shape=(
                int(stored.shape[0]),
                int(stored.shape[1]),
                int(stored.shape[2]),
                int(stored.shape[3]),
            ),
            dtype=str(stored.dtype),
            byte_order=_canonical_byte_order(stored.dtype),
            source_checksum=source_checksum,
            array_checksum=array_checksum,
            created_ns=time.time_ns(),
        )
        _write_json_atomic(self.manifest_path, manifest.as_json())
        return manifest


class CameraCaptureZarrReplay:
    """Verify and read one completed single-camera capture bundle."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.zarr_path = root / "data.ome.zarr"
        self.manifest_path = root / _MANIFEST_NAME

    def verify(self) -> CameraCaptureManifest:
        """Verify manifest metadata and source/stored checksums."""
        manifest = self._read_manifest()
        group = zarr.open_group(self.zarr_path, mode="r")
        metadata = cast("Mapping[str, object]", group.attrs["ahtCameraCapture"])
        if (
            metadata.get("status") != "complete"
            or metadata.get("runId") != manifest.run_id
            or metadata.get("serviceId") != manifest.service_id
            or metadata.get("frameId") != manifest.frame_id
        ):
            raise ValueError("OME-Zarr hierarchy is not a matching completed capture")
        stored = np.asarray(cast("Any", group[manifest.array_path])[:])
        checksum = hashlib.sha256(stored.tobytes(order="C")).hexdigest()
        if (
            tuple(stored.shape) != manifest.shape
            or str(stored.dtype) != manifest.dtype
            or _canonical_byte_order(stored.dtype) != manifest.byte_order
            or checksum != manifest.array_checksum
            or hashlib.sha256(stored[0, 0].tobytes(order="C")).hexdigest()
            != manifest.source_checksum
        ):
            raise ValueError("camera-capture array integrity check failed")
        return manifest

    def read(self) -> np.ndarray[Any, Any]:
        """Return a verified CZYX array copy."""
        manifest = self.verify()
        group = zarr.open_group(self.zarr_path, mode="r")
        return np.asarray(cast("Any", group[manifest.array_path])[:])

    def _read_manifest(self) -> CameraCaptureManifest:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
                raise TypeError("camera-capture manifest fields differ")
            payload["shape"] = tuple(payload["shape"])
            return CameraCaptureManifest(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid camera-capture manifest") from error


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
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


__all__ = ["CameraCaptureManifest", "CameraCaptureZarrReplay", "CameraCaptureZarrStore"]
