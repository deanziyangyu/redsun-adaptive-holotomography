"""Durable immutable calibration artifacts and correction recipes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse
from urllib.request import url2pathname
from uuid import uuid4

import numpy as np

from redsun_aht.domain import ArrayReference, CalibrationArtifact, CalibrationKind
from redsun_aht.domain.models import thaw_json

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from redsun_aht.domain.models import JsonValue


class ArtifactIntegrityError(RuntimeError):
    """Raised when persisted calibration bytes do not match their checksum."""


class CalibrationArtifactStore:
    """Persist calibration arrays and metadata through atomic replacements."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def create(
        self,
        *,
        kind: CalibrationKind,
        array: npt.NDArray[Any],
        validity_scope: Mapping[str, JsonValue],
        source_run_id: str,
        settings: Mapping[str, JsonValue],
        created_ns: int,
    ) -> CalibrationArtifact:
        """Copy and atomically persist one immutable artifact."""
        value = np.ascontiguousarray(array)
        artifact_id = str(uuid4())
        settings_bytes = json.dumps(
            thaw_json(settings), sort_keys=True, separators=(",", ":")
        ).encode()
        fingerprint = hashlib.sha256(settings_bytes).hexdigest()
        checksum = hashlib.sha256(value.tobytes(order="C")).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        array_path = self.root / f"{artifact_id}.npy"
        self._write_array(array_path, value)
        artifact = CalibrationArtifact(
            artifact_id=artifact_id,
            kind=kind,
            validity_scope=validity_scope,
            source_run_id=source_run_id,
            settings_fingerprint=fingerprint,
            created_ns=created_ns,
            payload=ArrayReference(
                uri=array_path.as_uri(),
                checksum=checksum,
                shape=value.shape,
                dtype=value.dtype.str,
                byte_order=value.dtype.byteorder,
            ),
        )
        metadata = {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind.value,
            "validity_scope": thaw_json(artifact.validity_scope),
            "source_run_id": artifact.source_run_id,
            "settings_fingerprint": artifact.settings_fingerprint,
            "created_ns": artifact.created_ns,
            "payload": {
                "uri": artifact.payload.uri,
                "checksum": artifact.payload.checksum,
                "shape": list(artifact.payload.shape),
                "dtype": artifact.payload.dtype,
                "byte_order": artifact.payload.byte_order,
            },
        }
        self._write_bytes(
            self.root / f"{artifact_id}.json",
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(),
        )
        return artifact

    def read(self, artifact_id: str) -> tuple[CalibrationArtifact, npt.NDArray[Any]]:
        """Load an artifact and verify its array checksum."""
        metadata = json.loads(
            (self.root / f"{artifact_id}.json").read_text(encoding="utf-8")
        )
        payload = metadata["payload"]
        artifact = CalibrationArtifact(
            artifact_id=metadata["artifact_id"],
            kind=CalibrationKind(metadata["kind"]),
            validity_scope=metadata["validity_scope"],
            source_run_id=metadata["source_run_id"],
            settings_fingerprint=metadata["settings_fingerprint"],
            created_ns=metadata["created_ns"],
            payload=ArrayReference(
                uri=payload["uri"],
                checksum=payload["checksum"],
                shape=tuple(payload["shape"]),
                dtype=payload["dtype"],
                byte_order=payload["byte_order"],
            ),
        )
        parsed = urlparse(artifact.payload.uri)
        array = np.load(Path(url2pathname(parsed.path)), allow_pickle=False)
        checksum = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        if checksum != artifact.payload.checksum:
            raise ArtifactIntegrityError(f"checksum mismatch for {artifact_id}")
        return artifact, array

    @staticmethod
    def _write_array(path: Path, array: npt.NDArray[Any]) -> None:
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def temporal_median(frames: Sequence[npt.NDArray[Any]]) -> npt.NDArray[Any]:
    """Compute the document-routed temporal-median recipe result."""
    if not frames:
        raise ValueError("median calibration requires at least one frame")
    stack = np.stack(frames, axis=0)
    result = np.median(stack, axis=0).astype(stack.dtype)
    return cast("npt.NDArray[Any]", result)


def apply_background(
    frame: npt.NDArray[Any], background: npt.NDArray[Any]
) -> npt.NDArray[np.float32]:
    """Divide raw data by a background, mapping zero background pixels to one."""
    if frame.shape != background.shape:
        raise ValueError("raw frame and background shapes differ")
    result = np.divide(
        frame,
        background,
        out=np.ones_like(frame, dtype=np.float32),
        where=background != 0,
    )
    return cast("npt.NDArray[np.float32]", result)


__all__ = [
    "ArtifactIntegrityError",
    "CalibrationArtifactStore",
    "apply_background",
    "temporal_median",
]
