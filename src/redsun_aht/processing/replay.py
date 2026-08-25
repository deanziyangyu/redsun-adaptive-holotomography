"""Verified committed-observation assembly for offline DPCT replay."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit

import numpy as np
import zarr

from redsun_aht.domain import ArrayReference, Observation
from redsun_aht.storage import CameraCaptureZarrReplay, DpctZarrReplay

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


def observations_from_dpct_bundle(root: Path) -> tuple[Observation, ...]:
    """Verify one run bundle and expose one immutable batch per detector."""
    replay = DpctZarrReplay(root)
    manifest = replay.verify()
    group = zarr.open_group(root / "data.ome.zarr", mode="r")
    observations: list[Observation] = []
    for sequence, detector_id in enumerate(sorted(manifest.detector_arrays)):
        array_path = manifest.detector_arrays[detector_id]
        array = cast("Any", group[array_path])
        payload = np.asarray(array[:])
        checksum = hashlib.sha256(payload.tobytes(order="C")).hexdigest()
        array_uri = f"{(root / 'data.ome.zarr').resolve().as_uri()}#{array_path}"
        replay_key = hashlib.sha256(
            f"{manifest.run_id}\0{detector_id}\0{checksum}".encode()
        ).hexdigest()
        observations.append(
            Observation(
                run_id=manifest.run_id,
                observation_id=f"{manifest.run_id}:{detector_id}:batch",
                detector_id=detector_id,
                channel_id="dpct-pattern-cycle",
                sequence=sequence,
                array=ArrayReference(
                    uri=array_uri,
                    checksum=checksum,
                    shape=tuple(payload.shape),
                    dtype=str(payload.dtype),
                    byte_order=_canonical_byte_order(payload.dtype),
                ),
                committed=True,
                configuration_revision=manifest.commit_journal_checksum,
                calibration_ids=(),
                replay_key=replay_key,
                latency_class="offline",
                metadata={
                    "axes": ["c", "z", "y", "x"],
                    "pattern_ids": list(manifest.pattern_ids),
                    "scan_count": manifest.scan_count,
                    "ome_zarr_version": manifest.ome_zarr_version,
                },
            )
        )
    return tuple(observations)


def observations_from_offline_bundle(root: Path) -> tuple[Observation, ...]:
    """Expose verified DPCT or explicitly single-camera offline observations."""
    if (root / "camera-capture-manifest.json").is_file():
        return observations_from_camera_capture_bundle(root)
    return observations_from_dpct_bundle(root)


def observations_from_camera_capture_bundle(root: Path) -> tuple[Observation, ...]:
    """Expose one verified CZYX frame without declaring tomography semantics."""
    replay = CameraCaptureZarrReplay(root)
    manifest = replay.verify()
    payload = replay.read()
    checksum = hashlib.sha256(payload.tobytes(order="C")).hexdigest()
    array_uri = f"{(root / 'data.ome.zarr').resolve().as_uri()}#{manifest.array_path}"
    replay_key = hashlib.sha256(
        f"{manifest.run_id}\0{manifest.service_id}\0{checksum}".encode()
    ).hexdigest()
    return (
        Observation(
            run_id=manifest.run_id,
            observation_id=f"{manifest.run_id}:{manifest.service_id}:capture",
            detector_id=manifest.service_id,
            channel_id="single-camera-frame",
            sequence=manifest.sequence,
            array=ArrayReference(
                uri=array_uri,
                checksum=checksum,
                shape=tuple(payload.shape),
                dtype=str(payload.dtype),
                byte_order=_canonical_byte_order(payload.dtype),
            ),
            committed=True,
            configuration_revision=manifest.array_checksum,
            calibration_ids=(),
            replay_key=replay_key,
            latency_class="offline",
            metadata={
                "axes": ["c", "z", "y", "x"],
                "capture_kind": "single-camera-frame",
                "source_checksum": manifest.source_checksum,
            },
        ),
    )


def read_observation_array(observation: Observation) -> npt.NDArray[Any]:
    """Read and checksum one file-backed Zarr observation reference."""
    parsed = urlsplit(observation.array.uri)
    if parsed.scheme != "file" or not parsed.fragment:
        raise ValueError("processing observation must reference a file Zarr array")
    path_text = unquote(parsed.path)
    if os.name == "nt" and path_text.startswith("/"):
        path_text = path_text[1:]
    if parsed.netloc:
        path_text = f"//{parsed.netloc}{path_text}"
    group = zarr.open_group(path_text, mode="r")
    array = cast("Any", group[parsed.fragment])
    payload = np.asarray(array[:])
    checksum = hashlib.sha256(payload.tobytes(order="C")).hexdigest()
    if (
        checksum != observation.array.checksum
        or tuple(payload.shape) != observation.array.shape
        or str(payload.dtype) != observation.array.dtype
        or _canonical_byte_order(payload.dtype) != observation.array.byte_order
    ):
        raise ValueError("processing observation array integrity check failed")
    return payload


def _canonical_byte_order(dtype: np.dtype[Any]) -> str:
    if dtype.itemsize == 1:
        return "|"
    if dtype.byteorder == "=":
        return "<" if np.little_endian else ">"
    return dtype.byteorder


__all__ = [
    "observations_from_camera_capture_bundle",
    "observations_from_dpct_bundle",
    "observations_from_offline_bundle",
    "read_observation_array",
]
