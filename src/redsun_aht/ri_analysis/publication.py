"""Atomic OME-Zarr publication and verification for RI analysis artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

if TYPE_CHECKING:
    from pathlib import Path

    from .config import RiAnalysisConfig
    from .pipeline import AnalysisArtifacts, AnalysisTable, RiTiffVolume


MANIFEST_NAME = "result-manifest.json"
BUNDLE_NAME = "analysis.ome.zarr"
BUNDLE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class PublishedAnalysis:
    """Locations created by one successful atomic publication."""

    output_root: Path
    bundle_path: Path
    manifest_path: Path


def publish_analysis(
    source: RiTiffVolume,
    artifacts: AnalysisArtifacts,
    config: RiAnalysisConfig,
    output_root: Path,
) -> PublishedAnalysis:
    """Write a fresh self-contained bundle and atomically expose it."""
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"RI analysis output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.tmp"
    checksums: dict[str, str] = {}
    try:
        staging.mkdir()
        bundle = staging / BUNDLE_NAME
        root = zarr.open_group(bundle, mode="w", zarr_format=3)
        root.attrs.update(
            {
                "aht": {
                    "schemaVersion": BUNDLE_SCHEMA_VERSION,
                    "kind": "ri_analysis",
                    "sourceChecksum": source.source_checksum,
                    "voxelSizeZYXUm": list(source.voxel_size_zyx_um),
                    "riEncoding": "uint16_ri_times_10000",
                }
            }
        )
        images = root.create_group("images")
        _write_image(
            images, "ri", artifacts.ri_zyx, source.voxel_size_zyx_um, checksums
        )
        _write_image(
            images,
            "sobel_magnitude",
            artifacts.sobel_magnitude_zyx,
            source.voxel_size_zyx_um,
            checksums,
        )
        masks = root.create_group("masks")
        for name, mask in artifacts.candidate_masks.items():
            _write_image(
                masks,
                f"candidate_{name}",
                mask.astype(np.uint8),
                source.voxel_size_zyx_um,
                checksums,
            )
        _write_image(
            masks,
            "selected_roi",
            artifacts.selected_mask_zyx.astype(np.uint8),
            source.voxel_size_zyx_um,
            checksums,
        )
        _write_image(
            masks,
            "labels_unfiltered",
            artifacts.unfiltered_labels_zyx.astype(np.uint32),
            source.voxel_size_zyx_um,
            checksums,
        )
        _write_image(
            masks,
            "labels",
            artifacts.labels_zyx.astype(np.uint32),
            source.voxel_size_zyx_um,
            checksums,
        )
        tables = root.create_group("tables")
        _write_table(tables, "roi", artifacts.roi_table, checksums)
        _write_table(tables, "grid", artifacts.grid_table, checksums)
        _write_table(tables, "labels", artifacts.label_table, checksums)
        _write_table(tables, "label_grid", artifacts.label_grid_table, checksums)
        histograms = root.create_group("histograms")
        _write_array(histograms, "bin_edges", artifacts.bin_edges, checksums)
        _write_array(histograms, "roi_counts", artifacts.roi_histogram, checksums)
        _write_array(histograms, "grid_counts", artifacts.grid_histograms, checksums)
        _write_array(histograms, "label_counts", artifacts.label_histograms, checksums)
        _write_array(
            histograms,
            "label_grid_counts",
            artifacts.label_grid_histograms,
            checksums,
        )
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "source": {
                "path": str(source.path),
                "sha256": source.source_checksum,
                "tiff_metadata": source.tiff_metadata,
                "voxel_size_zyx_um": list(source.voxel_size_zyx_um),
                "ri_encoding": "uint16_ri_times_10000",
            },
            "configuration": config.to_mapping(),
            "diagnostics": artifacts.diagnostics,
            "artifacts": checksums,
        }
        manifest_path = staging / MANIFEST_NAME
        _write_json(manifest_path, manifest)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PublishedAnalysis(
        output_root, output_root / BUNDLE_NAME, output_root / MANIFEST_NAME
    )


def verify_analysis_bundle(path: Path) -> dict[str, Any]:
    """Check manifest structure and every recorded Zarr array checksum."""
    output_root = _output_root(path)
    manifest_path = output_root / MANIFEST_NAME
    bundle_path = output_root / BUNDLE_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("RI analysis result manifest is missing or invalid") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {1, 2}:
        raise ValueError("RI analysis result manifest schema is unsupported")
    checksums = manifest.get("artifacts")
    if not isinstance(checksums, dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in checksums.items()
    ):
        raise ValueError("RI analysis result manifest artifact checksums are invalid")
    try:
        root = zarr.open_group(bundle_path, mode="r")
        tables: Any = root["tables"]
        histograms: Any = root["histograms"]
        if manifest["schema_version"] >= 2 and (
            "label_grid" not in tables or "label_grid_counts" not in histograms
        ):
            raise ValueError("RI analysis label-grid artifacts are missing")
        for name, expected in checksums.items():
            array: Any = root[name]
            actual = _array_checksum(np.asarray(array[:]))
            if actual != expected:
                raise ValueError(f"RI analysis artifact checksum mismatch: {name}")
    except (KeyError, OSError, ValueError) as error:
        if isinstance(error, ValueError) and "checksum mismatch" in str(error):
            raise
        raise ValueError("RI analysis bundle is incomplete or invalid") from error
    return manifest


def _write_image(
    parent: Any,
    name: str,
    array: np.ndarray,
    spacing: tuple[float, float, float],
    checksums: dict[str, str],
) -> None:
    image = parent.create_group(name)
    image.attrs.update(
        {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": name,
                        "axes": [
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"},
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": list(spacing)}
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    )
    key = f"{parent.path}/{name}/0"
    stored = image.create_array(
        "0",
        data=array,
        chunks=_image_chunks(array.shape),
        dimension_names=("z", "y", "x"),
    )
    checksums[key] = _array_checksum(np.asarray(stored[:]))


def _write_table(
    parent: Any, name: str, table: AnalysisTable, checksums: dict[str, str]
) -> None:
    group = parent.create_group(name)
    group.attrs.update(
        {
            "aht": {
                "table": {"columns": list(table.columns), "rowCount": table.row_count}
            }
        }
    )
    for column, values in table.columns.items():
        _write_array(group, column, values, checksums)


def _write_array(
    parent: Any, name: str, values: np.ndarray, checksums: dict[str, str]
) -> None:
    array = np.asarray(values)
    stored = parent.create_array(
        name,
        data=array,
        chunks=_array_chunks(array.shape),
        dimension_names=tuple(f"dim_{index}" for index in range(array.ndim)),
    )
    checksums[f"{parent.path}/{name}"] = _array_checksum(np.asarray(stored[:]))


def _image_chunks(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError("RI analysis image arrays must be ZYX")
    return min(shape[0], 16), min(shape[1], 256), min(shape[2], 256)


def _array_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    return tuple(max(1, min(item, 1024)) for item in shape)


def _array_checksum(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.name == BUNDLE_NAME:
        return resolved.parent
    return resolved


__all__ = [
    "BUNDLE_NAME",
    "BUNDLE_SCHEMA_VERSION",
    "MANIFEST_NAME",
    "PublishedAnalysis",
    "publish_analysis",
    "verify_analysis_bundle",
]
