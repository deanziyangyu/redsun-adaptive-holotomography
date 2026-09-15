"""Non-authoritative TIFF and CSV exports from verified RI analysis bundles."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Any

import numpy as np
import tifffile
import zarr

from .publication import BUNDLE_NAME, verify_analysis_bundle

if TYPE_CHECKING:
    from pathlib import Path


def export_analysis_bundle(
    input_path: Path,
    output_root: Path,
    *,
    formats: tuple[str, ...] = ("tiff", "csv"),
) -> tuple[Path, ...]:
    """Export selected inspection formats after integrity verification."""
    requested = set(formats)
    unknown = requested - {"tiff", "csv"}
    if unknown:
        raise ValueError(f"unsupported RI analysis export formats: {sorted(unknown)}")
    if not requested:
        raise ValueError("at least one RI analysis export format is required")
    manifest = verify_analysis_bundle(input_path)
    analysis_root = input_path.resolve()
    if analysis_root.name == BUNDLE_NAME:
        analysis_root = analysis_root.parent
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"RI analysis export output already exists: {output_root}"
        )
    output_root.mkdir(parents=True)
    bundle = zarr.open_group(analysis_root / BUNDLE_NAME, mode="r")
    spacing = _spacing(manifest)
    outputs: list[Path] = []
    if "tiff" in requested:
        tiff_dir = output_root / "tiff"
        tiff_dir.mkdir()
        for destination, array in _tiff_arrays(bundle):
            path = tiff_dir / f"{destination}.tif"
            imagej_compatible = array.dtype in {
                np.dtype("uint8"),
                np.dtype("uint16"),
                np.dtype("float32"),
            }
            tifffile.imwrite(
                path,
                array,
                imagej=imagej_compatible,
                resolution=(1.0 / spacing[2], 1.0 / spacing[1]),
                metadata={"axes": "ZYX", "unit": "um", "spacing": spacing[0]},
            )
            outputs.append(path)
    if "csv" in requested:
        csv_dir = output_root / "csv"
        csv_dir.mkdir()
        tables: Any = bundle["tables"]
        for name in ("roi", "grid", "labels", "label_grid"):
            if name not in tables:
                continue
            path = csv_dir / f"{name}.csv"
            _write_table_csv(tables[name], path)
            outputs.append(path)
        path = csv_dir / "histograms.csv"
        _write_histogram_csv(bundle["histograms"], path)
        outputs.append(path)
    return tuple(outputs)


def _spacing(manifest: dict[str, Any]) -> tuple[float, float, float]:
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("RI analysis manifest source is invalid")
    values = source.get("voxel_size_zyx_um")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("RI analysis manifest voxel size is invalid")
    spacing = tuple(float(value) for value in values)
    if any(value <= 0 for value in spacing):
        raise ValueError("RI analysis manifest voxel size must be positive")
    return spacing  # type: ignore[return-value]


def _tiff_arrays(bundle: Any) -> tuple[tuple[str, np.ndarray], ...]:
    images = bundle["images"]
    masks = bundle["masks"]
    arrays: list[tuple[str, np.ndarray]] = [
        ("ri", np.asarray(images["ri"]["0"][:])),
        ("sobel_magnitude", np.asarray(images["sobel_magnitude"]["0"][:])),
        ("selected_roi", np.asarray(masks["selected_roi"]["0"][:])),
        ("labels", np.asarray(masks["labels"]["0"][:])),
    ]
    if "labels_unfiltered" in masks:
        arrays.append(
            ("labels_unfiltered", np.asarray(masks["labels_unfiltered"]["0"][:]))
        )
    arrays.extend(
        (f"candidate_{name}", np.asarray(masks[f"candidate_{name}"]["0"][:]))
        for name in ("ri_only", "edge_only", "hybrid")
    )
    return tuple(arrays)


def _write_table_csv(group: Any, path: Path) -> None:
    metadata = group.attrs.get("aht")
    if not isinstance(metadata, dict):
        raise ValueError("RI analysis table metadata is invalid")
    table = metadata.get("table")
    if not isinstance(table, dict) or not isinstance(table.get("columns"), list):
        raise ValueError("RI analysis table schema is invalid")
    columns = tuple(str(item) for item in table["columns"])
    values = {name: np.asarray(group[name][:]) for name in columns}
    row_count = len(next(iter(values.values()), ()))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index in range(row_count):
            writer.writerow({name: _scalar(values[name][index]) for name in columns})


def _write_histogram_csv(group: Any, path: Path) -> None:
    edges = np.asarray(group["bin_edges"][:])
    roi = np.asarray(group["roi_counts"][:])
    grid = np.asarray(group["grid_counts"][:])
    labels = np.asarray(group["label_counts"][:])
    label_grid = (
        np.asarray(group["label_grid_counts"][:])
        if "label_grid_counts" in group
        else np.empty((0, len(edges) - 1), dtype=np.uint64)
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "population",
                "row_id",
                "bin_index",
                "ri_start",
                "ri_stop",
                "count",
            ),
        )
        writer.writeheader()
        for index, count in enumerate(roi):
            writer.writerow(_histogram_row("roi", 0, index, edges, count))
        for row_id, counts in enumerate(grid):
            for index, count in enumerate(counts):
                writer.writerow(_histogram_row("grid", row_id, index, edges, count))
        for row_id, counts in enumerate(labels):
            for index, count in enumerate(counts):
                writer.writerow(_histogram_row("label", row_id, index, edges, count))
        for row_id, counts in enumerate(label_grid):
            for index, count in enumerate(counts):
                writer.writerow(
                    _histogram_row("label_grid", row_id, index, edges, count)
                )


def _histogram_row(
    population: str, row_id: int, index: int, edges: np.ndarray, count: Any
) -> dict[str, object]:
    return {
        "population": population,
        "row_id": row_id,
        "bin_index": index,
        "ri_start": float(edges[index]),
        "ri_stop": float(edges[index + 1]),
        "count": _scalar(count),
    }


def _scalar(value: Any) -> object:
    return value.item() if isinstance(value, np.generic) else value


__all__ = ["export_analysis_bundle"]
