"""Pure offline stages for quantitative refractive-index TIFF analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import cc3d
import numpy as np
import scipy.ndimage as ndi
import tifffile
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

from .config import RiAnalysisConfig
from .publication import publish_analysis

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class RiTiffVolume:
    """One validated donor-format TIFF and its decoded physical RI volume."""

    path: Path
    ri_zyx: np.ndarray
    voxel_size_zyx_um: tuple[float, float, float]
    source_checksum: str
    tiff_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnalysisTable:
    """Columnar, length-consistent tabular output."""

    columns: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        """Require each column to have an equal number of rows."""
        lengths = {len(value) for value in self.columns.values()}
        if len(lengths) > 1:
            raise ValueError("analysis table columns must share one length")

    @property
    def row_count(self) -> int:
        """Return the number of rows, including zero-column table handling."""
        return next(iter(len(value) for value in self.columns.values()), 0)


@dataclass(frozen=True, slots=True)
class AnalysisArtifacts:
    """In-memory results of the numerical stages before persistence."""

    ri_zyx: np.ndarray
    sobel_magnitude_zyx: np.ndarray
    candidate_masks: dict[str, np.ndarray]
    selected_mask_zyx: np.ndarray
    unfiltered_labels_zyx: np.ndarray
    labels_zyx: np.ndarray
    roi_table: AnalysisTable
    grid_table: AnalysisTable
    label_table: AnalysisTable
    label_grid_table: AnalysisTable
    bin_edges: np.ndarray
    roi_histogram: np.ndarray
    grid_histograms: np.ndarray
    label_histograms: np.ndarray
    label_grid_histograms: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RiAnalysisResult:
    """Published analysis location and compact output summary."""

    output_root: Path
    bundle_path: Path
    manifest_path: Path
    label_count: int
    grid_cell_count: int


def load_config(path: Path | None) -> RiAnalysisConfig:
    """Load an optional JSON configuration without silently accepting YAML."""
    if path is None:
        return RiAnalysisConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid RI analysis JSON config: {path}") from error
    return RiAnalysisConfig.from_mapping(raw)


def load_ri_tiff(
    path: Path,
    *,
    voxel_size_zyx_um: tuple[float, float, float] | None = None,
) -> RiTiffVolume:
    """Read a donor uint16 TIFF, optionally with an explicit physical override.

    The public analysis CLI does not supply an override and therefore continues
    to require complete ImageJ physical metadata. An override is for an
    auditable, caller-owned calibration decision such as retrospective plotting
    of known donor TIFFs whose metadata is incomplete.
    """
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"RI TIFF input is not a file: {path}")
    try:
        with tifffile.TiffFile(path) as handle:
            if len(handle.series) != 1:
                raise ValueError("RI TIFF must contain exactly one image series")
            series = handle.series[0]
            if series.axes != "ZYX":
                raise ValueError("RI TIFF must declare ImageJ ZYX axes")
            encoded = np.asarray(series.asarray())
            if encoded.ndim != 3 or encoded.dtype != np.uint16:
                raise ValueError("RI TIFF must be a three-dimensional uint16 stack")
            imagej = dict(handle.imagej_metadata or {})
            voxel_size = (
                _validated_voxel_size(voxel_size_zyx_um)
                if voxel_size_zyx_um is not None
                else _voxel_size_from_imagej(handle.pages[0], imagej)
            )
    except (OSError, tifffile.TiffFileError) as error:
        raise ValueError(f"unable to read RI TIFF: {path}") from error
    if not np.isfinite(encoded).all():  # defensive; uint16 is always finite
        raise ValueError("RI TIFF contains non-finite values")
    metadata: dict[str, Any] = {
        "series_axes": series.axes,
        "series_shape": list(encoded.shape),
        "dtype": str(encoded.dtype),
        "imagej": imagej,
        "ri_encoding": "uint16_ri_times_10000",
        "voxel_size_source": (
            "explicit_override" if voxel_size_zyx_um is not None else "imagej"
        ),
    }
    return RiTiffVolume(
        path=path,
        ri_zyx=(encoded.astype(np.float32) / np.float32(10_000.0)),
        voxel_size_zyx_um=voxel_size,
        source_checksum=_file_checksum(path),
        tiff_metadata=metadata,
    )


def analyze_volume(volume: RiTiffVolume, config: RiAnalysisConfig) -> AnalysisArtifacts:
    """Run the deterministic numerical stages without reading or writing files."""
    ri = volume.ri_zyx
    magnitude = _sobel_magnitude(ri, volume.voxel_size_zyx_um, config)
    ri_mask, threshold = _ri_candidate(ri, config)
    edge_mask, edge_threshold = _edge_candidate(
        magnitude, volume.voxel_size_zyx_um, config
    )
    candidates = {
        "ri_only": ri_mask,
        "edge_only": edge_mask,
        "hybrid": ri_mask & edge_mask,
    }
    selected = _morphology_cleanup(
        candidates[config.segmentation.strategy],
        volume.voxel_size_zyx_um,
        config,
    )
    unfiltered_labels = _label_components(
        selected, ri, volume.voxel_size_zyx_um, config
    )
    labels, filter_diagnostics = _filter_labels(unfiltered_labels, config)
    selected = labels > 0
    bin_edges = np.linspace(
        config.histogram.ri_min,
        config.histogram.ri_max,
        config.histogram.bin_count + 1,
        dtype=np.float32,
    )
    roi_values = ri[selected]
    roi_summary, roi_histogram = _summary_and_histogram(roi_values, bin_edges)
    roi_table = AnalysisTable(_table_from_rows([{"row_id": 0, **roi_summary}]))
    (
        grid_table,
        grid_histograms,
        label_grid_table,
        label_grid_histograms,
    ) = _grid_statistics(
        ri, selected, labels, volume.voxel_size_zyx_um, config, bin_edges
    )
    label_table, label_histograms = _label_statistics(
        ri, labels, volume.voxel_size_zyx_um, bin_edges
    )
    diagnostics = {
        "sobel_edge_threshold": float(edge_threshold),
        "ri_seed_threshold": float(threshold),
        "candidate_voxels": {
            name: int(mask.sum()) for name, mask in candidates.items()
        },
        "selected_voxels": int(selected.sum()),
        "unfiltered_label_count": _component_count(unfiltered_labels),
        "label_count": _component_count(labels),
        "label_grid_row_count": label_grid_table.row_count,
        "label_filter": filter_diagnostics,
        "strategy": config.segmentation.strategy,
    }
    return AnalysisArtifacts(
        ri_zyx=ri,
        sobel_magnitude_zyx=magnitude,
        candidate_masks=candidates,
        selected_mask_zyx=selected,
        unfiltered_labels_zyx=unfiltered_labels,
        labels_zyx=labels,
        roi_table=roi_table,
        grid_table=grid_table,
        label_table=label_table,
        label_grid_table=label_grid_table,
        bin_edges=bin_edges,
        roi_histogram=roi_histogram,
        grid_histograms=grid_histograms,
        label_histograms=label_histograms,
        label_grid_histograms=label_grid_histograms,
        diagnostics=diagnostics,
    )


def analyze_ri_tiff(
    input_path: Path,
    output_root: Path,
    config: RiAnalysisConfig,
) -> RiAnalysisResult:
    """Analyze one source TIFF and atomically publish a self-contained bundle."""
    source = load_ri_tiff(input_path)
    artifacts = analyze_volume(source, config)
    published = publish_analysis(source, artifacts, config, output_root)
    return RiAnalysisResult(
        output_root=published.output_root,
        bundle_path=published.bundle_path,
        manifest_path=published.manifest_path,
        label_count=_component_count(artifacts.labels_zyx),
        grid_cell_count=artifacts.grid_table.row_count,
    )


def _voxel_size_from_imagej(
    page: Any, imagej: dict[str, Any]
) -> tuple[float, float, float]:
    unit = imagej.get("unit")
    if unit not in {"um", "µm", "μm", "\\u00B5m"}:
        raise ValueError("RI TIFF ImageJ metadata must declare micrometre units")
    spacing = _positive_float(imagej.get("spacing"), "ImageJ spacing")
    x_resolution = _resolution_um_per_pixel(page, "XResolution")
    y_resolution = _resolution_um_per_pixel(page, "YResolution")
    return spacing, y_resolution, x_resolution


def _validated_voxel_size(
    voxel_size_zyx_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Validate an explicit physical Z/Y/X voxel-size calibration."""
    if len(voxel_size_zyx_um) != 3:
        raise ValueError("explicit voxel size must contain Z, Y, and X values")
    return tuple(
        _positive_float(value, f"explicit voxel size {axis}")
        for axis, value in zip("ZYX", voxel_size_zyx_um, strict=True)
    )  # type: ignore[return-value]


def _resolution_um_per_pixel(page: Any, name: str) -> float:
    tag = page.tags.get(name)
    if tag is None:
        raise ValueError(f"RI TIFF is missing {name}")
    value = tag.value
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"RI TIFF {name} must be a rational resolution")
    numerator, denominator = value
    pixels_per_um = _positive_float(numerator, name) / _positive_float(
        denominator, name
    )
    return 1.0 / pixels_per_um


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _sobel_magnitude(
    ri: np.ndarray,
    spacing: tuple[float, float, float],
    config: RiAnalysisConfig,
) -> np.ndarray:
    source = ri
    sigma_um = config.sobel.pre_smoothing_sigma_um
    if sigma_um > 0:
        sigma = tuple(sigma_um / item for item in spacing)
        source = ndi.gaussian_filter(source, sigma=sigma).astype(np.float32)
    gradient = [
        ndi.sobel(source, axis=axis, mode="nearest") / spacing[axis]
        for axis in range(3)
    ]
    squared = sum(item * item for item in gradient)
    return cast("np.ndarray", np.sqrt(squared, dtype=np.float32))


def _ri_candidate(ri: np.ndarray, config: RiAnalysisConfig) -> tuple[np.ndarray, float]:
    segmentation = config.segmentation
    if segmentation.seed_mode == "manual_window":
        if segmentation.ri_lower is None or segmentation.ri_upper is None:
            raise AssertionError("validated manual window is missing bounds")
        mask = (ri >= segmentation.ri_lower) & (ri <= segmentation.ri_upper)
        return mask, float(segmentation.ri_lower)
    threshold = float(threshold_otsu(ri))  # type: ignore[no-untyped-call]
    return ri >= threshold, threshold


def _edge_candidate(
    magnitude: np.ndarray,
    spacing: tuple[float, float, float],
    config: RiAnalysisConfig,
) -> tuple[np.ndarray, float]:
    threshold = float(np.percentile(magnitude, config.sobel.edge_percentile))
    edges = magnitude >= threshold
    footprint = _physical_footprint(config.sobel.edge_close_radius_um, spacing)
    closed = ndi.binary_closing(edges, structure=footprint)
    return ndi.binary_fill_holes(closed), threshold


def _morphology_cleanup(
    candidate: np.ndarray,
    spacing: tuple[float, float, float],
    config: RiAnalysisConfig,
) -> np.ndarray:
    segmentation = config.segmentation
    cleaned = candidate
    if segmentation.close_radius_um > 0:
        cleaned = ndi.binary_closing(
            cleaned,
            structure=_physical_footprint(segmentation.close_radius_um, spacing),
        )
    if segmentation.open_radius_um > 0:
        cleaned = ndi.binary_opening(
            cleaned,
            structure=_physical_footprint(segmentation.open_radius_um, spacing),
        )
    minimum_voxels = int(
        np.ceil(segmentation.min_component_volume_um3 / np.prod(spacing))
    )
    cleaned = remove_small_objects(  # type: ignore[no-untyped-call]
        cleaned, min_size=max(1, minimum_voxels)
    )
    if segmentation.dilation_radius_um > 0:
        cleaned = ndi.binary_dilation(
            cleaned,
            structure=_physical_footprint(segmentation.dilation_radius_um, spacing),
        )
    return np.asarray(cleaned, dtype=bool)


def _physical_footprint(
    radius_um: float, spacing: tuple[float, float, float]
) -> np.ndarray:
    if radius_um <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    radii = [max(1, int(np.ceil(radius_um / item))) for item in spacing]
    coordinates = np.ogrid[tuple(slice(-item, item + 1) for item in radii)]
    scaled = sum(
        (coordinate * spacing[index] / radius_um) ** 2
        for index, coordinate in enumerate(coordinates)
    )
    return np.asarray(scaled <= 1.0, dtype=bool)


def _label_components(
    mask: np.ndarray,
    ri: np.ndarray,
    spacing: tuple[float, float, float],
    config: RiAnalysisConfig,
) -> np.ndarray:
    labels = np.asarray(
        cc3d.connected_components(mask, connectivity=config.segmentation.connectivity),
        dtype=np.uint32,
    )
    maximum = config.segmentation.refinement_max_component_voxels
    if maximum is None or not labels.any():
        return labels
    stats = cc3d.statistics(labels)
    counts = np.asarray(stats["voxel_counts"], dtype=np.int64)
    boxes = stats["bounding_boxes"]
    refined = labels > 0
    window = tuple(
        max(1, round(config.segmentation.refinement_window_um / item))
        for item in spacing
    )
    for label_id in range(1, len(counts)):
        if counts[label_id] <= maximum:
            continue
        box = boxes[label_id]
        if box is None:
            continue
        region = labels[box] == label_id
        local_mean = ndi.uniform_filter(ri[box], size=window, mode="nearest")
        split = region & (
            ri[box] > local_mean * config.segmentation.refinement_multiplier
        )
        if (
            int(
                cc3d.connected_components(
                    split, connectivity=config.segmentation.connectivity
                ).max()
            )
            > 1
        ):
            target = refined[box]
            target[region] = split[region]
            refined[box] = target
    return np.asarray(
        cc3d.connected_components(
            refined, connectivity=config.segmentation.connectivity
        ),
        dtype=np.uint32,
    )


def _filter_labels(
    labels: np.ndarray, config: RiAnalysisConfig
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply auditable inclusive component-size bounds without relabeling IDs."""
    counts = np.bincount(labels.ravel())
    present = np.flatnonzero(counts)[1:]
    label_filter = config.label_filter
    keep = present[counts[present] >= label_filter.min_voxel_count]
    if label_filter.max_voxel_count is not None:
        keep = keep[counts[keep] <= label_filter.max_voxel_count]
    filtered = np.where(np.isin(labels, keep), labels, 0).astype(np.uint32)
    return filtered, {
        "input_labels": len(present),
        "retained_labels": len(keep),
        "removed_labels": int(len(present) - len(keep)),
    }


def _component_count(labels: np.ndarray) -> int:
    """Return the count of present non-background labels even when IDs have gaps."""
    return int(np.count_nonzero(np.bincount(labels.ravel())[1:]))


def _summary_and_histogram(
    values: np.ndarray, bin_edges: np.ndarray
) -> tuple[dict[str, float | int], np.ndarray]:
    histogram, _ = np.histogram(values, bins=bin_edges)
    if values.size == 0:
        return (
            {
                "voxel_count": 0,
                "ri_mean": float("nan"),
                "ri_std": float("nan"),
                "ri_min": float("nan"),
                "ri_max": float("nan"),
                "ri_median": float("nan"),
                "ri_p05": float("nan"),
                "ri_p95": float("nan"),
                "histogram_underflow": 0,
                "histogram_overflow": 0,
            },
            histogram.astype(np.uint64),
        )
    return (
        {
            "voxel_count": int(values.size),
            "ri_mean": float(np.mean(values)),
            "ri_std": float(np.std(values)),
            "ri_min": float(np.min(values)),
            "ri_max": float(np.max(values)),
            "ri_median": float(np.median(values)),
            "ri_p05": float(np.percentile(values, 5)),
            "ri_p95": float(np.percentile(values, 95)),
            "histogram_underflow": int(np.count_nonzero(values < bin_edges[0])),
            "histogram_overflow": int(np.count_nonzero(values > bin_edges[-1])),
        },
        histogram.astype(np.uint64),
    )


def _grid_statistics(
    ri: np.ndarray,
    roi: np.ndarray,
    labels: np.ndarray,
    spacing: tuple[float, float, float],
    config: RiAnalysisConfig,
    bin_edges: np.ndarray,
) -> tuple[AnalysisTable, np.ndarray, AnalysisTable, np.ndarray]:
    points = np.argwhere(roi)
    if points.size == 0:
        empty_histograms = np.empty((0, len(bin_edges) - 1), np.uint64)
        return (
            AnalysisTable(_empty_grid_columns()),
            empty_histograms,
            AnalysisTable(_empty_label_grid_columns()),
            empty_histograms.copy(),
        )
    stride = tuple(
        max(1, round(size / voxel))
        for size, voxel in zip(config.grid.cell_size_zyx_um, spacing, strict=True)
    )
    lower = (points.min(axis=0) // np.asarray(stride)) * np.asarray(stride)
    upper = points.max(axis=0) + 1
    rows: list[dict[str, float | int]] = []
    histograms: list[np.ndarray] = []
    label_grid_rows: list[dict[str, float | int]] = []
    label_grid_histograms: list[np.ndarray] = []
    grid_id = 0
    for z in range(int(lower[0]), int(upper[0]), stride[0]):
        for y in range(int(lower[1]), int(upper[1]), stride[1]):
            for x in range(int(lower[2]), int(upper[2]), stride[2]):
                stop = (
                    min(z + stride[0], roi.shape[0]),
                    min(y + stride[1], roi.shape[1]),
                    min(x + stride[2], roi.shape[2]),
                )
                slices = (slice(z, stop[0]), slice(y, stop[1]), slice(x, stop[2]))
                values = ri[slices][roi[slices]]
                if values.size == 0:
                    continue
                summary, histogram = _summary_and_histogram(values, bin_edges)
                rows.append(
                    {
                        "grid_id": grid_id,
                        "z_index": z // stride[0],
                        "y_index": y // stride[1],
                        "x_index": x // stride[2],
                        "z_start_um": z * spacing[0],
                        "y_start_um": y * spacing[1],
                        "x_start_um": x * spacing[2],
                        "z_stop_um": stop[0] * spacing[0],
                        "y_stop_um": stop[1] * spacing[1],
                        "x_stop_um": stop[2] * spacing[2],
                        "roi_volume_um3": values.size * np.prod(spacing),
                        **summary,
                    }
                )
                histograms.append(histogram)
                local_labels = labels[slices]
                local_ri = ri[slices]
                for label_id in np.unique(local_labels):
                    if label_id == 0:
                        continue
                    label_values = local_ri[local_labels == label_id]
                    label_summary, label_histogram = _summary_and_histogram(
                        label_values, bin_edges
                    )
                    label_grid_rows.append(
                        {
                            "grid_id": grid_id,
                            "label_id": int(label_id),
                            "label_volume_um3": label_values.size * np.prod(spacing),
                            **label_summary,
                        }
                    )
                    label_grid_histograms.append(label_histogram)
                grid_id += 1
    return (
        AnalysisTable(_table_from_rows(rows, _empty_grid_columns())),
        _histogram_matrix(histograms, len(bin_edges) - 1),
        AnalysisTable(_table_from_rows(label_grid_rows, _empty_label_grid_columns())),
        _histogram_matrix(label_grid_histograms, len(bin_edges) - 1),
    )


def _label_statistics(
    ri: np.ndarray,
    labels: np.ndarray,
    spacing: tuple[float, float, float],
    bin_edges: np.ndarray,
) -> tuple[AnalysisTable, np.ndarray]:
    rows: list[dict[str, float | int]] = []
    histograms: list[np.ndarray] = []
    boxes = ndi.find_objects(labels)
    for label_id, box in enumerate(boxes, start=1):
        if box is None:
            continue
        local = labels[box] == label_id
        values = ri[box][local]
        summary, histogram = _summary_and_histogram(values, bin_edges)
        points = np.argwhere(local)
        centroid = points.mean(axis=0)
        starts = np.asarray([item.start for item in box])
        stops = np.asarray([item.stop for item in box])
        rows.append(
            {
                "label_id": label_id,
                "centroid_z_um": (starts[0] + centroid[0]) * spacing[0],
                "centroid_y_um": (starts[1] + centroid[1]) * spacing[1],
                "centroid_x_um": (starts[2] + centroid[2]) * spacing[2],
                "z_start_um": starts[0] * spacing[0],
                "y_start_um": starts[1] * spacing[1],
                "x_start_um": starts[2] * spacing[2],
                "z_stop_um": stops[0] * spacing[0],
                "y_stop_um": stops[1] * spacing[1],
                "x_stop_um": stops[2] * spacing[2],
                "roi_volume_um3": values.size * np.prod(spacing),
                **summary,
            }
        )
        histograms.append(histogram)
    return AnalysisTable(
        _table_from_rows(rows, _empty_label_columns())
    ), _histogram_matrix(histograms, len(bin_edges) - 1)


def _histogram_matrix(rows: list[np.ndarray], bins: int) -> np.ndarray:
    if not rows:
        return np.empty((0, bins), dtype=np.uint64)
    return np.stack(rows).astype(np.uint64, copy=False)


def _table_from_rows(
    rows: list[dict[str, float | int]],
    empty: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    if not rows:
        return dict(empty or {})
    names = tuple(rows[0])
    if any(tuple(row) != names for row in rows):
        raise ValueError("table rows must share one schema")
    return {name: np.asarray([row[name] for row in rows]) for name in names}


def _empty_grid_columns() -> dict[str, np.ndarray]:
    return _empty_columns(
        "grid_id z_index y_index x_index z_start_um y_start_um x_start_um "
        "z_stop_um y_stop_um x_stop_um roi_volume_um3"
    )


def _empty_label_columns() -> dict[str, np.ndarray]:
    return _empty_columns(
        "label_id centroid_z_um centroid_y_um centroid_x_um z_start_um y_start_um "
        "x_start_um z_stop_um y_stop_um x_stop_um roi_volume_um3"
    )


def _empty_label_grid_columns() -> dict[str, np.ndarray]:
    """Return the schema for one retained label's measurements in one cube."""
    return _empty_columns("grid_id label_id label_volume_um3")


def _empty_columns(prefix: str) -> dict[str, np.ndarray]:
    summary = (
        "voxel_count ri_mean ri_std ri_min ri_max ri_median ri_p05 ri_p95 "
        "histogram_underflow histogram_overflow"
    )
    integer = {
        "grid_id",
        "z_index",
        "y_index",
        "x_index",
        "label_id",
        "voxel_count",
        "histogram_underflow",
        "histogram_overflow",
    }
    return {
        name: np.empty(0, dtype=np.int64 if name in integer else np.float64)
        for name in f"{prefix} {summary}".split()
    }


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "AnalysisArtifacts",
    "AnalysisTable",
    "RiAnalysisResult",
    "RiTiffVolume",
    "analyze_ri_tiff",
    "analyze_volume",
    "load_config",
    "load_ri_tiff",
]
