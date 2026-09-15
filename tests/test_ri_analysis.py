"""Regression tests for offline donor-format RI TIFF analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile
import zarr

from redsun_aht.__main__ import main
from redsun_aht.ri_analysis import (
    RiAnalysisConfig,
    analyze_ri_tiff,
    export_analysis_bundle,
)
from redsun_aht.ri_analysis.config import (
    GridConfig,
    HistogramConfig,
    LabelFilterConfig,
    SegmentationConfig,
)
from redsun_aht.ri_analysis.pipeline import load_ri_tiff
from redsun_aht.ri_analysis.publication import verify_analysis_bundle


def _source_tiff(tmp_path: Path) -> Path:
    """Create a compact ImageJ ZYX uint16 donor-format RI volume."""
    encoded = np.full((8, 20, 20), 13_400, dtype=np.uint16)
    encoded[2:6, 6:14, 7:15] = 14_200
    encoded[2, 6, 7] = 14_100
    path = tmp_path / "ri.tif"
    tifffile.imwrite(
        path,
        encoded,
        imagej=True,
        resolution=(10.0, 8.0),
        metadata={"axes": "ZYX", "unit": "um", "spacing": 0.5},
    )
    return path


def _config() -> RiAnalysisConfig:
    return RiAnalysisConfig(
        segmentation=SegmentationConfig(
            strategy="ri_only",
            seed_mode="manual_window",
            ri_lower=1.4,
            ri_upper=1.5,
            close_radius_um=0.0,
            min_component_volume_um3=0.0,
        ),
        grid=GridConfig(cell_size_zyx_um=(1.0, 1.0, 1.0)),
        histogram=HistogramConfig(ri_min=1.3, ri_max=1.5, bin_count=4),
    )


def test_loads_donor_uint16_imagej_tiff_with_physical_spacing(tmp_path: Path) -> None:
    source = load_ri_tiff(_source_tiff(tmp_path))

    assert source.ri_zyx.shape == (8, 20, 20)
    assert source.ri_zyx.dtype == np.float32
    assert source.ri_zyx[3, 8, 9] == pytest.approx(1.42)
    assert source.voxel_size_zyx_um == pytest.approx((0.5, 0.125, 0.1))
    assert len(source.source_checksum) == 64


def test_rejects_tiff_without_imagej_zyx_metadata(tmp_path: Path) -> None:
    source = tmp_path / "invalid.tif"
    tifffile.imwrite(source, np.zeros((4, 4, 4), dtype=np.uint16))

    with pytest.raises(ValueError, match="ImageJ ZYX"):
        load_ri_tiff(source)


def test_explicit_voxel_size_override_bypasses_incomplete_imagej_spacing(
    tmp_path: Path,
) -> None:
    source = _source_tiff(tmp_path)

    volume = load_ri_tiff(source, voxel_size_zyx_um=(1.0, 0.112, 0.112))

    assert volume.voxel_size_zyx_um == pytest.approx((1.0, 0.112, 0.112))
    assert volume.tiff_metadata["voxel_size_source"] == "explicit_override"


def test_analysis_publishes_statistics_histograms_and_verified_exports(
    tmp_path: Path,
) -> None:
    source = _source_tiff(tmp_path)
    output = tmp_path / "analysis"
    result = analyze_ri_tiff(source, output, _config())

    assert result.label_count == 1
    assert result.grid_cell_count > 0
    manifest = verify_analysis_bundle(output)
    assert manifest["schema_version"] == 2
    assert manifest["source"]["ri_encoding"] == "uint16_ri_times_10000"
    assert manifest["source"]["voxel_size_zyx_um"] == pytest.approx([0.5, 0.125, 0.1])
    bundle = zarr.open_group(result.bundle_path, mode="r")
    labels: Any = bundle["masks/labels/0"]
    roi_counts: Any = bundle["tables/roi/voxel_count"]
    grid_histograms: Any = bundle["histograms/grid_counts"]
    label_grid: Any = bundle["tables/label_grid"]
    label_grid_histograms: Any = bundle["histograms/label_grid_counts"]
    assert labels.dtype == np.dtype("uint32")
    assert int(roi_counts[0]) == 4 * 8 * 8
    assert grid_histograms.shape[1] == 4
    assert label_grid["grid_id"].shape[0] == result.grid_cell_count
    assert np.array_equal(label_grid["label_id"][:], np.ones(result.grid_cell_count))
    assert label_grid_histograms.shape == (result.grid_cell_count, 4)
    assert {
        "voxel_count",
        "label_volume_um3",
        "ri_mean",
        "ri_median",
        "ri_std",
        "ri_p05",
        "ri_p95",
    } <= set(label_grid.array_keys())
    assert np.any(np.asarray(label_grid["ri_std"][:]) > 0)
    exports = export_analysis_bundle(output, tmp_path / "exports")
    assert len(exports) == 13
    assert (tmp_path / "exports/csv/grid.csv").is_file()
    assert (tmp_path / "exports/csv/label_grid.csv").is_file()
    assert (tmp_path / "exports/tiff/labels.tif").is_file()


def test_post_label_filter_removes_small_components_without_relabeling(
    tmp_path: Path,
) -> None:
    encoded = np.full((8, 20, 20), 13_400, dtype=np.uint16)
    encoded[2:6, 6:14, 7:15] = 14_200
    encoded[0:2, 0:2, 0:2] = 14_200
    source = tmp_path / "filtered-ri.tif"
    tifffile.imwrite(
        source,
        encoded,
        imagej=True,
        resolution=(10.0, 8.0),
        metadata={"axes": "ZYX", "unit": "um", "spacing": 0.5},
    )
    output = tmp_path / "analysis"
    config = RiAnalysisConfig(
        segmentation=_config().segmentation,
        label_filter=LabelFilterConfig(min_voxel_count=50),
        grid=_config().grid,
        histogram=_config().histogram,
    )

    result = analyze_ri_tiff(source, output, config)

    manifest = verify_analysis_bundle(output)
    assert result.label_count == 1
    assert manifest["diagnostics"]["label_filter"] == {
        "input_labels": 2,
        "retained_labels": 1,
        "removed_labels": 1,
    }


def test_cli_runs_analysis_and_export_with_strict_json_config(tmp_path: Path) -> None:
    source = _source_tiff(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config().to_mapping()), encoding="utf-8")
    output = tmp_path / "analysis"

    assert (
        main(
            [
                "analyze-ri",
                "--input",
                str(source),
                "--output",
                str(output),
                "--config",
                str(config),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "export-ri-analysis",
                "--input",
                str(output),
                "--output",
                str(tmp_path / "exports"),
                "--format",
                "csv",
            ]
        )
        == 0
    )
