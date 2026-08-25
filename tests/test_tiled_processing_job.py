from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import zarr

from redsun_aht.configurations import build_dpct_simulation
from redsun_aht.processing import (
    QuantitativeTileConfig,
    TiledProcessingRequest,
    TilingConfig,
    resolve_tiled_processing,
    run_tiled_processing_plan,
)
from redsun_aht.processing.solvers import build_solver


def _bundle(tmp_path: Path) -> Path:
    input_root = tmp_path / "four-shot-input"
    asyncio.run(
        build_dpct_simulation(
            "tiled-job-run",
            output_root=input_root,
            pattern_ids=(10, 11, 12, 13),
        ).run()
    )
    return input_root


def _optical(model: Literal["dpct", "qobt"]) -> QuantitativeTileConfig:
    if model == "dpct":
        return QuantitativeTileConfig(
            model="dpct",
            wavelength_um=0.625,
            numerical_aperture=0.8,
            pixel_size_um=0.108333,
            pixel_size_z_um=1.0,
            source_azimuth_deg=(90, 180, 270, 0),
        )
    return QuantitativeTileConfig(
        model="qobt",
        wavelength_um=0.660,
        numerical_aperture=0.8,
        pixel_size_um=0.108333,
        pixel_size_z_um=1.0,
        refractive_index_medium=1.33,
        source_azimuth_deg=(45, 315, 225, 135),
        source_elevation_deg=50,
        background_division="sample_over_background",
    )


def _tiling() -> TilingConfig:
    return TilingConfig(
        explicit_shape_yx=(6, 6),
        explicit_overlap_yx=(2, 2),
        min_shape_yx=(2, 2),
        max_shape_yx=(8, 8),
        alignment_px=2,
    )


@pytest.mark.parametrize("model", ["dpct", "qobt"])
def test_tiled_quantitative_job_runs_in_spawned_offline_worker_with_provenance(
    tmp_path: Path,
    model: Literal["dpct", "qobt"],
) -> None:
    input_root = _bundle(tmp_path)
    request = TiledProcessingRequest(
        input_root,
        tmp_path / f"{model}-products",
        "dhm",
        _optical(model),
        _tiling(),
    )

    plan = resolve_tiled_processing(request)
    kernel = build_solver(plan.job.solver_id)
    batch = run_tiled_processing_plan(plan)

    assert plan.source_run_id == "tiled-job-run"
    assert len(plan.observations) == 1
    assert plan.observations[0].array.shape == (4, 2, 8, 8)
    assert plan.job.latency_class == "offline"
    assert kernel.capabilities.live_supported is False
    assert kernel.capabilities.required_shots == 4
    kernel.close()
    assert not batch.failures
    result = batch.results[plan.job.solver_id]
    assert result.output.shape == (2, 8, 8)
    assert result.metrics["model"] == model
    assert result.metrics["tile_count"] == 4
    assert isinstance(result.metrics["provenance_sha256"], str)

    group = zarr.open_group(
        Path(plan.job.output_destination) / "product.ome.zarr", mode="r"
    )
    aht = cast("dict[str, object]", group.attrs["aht"])
    provenance = cast("dict[str, object]", aht["provenance"])
    tiling = cast("dict[str, object]", provenance["tiling"])
    donor = cast("dict[str, object]", provenance["donor"])
    source = cast("dict[str, object]", provenance["source"])
    assert aht["provenanceSha256"] == result.metrics["provenance_sha256"]
    assert provenance["model"] == model
    assert provenance["scientific_status"] == "donor-characterized-cpu-reference"
    assert tiling["resolved_tile_shape_yx"] == [6, 6]
    assert tiling["resolved_overlap_yx"] == [2, 2]
    assert len(cast("list[object]", tiling["placements"])) == 4
    assert donor["commit"] == "405e53b35498898dc7e973f9840c0259ace7671a"
    assert source["sample_detector_id"] == "dhm"

    replayed = run_tiled_processing_plan(plan)
    assert replayed.results[plan.job.solver_id].cached


def test_cached_tiled_product_rejects_mutated_resolved_provenance(
    tmp_path: Path,
) -> None:
    plan = resolve_tiled_processing(
        TiledProcessingRequest(
            _bundle(tmp_path),
            tmp_path / "tamper-products",
            "dhm",
            _optical("dpct"),
            _tiling(),
        )
    )
    first = run_tiled_processing_plan(plan)
    assert not first.failures
    group = zarr.open_group(
        Path(plan.job.output_destination) / "product.ome.zarr", mode="a"
    )
    aht = dict(cast("dict[str, object]", group.attrs["aht"]))
    provenance = dict(cast("dict[str, object]", aht["provenance"]))
    provenance["model"] = "qobt"
    aht["provenance"] = provenance
    cast("Any", group.attrs)["aht"] = aht

    replayed = run_tiled_processing_plan(plan)

    assert "provenance check failed" in replayed.failures[plan.job.solver_id]


def test_tiled_job_resolution_rejects_invalid_roots_and_detector_selection(
    tmp_path: Path,
) -> None:
    input_root = _bundle(tmp_path)
    base = TiledProcessingRequest(
        input_root,
        tmp_path / "products",
        "dhm",
        _optical("dpct"),
        _tiling(),
    )
    with pytest.raises(ValueError, match="inside the input"):
        replace(base, output_root=input_root / "products")
    with pytest.raises(ValueError, match="identity cannot be empty"):
        replace(base, detector_id="")
    with pytest.raises(ValueError, match="must differ"):
        replace(base, background_detector_id="dhm")
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(base, background_detector_id="fluorescence", background_z_index=-1)
    with pytest.raises(ValueError, match="requires a background"):
        replace(base, background_z_index=0)
    with pytest.raises(ValueError, match="sample detector is absent"):
        resolve_tiled_processing(replace(base, detector_id="missing"))
    with pytest.raises(ValueError, match="background detector is absent"):
        resolve_tiled_processing(replace(base, background_detector_id="missing"))
    with pytest.raises(ValueError, match="requires an explicit GPU"):
        replace(base, backend="cupy")
    with pytest.raises(ValueError, match="cannot select a GPU"):
        replace(base, gpu_id=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(base, backend="cupy", gpu_id=-1)
    with pytest.raises(ValueError, match="must be positive"):
        replace(base, gpu_reservation_bytes=0)
    with pytest.raises(ValueError, match="whole mebibytes"):
        replace(base, gpu_reservation_bytes=1024 * 1024 + 1)

    gpu_plan = resolve_tiled_processing(replace(base, backend="cupy", gpu_id=1))
    assert gpu_plan.job.solver_id == "tiled-dpct-cupy"
    assert gpu_plan.job.resource_request["gpu_ids"] == (1,)


def test_four_shot_simulation_rejects_empty_or_excess_pattern_sets() -> None:
    with pytest.raises(ValueError, match="one to four"):
        build_dpct_simulation(pattern_ids=())
    with pytest.raises(ValueError, match="one to four"):
        build_dpct_simulation(pattern_ids=(1, 2, 3, 4, 5))
