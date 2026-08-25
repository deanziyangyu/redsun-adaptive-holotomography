from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from redsun_aht.configurations import build_dpct_simulation
from redsun_aht.processing import (
    NumpyQuantitativeTileSolver,
    OfflineProcessingRequest,
    OfflineTiledJobRequest,
    QuantitativeTileConfig,
    TilingConfig,
    observations_from_dpct_bundle,
    read_observation_array,
    read_processing_result_array,
    resolve_offline_processing,
    run_offline_processing_plan,
)

GPU_IDS = tuple(
    int(value.strip())
    for value in os.environ.get(
        "AHT_TEST_GPU_IDS", os.environ.get("AHT_TEST_GPU_ID", "0")
    ).split(",")
)


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


@pytest.mark.hardware
@pytest.mark.parametrize("gpu_id", GPU_IDS)
@pytest.mark.parametrize("model", ["dpct", "qobt"])
@pytest.mark.skipif(
    os.environ.get("AHT_TEST_GPU_PROCESSING") != "1",
    reason="set AHT_TEST_GPU_PROCESSING=1 for explicit CUDA characterization",
)
def test_tiled_cupy_worker_matches_cpu_reference_on_physical_gpu(
    tmp_path: Path,
    gpu_id: int,
    model: Literal["dpct", "qobt"],
) -> None:
    input_root = tmp_path / "input"
    asyncio.run(
        build_dpct_simulation(
            f"gpu-{gpu_id}-{model}",
            output_root=input_root,
            pattern_ids=(10, 11, 12, 13),
        ).run()
    )
    sample = next(
        item
        for item in observations_from_dpct_bundle(input_root)
        if item.detector_id == "dhm"
    )
    sample_zsyx = np.transpose(read_observation_array(sample), (1, 0, 2, 3))
    optical = _optical(model)
    cpu = NumpyQuantitativeTileSolver(optical)
    cpu.prepare((sample_zsyx.shape[0], *sample_zsyx.shape[-2:]))
    expected = cpu.reconstruct_tile(sample_zsyx, None)
    cpu.close()

    plan = resolve_offline_processing(
        OfflineProcessingRequest(
            input_root,
            tmp_path / "gpu-output",
            tiled=OfflineTiledJobRequest(
                "dhm",
                optical,
                TilingConfig(mode="off"),
                backend="cupy",
                gpu_id=gpu_id,
            ),
        )
    )
    batch = run_offline_processing_plan(plan)

    assert not batch.failures
    solver_id = f"tiled-{model}-cupy"
    result = batch.results[solver_id]
    actual = read_processing_result_array(result)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert result.metrics["backend"] == "cupy"
    assert result.metrics["gpu_device_id"] == gpu_id
    assert isinstance(result.metrics["cuda_runtime_version"], int)
    assert isinstance(result.metrics["cuda_driver_version"], int)
    allocator_use = result.metrics["gpu_allocator_used_bytes"]
    assert isinstance(allocator_use, int) and allocator_use > 0
    assert batch.statuses[solver_id].gpu_id == gpu_id
    assert batch.statuses[solver_id].observed_memory_bytes > 0
    assert result.metrics["provenance_sha256"]
