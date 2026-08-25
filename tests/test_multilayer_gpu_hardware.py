from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from redsun_aht.configurations import build_dpct_simulation
from redsun_aht.processing import (
    MultiLayerProcessingRequest,
    MultiLayerReconstructionConfig,
    MultiLayerSolveConfig,
    read_processing_result_array,
    resolve_multilayer_processing,
    run_multilayer_processing_plan,
)

GPU_IDS = tuple(
    int(value.strip())
    for value in os.environ.get(
        "AHT_TEST_GPU_IDS", os.environ.get("AHT_TEST_GPU_ID", "0")
    ).split(",")
)


@pytest.mark.hardware
@pytest.mark.parametrize("gpu_id", GPU_IDS)
@pytest.mark.parametrize("model", ["multi_born", "multislice"])
@pytest.mark.skipif(
    os.environ.get("AHT_TEST_GPU_PROCESSING") != "1",
    reason="set AHT_TEST_GPU_PROCESSING=1 for explicit CUDA characterization",
)
def test_multilayer_cupy_worker_matches_cpu_on_physical_gpu(
    tmp_path: Path,
    gpu_id: int,
    model: Literal["multi_born", "multislice"],
) -> None:
    input_root = tmp_path / "input"
    asyncio.run(
        build_dpct_simulation(
            f"multilayer-gpu-{gpu_id}-{model}",
            output_root=input_root,
            pattern_ids=(10, 11, 12, 13),
        ).run()
    )
    reconstruction = MultiLayerReconstructionConfig(
        model=model,
        depth_layers=3,
        voxel_size_zyx_um=(0.7, 0.12, 0.11),
        wavelength_um=0.625,
        numerical_aperture=0.8,
        refractive_index_medium=1.33,
        illumination_na=0.45,
        padding_yx=(1, 2),
        defocus_um=0.3,
        solve=MultiLayerSolveConfig(max_iterations=1),
    )
    cpu_plan = resolve_multilayer_processing(
        MultiLayerProcessingRequest(
            input_root,
            tmp_path / "cpu-output",
            "dhm",
            reconstruction,
        )
    )
    cpu_batch = run_multilayer_processing_plan(cpu_plan)
    assert not cpu_batch.failures
    expected = read_processing_result_array(cpu_batch.results[cpu_plan.job.solver_id])

    gpu_plan = resolve_multilayer_processing(
        MultiLayerProcessingRequest(
            input_root,
            tmp_path / "gpu-output",
            "dhm",
            reconstruction,
            backend="cupy",
            gpu_id=gpu_id,
        )
    )
    gpu_batch = run_multilayer_processing_plan(gpu_plan)

    assert not gpu_batch.failures
    result = gpu_batch.results[gpu_plan.job.solver_id]
    actual = read_processing_result_array(result)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert result.metrics["backend"] == "cupy"
    assert result.metrics["gpu_device_id"] == gpu_id
    assert isinstance(result.metrics["cuda_runtime_version"], int)
    assert isinstance(result.metrics["cuda_driver_version"], int)
    allocator_use = result.metrics["gpu_allocator_used_bytes"]
    assert isinstance(allocator_use, int) and allocator_use > 0
    assert gpu_batch.statuses[gpu_plan.job.solver_id].gpu_id == gpu_id
    assert gpu_batch.statuses[gpu_plan.job.solver_id].observed_memory_bytes > 0
    assert result.metrics["provenance_sha256"]
