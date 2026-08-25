from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from redsun_aht.configurations import (
    build_dpct_simulation,
    run_processing_simulation,
)
from redsun_aht.domain import ProcessingJob, SolverState
from redsun_aht.processing import (
    SolverWorker,
    observations_from_dpct_bundle,
)


@pytest.mark.hardware
@pytest.mark.skipif(
    os.environ.get("AHT_TEST_GPU_PROCESSING") != "1",
    reason="set AHT_TEST_GPU_PROCESSING=1 for the explicit CUDA smoke test",
)
def test_real_gpu_worker_owns_context_and_publishes_telemetry(tmp_path: Path) -> None:
    gpu_id = int(os.environ.get("AHT_TEST_GPU_ID", "0"))
    input_root = tmp_path / "gpu-input"
    asyncio.run(
        build_dpct_simulation("gpu-processing-run", output_root=input_root).run()
    )

    result = run_processing_simulation(
        input_root,
        tmp_path / "gpu-output",
        gpu_id=gpu_id,
        gpu_reservation_bytes=256 * 1024 * 1024,
    )

    assert not result.failures
    gpu_result = result.results["gpu-mean-projection"]
    gpu_status = result.statuses["gpu-mean-projection"]
    assert gpu_result.metrics["gpu_device_id"] == gpu_id
    assert isinstance(gpu_result.metrics["gpu_elapsed_ns"], int)
    assert gpu_result.metrics["gpu_elapsed_ns"] > 0
    assert isinstance(gpu_result.metrics["gpu_allocator_reserved_bytes"], int)
    assert gpu_result.metrics["gpu_allocator_reserved_bytes"] > 0
    assert gpu_status.state is SolverState.WAITING_INPUT
    assert gpu_status.gpu_id == gpu_id
    assert gpu_status.allocated_memory_bytes >= 256 * 1024 * 1024
    assert gpu_status.observed_memory_bytes > 0
    assert gpu_result.output.uri.endswith("product.ome.zarr#0")


@pytest.mark.hardware
@pytest.mark.skipif(
    os.environ.get("AHT_TEST_GPU_PROCESSING") != "1",
    reason="set AHT_TEST_GPU_PROCESSING=1 for the explicit CUDA smoke test",
)
def test_real_gpu_worker_restarts_and_replays_cached_cursor(tmp_path: Path) -> None:
    gpu_id = int(os.environ.get("AHT_TEST_GPU_ID", "0"))
    input_root = tmp_path / "restart-input"
    asyncio.run(build_dpct_simulation("gpu-restart-run", output_root=input_root).run())
    observations = observations_from_dpct_bundle(input_root)
    job = ProcessingJob(
        job_id="gpu-restart-job",
        solver_id="gpu-mean-projection",
        solver_version="1.0.0",
        input_selectors=tuple(item.observation_id for item in observations),
        configuration={},
        priority=0,
        latency_class="offline",
        resource_request={
            "gpu_ids": [gpu_id],
            "gpu_memory_reservation_bytes": 256 * 1024 * 1024,
        },
        output_destination=str((tmp_path / "restart-output").resolve()),
        cancellation_policy="cooperative",
    )
    worker = SolverWorker(
        "gpu-mean-projection",
        startup_timeout_s=60,
        result_timeout_s=120,
        gpu_id=gpu_id,
        memory_reservation_bytes=256 * 1024 * 1024,
        gpu_cache_directory=str((tmp_path / "restart-cache").resolve()),
    )
    try:
        worker.start()
        first_pid = worker.process_id
        worker.submit(job, observations)
        assert not worker.wait().cached

        worker.restart()
        assert worker.process_id != first_pid
        assert worker.status.restart_count == 1
        worker.submit(job, observations)
        replayed = worker.wait()

        assert replayed.cached
        assert worker.status.gpu_id == gpu_id
        assert worker.status.restart_count == 1
    finally:
        worker.close()
