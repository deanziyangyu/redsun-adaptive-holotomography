from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from redsun_aht.domain import ArrayReference, Observation, ProcessingJob
from redsun_aht.processing import solvers
from redsun_aht.processing.worker import _validate_worker_resources


class _Context:
    def __enter__(self) -> _Context:
        return self

    def __exit__(self, *_: object) -> None:
        pass


class _Stream(_Context):
    def __init__(self, *, non_blocking: bool) -> None:
        assert non_blocking

    def synchronize(self) -> None:
        pass


class _MemoryPool:
    def total_bytes(self) -> int:
        return 4096

    def used_bytes(self) -> int:
        return 2048

    def free_all_blocks(self) -> None:
        pass


def _fake_cupy() -> Any:
    runtime = SimpleNamespace(
        getDeviceCount=lambda: 2,
        runtimeGetVersion=lambda: 12020,
        driverGetVersion=lambda: 12040,
    )
    return SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=runtime, Device=lambda _: _Context(), Stream=_Stream
        ),
        get_default_memory_pool=lambda: _MemoryPool(),
        zeros=np.zeros,
        asarray=np.asarray,
        mean=np.mean,
        asnumpy=lambda value, stream=None: np.asarray(value),
        float32=np.float32,
        float64=np.float64,
    )


def _observation() -> Observation:
    return Observation(
        run_id="gpu-run",
        observation_id="gpu-observation",
        detector_id="detector",
        channel_id="channel",
        sequence=0,
        array=ArrayReference("file:///unused#0", "0" * 64, (1, 1, 2, 2), "uint8", "|"),
        committed=True,
        configuration_revision="revision",
        calibration_ids=(),
        replay_key="replay",
        latency_class="offline",
    )


def _job(tmp_path: Path) -> ProcessingJob:
    return ProcessingJob(
        "gpu-job",
        "gpu-mean-projection",
        "1.0.0",
        ("gpu-observation",),
        {},
        0,
        "offline",
        {"gpu_ids": [1], "gpu_memory_reservation_bytes": 1024},
        str((tmp_path / "output").resolve()),
        "cooperative",
    )


def test_gpu_solver_binds_runtime_stream_memory_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib, "import_module", lambda _: _fake_cupy())
    monkeypatch.setattr(
        solvers,
        "read_observation_array",
        lambda _: np.arange(4, dtype=np.uint8).reshape(1, 1, 2, 2),
    )
    cache = str((tmp_path / "cupy-cache").resolve())

    solver = solvers.GpuMeanProjectionSolver(1, cache_directory=cache)
    output = solver.run(_job(tmp_path), (_observation(),), lambda: False)

    assert solver.gpu_id == 1
    assert solver.allocated_memory_bytes == 4096
    assert solver.observed_memory_bytes == 2048
    assert output.array.shape == (1, 2, 2)
    assert output.metrics["gpu_device_id"] == 1
    assert output.metrics["cuda_runtime_version"] == 12020
    startup_ns = output.metrics["gpu_context_startup_ns"]
    assert isinstance(startup_ns, int) and startup_ns > 0
    solver.close()
    solver.close()
    with pytest.raises(RuntimeError, match="is closed"):
        solver.run(_job(tmp_path), (_observation(),), lambda: False)


def test_gpu_solver_rejects_missing_runtime_device_and_relative_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        solvers.GpuMeanProjectionSolver(-1)
    monkeypatch.setattr(importlib, "import_module", lambda _: _fake_cupy())
    with pytest.raises(ValueError, match="cache directory must be absolute"):
        solvers.GpuMeanProjectionSolver(0, cache_directory="relative")
    with pytest.raises(ValueError, match="does not exist"):
        solvers.GpuMeanProjectionSolver(2)

    def missing(_: str) -> Any:
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="install the gpu-cupy extra"):
        solvers.GpuMeanProjectionSolver(0)


def test_gpu_job_resources_must_match_worker_placement(tmp_path: Path) -> None:
    job = _job(tmp_path)
    _validate_worker_resources(job, 1, 1024)
    _validate_worker_resources(job, None, 0)
    with pytest.raises(ValueError, match="does not match worker placement"):
        _validate_worker_resources(job, 0, 1024)
    with pytest.raises(ValueError, match="does not match worker placement"):
        _validate_worker_resources(job, 1, 2048)
