from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from redsun_aht.processing import (
    GpuAdmissionError,
    GpuDeviceInfo,
    GpuDiscoveryError,
    GpuLease,
    GpuResourcePool,
    GpuResourceRequest,
    ProcessingSupervisor,
    SolverWorker,
    discover_nvidia_gpus,
)
from redsun_aht.processing.solvers import build_solver


def _devices() -> tuple[GpuDeviceInfo, ...]:
    gib = 1024**3
    return (
        GpuDeviceInfo(0, "large", 16 * gib, 12 * gib, "551.23"),
        GpuDeviceInfo(1, "small", 8 * gib, 7 * gib, "551.23"),
    )


def test_nvidia_inventory_discovery_is_context_free_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args[0] == [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs["timeout"] == 2.0
        return subprocess.CompletedProcess(
            args[0], 0, "0, Tesla P100, 16384, 16000, 551.23\n", ""
        )

    monkeypatch.setattr(subprocess, "run", completed)

    devices = discover_nvidia_gpus(timeout_s=2.0)

    assert devices == (
        GpuDeviceInfo(
            0,
            "Tesla P100",
            16384 * 1024 * 1024,
            16000 * 1024 * 1024,
            "551.23",
        ),
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )
    with pytest.raises(GpuDiscoveryError, match="empty or ambiguous"):
        discover_nvidia_gpus()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "bad", ""),
    )
    with pytest.raises(GpuDiscoveryError, match="query failed"):
        discover_nvidia_gpus()
    with pytest.raises(ValueError, match="executable and timeout"):
        discover_nvidia_gpus(timeout_s=0)


def test_gpu_resource_pool_enforces_preferences_sharing_and_margin() -> None:
    gib = 1024**3
    pool = GpuResourcePool(_devices(), safety_margin_fraction=0.1)

    shared_a = pool.reserve(
        "solver-a",
        GpuResourceRequest(
            3 * gib,
            allowed_device_ids=(0, 1),
            preferred_device_ids=(1,),
            exclusive=False,
        ),
    )
    shared_b = pool.reserve(
        "solver-b",
        GpuResourceRequest(3 * gib, allowed_device_ids=(1,), exclusive=False),
    )
    exclusive = pool.reserve(
        "solver-exclusive",
        GpuResourceRequest(10 * gib, allowed_device_ids=(0,), exclusive=True),
    )

    assert shared_a.device_id == shared_b.device_id == 1
    assert exclusive.device_id == 0
    assert pool.devices[0].name == "large"
    assert set(pool.leases) == {
        shared_a.lease_id,
        shared_b.lease_id,
        exclusive.lease_id,
    }
    with pytest.raises(GpuAdmissionError, match="no GPU can satisfy"):
        pool.reserve(
            "blocked",
            GpuResourceRequest(gib, allowed_device_ids=(0,), exclusive=False),
        )
    with pytest.raises(GpuAdmissionError, match="unknown allowed GPU"):
        pool.reserve("unknown", GpuResourceRequest(gib, allowed_device_ids=(9,)))

    pool.release(shared_a)
    pool.release(shared_a)
    with pytest.raises(ValueError, match="identity does not match"):
        pool.release(
            GpuLease(
                shared_b.lease_id,
                "different",
                shared_b.device_id,
                shared_b.reservation_bytes,
                shared_b.exclusive,
            )
        )


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: GpuDeviceInfo(-1, "gpu", 1, 1, "driver"), "identity"),
        (lambda: GpuDeviceInfo(0, "gpu", 1, 2, "driver"), "memory"),
        (lambda: GpuResourceRequest(0), "positive"),
        (
            lambda: GpuResourceRequest(1, allowed_device_ids=(0, 0)),
            "unique and non-negative",
        ),
        (
            lambda: GpuResourceRequest(
                1, allowed_device_ids=(0,), preferred_device_ids=(1,)
            ),
            "included in the allowed",
        ),
        (lambda: GpuLease("", "solver", 0, 1, True), "lease fields"),
        (lambda: GpuResourcePool(()), "unique devices"),
        (lambda: GpuResourcePool(_devices(), safety_margin_fraction=1), "margin"),
    ],
)
def test_gpu_resource_contract_validation(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_gpu_worker_placement_requires_matching_lease(tmp_path: Path) -> None:
    gib = 1024**3
    lease = GpuLease("lease", "gpu-mean-projection", 1, gib, True)
    cache_directory = str((tmp_path / "cupy-cache").resolve())
    supervisor = ProcessingSupervisor(
        ("gpu-mean-projection",),
        gpu_leases={"gpu-mean-projection": lease},
        gpu_cache_directories={"gpu-mean-projection": cache_directory},
    )

    worker = supervisor.workers["gpu-mean-projection"]
    assert worker.gpu_id == 1
    assert worker.memory_reservation_bytes == gib
    assert worker.gpu_cache_directory == cache_directory
    assert worker.status.gpu_id == 1
    assert worker.status.allocated_memory_bytes == gib

    with pytest.raises(ValueError, match="match supervised solver"):
        ProcessingSupervisor(("quality-metrics",), gpu_leases={"other": lease})
    with pytest.raises(ValueError, match="exactly one cache directory"):
        ProcessingSupervisor(
            ("gpu-mean-projection",),
            gpu_cache_directories={"gpu-mean-projection": cache_directory},
        )
    with pytest.raises(ValueError, match="exactly one cache directory"):
        ProcessingSupervisor(
            ("gpu-mean-projection",),
            gpu_leases={"gpu-mean-projection": lease},
        )
    with pytest.raises(ValueError, match="CPU worker cannot reserve"):
        SolverWorker("mean-projection", memory_reservation_bytes=1)
    with pytest.raises(ValueError, match="placement and reservation"):
        SolverWorker("gpu-mean-projection", gpu_id=0)
    with pytest.raises(ValueError, match="cache directory requires"):
        SolverWorker("mean-projection", gpu_cache_directory=cache_directory)
    with pytest.raises(ValueError, match="explicit device ID"):
        build_solver("gpu-mean-projection")
