"""Small deterministic reference solvers for detached-worker admission."""

from __future__ import annotations

import importlib
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from redsun_aht.domain import Observation, ProcessingJob, SolverCapabilities
from redsun_aht.processing.replay import read_observation_array

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Literal

    import numpy.typing as npt

    from redsun_aht.domain.models import JsonValue


class ProcessingCancelled(RuntimeError):
    """Raised cooperatively when a detached processing job is cancelled."""


@dataclass(frozen=True, slots=True)
class KernelOutput:
    """Array and lightweight metrics returned inside one worker process."""

    array: npt.NDArray[Any]
    dimension_names: tuple[str, ...]
    metrics: Mapping[str, JsonValue]
    ome_image: bool = False
    provenance: Mapping[str, JsonValue] = field(default_factory=dict)


class SolverKernel(Protocol):
    """Synchronous worker-owned numerical kernel contract."""

    @property
    def capabilities(self) -> SolverCapabilities:
        """Return immutable admission metadata."""

    @property
    def gpu_id(self) -> int | None:
        """Return the worker-owned GPU placement, if any."""

    @property
    def allocated_memory_bytes(self) -> int:
        """Return current allocator-reserved memory."""

    @property
    def observed_memory_bytes(self) -> int:
        """Return the worker-observed memory high-water mark."""

    def run(
        self,
        job: ProcessingJob,
        observations: tuple[Observation, ...],
        cancelled: Callable[[], bool],
    ) -> KernelOutput:
        """Run without touching acquisition hardware or shared mutable inputs."""

    def close(self) -> None:
        """Release persistent worker-owned resources idempotently."""


class MeanProjectionSolver:
    """CPU reference preview averaging detectors and illumination channels."""

    capabilities = SolverCapabilities(
        solver_id="mean-projection",
        solver_version="1.0.0",
        dimensionality=(4,),
        required_channels=1,
        required_shots=1,
        streaming_supported=False,
        batch_supported=True,
        incremental_supported=False,
        live_supported=False,
        output_types=("mean_projection_zyx",),
    )
    gpu_id = None
    allocated_memory_bytes = 0
    observed_memory_bytes = 0

    def run(
        self,
        job: ProcessingJob,
        observations: tuple[Observation, ...],
        cancelled: Callable[[], bool],
    ) -> KernelOutput:
        """Return one ZYX mean projection from committed CZYX inputs."""
        arrays = _read_czyx(job, observations, cancelled)
        _optional_delay(job.configuration, cancelled)
        stacked = np.stack(arrays, axis=0).astype(np.float32, copy=False)
        result = np.mean(stacked, axis=(0, 1), dtype=np.float64).astype(np.float32)
        return KernelOutput(
            result,
            ("z", "y", "x"),
            {
                "input_observations": len(observations),
                "mean": float(np.mean(result)),
                "minimum": float(np.min(result)),
                "maximum": float(np.max(result)),
            },
            ome_image=True,
        )

    def close(self) -> None:
        """Release no-op CPU reference resources."""


class QualityMetricsSolver:
    """CPU reference quality pipeline publishing per-shot mean and contrast."""

    capabilities = SolverCapabilities(
        solver_id="quality-metrics",
        solver_version="1.0.0",
        dimensionality=(4,),
        required_channels=1,
        required_shots=1,
        streaming_supported=False,
        batch_supported=True,
        incremental_supported=False,
        live_supported=False,
        output_types=("quality_metrics_czm",),
    )
    gpu_id = None
    allocated_memory_bytes = 0
    observed_memory_bytes = 0

    def run(
        self,
        job: ProcessingJob,
        observations: tuple[Observation, ...],
        cancelled: Callable[[], bool],
    ) -> KernelOutput:
        """Return C,Z,(mean,stddev) metrics averaged across detectors."""
        arrays = _read_czyx(job, observations, cancelled)
        metrics: list[npt.NDArray[np.float32]] = []
        for array in arrays:
            _raise_if_cancelled(cancelled)
            mean = np.mean(array, axis=(-2, -1), dtype=np.float64)
            stddev = np.std(array, axis=(-2, -1), dtype=np.float64)
            metrics.append(np.stack((mean, stddev), axis=-1).astype(np.float32))
            _optional_delay(job.configuration, cancelled)
        result = np.mean(np.stack(metrics, axis=0), axis=0, dtype=np.float64).astype(
            np.float32
        )
        return KernelOutput(
            result,
            ("c", "z", "metric"),
            {
                "input_observations": len(observations),
                "global_mean": float(np.mean(result[..., 0])),
                "global_contrast": float(np.mean(result[..., 1])),
            },
        )

    def close(self) -> None:
        """Release no-op CPU reference resources."""


class GpuMeanProjectionSolver:
    """Persistent CuPy mean projection with one worker-owned CUDA context."""

    capabilities = SolverCapabilities(
        solver_id="gpu-mean-projection",
        solver_version="1.0.0",
        dimensionality=(4,),
        required_channels=1,
        required_shots=1,
        streaming_supported=False,
        batch_supported=True,
        incremental_supported=False,
        live_supported=False,
        output_types=("mean_projection_zyx",),
        requires_gpu=True,
        gpu_runtime="cupy-cuda12x",
    )

    def __init__(self, gpu_id: int, *, cache_directory: str | None = None) -> None:
        if gpu_id < 0:
            raise ValueError("GPU solver device ID must be non-negative")
        if cache_directory is not None:
            cache_path = Path(cache_directory)
            if not cache_path.is_absolute():
                raise ValueError("CuPy cache directory must be absolute")
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ["CUPY_CACHE_DIR"] = str(cache_path)
        elif "CUPY_CACHE_DIR" not in os.environ:
            cache_path = Path(tempfile.gettempdir()) / "redsun-aht-cupy-cache"
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ["CUPY_CACHE_DIR"] = str(cache_path)
        try:
            cupy = importlib.import_module("cupy")
        except ImportError as error:
            raise RuntimeError(
                "CuPy GPU runtime is unavailable; install the gpu-cupy extra"
            ) from error
        if gpu_id >= int(cupy.cuda.runtime.getDeviceCount()):
            raise ValueError(f"GPU solver device does not exist: {gpu_id}")
        self._cupy = cupy
        self._gpu_id = gpu_id
        self._device = cupy.cuda.Device(gpu_id)
        with self._device:
            self._stream = cupy.cuda.Stream(non_blocking=True)
            self._memory_pool = cupy.get_default_memory_pool()
            self._runtime_version = int(cupy.cuda.runtime.runtimeGetVersion())
            self._driver_version = int(cupy.cuda.runtime.driverGetVersion())
            warmup_started_ns = time.perf_counter_ns()
            with self._stream:
                warmup_input = cupy.zeros((1, 1, 1, 2, 2), dtype=cupy.float32)
                warmup_result = cupy.mean(
                    warmup_input, axis=(0, 1), dtype=cupy.float64
                ).astype(cupy.float32)
                cupy.asnumpy(warmup_result, stream=self._stream)
            self._stream.synchronize()
            self._startup_context_ns = time.perf_counter_ns() - warmup_started_ns
            del warmup_input, warmup_result
            self._memory_pool.free_all_blocks()
        self._observed_memory_bytes = 0
        self._closed = False

    @property
    def gpu_id(self) -> int:
        """Return the CUDA device owned by this worker process."""
        return self._gpu_id

    @property
    def allocated_memory_bytes(self) -> int:
        """Return CuPy memory-pool reservation in bytes."""
        with self._device:
            return int(self._memory_pool.total_bytes())

    @property
    def observed_memory_bytes(self) -> int:
        """Return the largest observed CuPy memory-pool use."""
        return self._observed_memory_bytes

    def run(
        self,
        job: ProcessingJob,
        observations: tuple[Observation, ...],
        cancelled: Callable[[], bool],
    ) -> KernelOutput:
        """Transfer committed inputs and compute one ZYX projection on CUDA."""
        if self._closed:
            raise RuntimeError("GPU solver is closed")
        arrays = _read_czyx(job, observations, cancelled)
        _optional_delay(job.configuration, cancelled)
        host = np.stack(arrays, axis=0).astype(np.float32, copy=False)
        started_ns = time.perf_counter_ns()
        with self._device, self._stream:
            gpu_input = self._cupy.asarray(host)
            gpu_result = self._cupy.mean(
                gpu_input, axis=(0, 1), dtype=self._cupy.float64
            ).astype(self._cupy.float32)
            result = self._cupy.asnumpy(gpu_result, stream=self._stream)
        self._stream.synchronize()
        elapsed_ns = time.perf_counter_ns() - started_ns
        with self._device:
            current_memory_bytes = int(self._memory_pool.used_bytes())
        self._observed_memory_bytes = max(
            self._observed_memory_bytes, current_memory_bytes
        )
        return KernelOutput(
            result,
            ("z", "y", "x"),
            {
                "input_observations": len(observations),
                "mean": float(np.mean(result)),
                "minimum": float(np.min(result)),
                "maximum": float(np.max(result)),
                "gpu_device_id": self._gpu_id,
                "gpu_elapsed_ns": elapsed_ns,
                "cuda_runtime_version": self._runtime_version,
                "cuda_driver_version": self._driver_version,
                "gpu_context_startup_ns": self._startup_context_ns,
                "gpu_allocator_reserved_bytes": self.allocated_memory_bytes,
                "gpu_allocator_used_bytes": self.observed_memory_bytes,
            },
            ome_image=True,
        )

    def close(self) -> None:
        """Synchronize and release this process-owned CuPy allocator."""
        if self._closed:
            return
        with self._device:
            self._stream.synchronize()
            self._memory_pool.free_all_blocks()
        self._closed = True


def build_solver(
    solver_id: str,
    *,
    gpu_id: int | None = None,
    gpu_cache_directory: str | None = None,
) -> SolverKernel:
    """Construct one allowlisted worker-owned reference solver."""
    if solver_id == "gpu-mean-projection":
        if gpu_id is None:
            raise ValueError("GPU mean projection requires an explicit device ID")
        return GpuMeanProjectionSolver(gpu_id, cache_directory=gpu_cache_directory)
    if solver_id in {"tiled-dpct-numpy", "tiled-qobt-numpy"}:
        from redsun_aht.processing.tiled_kernel import TiledQuantitativeKernel

        if solver_id == "tiled-dpct-numpy":
            return TiledQuantitativeKernel("dpct")
        return TiledQuantitativeKernel("qobt")
    if solver_id in {"tiled-dpct-cupy", "tiled-qobt-cupy"}:
        from redsun_aht.processing.tiled_kernel import TiledQuantitativeKernel

        if gpu_id is None:
            raise ValueError("tiled CuPy reconstruction requires an explicit GPU ID")
        if solver_id == "tiled-dpct-cupy":
            return TiledQuantitativeKernel(
                "dpct",
                backend="cupy",
                gpu_id=gpu_id,
                gpu_cache_directory=gpu_cache_directory,
            )
        return TiledQuantitativeKernel(
            "qobt",
            backend="cupy",
            gpu_id=gpu_id,
            gpu_cache_directory=gpu_cache_directory,
        )
    if solver_id in {
        "multilayer-multi_born-numpy",
        "multilayer-multislice-numpy",
    }:
        from redsun_aht.processing.multilayer_kernel import MultiLayerKernel

        model: Literal["multi_born", "multislice"] = (
            "multi_born" if "multi_born" in solver_id else "multislice"
        )
        return MultiLayerKernel(model)
    if solver_id in {
        "multilayer-multi_born-cupy",
        "multilayer-multislice-cupy",
    }:
        from redsun_aht.processing.multilayer_kernel import MultiLayerKernel

        if gpu_id is None:
            raise ValueError("multi-layer CuPy reconstruction requires a GPU ID")
        model = "multi_born" if "multi_born" in solver_id else "multislice"
        return MultiLayerKernel(
            model,
            backend="cupy",
            gpu_id=gpu_id,
            gpu_cache_directory=gpu_cache_directory,
        )
    factories: dict[str, type[MeanProjectionSolver] | type[QualityMetricsSolver]] = {
        "mean-projection": MeanProjectionSolver,
        "quality-metrics": QualityMetricsSolver,
    }
    try:
        return factories[solver_id]()
    except KeyError as error:
        raise ValueError(f"unknown solver plugin: {solver_id}") from error


def _read_czyx(
    job: ProcessingJob,
    observations: tuple[Observation, ...],
    cancelled: Callable[[], bool],
) -> tuple[npt.NDArray[Any], ...]:
    if not observations:
        raise ValueError("solver job requires committed observations")
    arrays: list[npt.NDArray[Any]] = []
    shape: tuple[int, ...] | None = None
    for observation in observations:
        _raise_if_cancelled(cancelled)
        array = read_observation_array(observation)
        if array.ndim != 4:
            raise ValueError("reference solvers require CZYX observations")
        if shape is not None and array.shape != shape:
            raise ValueError("solver observations must share one CZYX shape")
        shape = tuple(array.shape)
        arrays.append(array)
    if job.solver_version != "1.0.0":
        raise ValueError("reference solver version does not match job")
    return tuple(arrays)


def _optional_delay(
    configuration: Mapping[str, object], cancelled: Callable[[], bool]
) -> None:
    value = configuration.get("simulation_delay_s", 0.0)
    if not isinstance(value, (int, float)) or not 0 <= value <= 0.5:
        raise ValueError("simulation_delay_s must be between 0 and 0.5")
    if value:
        deadline = time.monotonic() + float(value)
        while time.monotonic() < deadline:
            _raise_if_cancelled(cancelled)
            time.sleep(min(0.01, deadline - time.monotonic()))


def _raise_if_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise ProcessingCancelled("processing job was cancelled")


__all__ = [
    "GpuMeanProjectionSolver",
    "KernelOutput",
    "MeanProjectionSolver",
    "ProcessingCancelled",
    "QualityMetricsSolver",
    "SolverKernel",
    "build_solver",
]
