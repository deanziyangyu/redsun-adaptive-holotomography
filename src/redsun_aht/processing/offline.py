"""Typed hardware-free processing plans and platform-aware CLI rendering."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import unquote, urlsplit

import numpy as np
import zarr

from redsun_aht.domain import ProcessingJob
from redsun_aht.processing.multilayer_job import (
    MultiLayerProcessingRequest,
    resolve_multilayer_processing,
)
from redsun_aht.processing.replay import observations_from_offline_bundle
from redsun_aht.processing.resources import (
    GpuResourcePool,
    GpuResourceRequest,
    discover_nvidia_gpus,
)
from redsun_aht.processing.supervisor import (
    ProcessingBatchResult,
    ProcessingSupervisor,
)
from redsun_aht.processing.tiled_job import (
    TiledProcessingRequest,
    resolve_tiled_processing,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from redsun_aht.domain import Observation, ProcessingResult
    from redsun_aht.processing.multilayer_job import MultiLayerReconstructionConfig
    from redsun_aht.processing.quantitative import QuantitativeTileConfig
    from redsun_aht.processing.tiling import TilingConfig


@dataclass(frozen=True, slots=True)
class OfflineProcessingRequest:
    """Validated user intent for one hardware-free two-solver run."""

    input_root: Path
    output_root: Path
    gpu_id: int | None = None
    gpu_reservation_bytes: int = 256 * 1024 * 1024
    tiled: OfflineTiledJobRequest | None = None
    multilayer: OfflineMultiLayerJobRequest | None = None

    def __post_init__(self) -> None:
        """Normalize paths and reject source-mutating output placement."""
        input_root = self.input_root.resolve()
        output_root = self.output_root.resolve()
        if not input_root.is_dir():
            raise ValueError(f"processing input is not a directory: {input_root}")
        if output_root == input_root or input_root in output_root.parents:
            raise ValueError(
                "processing output cannot be inside the immutable input run"
            )
        if self.gpu_id is not None and self.gpu_id < 0:
            raise ValueError("GPU ID cannot be negative")
        if self.gpu_reservation_bytes <= 0:
            raise ValueError("GPU reservation must be positive")
        if self.gpu_reservation_bytes % (1024 * 1024):
            raise ValueError("GPU reservation must use whole mebibytes")
        object.__setattr__(self, "input_root", input_root)
        object.__setattr__(self, "output_root", output_root)


@dataclass(frozen=True, slots=True)
class OfflineTiledJobRequest:
    """Optional quantitative job attached to an offline processing request."""

    detector_id: str
    optical: QuantitativeTileConfig
    tiling: TilingConfig
    background_detector_id: str | None = None
    background_z_index: int | None = None
    fixed_median_from_sample: bool = False
    backend: Literal["numpy", "cupy"] = "numpy"
    gpu_id: int | None = None
    gpu_reservation_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        """Reject incomplete detector and background selection."""
        if not self.detector_id:
            raise ValueError("tiled detector identity cannot be empty")
        if self.background_detector_id == self.detector_id:
            raise ValueError("sample and background detector identities must differ")
        if self.fixed_median_from_sample and self.background_detector_id is not None:
            raise ValueError("fixed median and background detector are exclusive")
        if self.fixed_median_from_sample and self.background_z_index is not None:
            raise ValueError("fixed median does not accept a background Z index")
        if self.background_z_index is not None and self.background_z_index < 0:
            raise ValueError("background Z index cannot be negative")
        if self.background_z_index is not None and self.background_detector_id is None:
            raise ValueError("background Z index requires a background detector")
        if self.backend not in {"numpy", "cupy"}:
            raise ValueError("tiled backend must be NumPy or CuPy")
        if self.backend == "numpy" and self.gpu_id is not None:
            raise ValueError("tiled NumPy processing cannot select a GPU")
        if self.backend == "cupy" and self.gpu_id is None:
            raise ValueError("tiled CuPy processing requires an explicit GPU ID")
        if self.gpu_id is not None and self.gpu_id < 0:
            raise ValueError("tiled GPU ID cannot be negative")
        if self.gpu_reservation_bytes <= 0:
            raise ValueError("tiled GPU reservation must be positive")
        if self.gpu_reservation_bytes % (1024 * 1024):
            raise ValueError("tiled GPU reservation must use whole mebibytes")


@dataclass(frozen=True, slots=True)
class OfflineMultiLayerJobRequest:
    """Optional nonlinear reconstruction attached to an offline request."""

    detector_id: str
    reconstruction: MultiLayerReconstructionConfig
    background_detector_id: str | None = None
    background_z_index: int | None = None
    backend: Literal["numpy", "cupy"] = "numpy"
    gpu_id: int | None = None
    gpu_reservation_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        """Reject incomplete source, normalization, and GPU selections."""
        if not self.detector_id:
            raise ValueError("multi-layer detector identity cannot be empty")
        if self.background_detector_id == self.detector_id:
            raise ValueError("multi-layer sample and background detectors must differ")
        if self.reconstruction.normalization == "background":
            if self.background_detector_id is None:
                raise ValueError(
                    "multi-layer background normalization requires a detector"
                )
        elif self.background_detector_id is not None:
            raise ValueError(
                "multi-layer background detector requires background normalization"
            )
        if self.background_z_index is not None and self.background_z_index < 0:
            raise ValueError("multi-layer background Z index cannot be negative")
        if self.background_z_index is not None and self.background_detector_id is None:
            raise ValueError("multi-layer background Z index requires a detector")
        if self.backend not in {"numpy", "cupy"}:
            raise ValueError("multi-layer backend must be NumPy or CuPy")
        if self.backend == "numpy" and self.gpu_id is not None:
            raise ValueError("multi-layer NumPy processing cannot select a GPU")
        if self.backend == "cupy" and self.gpu_id is None:
            raise ValueError("multi-layer CuPy processing requires an explicit GPU ID")
        if self.gpu_id is not None and self.gpu_id < 0:
            raise ValueError("multi-layer GPU ID cannot be negative")
        if self.gpu_reservation_bytes <= 0:
            raise ValueError("multi-layer GPU reservation must be positive")
        if self.gpu_reservation_bytes % (1024 * 1024):
            raise ValueError("multi-layer GPU reservation must use whole mebibytes")


@dataclass(frozen=True, slots=True)
class OfflineProcessingPlan:
    """Resolved observations and typed jobs shared by CLI and GUI."""

    request: OfflineProcessingRequest
    source_run_id: str
    observations: tuple[Observation, ...]
    jobs: Mapping[str, ProcessingJob]
    job_observations: Mapping[str, tuple[Observation, ...]]

    def __post_init__(self) -> None:
        """Freeze job identity after source verification and resolution."""
        if not self.source_run_id or not self.observations or not self.jobs:
            raise ValueError("offline processing plan is incomplete")
        if set(self.jobs) != set(self.job_observations):
            raise ValueError("every offline job requires an observation selection")
        object.__setattr__(self, "jobs", MappingProxyType(dict(self.jobs)))
        object.__setattr__(
            self,
            "job_observations",
            MappingProxyType(
                {
                    solver_id: tuple(selected)
                    for solver_id, selected in self.job_observations.items()
                }
            ),
        )

    def command_argv(self, *, executable: str = "aht") -> tuple[str, ...]:
        """Return the equivalent headless argument vector without a shell."""
        if not executable:
            raise ValueError("processing executable cannot be empty")
        arguments = [
            executable,
            "process-headless",
            "--input",
            str(self.request.input_root),
            "--output",
            str(self.request.output_root),
        ]
        if self.request.gpu_id is not None:
            arguments.extend(
                (
                    "--gpu-id",
                    str(self.request.gpu_id),
                    "--gpu-reservation-mib",
                    str(self.request.gpu_reservation_bytes // (1024 * 1024)),
                )
            )
        if self.request.tiled is not None:
            arguments.extend(_tiled_command_arguments(self.request.tiled))
        if self.request.multilayer is not None:
            arguments.extend(_multilayer_command_arguments(self.request.multilayer))
        return tuple(arguments)

    def render_command(
        self,
        *,
        executable: str = "aht",
        platform: Literal["windows", "posix"] | None = None,
    ) -> str:
        """Quote the equivalent command for display or clipboard use."""
        selected = platform or ("windows" if os.name == "nt" else "posix")
        arguments = self.command_argv(executable=executable)
        if selected == "windows":
            return subprocess.list2cmdline(arguments)
        if selected == "posix":
            return shlex.join(arguments)
        raise ValueError(f"unsupported command-rendering platform: {selected}")


def resolve_offline_processing(
    request: OfflineProcessingRequest,
) -> OfflineProcessingPlan:
    """Verify a DPCT bundle and resolve the two reference solver jobs."""
    observations = observations_from_offline_bundle(request.input_root)
    if not observations:
        raise ValueError("processing request requires detector observations")
    source_run_id = observations[0].run_id
    if any(item.run_id != source_run_id for item in observations):
        raise ValueError("processing request observations span multiple runs")
    selectors = tuple(item.observation_id for item in observations)
    projection_solver = (
        "mean-projection" if request.gpu_id is None else "gpu-mean-projection"
    )
    solver_ids = (projection_solver, "quality-metrics")
    jobs: dict[str, ProcessingJob] = {
        solver_id: ProcessingJob(
            job_id=f"{source_run_id}-{solver_id}",
            solver_id=solver_id,
            solver_version="1.0.0",
            input_selectors=selectors,
            configuration={"simulation_delay_s": 0.05},
            priority=0,
            latency_class="offline",
            resource_request=(
                {"cpu_threads": 1, "gpu_ids": []}
                if solver_id != "gpu-mean-projection"
                else {
                    "cpu_threads": 1,
                    "gpu_ids": [request.gpu_id],
                    "gpu_memory_reservation_bytes": request.gpu_reservation_bytes,
                }
            ),
            output_destination=str((request.output_root / solver_id).resolve()),
            cancellation_policy="cooperative",
        )
        for solver_id in solver_ids
    }
    job_observations = {solver_id: observations for solver_id in jobs}
    if request.tiled is not None:
        tiled = resolve_tiled_processing(
            TiledProcessingRequest(
                request.input_root,
                request.output_root,
                request.tiled.detector_id,
                request.tiled.optical,
                request.tiled.tiling,
                background_detector_id=request.tiled.background_detector_id,
                background_z_index=request.tiled.background_z_index,
                fixed_median_from_sample=request.tiled.fixed_median_from_sample,
                backend=request.tiled.backend,
                gpu_id=request.tiled.gpu_id,
                gpu_reservation_bytes=request.tiled.gpu_reservation_bytes,
            )
        )
        if tiled.source_run_id != source_run_id:
            raise ValueError("tiled and summary jobs resolved different source runs")
        jobs[tiled.job.solver_id] = tiled.job
        job_observations[tiled.job.solver_id] = tiled.observations
    if request.multilayer is not None:
        multilayer = resolve_multilayer_processing(
            MultiLayerProcessingRequest(
                request.input_root,
                request.output_root,
                request.multilayer.detector_id,
                request.multilayer.reconstruction,
                background_detector_id=request.multilayer.background_detector_id,
                background_z_index=request.multilayer.background_z_index,
                backend=request.multilayer.backend,
                gpu_id=request.multilayer.gpu_id,
                gpu_reservation_bytes=request.multilayer.gpu_reservation_bytes,
            )
        )
        if multilayer.source_run_id != source_run_id:
            raise ValueError("multi-layer and summary jobs resolved different runs")
        jobs[multilayer.job.solver_id] = multilayer.job
        job_observations[multilayer.job.solver_id] = multilayer.observations
    return OfflineProcessingPlan(
        request,
        source_run_id,
        observations,
        jobs,
        job_observations,
    )


def _tiled_command_arguments(request: OfflineTiledJobRequest) -> tuple[str, ...]:
    """Render every scientific input needed to recreate one tiled job."""
    optical = request.optical
    tiling = request.tiling
    arguments = [
        "--tiled-model",
        optical.model,
        "--tiled-detector",
        request.detector_id,
        "--tiled-backend",
        request.backend,
        "--wavelength-um",
        str(optical.wavelength_um),
        "--numerical-aperture",
        str(optical.numerical_aperture),
        "--pixel-size-um",
        str(optical.pixel_size_um),
        "--pixel-size-z-um",
        str(optical.pixel_size_z_um),
        "--refractive-index-medium",
        str(optical.refractive_index_medium),
        "--refractive-index-sample",
        str(optical.refractive_index_sample),
        "--source-azimuth-deg",
        *(str(value) for value in optical.source_azimuth_deg),
        "--source-elevation-deg",
        str(optical.source_elevation_deg),
        "--source-sigma",
        str(optical.source_sigma),
        "--background-division",
        optical.background_division,
        "--regularization-real",
        str(optical.regularization_real),
        "--regularization-imaginary",
        str(optical.regularization_imaginary),
        "--regularization-scalar",
        str(optical.regularization_scalar),
        "--axial-window",
        optical.axial_window,
        "--tile-mode",
        tiling.mode,
        "--tile-voxel-budget",
        str(tiling.voxel_budget),
        "--tile-max-shape-yx",
        *(str(value) for value in tiling.max_shape_yx),
        "--tile-min-shape-yx",
        *(str(value) for value in tiling.min_shape_yx),
        "--tile-alignment-px",
        str(tiling.alignment_px),
        "--tile-overlap-fraction",
        str(tiling.overlap_fraction),
        "--tile-pad-mode",
        tiling.pad_mode,
    ]
    if request.background_detector_id is not None:
        arguments.extend(("--background-detector", request.background_detector_id))
    if request.background_z_index is not None:
        arguments.extend(("--background-z-index", str(request.background_z_index)))
    if request.fixed_median_from_sample:
        arguments.append("--fixed-median-from-sample")
    if request.gpu_id is not None:
        arguments.extend(
            (
                "--tiled-gpu-id",
                str(request.gpu_id),
                "--tiled-gpu-reservation-mib",
                str(request.gpu_reservation_bytes // (1024 * 1024)),
            )
        )
    if tiling.explicit_shape_yx is not None:
        arguments.extend(
            ("--tile-shape-yx", *(str(value) for value in tiling.explicit_shape_yx))
        )
    if tiling.explicit_overlap_yx is not None:
        arguments.extend(
            (
                "--tile-overlap-yx",
                *(str(value) for value in tiling.explicit_overlap_yx),
            )
        )
    return tuple(arguments)


def _multilayer_command_arguments(
    request: OfflineMultiLayerJobRequest,
) -> tuple[str, ...]:
    """Render every scientific input needed to recreate one multi-layer job."""
    reconstruction = request.reconstruction
    solve = reconstruction.solve
    arguments = [
        "--multilayer-model",
        reconstruction.model,
        "--multilayer-detector",
        request.detector_id,
        "--multilayer-backend",
        request.backend,
        "--multilayer-depth-layers",
        str(reconstruction.depth_layers),
        "--multilayer-voxel-size-zyx-um",
        *(str(value) for value in reconstruction.voxel_size_zyx_um),
        "--multilayer-wavelength-um",
        str(reconstruction.wavelength_um),
        "--multilayer-na",
        str(reconstruction.numerical_aperture),
        "--multilayer-medium-index",
        str(reconstruction.refractive_index_medium),
        "--multilayer-illumination-na",
        str(reconstruction.illumination_na),
        "--multilayer-source-z-index",
        str(reconstruction.source_z_index),
        "--multilayer-padding-yx",
        *(str(value) for value in reconstruction.padding_yx),
        "--multilayer-defocus-um",
        str(reconstruction.defocus_um),
        "--multilayer-normalization",
        reconstruction.normalization,
        "--multilayer-memory-budget-mib",
        str(reconstruction.memory_budget_bytes // (1024 * 1024)),
        "--multilayer-max-iterations",
        str(solve.max_iterations),
        "--multilayer-optimizer",
        solve.optimizer,
        "--multilayer-seed",
        str(solve.seed),
        "--multilayer-measurement-domain",
        solve.measurement_domain,
        "--multilayer-l2-weight",
        str(solve.l2_weight),
        "--multilayer-tv-weight",
        str(solve.tv_weight),
        "--multilayer-tv-iterations",
        str(solve.tv_iterations),
        "--multilayer-pupil-step-size",
        str(solve.pupil_step_size),
        "--multilayer-pupil-update-method",
        solve.pupil_update_method,
    ]
    if solve.step_size is not None:
        arguments.extend(("--multilayer-step-size", str(solve.step_size)))
    if not solve.restart_on_loss_increase:
        arguments.append("--no-multilayer-restart")
    if solve.random_order:
        arguments.append("--multilayer-random-order")
    if not solve.enforce_physical_sign:
        arguments.append("--no-multilayer-enforce-physical-sign")
    if solve.recover_pupil:
        arguments.append("--multilayer-recover-pupil")
    if request.background_detector_id is not None:
        arguments.extend(
            ("--multilayer-background-detector", request.background_detector_id)
        )
    if request.background_z_index is not None:
        arguments.extend(
            ("--multilayer-background-z-index", str(request.background_z_index))
        )
    if request.gpu_id is not None:
        arguments.extend(
            (
                "--multilayer-gpu-id",
                str(request.gpu_id),
                "--multilayer-gpu-reservation-mib",
                str(request.gpu_reservation_bytes // (1024 * 1024)),
            )
        )
    return tuple(arguments)


def read_processing_result_array(
    result: ProcessingResult,
) -> npt.NDArray[Any]:
    """Read and checksum-verify one published local Zarr result."""
    parsed = urlsplit(result.output.uri)
    if parsed.scheme != "file":
        raise ValueError("processing result must use a local file URI")
    path_text = unquote(parsed.path)
    if os.name == "nt" and path_text.startswith("/"):
        path_text = path_text[1:]
    if parsed.netloc:
        path_text = f"//{parsed.netloc}{path_text}"
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError("processing result URI did not resolve to an absolute path")
    node: Any = zarr.open(path, mode="r")
    if parsed.fragment:
        node = node[parsed.fragment]
    array = np.asarray(node[:])
    checksum = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    if checksum != result.output.checksum:
        raise ValueError("processing result checksum mismatch")
    return array


def run_offline_processing_plan(plan: OfflineProcessingPlan) -> ProcessingBatchResult:
    """Execute one resolved plan without constructing acquisition hardware."""
    gpu_requests: list[tuple[str, int, int]] = []
    if plan.request.gpu_id is not None:
        gpu_requests.append(
            (
                "gpu-mean-projection",
                plan.request.gpu_id,
                plan.request.gpu_reservation_bytes,
            )
        )
    if plan.request.tiled is not None and plan.request.tiled.gpu_id is not None:
        gpu_requests.append(
            (
                f"tiled-{plan.request.tiled.optical.model}-cupy",
                plan.request.tiled.gpu_id,
                plan.request.tiled.gpu_reservation_bytes,
            )
        )
    if (
        plan.request.multilayer is not None
        and plan.request.multilayer.gpu_id is not None
    ):
        gpu_requests.append(
            (
                f"multilayer-{plan.request.multilayer.reconstruction.model}-cupy",
                plan.request.multilayer.gpu_id,
                plan.request.multilayer.gpu_reservation_bytes,
            )
        )
    pool = GpuResourcePool(discover_nvidia_gpus()) if gpu_requests else None
    leases = []
    gpu_leases = {}
    supervisor: ProcessingSupervisor | None = None
    try:
        if pool is not None:
            for solver_id, gpu_id, reservation_bytes in gpu_requests:
                lease = pool.reserve(
                    solver_id,
                    GpuResourceRequest(
                        reservation_bytes=reservation_bytes,
                        allowed_device_ids=(gpu_id,),
                        preferred_device_ids=(gpu_id,),
                        exclusive=True,
                    ),
                )
                leases.append(lease)
                gpu_leases[solver_id] = lease
        gpu_cache_directories = {
            solver_id: str(
                (plan.request.output_root / ".cupy-cache" / solver_id).resolve()
            )
            for solver_id in gpu_leases
        }
        supervisor = ProcessingSupervisor(
            tuple(plan.jobs),
            gpu_leases=gpu_leases or None,
            gpu_cache_directories=gpu_cache_directories or None,
        )
        supervisor.start()
        return supervisor.run_batch(
            {
                solver_id: (job, plan.job_observations[solver_id])
                for solver_id, job in plan.jobs.items()
            }
        )
    finally:
        try:
            if supervisor is not None:
                supervisor.close()
        finally:
            if pool is not None:
                for lease in reversed(leases):
                    pool.release(lease)


__all__ = [
    "OfflineMultiLayerJobRequest",
    "OfflineProcessingPlan",
    "OfflineProcessingRequest",
    "OfflineTiledJobRequest",
    "read_processing_result_array",
    "resolve_offline_processing",
    "run_offline_processing_plan",
]
