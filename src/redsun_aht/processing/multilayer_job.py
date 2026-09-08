"""Typed offline job composition for nonlinear multi-layer reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from redsun_aht.domain import MultiSliceAcquisitionMode, ProcessingJob
from redsun_aht.processing.multilayer import (
    ModelName,
    MultiLayerConfig,
    create_multilayer_model,
    ring_illumination_frequencies,
)
from redsun_aht.processing.multilayer_iterative import MultiLayerSolveConfig
from redsun_aht.processing.replay import observations_from_dpct_bundle
from redsun_aht.processing.resources import (
    GpuResourcePool,
    GpuResourceRequest,
    discover_nvidia_gpus,
)
from redsun_aht.processing.supervisor import ProcessingBatchResult, ProcessingSupervisor

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from redsun_aht.domain import Observation
    from redsun_aht.domain.models import JsonValue

MeasurementNormalization = Literal["none", "shot_mean", "background"]
BackendName = Literal["numpy", "cupy"]


@dataclass(frozen=True, slots=True)
class MultiLayerReconstructionConfig:
    """Shape-independent optical and optimization inputs for one offline job."""

    model: ModelName
    depth_layers: int
    voxel_size_zyx_um: tuple[float, float, float]
    wavelength_um: float
    numerical_aperture: float
    refractive_index_medium: float
    illumination_na: float
    acquisition_mode: MultiSliceAcquisitionMode = (
        MultiSliceAcquisitionMode.AHT_REAL_AMPLITUDE_FOCAL_STACK
    )
    source_z_index: int = 0
    padding_yx: tuple[int, int] = (0, 0)
    defocus_um: float = 0.0
    focus_offsets_slices: tuple[float, ...] = (0.0,)
    illumination_fxy: tuple[tuple[float, float], ...] | None = None
    skip_shots: tuple[int, ...] = ()
    normalization: MeasurementNormalization = "shot_mean"
    solve: MultiLayerSolveConfig = field(default_factory=MultiLayerSolveConfig)
    memory_budget_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        """Validate inputs that do not depend on the selected observation."""
        try:
            acquisition_mode = MultiSliceAcquisitionMode(self.acquisition_mode)
        except (TypeError, ValueError) as error:
            raise ValueError("multi-slice acquisition mode is unsupported") from error
        object.__setattr__(self, "acquisition_mode", acquisition_mode)
        if self.model not in {"multi_born", "multislice"}:
            raise ValueError("multi-layer reconstruction model is unsupported")
        if type(self.depth_layers) is not int or self.depth_layers <= 0:
            raise ValueError("multi-layer depth must be a positive integer")
        if type(self.source_z_index) is not int or self.source_z_index < 0:
            raise ValueError("multi-layer source Z index cannot be negative")
        if (
            not np.isfinite(self.illumination_na)
            or self.illumination_na < 0
            or self.illumination_na > self.refractive_index_medium
        ):
            raise ValueError("multi-layer illumination NA is invalid")
        if self.normalization not in {"none", "shot_mean", "background"}:
            raise ValueError("multi-layer normalization is unsupported")
        if not self.focus_offsets_slices or any(
            not np.isfinite(value) for value in self.focus_offsets_slices
        ):
            raise ValueError("multi-layer focus offsets must be non-empty and finite")
        if self.model != "multislice" and self.focus_offsets_slices != (0.0,):
            raise ValueError("focus diversity is available only for multislice")
        if (
            self.model == "multislice"
            and acquisition_mode
            is MultiSliceAcquisitionMode.AHT_REAL_AMPLITUDE_FOCAL_STACK
            and self.focus_offsets_slices != (0.0,)
        ):
            raise ValueError(
                "AHT focal-stack acquisition does not use numerical refocusing"
            )
        if self.illumination_fxy is not None and any(
            len(pair) != 2 or not all(np.isfinite(value) for value in pair)
            for pair in self.illumination_fxy
        ):
            raise ValueError("explicit illuminations must contain finite FX,FY pairs")
        if len(set(self.skip_shots)) != len(self.skip_shots) or any(
            type(index) is not int or index < 0 for index in self.skip_shots
        ):
            raise ValueError("skipped shots must be unique zero-based indices")
        if self.model != "multislice" and self.skip_shots:
            raise ValueError("skipped shots are available only for multislice")
        if self.model == "multislice" and self.solve.recover_pupil:
            raise ValueError(
                "the current multislice model does not support pupil recovery"
            )
        if (
            type(self.memory_budget_bytes) is not int
            or self.memory_budget_bytes <= 0
            or self.memory_budget_bytes % (1024 * 1024)
        ):
            raise ValueError(
                "multi-layer memory budget must be positive whole mebibytes"
            )
        self.model_config((1, 1))

    def model_config(self, shape_yx: tuple[int, int]) -> MultiLayerConfig:
        """Resolve the complete model geometry for one observation XY shape."""
        return MultiLayerConfig(
            (self.depth_layers, *shape_yx),
            self.voxel_size_zyx_um,
            self.wavelength_um,
            self.numerical_aperture,
            self.refractive_index_medium,
            padding_yx=self.padding_yx,
            defocus_um=self.defocus_um,
            allow_evanescent_pupil=self.model == "multislice",
        )


@dataclass(frozen=True, slots=True)
class MultiLayerProcessingRequest:
    """Validated source, placement, and destination for one multi-layer job."""

    input_root: Path
    output_root: Path
    detector_id: str
    reconstruction: MultiLayerReconstructionConfig
    background_detector_id: str | None = None
    background_z_index: int | None = None
    backend: BackendName = "numpy"
    gpu_id: int | None = None
    gpu_reservation_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        """Normalize roots and reject ambiguous source or GPU selection."""
        input_root = self.input_root.resolve()
        output_root = self.output_root.resolve()
        if not input_root.is_dir():
            raise ValueError(
                f"multi-layer processing input is not a directory: {input_root}"
            )
        if output_root == input_root or input_root in output_root.parents:
            raise ValueError("multi-layer output cannot be inside the input run")
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
        if self.background_z_index is not None and (
            type(self.background_z_index) is not int or self.background_z_index < 0
        ):
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
        if (
            type(self.gpu_reservation_bytes) is not int
            or self.gpu_reservation_bytes <= 0
            or self.gpu_reservation_bytes % (1024 * 1024)
        ):
            raise ValueError(
                "multi-layer GPU reservation must be positive whole mebibytes"
            )
        object.__setattr__(self, "input_root", input_root)
        object.__setattr__(self, "output_root", output_root)


@dataclass(frozen=True, slots=True)
class MultiLayerProcessingPlan:
    """Resolved observations, memory estimate, and immutable worker job."""

    request: MultiLayerProcessingRequest
    source_run_id: str
    observations: tuple[Observation, ...]
    working_set_estimate_bytes: int
    job: ProcessingJob

    @property
    def jobs(self) -> Mapping[str, ProcessingJob]:
        """Expose the single job in supervisor-compatible immutable form."""
        return MappingProxyType({self.job.solver_id: self.job})


def resolve_multilayer_processing(
    request: MultiLayerProcessingRequest,
) -> MultiLayerProcessingPlan:
    """Integrity-replay and preflight one offline nonlinear reconstruction."""
    available = observations_from_dpct_bundle(request.input_root)
    by_detector = {item.detector_id: item for item in available}
    try:
        sample = by_detector[request.detector_id]
    except KeyError as error:
        raise ValueError(
            f"multi-layer sample detector is absent: {request.detector_id}"
        ) from error
    if len(sample.array.shape) != 4:
        raise ValueError("multi-layer sample observation must have CZYX axes")
    shots, acquisition_z, ny, nx = sample.array.shape
    if request.reconstruction.source_z_index >= acquisition_z:
        raise ValueError("multi-layer source Z index is out of range")
    selected = [sample]
    background_z_index = request.background_z_index
    if request.background_detector_id is not None:
        try:
            background = by_detector[request.background_detector_id]
        except KeyError as error:
            raise ValueError(
                "multi-layer background detector is absent: "
                f"{request.background_detector_id}"
            ) from error
        if len(background.array.shape) != 4:
            raise ValueError("multi-layer background observation must have CZYX axes")
        background_shots, background_z, background_y, background_x = (
            background.array.shape
        )
        if (background_shots, background_y, background_x) != (shots, ny, nx):
            raise ValueError("multi-layer background must share sample SYX shape")
        selected_background_z = (
            request.reconstruction.source_z_index
            if background_z_index is None
            else background_z_index
        )
        if selected_background_z >= background_z:
            raise ValueError("multi-layer background Z index is out of range")
        background_z_index = selected_background_z
        selected.append(background)

    model_config = request.reconstruction.model_config((ny, nx))
    model = create_multilayer_model(request.reconstruction.model, model_config)
    estimate = model.estimated_working_set_bytes(shots)
    retained_planes = (
        len(request.reconstruction.focus_offsets_slices)
        if request.reconstruction.model == "multislice"
        else 1
    )
    estimate += 2 * shots * retained_planes * ny * nx * np.dtype(np.complex64).itemsize
    budget = (
        request.reconstruction.memory_budget_bytes
        if request.backend == "numpy"
        else request.gpu_reservation_bytes
    )
    if estimate > budget:
        raise MemoryError(
            f"multi-layer working-set estimate {estimate} exceeds budget {budget}"
        )
    if request.reconstruction.illumination_fxy is None:
        frequencies = ring_illumination_frequencies(
            shots,
            wavelength_um=request.reconstruction.wavelength_um,
            illumination_na=request.reconstruction.illumination_na,
        )
    else:
        frequencies = np.asarray(
            request.reconstruction.illumination_fxy, dtype=np.float64
        )
        if frequencies.shape != (shots, 2):
            raise ValueError("explicit illuminations must match the shot axis")
    if any(index >= shots for index in request.reconstruction.skip_shots):
        raise ValueError("skipped shot index is out of range")
    solver_id = f"multilayer-{request.reconstruction.model}-{request.backend}"
    configuration = cast(
        "dict[str, JsonValue]",
        {
            "reconstruction": asdict(request.reconstruction),
            "model_config": asdict(model_config),
            "illumination_fxy": frequencies.tolist(),
            "working_set_estimate_bytes": estimate,
            "background_z_index": background_z_index,
        },
    )
    resources: dict[str, JsonValue] = {
        "cpu_threads": 1,
        "gpu_ids": [] if request.gpu_id is None else [request.gpu_id],
        "memory_budget_bytes": budget,
        "working_set_estimate_bytes": estimate,
    }
    if request.gpu_id is not None:
        resources["gpu_memory_reservation_bytes"] = request.gpu_reservation_bytes
    job = ProcessingJob(
        job_id=f"{sample.run_id}-{solver_id}-{sample.detector_id}",
        solver_id=solver_id,
        solver_version=(
            "2.0.0" if request.reconstruction.model == "multislice" else "1.0.0"
        ),
        input_selectors=tuple(item.observation_id for item in selected),
        configuration=configuration,
        priority=0,
        latency_class="offline",
        resource_request=resources,
        output_destination=str((request.output_root / solver_id).resolve()),
        cancellation_policy="cooperative",
    )
    return MultiLayerProcessingPlan(
        request, sample.run_id, tuple(selected), estimate, job
    )


def run_multilayer_processing_plan(
    plan: MultiLayerProcessingPlan,
) -> ProcessingBatchResult:
    """Execute one resolved multi-layer job in a persistent detached worker."""
    pool = None
    lease = None
    gpu_leases = None
    cache_directories = None
    if plan.request.gpu_id is not None:
        pool = GpuResourcePool(discover_nvidia_gpus())
        lease = pool.reserve(
            plan.job.solver_id,
            GpuResourceRequest(
                reservation_bytes=plan.request.gpu_reservation_bytes,
                allowed_device_ids=(plan.request.gpu_id,),
                preferred_device_ids=(plan.request.gpu_id,),
                exclusive=True,
            ),
        )
        gpu_leases = {plan.job.solver_id: lease}
        cache_directories = {
            plan.job.solver_id: str(
                (plan.request.output_root / ".cupy-cache").resolve()
            )
        }
    supervisor = ProcessingSupervisor(
        (plan.job.solver_id,),
        gpu_leases=gpu_leases,
        gpu_cache_directories=cache_directories,
    )
    try:
        supervisor.start()
        return supervisor.run_batch({plan.job.solver_id: (plan.job, plan.observations)})
    finally:
        try:
            supervisor.close()
        finally:
            if pool is not None and lease is not None:
                pool.release(lease)


__all__ = [
    "BackendName",
    "MeasurementNormalization",
    "MultiLayerProcessingPlan",
    "MultiLayerProcessingRequest",
    "MultiLayerReconstructionConfig",
    "resolve_multilayer_processing",
    "run_multilayer_processing_plan",
]
