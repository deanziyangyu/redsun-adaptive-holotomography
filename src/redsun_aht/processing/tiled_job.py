"""Typed offline job composition for tiled quantitative reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from redsun_aht.domain import ProcessingJob
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
    from redsun_aht.processing.quantitative import QuantitativeTileConfig
    from redsun_aht.processing.tiling import TilingConfig


@dataclass(frozen=True, slots=True)
class TiledProcessingRequest:
    """Validated user intent for one offline-only tiled quantitative job."""

    input_root: Path
    output_root: Path
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
        """Normalize roots and reject source mutation or ambiguous background."""
        input_root = self.input_root.resolve()
        output_root = self.output_root.resolve()
        if not input_root.is_dir():
            raise ValueError(f"tiled processing input is not a directory: {input_root}")
        if output_root == input_root or input_root in output_root.parents:
            raise ValueError("tiled processing output cannot be inside the input run")
        if not self.detector_id:
            raise ValueError("tiled processing detector identity cannot be empty")
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
            raise ValueError("tiled processing backend must be NumPy or CuPy")
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
        object.__setattr__(self, "input_root", input_root)
        object.__setattr__(self, "output_root", output_root)


@dataclass(frozen=True, slots=True)
class TiledProcessingPlan:
    """Resolved source observations and one immutable offline job."""

    request: TiledProcessingRequest
    source_run_id: str
    observations: tuple[Observation, ...]
    job: ProcessingJob

    @property
    def jobs(self) -> Mapping[str, ProcessingJob]:
        """Expose the single job in supervisor-compatible immutable form."""
        return MappingProxyType({self.job.solver_id: self.job})


def resolve_tiled_processing(request: TiledProcessingRequest) -> TiledProcessingPlan:
    """Integrity-replay a bundle and select exact sample/background detectors."""
    available = observations_from_dpct_bundle(request.input_root)
    by_detector = {item.detector_id: item for item in available}
    try:
        sample = by_detector[request.detector_id]
    except KeyError as error:
        raise ValueError(
            f"tiled sample detector is absent from the run: {request.detector_id}"
        ) from error
    selected = [sample]
    if request.background_detector_id is not None:
        try:
            selected.append(by_detector[request.background_detector_id])
        except KeyError as error:
            raise ValueError(
                "tiled background detector is absent from the run: "
                f"{request.background_detector_id}"
            ) from error
    model: Literal["dpct", "qobt"] = request.optical.model
    solver_id = f"tiled-{model}-{request.backend}"
    configuration: dict[str, JsonValue] = {
        "model": model,
        "optical": asdict(request.optical),
        "tiling": asdict(request.tiling),
    }
    if request.background_z_index is not None:
        configuration["background_z_index"] = request.background_z_index
    if request.fixed_median_from_sample:
        configuration["fixed_median_from_sample"] = True
    job = ProcessingJob(
        job_id=f"{sample.run_id}-{solver_id}-{sample.detector_id}",
        solver_id=solver_id,
        solver_version="1.0.0",
        input_selectors=tuple(item.observation_id for item in selected),
        configuration=configuration,
        priority=0,
        latency_class="offline",
        resource_request=(
            {"cpu_threads": 1, "gpu_ids": []}
            if request.gpu_id is None
            else {
                "cpu_threads": 1,
                "gpu_ids": [request.gpu_id],
                "gpu_memory_reservation_bytes": request.gpu_reservation_bytes,
            }
        ),
        output_destination=str((request.output_root / solver_id).resolve()),
        cancellation_policy="cooperative",
    )
    return TiledProcessingPlan(request, sample.run_id, tuple(selected), job)


def run_tiled_processing_plan(plan: TiledProcessingPlan) -> ProcessingBatchResult:
    """Execute one resolved tiled plan in its own persistent spawned worker."""
    pool = None
    lease = None
    gpu_leases = None
    gpu_cache_directories = None
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
        gpu_cache_directories = {
            plan.job.solver_id: str(
                (plan.request.output_root / ".cupy-cache").resolve()
            )
        }
    supervisor = ProcessingSupervisor(
        (plan.job.solver_id,),
        gpu_leases=gpu_leases,
        gpu_cache_directories=gpu_cache_directories,
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
    "TiledProcessingPlan",
    "TiledProcessingRequest",
    "resolve_tiled_processing",
    "run_tiled_processing_plan",
]
