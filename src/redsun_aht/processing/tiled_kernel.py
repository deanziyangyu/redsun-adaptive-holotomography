"""Offline-only spawned-kernel adapter for tiled NumPy DPCT/qOBT."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from redsun_aht.domain import ProcessingJob, SolverCapabilities
from redsun_aht.domain.models import JsonValue, thaw_json
from redsun_aht.processing.fixed_median import fixed_median_from_czyx
from redsun_aht.processing.quantitative import (
    CupyQuantitativeRuntime,
    CupyQuantitativeTileSolver,
    NumpyQuantitativeTileSolver,
    QuantitativeTileConfig,
)
from redsun_aht.processing.replay import read_observation_array
from redsun_aht.processing.solvers import KernelOutput
from redsun_aht.processing.tiling import TiledVolumeReconstructor, TilingConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from redsun_aht.domain import Observation


TILED_DONOR_COMMIT = "405e53b35498898dc7e973f9840c0259ace7671a"


class TiledQuantitativeKernel:
    """Persistent worker kernel for one CPU or CUDA DPCT/qOBT model."""

    def __init__(
        self,
        model: Literal["dpct", "qobt"],
        *,
        backend: Literal["numpy", "cupy"] = "numpy",
        gpu_id: int | None = None,
        gpu_cache_directory: str | None = None,
    ) -> None:
        self.model = model
        self.backend = backend
        if backend == "cupy" and gpu_id is None:
            raise ValueError("tiled CuPy reconstruction requires an explicit GPU ID")
        if backend == "numpy" and gpu_id is not None:
            raise ValueError("tiled NumPy reconstruction cannot receive a GPU ID")
        self._runtime = (
            None
            if gpu_id is None
            else CupyQuantitativeRuntime(
                gpu_id,
                cache_directory=gpu_cache_directory,
            )
        )
        self.capabilities = SolverCapabilities(
            solver_id=f"tiled-{model}-{backend}",
            solver_version="1.0.0",
            dimensionality=(4,),
            required_channels=1,
            required_shots=4,
            streaming_supported=False,
            batch_supported=True,
            incremental_supported=False,
            live_supported=False,
            output_types=(f"tiled_{model}_ri_zyx",),
            requires_gpu=backend == "cupy",
            gpu_runtime="cupy-cuda12x" if backend == "cupy" else None,
        )
        self._signature: str | None = None
        self._reconstructor: TiledVolumeReconstructor | None = None

    @property
    def gpu_id(self) -> int | None:
        """Return the worker-owned CUDA device, if selected."""
        return None if self._runtime is None else self._runtime.gpu_id

    @property
    def allocated_memory_bytes(self) -> int:
        """Return current CuPy allocator reservation."""
        return 0 if self._runtime is None else self._runtime.allocated_memory_bytes

    @property
    def observed_memory_bytes(self) -> int:
        """Return the CuPy allocator high-water mark."""
        return 0 if self._runtime is None else self._runtime.observed_memory_bytes

    def run(
        self,
        job: ProcessingJob,
        observations: tuple[Observation, ...],
        cancelled: Callable[[], bool],
    ) -> KernelOutput:
        """Resolve one sample/background selection and publish a tiled RI volume."""
        if not 1 <= len(observations) <= 2:
            raise ValueError(
                "tiled quantitative job requires sample and optional background"
            )
        configuration = job.configuration
        optical = _quantitative_config(configuration, self.model)
        tiling = _tiling_config(configuration)
        signature = json.dumps(
            thaw_json(configuration), separators=(",", ":"), sort_keys=True
        )
        if self._signature != signature:
            if self._reconstructor is not None:
                self._reconstructor.close()
            tile_solver = (
                NumpyQuantitativeTileSolver(optical)
                if self._runtime is None
                else CupyQuantitativeTileSolver(optical, self._runtime)
            )
            self._reconstructor = TiledVolumeReconstructor(tile_solver, tiling)
            self._signature = signature

        sample_observation = observations[0]
        sample_czyx = read_observation_array(sample_observation)
        if sample_czyx.ndim != 4:
            raise ValueError("tiled quantitative sample must be one CZYX observation")
        sample_zsyx = np.transpose(sample_czyx, (1, 0, 2, 3))
        background_syx = None
        background_observation = None
        background_z_index = _optional_int(configuration, "background_z_index")
        fixed_median = bool(configuration.get("fixed_median_from_sample", False))
        fixed_median_checksum = None
        if fixed_median:
            if len(observations) != 1 or background_z_index is not None:
                raise ValueError("fixed median requires exactly one sample observation")
            median = fixed_median_from_czyx(
                sample_czyx,
                source_checksum=sample_observation.array.checksum,
            )
            background_syx = median.values_syx
            fixed_median_checksum = str(median.provenance["median_checksum"])
        elif len(observations) == 2:
            background_observation = observations[1]
            background_czyx = read_observation_array(background_observation)
            if background_czyx.ndim != 4:
                raise ValueError("tiled quantitative background must be CZYX")
            selected_z = 0 if background_z_index is None else background_z_index
            if not 0 <= selected_z < background_czyx.shape[1]:
                raise ValueError(
                    "tiled quantitative background Z index is out of range"
                )
            background_syx = background_czyx[:, selected_z]
        elif background_z_index is not None:
            raise ValueError("background Z index requires a background observation")

        if self._reconstructor is None:  # pragma: no cover - construction invariant
            raise RuntimeError("tiled quantitative reconstructor was not initialized")
        result = self._reconstructor.run(
            sample_zsyx,
            background_syx,
            cancelled=cancelled,
        )
        provenance = dict(result.as_provenance())
        provenance.update(
            {
                "donor": {
                    "repository": "pyhololab",
                    "commit": TILED_DONOR_COMMIT,
                    "path": "src/core/tiled_reconstruction.py",
                },
                "source": {
                    "sample_observation_id": sample_observation.observation_id,
                    "sample_detector_id": sample_observation.detector_id,
                    "sample_checksum": sample_observation.array.checksum,
                    "sample_configuration_revision": (
                        sample_observation.configuration_revision
                    ),
                    "sample_calibration_ids": sample_observation.calibration_ids,
                    "background_observation_id": (
                        None
                        if background_observation is None
                        else background_observation.observation_id
                    ),
                    "background_checksum": (
                        None
                        if background_observation is None
                        else background_observation.array.checksum
                    ),
                    "background_z_index": background_z_index,
                    "fixed_median_from_sample": fixed_median,
                    "fixed_median_checksum": fixed_median_checksum,
                },
                "scientific_status": (
                    "donor-characterized-cpu-reference"
                    if self.backend == "numpy"
                    else "cpu-equivalent-cupy-reference"
                ),
            }
        )
        tile_timings = result.tile_timing_ns
        metrics: dict[str, JsonValue] = {
            "model": self.model,
            "backend": self.backend,
            "tile_count": len(result.plan.placements),
            "tile_y": result.plan.tile_shape_yx[0],
            "tile_x": result.plan.tile_shape_yx[1],
            "overlap_y": result.plan.overlap_yx[0],
            "overlap_x": result.plan.overlap_yx[1],
            "total_timing_ns": result.total_timing_ns,
            "maximum_tile_timing_ns": max(tile_timings),
            "donor_commit": TILED_DONOR_COMMIT,
        }
        if self._runtime is not None:
            metrics.update(
                {
                    "gpu_device_id": self._runtime.gpu_id,
                    "cuda_runtime_version": self._runtime.runtime_version,
                    "cuda_driver_version": self._runtime.driver_version,
                    "gpu_context_startup_ns": self._runtime.startup_context_ns,
                    "gpu_allocator_reserved_bytes": (
                        self._runtime.allocated_memory_bytes
                    ),
                    "gpu_allocator_used_bytes": (self._runtime.observed_memory_bytes),
                }
            )
        return KernelOutput(
            result.volume_zyx,
            ("z", "y", "x"),
            metrics,
            ome_image=True,
            provenance=cast("Mapping[str, JsonValue]", provenance),
        )

    def close(self) -> None:
        """Release persistent transfer functions idempotently."""
        if self._reconstructor is not None:
            self._reconstructor.close()
            self._reconstructor = None
        self._signature = None
        if self._runtime is not None:
            self._runtime.close()


def _quantitative_config(
    configuration: Mapping[str, JsonValue], model: Literal["dpct", "qobt"]
) -> QuantitativeTileConfig:
    configured_model = configuration.get("model")
    if configured_model != model:
        raise ValueError("tiled quantitative job model does not match its worker")
    optical = _mapping(configuration, "optical")
    source_azimuth = _float_tuple(optical, "source_azimuth_deg", length=4)
    return QuantitativeTileConfig(
        model=model,
        wavelength_um=_float(optical, "wavelength_um"),
        numerical_aperture=_float(optical, "numerical_aperture"),
        pixel_size_um=_float(optical, "pixel_size_um"),
        pixel_size_z_um=_float(optical, "pixel_size_z_um"),
        refractive_index_medium=_float(optical, "refractive_index_medium", default=1.0),
        refractive_index_sample=_float(optical, "refractive_index_sample", default=1.4),
        source_azimuth_deg=source_azimuth,
        source_elevation_deg=_float(optical, "source_elevation_deg", default=45.0),
        source_sigma=_float(optical, "source_sigma", default=0.5),
        background_division=_literal(
            optical,
            "background_division",
            ("background_over_sample", "sample_over_background"),
            "background_over_sample",
        ),
        regularization_real=_float(optical, "regularization_real", default=5e-5),
        regularization_imaginary=_float(
            optical, "regularization_imaginary", default=5e-5
        ),
        regularization_scalar=_float(optical, "regularization_scalar", default=1e-2),
        axial_window=_literal(
            optical,
            "axial_window",
            ("hamming", "rectangular", "none"),
            "hamming",
        ),
    )


def _tiling_config(configuration: Mapping[str, JsonValue]) -> TilingConfig:
    tiling = _mapping(configuration, "tiling")
    explicit_shape = _optional_int_pair(tiling, "explicit_shape_yx")
    explicit_overlap = _optional_int_pair(tiling, "explicit_overlap_yx")
    return TilingConfig(
        mode=_literal(tiling, "mode", ("auto", "off"), "auto"),
        voxel_budget=_int(tiling, "voxel_budget", default=512 * 512 * 100),
        max_shape_yx=_int_pair(tiling, "max_shape_yx", default=(768, 768)),
        min_shape_yx=_int_pair(tiling, "min_shape_yx", default=(256, 256)),
        alignment_px=_int(tiling, "alignment_px", default=32),
        overlap_fraction=_float(tiling, "overlap_fraction", default=0.20),
        blend=_literal(tiling, "blend", ("raised_cosine",), "raised_cosine"),
        pad_mode=_literal(tiling, "pad_mode", ("reflect", "edge"), "reflect"),
        explicit_shape_yx=explicit_shape,
        explicit_overlap_yx=explicit_overlap,
    )


def _mapping(mapping: Mapping[str, JsonValue], key: str) -> Mapping[str, JsonValue]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"tiled quantitative configuration requires {key}")
    return value


def _float(
    mapping: Mapping[str, JsonValue], key: str, *, default: float | None = None
) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"tiled quantitative {key} must be numeric")
    return float(value)


def _int(
    mapping: Mapping[str, JsonValue], key: str, *, default: int | None = None
) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"tiled quantitative {key} must be an integer")
    return value


def _optional_int(mapping: Mapping[str, JsonValue], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"tiled quantitative {key} must be an integer")
    return value


def _sequence(mapping: Mapping[str, JsonValue], key: str) -> Sequence[JsonValue]:
    value = mapping.get(key)
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"tiled quantitative {key} must be a sequence")
    return value


def _float_tuple(
    mapping: Mapping[str, JsonValue], key: str, *, length: int
) -> tuple[float, ...]:
    values = _sequence(mapping, key)
    if len(values) != length or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise ValueError(f"tiled quantitative {key} must contain {length} numbers")
    return tuple(float(cast("float", value)) for value in values)


def _int_pair(
    mapping: Mapping[str, JsonValue],
    key: str,
    *,
    default: tuple[int, int],
) -> tuple[int, int]:
    if key not in mapping:
        return default
    value = _optional_int_pair(mapping, key)
    if value is None:  # pragma: no cover - key-presence invariant
        raise ValueError(f"tiled quantitative {key} cannot be null")
    return value


def _optional_int_pair(
    mapping: Mapping[str, JsonValue], key: str
) -> tuple[int, int] | None:
    value = mapping.get(key)
    if value is None:
        return None
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"tiled quantitative {key} must contain two integers")
    return cast("tuple[int, int]", tuple(value))


def _literal(
    mapping: Mapping[str, JsonValue],
    key: str,
    allowed: tuple[str, ...],
    default: str,
) -> Any:
    value = mapping.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"tiled quantitative {key} is unsupported")
    return value


__all__ = ["TILED_DONOR_COMMIT", "TiledQuantitativeKernel"]
