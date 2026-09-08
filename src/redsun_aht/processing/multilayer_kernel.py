"""Offline-only worker adapter for nonlinear multi-layer reconstruction."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from redsun_aht.domain import (
    MultiSliceAcquisitionMode,
    ProcessingJob,
    SolverCapabilities,
)
from redsun_aht.domain.models import JsonValue, thaw_json
from redsun_aht.processing.multilayer import (
    MultiLayerConfig,
    MultiSliceModel,
    create_multilayer_model,
)
from redsun_aht.processing.multilayer_iterative import (
    MultiLayerSolveConfig,
    solve_multilayer,
    solve_multislice,
)
from redsun_aht.processing.quantitative import CupyQuantitativeRuntime
from redsun_aht.processing.replay import read_observation_array
from redsun_aht.processing.solvers import (
    KernelOutput,
    ProcessingCancelled,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from redsun_aht.domain import Observation

MULTILAYER_DONOR_COMMIT = "e8dd96bfa2d54784e00209769d594f305863b01a"
MSBP_DONOR_VERSION = "2025-08-16"
MSBP_DONOR_SHA256: dict[str, str] = {
    "Recon_main.m": "b083a3a8974d1b082401b510da0833688f0a91a5e15fb83d0868e86e6f94c6b2",
    "Recon_corefunction/MultiSlice_Forward.m": (
        "1656d86eca1c746eff0a4821e983b7afa370d81c9ac3963a62d09b909baa2fd1"
    ),
    "Recon_corefunction/BPM_update.m": (
        "5b11140e2bb33e4c393f977a025e71861c5afc27d1ecb264385f9130fd7a5ed3"
    ),
    "Recon_corefunction/generate_Prop_dataset.m": (
        "93f6f7403a1a9f16be89aede97b807f3056c3f7e438e61eed5c5f2bf8ac2972"
    ),
    "Recon_corefunction/Rytov_recon_init.m": (
        "48c8fc38618d13ee41e395f157468ed20064f04943c77de471ba1fbb36eefc2c"
    ),
}


class MultiLayerKernel:
    """Persistent CPU or CUDA worker kernel for one nonlinear model family."""

    def __init__(
        self,
        model: Literal["multi_born", "multislice"],
        *,
        backend: Literal["numpy", "cupy"] = "numpy",
        gpu_id: int | None = None,
        gpu_cache_directory: str | None = None,
    ) -> None:
        if backend == "cupy" and gpu_id is None:
            raise ValueError("multi-layer CuPy reconstruction requires a GPU ID")
        if backend == "numpy" and gpu_id is not None:
            raise ValueError("multi-layer NumPy reconstruction cannot receive a GPU")
        self.model_name = model
        self.backend = backend
        self._runtime = (
            None
            if gpu_id is None
            else CupyQuantitativeRuntime(
                gpu_id,
                cache_directory=gpu_cache_directory,
            )
        )
        self.capabilities = SolverCapabilities(
            solver_id=f"multilayer-{model}-{backend}",
            solver_version="2.0.0" if model == "multislice" else "1.0.0",
            dimensionality=(4,),
            required_channels=1,
            required_shots=1,
            streaming_supported=False,
            batch_supported=True,
            incremental_supported=False,
            live_supported=False,
            output_types=(f"multilayer_{model}_ri_zyx",),
            requires_gpu=backend == "cupy",
            gpu_runtime="cupy-cuda12x" if backend == "cupy" else None,
        )

    @property
    def gpu_id(self) -> int | None:
        """Return the worker-owned CUDA placement, if selected."""
        return None if self._runtime is None else self._runtime.gpu_id

    @property
    def allocated_memory_bytes(self) -> int:
        """Return current CuPy allocator reservation."""
        return 0 if self._runtime is None else self._runtime.allocated_memory_bytes

    @property
    def observed_memory_bytes(self) -> int:
        """Return the largest observed CuPy allocator use."""
        return 0 if self._runtime is None else self._runtime.observed_memory_bytes

    def run(
        self,
        job: ProcessingJob,
        observations: tuple[Observation, ...],
        cancelled: Callable[[], bool],
    ) -> KernelOutput:
        """Normalize one selected SYX plane and reconstruct one RI volume."""
        if not 1 <= len(observations) <= 2:
            raise ValueError("multi-layer job requires sample and optional background")
        model_config = _model_config(job.configuration)
        reconstruction = _mapping(job.configuration, "reconstruction")
        configured_model = _string(reconstruction, "model")
        if configured_model != self.model_name:
            raise ValueError("multi-layer job model does not match worker")
        mode_value = reconstruction.get(
            "acquisition_mode",
            MultiSliceAcquisitionMode.AHT_REAL_AMPLITUDE_FOCAL_STACK.value,
        )
        if not isinstance(mode_value, str):
            raise ValueError("multi-layer acquisition mode is invalid")
        try:
            acquisition_mode = MultiSliceAcquisitionMode(mode_value)
        except ValueError as error:
            raise ValueError("multi-layer acquisition mode is unsupported") from error
        solve_config = _solve_config(reconstruction)
        source_z_index = _integer(reconstruction, "source_z_index")
        normalization = _string(reconstruction, "normalization")
        frequencies = np.asarray(
            _sequence(job.configuration, "illumination_fxy"), dtype=np.float64
        )
        focus_offsets = _number_tuple(reconstruction, "focus_offsets_slices", None)
        if (
            configured_model == "multislice"
            and acquisition_mode
            is MultiSliceAcquisitionMode.AHT_REAL_AMPLITUDE_FOCAL_STACK
            and focus_offsets != (0.0,)
        ):
            raise ValueError(
                "AHT focal-stack acquisition does not use numerical refocusing"
            )
        skip_shots = _integer_tuple(reconstruction, "skip_shots", None)
        expected_estimate = _integer(job.configuration, "working_set_estimate_bytes")
        budget = job.resource_request.get("memory_budget_bytes")
        requested_estimate = job.resource_request.get("working_set_estimate_bytes")
        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or not isinstance(requested_estimate, int)
            or isinstance(requested_estimate, bool)
            or requested_estimate != expected_estimate
            or expected_estimate > budget
        ):
            raise MemoryError("multi-layer worker memory preflight does not match job")

        sample_observation = observations[0]
        sample = read_observation_array(sample_observation)
        if sample.ndim != 4 or tuple(sample.shape[-2:]) != model_config.shape_zyx[-2:]:
            raise ValueError("multi-layer sample must match configured CZYX geometry")
        if not 0 <= source_z_index < sample.shape[1]:
            raise ValueError("multi-layer source Z index is out of range")
        measurements = np.asarray(sample[:, source_z_index])
        measurement_dtype = (
            np.complex64 if np.iscomplexobj(measurements) else np.float32
        )
        measurements = np.asarray(measurements, dtype=measurement_dtype)
        background_observation = None
        background_z_index = job.configuration.get("background_z_index")
        if normalization == "background":
            if len(observations) != 2:
                raise ValueError("background normalization requires an observation")
            if not isinstance(background_z_index, int) or isinstance(
                background_z_index, bool
            ):
                raise ValueError("background normalization requires a Z index")
            background_observation = observations[1]
            background = read_observation_array(background_observation)
            if (
                background.ndim != 4
                or not 0 <= background_z_index < background.shape[1]
                or (background.shape[0], *background.shape[-2:]) != measurements.shape
            ):
                raise ValueError("multi-layer background does not match sample SYX")
            denominator = np.asarray(background[:, background_z_index])
            safe = np.where(
                np.abs(denominator) > np.finfo(np.float32).eps,
                denominator,
                1,
            )
            measurements = measurements / safe
        elif len(observations) != 1 or background_z_index is not None:
            raise ValueError("multi-layer job contains an unused background")
        elif normalization == "shot_mean":
            shot_means = np.mean(np.abs(measurements), axis=(-2, -1), dtype=np.float64)
            if np.any(shot_means == 0):
                raise ValueError("multi-layer shot normalization encountered zero mean")
            measurements /= shot_means[:, None, None]
        elif normalization != "none":
            raise ValueError("multi-layer normalization is unsupported")

        shots, ny, nx = measurements.shape
        recomputed_estimate = _working_set_estimate(
            model_config,
            shots,
            model_name=self.model_name,
            focus_planes=len(focus_offsets),
        )
        if recomputed_estimate != expected_estimate or frequencies.shape != (shots, 2):
            raise ValueError("multi-layer resolved geometry changed after preflight")

        def check_cancelled() -> None:
            if cancelled():
                raise ProcessingCancelled("multi-layer processing was cancelled")

        started_ns = time.perf_counter_ns()
        if self._runtime is None:
            model = create_multilayer_model(self.model_name, model_config)
            if self.model_name == "multislice":
                if not isinstance(model, MultiSliceModel):
                    raise RuntimeError(
                        "multislice model factory returned the wrong type"
                    )
                solved = solve_multislice(
                    model,
                    measurements,
                    frequencies,
                    focus_offsets,
                    skip_shots,
                    solve_config,
                    check_cancelled=check_cancelled,
                )
            else:
                solved = solve_multilayer(
                    model,
                    measurements,
                    frequencies,
                    solve_config,
                    check_cancelled=check_cancelled,
                )
            output = np.asarray(solved.refractive_index_zyx, dtype=np.float32)
            pupil = np.asarray(solved.pupil_yx, dtype=np.complex64)
        else:
            runtime = self._runtime
            with runtime.device, runtime.stream:
                free_bytes, _ = runtime.xp.cuda.runtime.memGetInfo()
                if expected_estimate > int(0.8 * free_bytes):
                    raise MemoryError(
                        "multi-layer estimate exceeds 80% of free GPU memory"
                    )
                model = create_multilayer_model(
                    self.model_name, model_config, xp=runtime.xp
                )
                if self.model_name == "multislice":
                    if not isinstance(model, MultiSliceModel):
                        raise RuntimeError(
                            "multislice model factory returned the wrong type"
                        )
                    solved = solve_multislice(
                        model,
                        runtime.xp.asarray(measurements),
                        frequencies,
                        focus_offsets,
                        skip_shots,
                        solve_config,
                        check_cancelled=check_cancelled,
                    )
                else:
                    solved = solve_multilayer(
                        model,
                        runtime.xp.asarray(measurements),
                        frequencies,
                        solve_config,
                        check_cancelled=check_cancelled,
                    )
                output = runtime.xp.asnumpy(solved.refractive_index_zyx)
                pupil = runtime.xp.asnumpy(solved.pupil_yx)
            runtime.stream.synchronize()
            runtime.observe_memory()
            output = np.asarray(output, dtype=np.float32)
            pupil = np.asarray(pupil, dtype=np.complex64)
        elapsed_ns = time.perf_counter_ns() - started_ns
        if output.shape != (model_config.shape_zyx[0], ny, nx) or not np.all(
            np.isfinite(output)
        ):
            raise ValueError("multi-layer reconstruction produced an invalid volume")

        source: dict[str, JsonValue] = {
            "sample_observation_id": sample_observation.observation_id,
            "sample_detector_id": sample_observation.detector_id,
            "sample_checksum": sample_observation.array.checksum,
            "sample_z_index": source_z_index,
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
        }
        provenance: dict[str, JsonValue] = {
            "model": self.model_name,
            "backend": self.backend,
            "model_config": cast(
                "JsonValue", thaw_json(_mapping(job.configuration, "model_config"))
            ),
            "reconstruction": cast(
                "JsonValue", thaw_json(_mapping(job.configuration, "reconstruction"))
            ),
            "illumination_fxy_cycles_per_um": frequencies.tolist(),
            "loss_history": list(solved.loss_history),
            "step_history": list(solved.step_history),
            "restart_history": list(solved.restart_history),
            "working_set_estimate_bytes": expected_estimate,
            "pupil_sha256": hashlib.sha256(pupil.tobytes(order="C")).hexdigest(),
            "source": source,
            "donor": _donor_provenance(self.model_name),
            "scientific_status": (
                "cropped-40um-phantom-convergence-tested"
                if self.model_name == "multislice"
                else (
                    "donor-characterized-cpu-reference"
                    if self.backend == "numpy"
                    else "cpu-equivalent-cupy-reference"
                )
            ),
        }
        metrics: dict[str, JsonValue] = {
            "model": self.model_name,
            "backend": self.backend,
            "iterations": len(solved.loss_history),
            "final_loss": solved.loss_history[-1],
            "restart_count": sum(solved.restart_history),
            "phase_change_per_slice": solved.phase_change_per_slice,
            "working_set_estimate_bytes": expected_estimate,
            "total_timing_ns": elapsed_ns,
        }
        if self.model_name == "multislice":
            metrics["donor_version"] = MSBP_DONOR_VERSION
        else:
            metrics["donor_commit"] = MULTILAYER_DONOR_COMMIT
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
            output,
            ("z", "y", "x"),
            metrics,
            ome_image=True,
            provenance=provenance,
        )

    def close(self) -> None:
        """Release the optional persistent worker-owned CUDA runtime."""
        if self._runtime is not None:
            self._runtime.close()


def _working_set_estimate(
    config: MultiLayerConfig,
    shots: int,
    *,
    model_name: str = "multislice",
    focus_planes: int = 1,
) -> int:
    model = create_multilayer_model(model_name, config)
    _, ny, nx = config.shape_zyx
    return model.estimated_working_set_bytes(shots) + (
        2 * shots * focus_planes * ny * nx * np.dtype(np.complex64).itemsize
    )


def _donor_provenance(model_name: str) -> dict[str, JsonValue]:
    if model_name == "multislice":
        return {
            "repository": (
                "ut-cwo/Inverse-scattering-in-biological-samples-via-beam-propagation"
            ),
            "version": MSBP_DONOR_VERSION,
            "license": "BSD-3-Clause",
            "citation_license_metadata": "MIT",
            "paths": [
                "Recon_main.m",
                "Recon_corefunction/MultiSlice_Forward.m",
                "Recon_corefunction/BPM_update.m",
                "Recon_corefunction/generate_Prop_dataset.m",
                "Recon_corefunction/Rytov_recon_init.m",
            ],
            "source_sha256": MSBP_DONOR_SHA256,
        }
    return {
        "repository": "pyhololab",
        "commit": MULTILAYER_DONOR_COMMIT,
        "paths": [
            "src/core/multilayer_idt.py",
            "src/core/multilayer/models.py",
            "src/core/multilayer/regularizers.py",
        ],
    }


def _mapping(source: Mapping[str, JsonValue], key: str) -> Mapping[str, JsonValue]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"multi-layer configuration requires mapping {key}")
    return value


def _sequence(source: Mapping[str, JsonValue], key: str) -> tuple[JsonValue, ...]:
    value = source.get(key)
    if not isinstance(value, tuple):
        raise ValueError(f"multi-layer configuration requires sequence {key}")
    return value


def _string(source: Mapping[str, JsonValue], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str):
        raise ValueError(f"multi-layer configuration requires string {key}")
    return value


def _integer(source: Mapping[str, JsonValue], key: str) -> int:
    value = source.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"multi-layer configuration requires integer {key}")
    return value


def _optional_float(source: Mapping[str, JsonValue], key: str) -> float | None:
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"multi-layer configuration requires numeric {key}")
    return float(value)


def _number(source: Mapping[str, JsonValue], key: str) -> float:
    value = _optional_float(source, key)
    if value is None:
        raise ValueError(f"multi-layer configuration requires numeric {key}")
    return value


def _boolean(source: Mapping[str, JsonValue], key: str) -> bool:
    value = source.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"multi-layer configuration requires boolean {key}")
    return value


def _number_tuple(
    source: Mapping[str, JsonValue], key: str, length: int | None
) -> tuple[float, ...]:
    values = _sequence(source, key)
    if (length is not None and len(values) != length) or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in values
    ):
        raise ValueError(f"multi-layer configuration has invalid {key}")
    return tuple(float(cast("int | float", value)) for value in values)


def _integer_tuple(
    source: Mapping[str, JsonValue], key: str, length: int | None
) -> tuple[int, ...]:
    values = _sequence(source, key)
    if (length is not None and len(values) != length) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise ValueError(f"multi-layer configuration has invalid {key}")
    return tuple(int(cast("int", value)) for value in values)


def _model_config(configuration: Mapping[str, JsonValue]) -> MultiLayerConfig:
    values = _mapping(configuration, "model_config")
    return MultiLayerConfig(
        cast("tuple[int, int, int]", _integer_tuple(values, "shape_zyx", 3)),
        cast(
            "tuple[float, float, float]",
            _number_tuple(values, "voxel_size_zyx_um", 3),
        ),
        _number(values, "wavelength_um"),
        _number(values, "numerical_aperture"),
        _number(values, "refractive_index_medium"),
        padding_yx=cast("tuple[int, int]", _integer_tuple(values, "padding_yx", 2)),
        defocus_um=_number(values, "defocus_um"),
        slice_binning_factor=_integer(values, "slice_binning_factor"),
        allow_evanescent_pupil=_boolean(values, "allow_evanescent_pupil"),
    )


def _solve_config(reconstruction: Mapping[str, JsonValue]) -> MultiLayerSolveConfig:
    values = _mapping(reconstruction, "solve")
    return MultiLayerSolveConfig(
        max_iterations=_integer(values, "max_iterations"),
        step_size=_optional_float(values, "step_size"),
        optimizer=cast("Any", _string(values, "optimizer")),
        restart_on_loss_increase=_boolean(values, "restart_on_loss_increase"),
        random_order=_boolean(values, "random_order"),
        seed=_integer(values, "seed"),
        measurement_domain=cast("Any", _string(values, "measurement_domain")),
        l2_weight=_number(values, "l2_weight"),
        tv_weight=_number(values, "tv_weight"),
        tv_iterations=_integer(values, "tv_iterations"),
        enforce_physical_sign=_boolean(values, "enforce_physical_sign"),
        recover_pupil=_boolean(values, "recover_pupil"),
        pupil_step_size=_number(values, "pupil_step_size"),
        pupil_update_method=cast("Any", _string(values, "pupil_update_method")),
        early_stopping_relative=_optional_float(values, "early_stopping_relative"),
        subtract_first_slice_mean=_boolean(values, "subtract_first_slice_mean"),
    )


__all__ = [
    "MSBP_DONOR_VERSION",
    "MULTILAYER_DONOR_COMMIT",
    "MultiLayerKernel",
]
