"""Deterministic offline iterative optimization for multi-layer IDT models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from redsun_aht.processing.multilayer import (
        BaseMultiLayerModel,
        MeasurementDomain,
        MultiSliceModel,
    )

OptimizerName = Literal["fista", "gradient_descent"]
PupilUpdateMethod = Literal["gradient", "gauss_newton"]


def _to_numpy(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


@dataclass(frozen=True, slots=True)
class MultiLayerSolveConfig:
    """Frozen iterative, regularization, and optional pupil-recovery inputs."""

    max_iterations: int = 5
    step_size: float | None = None
    optimizer: OptimizerName = "fista"
    restart_on_loss_increase: bool = True
    random_order: bool = False
    seed: int = 0
    measurement_domain: MeasurementDomain = "intensity"
    l2_weight: float = 0.0
    tv_weight: float = 0.0
    tv_iterations: int = 15
    enforce_physical_sign: bool = True
    recover_pupil: bool = False
    pupil_step_size: float = 0.0
    pupil_update_method: PupilUpdateMethod = "gradient"
    early_stopping_relative: float | None = None
    subtract_first_slice_mean: bool = False

    def __post_init__(self) -> None:
        """Reject non-deterministic or numerically invalid solver inputs."""
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("multi-layer iterations must be a positive integer")
        if self.step_size is not None and (
            not np.isfinite(self.step_size) or self.step_size <= 0
        ):
            raise ValueError("multi-layer step size must be finite and positive")
        if self.optimizer not in {"fista", "gradient_descent"}:
            raise ValueError("multi-layer optimizer is unsupported")
        if type(self.seed) is not int:
            raise ValueError("multi-layer random seed must be an integer")
        if self.measurement_domain not in {"intensity", "amplitude", "field"}:
            raise ValueError("multi-layer measurement domain is unsupported")
        if any(
            not np.isfinite(value) or value < 0
            for value in (self.l2_weight, self.tv_weight)
        ):
            raise ValueError("multi-layer regularization weights must be non-negative")
        if type(self.tv_iterations) is not int or self.tv_iterations <= 0:
            raise ValueError("multi-layer TV iterations must be a positive integer")
        if not np.isfinite(self.pupil_step_size) or self.pupil_step_size < 0:
            raise ValueError("multi-layer pupil step size cannot be negative")
        if self.recover_pupil and self.pupil_step_size <= 0:
            raise ValueError("pupil recovery requires a positive pupil step size")
        if self.pupil_update_method not in {"gradient", "gauss_newton"}:
            raise ValueError("multi-layer pupil update method is unsupported")
        if self.early_stopping_relative is not None and (
            not np.isfinite(self.early_stopping_relative)
            or self.early_stopping_relative <= 0
        ):
            raise ValueError("multi-layer early stopping must be finite and positive")


@dataclass(frozen=True, slots=True)
class MultiLayerSolveResult:
    """Recovered RI, pupil, and immutable optimization history."""

    refractive_index_zyx: Any
    pupil_yx: Any
    loss_history: tuple[float, ...]
    step_history: tuple[float, ...]
    restart_history: tuple[bool, ...]
    phase_change_per_slice: float


def prox_tv_chambolle_3d(
    image: Any,
    weight: float,
    *,
    max_iterations: int,
    xp: Any,
    check_cancelled: Callable[[], None],
    epsilon: float = 2e-4,
) -> Any:
    """Apply the donor's backend-neutral isotropic 3-D TV proximal operator."""
    if weight <= 0:
        return image
    if image.ndim != 3:
        raise ValueError("multi-layer TV input must have ZYX axes")
    p_value = xp.zeros((3, *image.shape), dtype=image.dtype)
    gradient = xp.zeros_like(p_value)
    divergence = xp.zeros_like(image)
    output = image
    initial_energy: float | None = None
    previous_energy: float | None = None
    tau = 1.0 / 6.0
    for iteration in range(max_iterations):
        check_cancelled()
        if iteration:
            divergence = -p_value.sum(axis=0)
            for axis in range(3):
                destination = [slice(None)] * 3
                source: list[int | slice] = [slice(None)] * 4
                destination[axis] = slice(1, None)
                source[0] = axis
                source[axis + 1] = slice(0, -1)
                divergence[tuple(destination)] += p_value[tuple(source)]
            output = image + divergence
        energy = xp.sum(divergence * divergence)
        gradient.fill(0)
        for axis in range(3):
            gradient_destination: list[int | slice] = [slice(None)] * 4
            gradient_destination[0] = axis
            gradient_destination[axis + 1] = slice(0, -1)
            gradient[tuple(gradient_destination)] = xp.diff(output, axis=axis)
        norm = xp.sqrt(xp.sum(gradient * gradient, axis=0))[None, ...]
        energy += weight * xp.sum(norm)
        p_value -= tau * gradient
        p_value /= 1.0 + (tau / weight) * norm
        scalar_energy = float((energy / image.size).item())
        if initial_energy is None:
            initial_energy = max(abs(scalar_energy), 1e-12)
        elif previous_energy is not None and (
            abs(previous_energy - scalar_energy) < epsilon * initial_energy
        ):
            break
        previous_energy = scalar_energy
    return output


def solve_multilayer(
    model: BaseMultiLayerModel,
    measurements_syx: Any,
    illumination_fxy: Any,
    config: MultiLayerSolveConfig,
    *,
    check_cancelled: Callable[[], None] = lambda: None,
) -> MultiLayerSolveResult:
    """Run deterministic sequential-shot gradient or FISTA reconstruction."""
    if model.model_name == "multislice":
        return solve_multislice(
            model,  # type: ignore[arg-type]
            measurements_syx,
            illumination_fxy,
            (0.0,),
            (),
            config,
            check_cancelled=check_cancelled,
        )
    xp = model.xp
    measurements = xp.asarray(measurements_syx)
    frequencies = np.asarray(illumination_fxy, dtype=np.float64)
    shots = int(measurements.shape[0]) if measurements.ndim == 3 else 0
    if (
        measurements.ndim != 3
        or shots <= 0
        or measurements.shape[1:] != model.shape_zyx[-2:]
    ):
        raise ValueError("multi-layer measurements must have SYX axes")
    if frequencies.shape != (shots, 2) or not np.all(np.isfinite(frequencies)):
        raise ValueError("multi-layer illuminations must have finite S,2 shape")
    if not np.all(np.isfinite(_to_numpy(measurements))):
        raise ValueError("multi-layer measurements must be finite")
    domain: MeasurementDomain = config.measurement_domain
    if domain == "intensity":
        if bool(xp.any(measurements < 0).item()):
            raise ValueError("multi-layer intensity cannot be negative")
        measurements = xp.sqrt(xp.maximum(measurements, 0)).astype(xp.float32)
        domain = "amplitude"
    elif domain == "amplitude":
        if bool(xp.any(measurements < 0).item()):
            raise ValueError("multi-layer amplitude cannot be negative")
        measurements = measurements.astype(xp.float32)
    else:
        measurements = measurements.astype(xp.complex64)

    ri = xp.full(model.shape_zyx, model.n0, dtype=xp.float32)
    native = model.ri_to_native(ri).astype(xp.float32)
    step_size = config.step_size
    if step_size is None:
        step_size = 10.0 if model.model_name == "multi_born" else 2e-4
    model.reset_pupil()
    random = np.random.default_rng(config.seed)
    momentum_reference = native.copy()
    accepted_native = native.copy()
    accepted_pupil = model.pupil.copy()
    t_value = 1.0
    previous_cost: float | None = None
    losses: list[float] = []
    restarts: list[bool] = []

    for _iteration in range(config.max_iterations):
        check_cancelled()
        order = random.permutation(shots) if config.random_order else np.arange(shots)
        total_cost = 0.0
        pupil_gradient = xp.zeros_like(model.pupil)
        pupil_hessian = xp.zeros_like(model.pupil.real)
        for shot in order:
            check_cancelled()
            fx_value, fy_value = frequencies[int(shot)]
            cost, gradient, shot_pupil_gradient, shot_pupil_hessian, _ = (
                model.loss_and_gradient(
                    native,
                    measurements[int(shot)],
                    fx_value,
                    fy_value,
                    measurement_domain=domain,
                )
            )
            total_cost += cost
            native -= step_size * gradient
            if config.recover_pupil:
                pupil_gradient += shot_pupil_gradient
                pupil_hessian += shot_pupil_hessian
        if config.l2_weight:
            native -= step_size * config.l2_weight * native
        if config.tv_weight:
            native = prox_tv_chambolle_3d(
                native,
                config.tv_weight,
                max_iterations=config.tv_iterations,
                xp=xp,
                check_cancelled=check_cancelled,
            )
        if config.enforce_physical_sign:
            native = model.project_native(native)
        if config.recover_pupil:
            model.update_pupil(
                pupil_gradient,
                hessian=pupil_hessian,
                step_size=config.pupil_step_size,
                method=config.pupil_update_method,
            )

        restarted = False
        if (
            previous_cost is not None
            and total_cost > previous_cost
            and config.restart_on_loss_increase
        ):
            native = accepted_native.copy()
            model.pupil_yx = accepted_pupil.copy()
            momentum_reference = accepted_native.copy()
            t_value = 1.0
            restarted = True
        else:
            proximal_native = native.copy()
            accepted_native = proximal_native.copy()
            accepted_pupil = model.pupil.copy()
            previous_cost = total_cost
            if config.optimizer == "fista":
                t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_value**2))
                beta = (t_value - 1.0) / t_next
                native = proximal_native + beta * (proximal_native - momentum_reference)
                momentum_reference = proximal_native
                t_value = t_next
        losses.append(float(total_cost))
        restarts.append(restarted)

    refractive_index = model.native_to_ri(accepted_native).astype(xp.float32)
    return MultiLayerSolveResult(
        refractive_index,
        model.pupil.copy(),
        tuple(losses),
        tuple(float(step_size) for _ in losses),
        tuple(restarts),
        model.phase_change_per_slice(refractive_index),
    )


def solve_multislice(
    model: MultiSliceModel,
    measurements_syx: Any,
    illumination_fxy: Any,
    focus_offsets_slices: tuple[float, ...],
    skip_shots: tuple[int, ...],
    config: MultiLayerSolveConfig,
    *,
    initial_native_zyx: Any | None = None,
    check_cancelled: Callable[[], None] = lambda: None,
) -> MultiLayerSolveResult:
    """Run the 2025 angle/defocus-diverse MSBP inverse solver.

    Raw complex fields are digitally refocused before optimization. Real
    measurements can be used only at a zero focus offset because phase is
    required to synthesize other planes.
    """
    if config.recover_pupil:
        raise ValueError("the 2025 MSBP solver does not recover the pupil")
    xp = model.xp
    raw = xp.asarray(measurements_syx)
    frequencies = np.asarray(illumination_fxy, dtype=np.float64)
    shots = int(raw.shape[0]) if raw.ndim == 3 else 0
    if (
        raw.ndim != 3
        or shots <= 0
        or raw.shape[1:] != model.shape_zyx[-2:]
        or frequencies.shape != (shots, 2)
        or not np.all(np.isfinite(frequencies))
    ):
        raise ValueError("MSBP measurements and illuminations require SYX and S,2")
    if not focus_offsets_slices or any(
        not np.isfinite(value) for value in focus_offsets_slices
    ):
        raise ValueError("MSBP focus offsets must be a non-empty finite sequence")
    skipped = frozenset(skip_shots)
    if any(type(index) is not int or not 0 <= index < shots for index in skipped):
        raise ValueError("MSBP skipped shots must be unique zero-based indices")
    included = np.asarray([index for index in range(shots) if index not in skipped])
    if included.size == 0:
        raise ValueError("MSBP cannot skip every illumination")
    if not np.all(np.isfinite(_to_numpy(raw))):
        raise ValueError("MSBP measurements must be finite")

    domain: MeasurementDomain = config.measurement_domain
    if np.issubdtype(raw.dtype, np.complexfloating):
        refocused = model.refocus_measurements(
            raw.astype(xp.complex64), frequencies, focus_offsets_slices
        )
        if domain == "amplitude":
            prepared = xp.abs(refocused).astype(xp.float32)
        elif domain == "intensity":
            prepared = (xp.abs(refocused) ** 2).astype(xp.float32)
        else:
            prepared = refocused.astype(xp.complex64)
    else:
        if tuple(focus_offsets_slices) != (0.0,):
            raise ValueError("MSBP digital refocusing requires complex input fields")
        if domain == "field":
            raise ValueError("MSBP field-domain reconstruction requires complex fields")
        if bool(xp.any(raw < 0).item()):
            raise ValueError(f"MSBP {domain} measurements cannot be negative")
        prepared = raw[:, None, ...].astype(xp.float32)

    if initial_native_zyx is None:
        native = xp.zeros(model.shape_zyx, dtype=xp.float32)
    else:
        native = xp.asarray(initial_native_zyx, dtype=xp.float32)
        if native.shape != model.shape_zyx or not np.all(
            np.isfinite(_to_numpy(native))
        ):
            raise ValueError("MSBP initial object must match finite ZYX geometry")
        native = model.project_native(native)
    step_size = config.step_size if config.step_size is not None else 2e-4
    random = np.random.default_rng(config.seed)
    momentum_reference = native.copy()
    accepted_native = native.copy()
    t_value = 1.0
    previous_cost: float | None = None
    losses: list[float] = []
    restarts: list[bool] = []

    for _iteration in range(config.max_iterations):
        check_cancelled()
        order = random.permutation(included) if config.random_order else included
        total_cost = 0.0
        for focus_index, focus_offset in enumerate(focus_offsets_slices):
            for shot_value in order:
                check_cancelled()
                shot = int(shot_value)
                fx, fy = frequencies[shot]
                cost, gradient, _ = model.loss_and_gradient_refocused(
                    native,
                    prepared[shot, focus_index],
                    float(fx),
                    float(fy),
                    focus_offset,
                    measurement_domain=domain,
                )
                total_cost += cost
                native -= step_size * gradient
            if config.tv_weight and focus_index < len(focus_offsets_slices) - 1:
                native = prox_tv_chambolle_3d(
                    native.real,
                    config.tv_weight,
                    max_iterations=config.tv_iterations,
                    xp=xp,
                    check_cancelled=check_cancelled,
                )

        if config.l2_weight:
            native -= step_size * config.l2_weight * native
        if config.subtract_first_slice_mean:
            native -= xp.mean(native[0].real)
        if config.enforce_physical_sign:
            native = model.project_native(native)
        else:
            native = native.real.astype(xp.float32)
        if config.tv_weight:
            native = prox_tv_chambolle_3d(
                native,
                config.tv_weight,
                max_iterations=config.tv_iterations,
                xp=xp,
                check_cancelled=check_cancelled,
            )

        restarted = False
        if (
            previous_cost is not None
            and total_cost > previous_cost
            and config.restart_on_loss_increase
        ):
            native = accepted_native.copy()
            momentum_reference = accepted_native.copy()
            t_value = 1.0
            restarted = True
        else:
            proximal_native = native.copy()
            accepted_native = proximal_native.copy()
            if config.optimizer == "fista":
                t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_value**2))
                beta = (t_value - 1.0) / t_next
                native = proximal_native + beta * (proximal_native - momentum_reference)
                momentum_reference = proximal_native
                t_value = t_next
            previous_cost = total_cost
        losses.append(float(total_cost))
        restarts.append(restarted)
        if (
            config.early_stopping_relative is not None
            and len(losses) > 1
            and abs(losses[-2] - losses[-1]) / max(abs(losses[-1]), np.finfo(float).eps)
            < config.early_stopping_relative
        ):
            break

    refractive_index = model.native_to_ri(accepted_native).astype(xp.float32)
    return MultiLayerSolveResult(
        refractive_index,
        model.pupil.copy(),
        tuple(losses),
        tuple(float(step_size) for _ in losses),
        tuple(restarts),
        model.phase_change_per_slice(refractive_index),
    )


__all__ = [
    "MultiLayerSolveConfig",
    "MultiLayerSolveResult",
    "OptimizerName",
    "PupilUpdateMethod",
    "prox_tv_chambolle_3d",
    "solve_multilayer",
    "solve_multislice",
]
