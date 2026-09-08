from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from redsun_aht.processing import (
    MultiLayerConfig,
    MultiLayerSolveConfig,
    create_multilayer_model,
    prox_tv_chambolle_3d,
    ring_illumination_frequencies,
    solve_multilayer,
)
from redsun_aht.processing.solvers import ProcessingCancelled


def _problem(model_name: str) -> tuple[Any, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    config = MultiLayerConfig(
        (3, 8, 10),
        (0.7, 0.12, 0.11),
        0.625,
        0.8,
        1.33,
        padding_yx=(1, 2),
        defocus_um=0.3,
    )
    frequencies = ring_illumination_frequencies(
        4, wavelength_um=0.625, illumination_na=0.45
    )
    truth = np.full(config.shape_zyx, 1.335, dtype=np.float32)
    forward = create_multilayer_model(model_name, config)
    intensity = np.abs(forward.forward(truth, frequencies)) ** 2
    return create_multilayer_model(model_name, config), intensity, frequencies


@pytest.mark.parametrize(
    ("model_name", "step_size"),
    [("multi_born", 1.0), ("multislice", 2e-4)],
)
def test_iterative_solver_is_deterministic_and_decreases_loss(
    model_name: str, step_size: float
) -> None:
    model, intensity, frequencies = _problem(model_name)
    config = MultiLayerSolveConfig(
        max_iterations=2,
        step_size=step_size,
        tv_weight=1e-5,
        tv_iterations=3,
        l2_weight=1e-6,
    )

    first = solve_multilayer(model, intensity, frequencies, config)
    second = solve_multilayer(
        create_multilayer_model(model_name, model.config),
        intensity,
        frequencies,
        config,
    )

    assert first.refractive_index_zyx.shape == model.shape_zyx
    assert first.refractive_index_zyx.dtype == np.float32
    assert len(first.loss_history) == 2
    assert first.loss_history[-1] <= first.loss_history[0]
    assert first.phase_change_per_slice >= 0
    np.testing.assert_array_equal(
        first.refractive_index_zyx, second.refractive_index_zyx
    )
    assert first.loss_history == second.loss_history


def test_iterative_solver_supports_random_order_field_reconstruction() -> None:
    model, intensity, frequencies = _problem("multislice")
    field = np.sqrt(intensity).astype(np.complex64)
    result = solve_multilayer(
        model,
        field,
        frequencies,
        MultiLayerSolveConfig(
            max_iterations=1,
            optimizer="gradient_descent",
            random_order=True,
            seed=4,
            measurement_domain="field",
        ),
    )

    assert result.loss_history[0] >= 0
    assert result.step_history == (2e-4,)
    assert np.max(np.abs(result.pupil_yx)) <= 1.0 + 1e-6


def test_replacement_multislice_rejects_removed_pupil_recovery() -> None:
    model, intensity, frequencies = _problem("multislice")
    with pytest.raises(ValueError, match="does not recover the pupil"):
        solve_multilayer(
            model,
            intensity,
            frequencies,
            MultiLayerSolveConfig(
                max_iterations=1,
                recover_pupil=True,
                pupil_step_size=1e-5,
            ),
        )


def test_iterative_solver_checks_cancellation_at_shot_boundaries() -> None:
    model, intensity, frequencies = _problem("multislice")
    calls = 0

    def cancel() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ProcessingCancelled("test cancellation")

    with pytest.raises(ProcessingCancelled, match="test cancellation"):
        solve_multilayer(
            model,
            intensity,
            frequencies,
            MultiLayerSolveConfig(max_iterations=2),
            check_cancelled=cancel,
        )


def test_tv_prox_preserves_constants_and_checks_cancellation() -> None:
    image = np.ones((3, 4, 5), dtype=np.float32)
    result = prox_tv_chambolle_3d(
        image,
        0.1,
        max_iterations=3,
        xp=np,
        check_cancelled=lambda: None,
    )
    np.testing.assert_array_equal(result, image)
    assert (
        prox_tv_chambolle_3d(
            image,
            0,
            max_iterations=3,
            xp=np,
            check_cancelled=lambda: None,
        )
        is image
    )
    with pytest.raises(ProcessingCancelled):
        prox_tv_chambolle_3d(
            image,
            0.1,
            max_iterations=3,
            xp=np,
            check_cancelled=lambda: (_ for _ in ()).throw(ProcessingCancelled()),
        )
    with pytest.raises(ValueError, match="ZYX"):
        prox_tv_chambolle_3d(
            np.ones((3, 4)),
            0.1,
            max_iterations=3,
            xp=np,
            check_cancelled=lambda: None,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"max_iterations": 0}, "iterations"),
        ({"step_size": 0}, "step size"),
        ({"optimizer": "other"}, "optimizer"),
        ({"seed": 1.0}, "seed"),
        ({"measurement_domain": "other"}, "domain"),
        ({"l2_weight": -1}, "weights"),
        ({"tv_iterations": 0}, "TV iterations"),
        ({"pupil_step_size": -1}, "pupil step"),
        ({"recover_pupil": True}, "pupil recovery"),
        ({"pupil_update_method": "other"}, "pupil update"),
    ],
)
def test_solve_config_rejects_invalid_values(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(MultiLayerSolveConfig(), **updates)


def test_iterative_solver_rejects_invalid_measurements_and_illuminations() -> None:
    model, intensity, frequencies = _problem("multislice")
    config = MultiLayerSolveConfig(max_iterations=1)
    with pytest.raises(ValueError, match="SYX"):
        solve_multilayer(model, intensity[0], frequencies, config)
    with pytest.raises(ValueError, match="illuminations"):
        solve_multilayer(model, intensity, frequencies[:2], config)
    invalid = intensity.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_multilayer(model, invalid, frequencies, config)
    negative = intensity.copy()
    negative[0, 0, 0] = -1
    with pytest.raises(ValueError, match="intensity"):
        solve_multilayer(model, negative, frequencies, config)
    with pytest.raises(ValueError, match="amplitude"):
        solve_multilayer(
            model,
            negative,
            frequencies,
            replace(config, measurement_domain="amplitude"),
        )
