from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from redsun_aht.processing import (
    MultiLayerBornModel,
    MultiLayerConfig,
    MultiSliceModel,
    create_multilayer_model,
    ring_illumination_frequencies,
)


def _config() -> MultiLayerConfig:
    return MultiLayerConfig(
        shape_zyx=(3, 8, 10),
        voxel_size_zyx_um=(0.7, 0.12, 0.11),
        wavelength_um=0.625,
        numerical_aperture=0.8,
        refractive_index_medium=1.33,
        padding_yx=(1, 2),
        defocus_um=0.3,
    )


def _fixture() -> tuple[MultiLayerConfig, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    config = _config()
    generator = np.random.default_rng(21)
    ri = (1.33 + generator.uniform(0, 0.015, config.shape_zyx)).astype(np.float32)
    frequencies = ring_illumination_frequencies(
        4, wavelength_um=0.625, illumination_na=0.45
    )
    return config, ri, frequencies


@pytest.mark.parametrize(
    (
        "model_name",
        "field_sample_0",
        "field_sample_1",
        "field_mean",
        "gradient_sample",
        "gradient_norm",
        "loss",
    ),
    [
        (
            "multi_born",
            -1.0057327 + 0.07977319j,
            -0.81438315 + 0.5571522j,
            -0.19450636208057404 - 0.20432062447071075j,
            -2.4548472e-05,
            0.0009355362853966653,
            0.004032963886857033,
        ),
        (
            "multislice",
            -0.9975217 + 0.08694608j,
            -0.8055267 + 0.56251407j,
            -0.19478186964988708 - 0.20140528678894043j,
            -0.0008141917,
            0.2305813729763031,
            0.003994406200945377,
        ),
    ],
)
def test_models_match_pinned_pyhololab_fixture(
    model_name: str,
    field_sample_0: complex,
    field_sample_1: complex,
    field_mean: complex,
    gradient_sample: float,
    gradient_norm: float,
    loss: float,
) -> None:
    config, ri, frequencies = _fixture()
    model = create_multilayer_model(model_name, config)

    fields = model.forward(ri, frequencies)
    native = model.ri_to_native(ri)
    actual_loss, gradient, pupil_gradient, pupil_hessian, predicted = (
        model.loss_and_gradient(
            native,
            np.abs(fields[0]) * np.float32(0.99),
            *frequencies[0],
            measurement_domain="amplitude",
        )
    )

    assert fields.shape == (4, 8, 10)
    assert fields.dtype == np.complex64
    assert gradient.shape == config.shape_zyx
    assert gradient.dtype == np.float32
    assert pupil_gradient.shape == (10, 14)
    assert pupil_hessian.shape == (10, 14)
    np.testing.assert_allclose(fields[0, 2, 3], field_sample_0, rtol=2e-6)
    np.testing.assert_allclose(fields[3, 6, 8], field_sample_1, rtol=2e-6)
    np.testing.assert_allclose(fields.mean(), field_mean, rtol=2e-6)
    np.testing.assert_allclose(gradient[1, 4, 5], gradient_sample, rtol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(gradient), gradient_norm, rtol=2e-6)
    np.testing.assert_allclose(actual_loss, loss, rtol=2e-6)
    np.testing.assert_array_equal(predicted, fields[0])


@pytest.mark.parametrize(
    ("model_name", "epsilon"), [("multi_born", 0.1), ("multislice", 0.01)]
)
def test_model_gradient_matches_central_difference(
    model_name: str, epsilon: float
) -> None:
    config, ri, frequencies = _fixture()
    model = create_multilayer_model(model_name, config)
    native = model.ri_to_native(ri)
    prediction = model.forward_native(native, *frequencies[0]).field_yx
    target = np.abs(prediction) * np.float32(0.99)
    _, gradient, *_ = model.loss_and_gradient(
        native, target, *frequencies[0], measurement_domain="amplitude"
    )
    direction = gradient / np.linalg.norm(gradient)
    analytic = float(np.sum(gradient * direction))

    plus = model.loss_and_gradient(
        native + epsilon * direction,
        target,
        *frequencies[0],
        measurement_domain="amplitude",
    )[0]
    minus = model.loss_and_gradient(
        native - epsilon * direction,
        target,
        *frequencies[0],
        measurement_domain="amplitude",
    )[0]

    np.testing.assert_allclose((plus - minus) / (2 * epsilon), analytic, rtol=5e-3)


@pytest.mark.parametrize("model_name", ["multi_born", "multislice"])
@pytest.mark.parametrize("measurement_domain", ["field", "intensity"])
def test_loss_domains_return_finite_derivatives(
    model_name: str, measurement_domain: str
) -> None:
    config, ri, frequencies = _fixture()
    model = create_multilayer_model(model_name, config)
    native = model.ri_to_native(ri)
    prediction = model.forward_native(native, *frequencies[0]).field_yx
    measurement = prediction * np.complex64(0.99)
    if measurement_domain == "intensity":
        measurement = np.abs(measurement) ** 2

    loss, gradient, pupil_gradient, pupil_hessian, _ = model.loss_and_gradient(
        native,
        measurement,
        *frequencies[0],
        measurement_domain=measurement_domain,  # type: ignore[arg-type]
    )

    assert loss > 0
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(pupil_gradient))
    assert np.all(np.isfinite(pupil_hessian))


def test_native_conversions_projection_and_phase_metric() -> None:
    config = _config()
    ri = np.full(config.shape_zyx, 1.34, dtype=np.float32)
    born = MultiLayerBornModel(config)
    multislice = MultiSliceModel(config)

    np.testing.assert_allclose(born.native_to_ri(born.ri_to_native(ri)), ri)
    np.testing.assert_allclose(multislice.native_to_ri(multislice.ri_to_native(ri)), ri)
    np.testing.assert_array_equal(
        born.project_native(np.array([-1.0, 2.0], dtype=np.float32)), [-1.0, 0.0]
    )
    np.testing.assert_array_equal(
        multislice.project_native(np.array([-1.0, 2.0], dtype=np.float32)), [0.0, 2.0]
    )
    np.testing.assert_allclose(
        multislice.phase_change_per_slice(ri),
        2 * np.pi * 0.01 * 0.7 / 0.625,
        rtol=2e-6,
    )


def test_pupil_updates_are_projected_to_support() -> None:
    model = MultiSliceModel(_config())
    native_pupil = np.full((8, 10), 2 + 1j, dtype=np.complex64)
    model.set_pupil(native_pupil)
    assert model.pupil.shape == (10, 14)
    assert np.max(np.abs(model.pupil)) > 1

    gradient = np.ones((10, 14), dtype=np.complex64)
    model.update_pupil(gradient, step_size=0.1)
    assert np.max(np.abs(model.pupil)) <= 1.0 + 1e-6
    assert np.all(model.pupil[model.pupil_support_yx == 0] == 0)
    model.update_pupil(
        gradient,
        step_size=0.1,
        method="gauss_newton",
        hessian=np.ones((10, 14), dtype=np.float32),
    )
    model.reset_pupil()
    np.testing.assert_array_equal(model.pupil, model.pupil_support_yx)


def test_memory_estimates_and_ring_geometry_are_deterministic() -> None:
    model = MultiSliceModel(_config())
    frequencies = ring_illumination_frequencies(
        4, wavelength_um=0.625, illumination_na=0.45
    )

    assert model.estimated_cache_bytes() == 6720
    assert model.estimated_working_set_bytes(4) == 39600
    np.testing.assert_allclose(np.linalg.norm(frequencies, axis=1), 0.72)
    np.testing.assert_allclose(frequencies[0], [0.72, 0.0], atol=1e-15)


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"shape_zyx": (3, 0, 10)}, ValueError),
        ({"shape_zyx": (3, 8.0, 10)}, ValueError),
        ({"voxel_size_zyx_um": (0.7, -0.12, 0.11)}, ValueError),
        ({"wavelength_um": 0.0}, ValueError),
        ({"numerical_aperture": 1.4}, ValueError),
        ({"padding_yx": (1, -1)}, ValueError),
        ({"padding_yx": (1, 1.0)}, ValueError),
        ({"defocus_um": np.inf}, ValueError),
        ({"slice_binning_factor": 2}, NotImplementedError),
        ({"slice_binning_factor": 1.0}, ValueError),
    ],
)
def test_config_rejects_invalid_inputs(
    updates: dict[str, Any], error: type[Exception]
) -> None:
    values: dict[str, Any] = {
        "shape_zyx": (3, 8, 10),
        "voxel_size_zyx_um": (0.7, 0.12, 0.11),
        "wavelength_um": 0.625,
        "numerical_aperture": 0.8,
        "refractive_index_medium": 1.33,
    }
    values.update(updates)
    with pytest.raises(error):
        MultiLayerConfig(**values)


def test_model_rejects_invalid_runtime_inputs() -> None:
    model = MultiSliceModel(_config())
    native = np.zeros(model.shape_zyx, dtype=np.float32)
    frequency = (0.0, 0.0)
    prediction = model.forward_native(native, *frequency).field_yx

    with pytest.raises(ValueError, match="unknown multi-layer"):
        create_multilayer_model("unknown", _config())
    with pytest.raises(ValueError, match="shape S,2"):
        model.forward(model.native_to_ri(native), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="frequencies must be finite"):
        model.forward(model.native_to_ri(native), np.array([[np.nan, 0.0]]))
    with pytest.raises(ValueError, match="measurement shape"):
        model.loss_and_gradient(native, np.zeros((2, 2)), *frequency)
    with pytest.raises(ValueError, match="domain"):
        model.loss_and_gradient(
            native,
            prediction,
            *frequency,
            measurement_domain="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="object"):
        model.forward_native(np.zeros((2, 2, 2)), *frequency)
    with pytest.raises(ValueError, match="pupil"):
        model.set_pupil(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="step size"):
        model.update_pupil(np.zeros((10, 14)), step_size=-1)
    with pytest.raises(ValueError, match="gradient shape"):
        model.update_pupil(np.zeros((2, 2)), step_size=1)
    with pytest.raises(ValueError, match="Hessian"):
        model.update_pupil(np.zeros((10, 14)), step_size=1, method="gauss_newton")
    with pytest.raises(ValueError, match="Hessian shape"):
        model.update_pupil(
            np.zeros((10, 14)),
            step_size=1,
            method="gauss_newton",
            hessian=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="epsilon"):
        model.update_pupil(np.zeros((10, 14)), step_size=1, epsilon=0)
    with pytest.raises(ValueError, match="unsupported"):
        model.update_pupil(
            np.zeros((10, 14)),
            step_size=1,
            method="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="invalid"):
        ring_illumination_frequencies(0, wavelength_um=0.625, illumination_na=0.45)
    with pytest.raises(ValueError, match="invalid"):
        ring_illumination_frequencies(1, wavelength_um=np.nan, illumination_na=0.45)
