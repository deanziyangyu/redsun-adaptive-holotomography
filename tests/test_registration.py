from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from redsun_aht.calibration import (
    CalibrationArtifactStore,
    RegistrationError,
    RegistrationSettings,
    RegistrationTransform,
    apply_registration,
    fit_registration,
    registration_plane,
)
from redsun_aht.domain import CalibrationKind


def _pattern() -> np.ndarray[Any, np.dtype[np.float32]]:
    y, x = np.mgrid[:128, :128]
    first = np.exp(-((x - 38) ** 2 + (y - 46) ** 2) / 180)
    second = 0.7 * np.exp(-((x - 88) ** 2 + (y - 79) ** 2) / 90)
    stripe = 0.3 * ((x > 57) & (x < 65) & (y > 18) & (y < 105))
    return np.asarray(first + second + stripe, dtype=np.float32)


def test_fit_once_then_apply_to_stack_with_downsample() -> None:
    reference = _pattern()
    known = cv2.getRotationMatrix2D((64, 64), 1.5, 1.0)
    known[0, 2] += 4.0
    known[1, 2] -= 3.0
    moving = cv2.warpAffine(reference, known, (128, 128))
    before = float(np.mean((reference - moving) ** 2))

    transform = fit_registration(
        np.stack((reference * 0.5, reference)),
        np.stack((moving * 0.5, moving)),
        reference_detector_id="pvcam",
        moving_detector_id="flir",
        settings=RegistrationSettings(plane_selection="last", downsample=2),
    )
    registered = apply_registration(np.stack((moving, moving)), transform)

    assert transform.direction == "moving-to-reference"
    assert transform.reference_shape == (128, 128)
    assert transform.score is not None and transform.score > 0.99
    assert registered.shape == (2, 128, 128)
    assert float(np.mean((reference - registered[0]) ** 2)) < before * 0.15


def test_registration_plane_has_no_split_detector_assumption() -> None:
    stack = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    np.testing.assert_array_equal(registration_plane(stack, "first"), stack[0])
    np.testing.assert_array_equal(registration_plane(stack, "middle"), stack[1])
    np.testing.assert_array_equal(registration_plane(stack, "last"), stack[2])
    np.testing.assert_array_equal(registration_plane(stack, "max"), stack.max(0))


def test_transform_is_immutable_and_shape_locked() -> None:
    transform = RegistrationTransform(
        reference_detector_id="pvcam",
        moving_detector_id="flir",
        matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        reference_shape=(32, 40),
        moving_shape=(32, 40),
        score=None,
        plane_selection="first",
        downsample=1,
    )
    matrix = transform.as_array()
    matrix[0, 0] = 9
    assert transform.matrix[0] == 1.0
    with pytest.raises(ValueError, match="calibrated shape"):
        apply_registration(np.zeros((31, 40), dtype=np.uint16), transform)


def test_fit_rejects_incompatible_detector_grids() -> None:
    with pytest.raises(RegistrationError, match="share a pixel grid"):
        fit_registration(
            np.zeros((32, 32), dtype=np.float32),
            np.zeros((32, 31), dtype=np.float32),
            reference_detector_id="pvcam",
            moving_detector_id="flir",
        )


def test_registration_fit_persists_with_direction_and_provenance(
    tmp_path: Path,
) -> None:
    reference = _pattern()
    transform = fit_registration(
        reference,
        reference.copy(),
        reference_detector_id="pvcam",
        moving_detector_id="flir",
    )
    store = CalibrationArtifactStore(tmp_path)
    artifact = store.create(
        kind=CalibrationKind.REGISTRATION,
        array=transform.as_array(),
        validity_scope={"optical_configuration": "60x"},
        source_run_id="dual-camera-registration-run",
        settings=transform.artifact_settings(),
        created_ns=123,
    )

    restored, matrix = store.read(artifact.artifact_id)
    np.testing.assert_array_equal(matrix, transform.as_array())
    assert restored.kind is CalibrationKind.REGISTRATION
    assert transform.artifact_settings()["direction"] == "moving-to-reference"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"downsample": 0}, "downsample"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"gaussian_filter_size": 2}, "positive and odd"),
    ],
)
def test_registration_settings_reject_invalid_fit_controls(
    changes: dict[str, int | float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RegistrationSettings(**changes)  # type: ignore[arg-type]


def test_registration_rejects_ambiguous_or_nonfinite_transform() -> None:
    common: dict[str, Any] = {
        "reference_detector_id": "camera",
        "moving_detector_id": "camera",
        "matrix": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "reference_shape": (32, 32),
        "moving_shape": (32, 32),
        "score": None,
        "plane_selection": "first",
        "downsample": 1,
    }
    with pytest.raises(ValueError, match="distinct"):
        RegistrationTransform(**common)
    common["moving_detector_id"] = "other"
    common["matrix"] = (1.0, 0.0, np.nan, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        RegistrationTransform(**common)


def test_registration_rejects_invalid_plane_and_initial_matrix() -> None:
    with pytest.raises(ValueError, match="at least two"):
        registration_plane(np.zeros(4), "first")
    with pytest.raises(ValueError, match="unsupported"):
        registration_plane(np.zeros((2, 32, 32)), "unknown")  # type: ignore[arg-type]
    with pytest.raises(RegistrationError, match="at least 16"):
        fit_registration(
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            reference_detector_id="pvcam",
            moving_detector_id="flir",
        )
    with pytest.raises(ValueError, match="6 finite"):
        fit_registration(
            _pattern(),
            _pattern(),
            reference_detector_id="pvcam",
            moving_detector_id="flir",
            initial_matrix=(1.0, 2.0),
        )
