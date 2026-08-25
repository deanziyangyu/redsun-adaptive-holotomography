"""Fit-once registration between independent detector coordinate systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from redsun_aht.domain.models import JsonValue

type PlaneSelection = Literal["first", "middle", "last", "max"]


class RegistrationError(ValueError):
    """Raised when a registration fit is invalid or fails to converge."""


@dataclass(frozen=True, slots=True)
class RegistrationSettings:
    """Reproducible ECC fit controls."""

    plane_selection: PlaneSelection = "first"
    downsample: int = 1
    max_iterations: int = 5000
    epsilon: float = 1e-12
    gaussian_filter_size: int = 1

    def __post_init__(self) -> None:
        """Validate deterministic fit settings."""
        if self.downsample < 1:
            raise ValueError("registration downsample must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("registration max_iterations must be >= 1")
        if self.epsilon <= 0:
            raise ValueError("registration epsilon must be positive")
        if self.gaussian_filter_size < 1 or self.gaussian_filter_size % 2 == 0:
            raise ValueError("gaussian_filter_size must be positive and odd")


@dataclass(frozen=True, slots=True)
class RegistrationTransform:
    """Immutable mapping from moving-detector pixels into a reference grid."""

    reference_detector_id: str
    moving_detector_id: str
    matrix: tuple[float, float, float, float, float, float]
    reference_shape: tuple[int, int]
    moving_shape: tuple[int, int]
    score: float | None
    plane_selection: PlaneSelection
    downsample: int
    algorithm: str = "opencv-ecc-euclidean"
    interpolation: str = "linear"
    border_mode: str = "constant-zero"
    direction: str = "moving-to-reference"

    def __post_init__(self) -> None:
        """Reject ambiguous detector identities and non-finite transforms."""
        if not self.reference_detector_id or not self.moving_detector_id:
            raise ValueError("registration detector IDs must not be empty")
        if self.reference_detector_id == self.moving_detector_id:
            raise ValueError("registration requires two distinct detector IDs")
        if any(dimension <= 0 for dimension in self.reference_shape):
            raise ValueError("reference_shape dimensions must be positive")
        if any(dimension <= 0 for dimension in self.moving_shape):
            raise ValueError("moving_shape dimensions must be positive")
        if not np.isfinite(self.matrix).all():
            raise ValueError("registration matrix must contain finite values")
        if self.score is not None and not np.isfinite(self.score):
            raise ValueError("registration score must be finite")

    def as_array(self) -> npt.NDArray[np.float32]:
        """Return a detached OpenCV-compatible 2 x 3 matrix."""
        return np.asarray(self.matrix, dtype=np.float32).reshape(2, 3).copy()

    def artifact_settings(self) -> Mapping[str, JsonValue]:
        """Return complete fit provenance for immutable artifact storage."""
        return {
            "algorithm": self.algorithm,
            "direction": self.direction,
            "reference_detector_id": self.reference_detector_id,
            "moving_detector_id": self.moving_detector_id,
            "reference_shape": self.reference_shape,
            "moving_shape": self.moving_shape,
            "matrix": self.matrix,
            "score": self.score,
            "plane_selection": self.plane_selection,
            "downsample": self.downsample,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
        }


def registration_plane(
    array: npt.NDArray[np.generic], selection: PlaneSelection
) -> npt.NDArray[np.generic]:
    """Select one 2D fit plane without any detector-splitting semantics."""
    value = np.asarray(array)
    if value.ndim < 2:
        raise ValueError("registration input must have at least two dimensions")
    if value.ndim == 2:
        return value
    flattened = value.reshape((-1, *value.shape[-2:]))
    if selection == "first":
        return cast("npt.NDArray[np.generic]", flattened[0])
    if selection == "middle":
        return cast("npt.NDArray[np.generic]", flattened[len(flattened) // 2])
    if selection == "last":
        return cast("npt.NDArray[np.generic]", flattened[-1])
    if selection == "max":
        return value.max(axis=tuple(range(value.ndim - 2)))
    raise ValueError(f"unsupported registration plane selection: {selection}")


def fit_registration(
    reference: npt.NDArray[np.generic],
    moving: npt.NDArray[np.generic],
    *,
    reference_detector_id: str,
    moving_detector_id: str,
    settings: RegistrationSettings | None = None,
    initial_matrix: Sequence[float] | None = None,
) -> RegistrationTransform:
    """Fit one Euclidean transform for later immutable reuse."""
    settings = RegistrationSettings() if settings is None else settings
    reference_plane = registration_plane(reference, settings.plane_selection)
    moving_plane = registration_plane(moving, settings.plane_selection)
    if reference_plane.shape != moving_plane.shape:
        raise RegistrationError(
            "registration calibration planes must share a pixel grid; "
            f"got {reference_plane.shape} and {moving_plane.shape}"
        )
    if min(reference_plane.shape) < 16:
        raise RegistrationError("registration planes must be at least 16 x 16")

    matrix = _initial_matrix(initial_matrix)
    fit_reference = reference_plane.astype(np.float32, copy=False)
    fit_moving = moving_plane.astype(np.float32, copy=False)
    scale_x = 1.0
    scale_y = 1.0
    if settings.downsample > 1:
        height, width = reference_plane.shape
        fit_size = (
            max(16, width // settings.downsample),
            max(16, height // settings.downsample),
        )
        scale_x = width / fit_size[0]
        scale_y = height / fit_size[1]
        fit_reference = cast(
            "npt.NDArray[np.float32]",
            cv2.resize(fit_reference, fit_size, interpolation=cv2.INTER_AREA),
        )
        fit_moving = cast(
            "npt.NDArray[np.float32]",
            cv2.resize(fit_moving, fit_size, interpolation=cv2.INTER_AREA),
        )
        matrix[0, 2] /= scale_x
        matrix[1, 2] /= scale_y

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        settings.max_iterations,
        settings.epsilon,
    )
    try:
        mask = np.full(fit_reference.shape, 255, dtype=np.uint8)
        score, fitted_value = cv2.findTransformECC(
            fit_reference,
            fit_moving,
            matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            mask,
            settings.gaussian_filter_size,
        )
    except cv2.error as exc:
        raise RegistrationError(f"OpenCV ECC failed to converge: {exc}") from exc
    fitted = cast("npt.NDArray[np.float32]", fitted_value)
    if settings.downsample > 1:
        fitted[0, 2] *= scale_x
        fitted[1, 2] *= scale_y

    values = cast(
        "tuple[float, float, float, float, float, float]",
        tuple(float(item) for item in fitted.reshape(-1)),
    )
    return RegistrationTransform(
        reference_detector_id=reference_detector_id,
        moving_detector_id=moving_detector_id,
        matrix=values,
        reference_shape=cast("tuple[int, int]", tuple(reference_plane.shape)),
        moving_shape=cast("tuple[int, int]", tuple(moving_plane.shape)),
        score=float(score),
        plane_selection=settings.plane_selection,
        downsample=settings.downsample,
    )


def apply_registration(
    moving: npt.NDArray[np.generic], transform: RegistrationTransform
) -> npt.NDArray[np.generic]:
    """Warp a frame or stack onto the locked reference-detector grid."""
    value = np.asarray(moving)
    if value.ndim < 2:
        raise ValueError("moving registration input must have at least two dimensions")
    if value.shape[-2:] != transform.moving_shape:
        raise ValueError(
            f"moving shape {value.shape[-2:]} does not match calibrated "
            f"shape {transform.moving_shape}"
        )
    output_shape = (*value.shape[:-2], *transform.reference_shape)
    registered = np.empty(output_shape, dtype=value.dtype)
    matrix = transform.as_array()
    output_size = (transform.reference_shape[1], transform.reference_shape[0])
    flattened_input = value.reshape((-1, *transform.moving_shape))
    flattened_output = registered.reshape((-1, *transform.reference_shape))
    for index, frame in enumerate(flattened_input):
        flattened_output[index] = cv2.warpAffine(
            frame,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return registered


def _initial_matrix(
    values: Sequence[float] | None,
) -> npt.NDArray[np.float32]:
    if values is None:
        return np.eye(2, 3, dtype=np.float32)
    matrix = np.asarray(tuple(values), dtype=np.float32)
    if matrix.size != 6 or not np.isfinite(matrix).all():
        raise ValueError("initial registration matrix must contain 6 finite values")
    return matrix.reshape(2, 3).copy()


__all__ = [
    "PlaneSelection",
    "RegistrationError",
    "RegistrationSettings",
    "RegistrationTransform",
    "apply_registration",
    "fit_registration",
    "registration_plane",
]
