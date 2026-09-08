"""Canonical OME-Zarr metadata contract for multi-slice measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from redsun_aht.domain import MultiSliceAcquisitionMode

if TYPE_CHECKING:
    from collections.abc import Mapping

    from redsun_aht.domain.models import JsonValue

MULTISLICE_SCHEMA_VERSION = 1
MULTISLICE_AXES = ("c", "z", "y", "x")


@dataclass(frozen=True, slots=True)
class MultiSliceMeasurementSchema:
    """Describe one multi-slice field representation inside an OME-Zarr run.

    ``real_array_path`` is the canonical measurement array for both modes.
    Interferometric measurements additionally require ``imaginary_array_path``
    and may declare numerical focus offsets.  AHT measurements use the raw
    acquisition ``z`` axis for physical focal positions and never use
    numerical refocusing.
    """

    acquisition_mode: MultiSliceAcquisitionMode
    real_array_path: str
    imaginary_array_path: str | None = None
    source_amplitude_array_path: str | None = None
    source_phase_array_path: str | None = None
    focal_position_coordinate_path: str | None = None
    focus_offsets_slices: tuple[float, ...] = (0.0,)

    def __post_init__(self) -> None:
        """Validate mode-specific paths and focus semantics."""
        try:
            mode = MultiSliceAcquisitionMode(self.acquisition_mode)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported multi-slice acquisition mode") from error
        if not self.real_array_path:
            raise ValueError("multi-slice real measurement array path is required")
        paths = (
            self.imaginary_array_path,
            self.source_amplitude_array_path,
            self.source_phase_array_path,
            self.focal_position_coordinate_path,
        )
        if any(path is not None and not path for path in paths):
            raise ValueError("multi-slice array and coordinate paths cannot be empty")
        if mode.requires_complex_field and not self.imaginary_array_path:
            raise ValueError(
                "interferometric measurements require a split imaginary field array"
            )
        if not mode.requires_complex_field and self.imaginary_array_path is not None:
            raise ValueError(
                "AHT real-amplitude measurements cannot declare an imaginary field"
            )
        if mode is MultiSliceAcquisitionMode.AHT_REAL_AMPLITUDE_FOCAL_STACK:
            if self.focus_offsets_slices != (0.0,):
                raise ValueError(
                    "AHT focal-stack measurements do not use numerical refocusing"
                )
            if not self.focal_position_coordinate_path:
                raise ValueError(
                    "AHT focal-stack measurements require physical focal coordinates"
                )
        elif not self.focus_offsets_slices:
            raise ValueError("interferometric focus offsets cannot be empty")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            for value in self.focus_offsets_slices
        ):
            raise ValueError("multi-slice focus offsets must be finite numbers")
        object.__setattr__(self, "acquisition_mode", mode)
        object.__setattr__(
            self,
            "focus_offsets_slices",
            tuple(float(value) for value in self.focus_offsets_slices),
        )

    @property
    def field_encoding(self) -> str:
        """Return the canonical storage encoding for this acquisition mode."""
        return (
            "split-complex-float32"
            if self.acquisition_mode.requires_complex_field
            else "real-amplitude-float32"
        )

    def as_json(self) -> dict[str, JsonValue]:
        """Return the ``aht.multislice`` OME-Zarr attribute payload."""
        return {
            "schemaVersion": MULTISLICE_SCHEMA_VERSION,
            "acquisitionMode": self.acquisition_mode.value,
            "axes": list(MULTISLICE_AXES),
            "fieldEncoding": self.field_encoding,
            "realArrayPath": self.real_array_path,
            "imaginaryArrayPath": self.imaginary_array_path,
            "sourceAmplitudeArrayPath": self.source_amplitude_array_path,
            "sourcePhaseArrayPath": self.source_phase_array_path,
            "focalPositionCoordinatePath": self.focal_position_coordinate_path,
            "focusOffsetsSlices": list(self.focus_offsets_slices),
            "numericalRefocusing": self.acquisition_mode.uses_numerical_refocusing,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, JsonValue]) -> MultiSliceMeasurementSchema:
        """Validate and reconstruct a schema from OME-Zarr attributes."""
        if payload.get("schemaVersion") != MULTISLICE_SCHEMA_VERSION:
            raise ValueError("unsupported multi-slice schema version")
        if payload.get("axes") != list(MULTISLICE_AXES):
            raise ValueError("multi-slice schema axes must be CZYX")
        mode_value = payload.get("acquisitionMode")
        real_path = payload.get("realArrayPath")
        imaginary_path = payload.get("imaginaryArrayPath")
        source_amplitude_path = payload.get("sourceAmplitudeArrayPath")
        source_phase_path = payload.get("sourcePhaseArrayPath")
        focal_path = payload.get("focalPositionCoordinatePath")
        focus_offsets = payload.get("focusOffsetsSlices")
        if not isinstance(mode_value, str) or not isinstance(real_path, str):
            raise ValueError("multi-slice schema identity fields are invalid")
        optional_paths = (
            imaginary_path,
            source_amplitude_path,
            source_phase_path,
            focal_path,
        )
        if any(
            path is not None and not isinstance(path, str) for path in optional_paths
        ):
            raise ValueError("multi-slice schema paths are invalid")
        if not isinstance(focus_offsets, (list, tuple)):
            raise ValueError("multi-slice focus offsets are invalid")
        return cls(
            acquisition_mode=MultiSliceAcquisitionMode(mode_value),
            real_array_path=real_path,
            imaginary_array_path=cast("str | None", imaginary_path),
            source_amplitude_array_path=cast("str | None", source_amplitude_path),
            source_phase_array_path=cast("str | None", source_phase_path),
            focal_position_coordinate_path=cast("str | None", focal_path),
            focus_offsets_slices=tuple(
                cast("float | int", value) for value in focus_offsets
            ),
        )


__all__ = [
    "MULTISLICE_AXES",
    "MULTISLICE_SCHEMA_VERSION",
    "MultiSliceMeasurementSchema",
]
