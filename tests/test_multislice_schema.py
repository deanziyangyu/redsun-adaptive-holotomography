from __future__ import annotations

import pytest

from redsun_aht.domain import MultiSliceAcquisitionMode
from redsun_aht.storage import MultiSliceMeasurementSchema


def test_acquisition_modes_declare_refocusing_contract() -> None:
    aht = MultiSliceAcquisitionMode.AHT_REAL_AMPLITUDE_FOCAL_STACK
    interferometric = MultiSliceAcquisitionMode.INTERFEROMETRIC_COMPLEX_FIELD

    assert not aht.requires_complex_field
    assert not aht.uses_numerical_refocusing
    assert interferometric.requires_complex_field
    assert interferometric.uses_numerical_refocusing


def test_aht_schema_uses_physical_focal_coordinates() -> None:
    schema = MultiSliceMeasurementSchema(
        MultiSliceAcquisitionMode.AHT_REAL_AMPLITUDE_FOCAL_STACK,
        "detectors/dhm/field/real/0",
        focal_position_coordinate_path="coordinates/focal_position_um",
    )

    payload = schema.as_json()
    assert payload["fieldEncoding"] == "real-amplitude-float32"
    assert payload["numericalRefocusing"] is False
    assert payload["axes"] == ["c", "z", "y", "x"]
    assert MultiSliceMeasurementSchema.from_json(payload) == schema


def test_interferometric_schema_requires_split_complex_field() -> None:
    schema = MultiSliceMeasurementSchema(
        MultiSliceAcquisitionMode.INTERFEROMETRIC_COMPLEX_FIELD,
        "detectors/dhm/field/real/0",
        imaginary_array_path="detectors/dhm/field/imag/0",
        focus_offsets_slices=(-1.0, 0.0, 2.0),
    )

    payload = schema.as_json()
    assert payload["fieldEncoding"] == "split-complex-float32"
    assert payload["numericalRefocusing"] is True
    assert MultiSliceMeasurementSchema.from_json(payload) == schema


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "acquisition_mode": (
                MultiSliceAcquisitionMode.AHT_REAL_AMPLITUDE_FOCAL_STACK
            ),
            "real_array_path": "detectors/dhm/field/real/0",
        },
        {
            "acquisition_mode": MultiSliceAcquisitionMode.INTERFEROMETRIC_COMPLEX_FIELD,
            "real_array_path": "detectors/dhm/field/real/0",
            "focus_offsets_slices": (0.0, 1.0),
        },
    ],
)
def test_schema_rejects_incomplete_mode_specific_metadata(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        MultiSliceMeasurementSchema(**kwargs)  # type: ignore[arg-type]
