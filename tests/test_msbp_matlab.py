from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
scipy_io = pytest.importorskip("scipy.io")

from redsun_aht.domain import MultiSliceAcquisitionMode  # noqa: E402
from redsun_aht.processing import (  # noqa: E402
    load_msbp_matlab_dataset,
    load_msbp_matlab_reference,
    rytov_initial_guess,
    solve_msbp_matlab_dataset,
)


def _classic_dataset(path: Path) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    amplitude = np.arange(6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3) / 100 + 1
    phase = np.arange(6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3) / 1000
    scipy_io.savemat(
        path,
        {
            "Efield_amplitude": amplitude,
            "Efield_phase": phase,
            "FOV_size": 4,
            "FocalPlane": np.asarray([-1, 0, 2], dtype=np.float64),
            "NA": 0.8,
            "O": 3,
            "SkipList": np.asarray([2, 5], dtype=np.float64),
            "fx_illum_ref": np.asarray([0.0, 0.1, -0.1]),
            "fy_illum_ref": np.asarray([0.0, -0.1, 0.1]),
            "lambda": 0.625,
            "maxiter": 2,
            "n_imm": 1.33,
            "pdar": 1,
            "ps": 0.12,
            "psz": 0.7,
            "regParam": 0.001,
            "step_size": 0.0002,
            "x_start": 2,
            "y_start": 3,
            "z_plane": 0.3,
        },
    )
    return amplitude, phase


def test_classic_matlab_loader_applies_one_based_fov_crop_and_shot_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper.mat"
    amplitude, phase = _classic_dataset(path)

    dataset = load_msbp_matlab_dataset(
        path,
        crop_shape_yx=(2, 2),
        crop_start_yx=(1, 0),
        max_shots=2,
    )

    np.testing.assert_array_equal(
        dataset.amplitude_syx,
        amplitude[2:4, 2:4, :2].transpose(2, 0, 1),
    )
    np.testing.assert_array_equal(
        dataset.phase_syx,
        phase[2:4, 2:4, :2].transpose(2, 0, 1),
    )
    expected_frequencies = np.asarray([[0.0, 0.0], [0.1, -0.1]], dtype=np.float64)
    np.testing.assert_allclose(dataset.illumination_fxy, expected_frequencies)
    assert dataset.roi_start_yx == (1, 0)
    assert dataset.focus_offsets_slices == (-1.0, 0.0, 2.0)
    assert dataset.skip_shots == (1,)
    assert dataset.model_config().shape_zyx == (3, 2, 2)
    assert dataset.reconstruction_config().model == "multislice"
    assert (
        dataset.reconstruction_config().acquisition_mode
        is MultiSliceAcquisitionMode.INTERFEROMETRIC_COMPLEX_FIELD
    )


def test_v73_matlab_loader_reads_direct_hdf5_measurement_arrays(tmp_path: Path) -> None:
    path = tmp_path / "paper-v73.mat"
    amplitude = np.arange(3 * 6 * 7, dtype=np.float32).reshape(3, 6, 7) / 100 + 1
    phase = np.arange(3 * 6 * 7, dtype=np.float32).reshape(3, 6, 7) / 1000
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Efield_amplitude", data=amplitude)
        handle.create_dataset("Efield_phase", data=phase)
        metadata = {
            "FOV_size": 4,
            "FocalPlane": np.asarray([], dtype=np.float64),
            "NA": 0.8,
            "O": 3,
            "SkipList": np.asarray([], dtype=np.float64),
            "fx_illum_ref": np.asarray([0.0, 0.1, -0.1]),
            "fy_illum_ref": np.asarray([0.0, -0.1, 0.1]),
            "lambda": 0.625,
            "maxiter": 2,
            "n_imm": 1.33,
            "pdar": 1,
            "ps": 0.12,
            "psz": 0.7,
            "regParam": 0.001,
            "step_size": 0.0002,
            "x_start": 2,
            "y_start": 2,
            "z_plane": 0.3,
        }
        for name, value in metadata.items():
            dataset = handle.create_dataset(name, data=value)
            if name == "SkipList":
                dataset.attrs["MATLAB_empty"] = np.uint8(1)

    dataset = load_msbp_matlab_dataset(
        path,
        crop_shape_yx=(2, 2),
        crop_start_yx=(1, 1),
        max_shots=2,
    )

    np.testing.assert_array_equal(
        dataset.amplitude_syx,
        amplitude[:2, 2:4, 2:4].transpose(0, 2, 1),
    )
    np.testing.assert_array_equal(
        dataset.phase_syx,
        phase[:2, 2:4, 2:4].transpose(0, 2, 1),
    )
    assert dataset.focus_offsets_slices == (0.0,)
    assert dataset.skip_shots == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop_shape_yx": (5, 2)},
        {"crop_shape_yx": (2, 2), "crop_start_yx": (3, 0)},
        {"max_shots": 0},
        {"max_shots": 1.0},
    ],
)
def test_matlab_loader_rejects_invalid_crop_and_shot_inputs(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    path = tmp_path / "paper.mat"
    _classic_dataset(path)
    with pytest.raises(ValueError, match=r"crop|shot"):
        load_msbp_matlab_dataset(path, **kwargs)


@pytest.mark.parametrize("structured", [False, True])
def test_matlab_reference_loader_accepts_real_and_complex_gpu_storage(
    tmp_path: Path, structured: bool
) -> None:
    path = tmp_path / "reference.mat"
    public = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    stored = public.transpose(0, 2, 1)
    stored = np.pad(stored, ((0, 0), (1, 1), (1, 1)))
    with h5py.File(path, "w") as handle:
        references = handle.create_group("#refs#")
        if structured:
            encoded = np.empty(stored.shape, dtype=[("real", "<f4"), ("imag", "<f4")])
            encoded["real"] = stored
            encoded["imag"] = 0
            references.create_dataset("value", data=encoded)
        else:
            references.create_dataset("value", data=stored)

    actual = load_msbp_matlab_reference(
        path,
        padding_yx=(1, 1),
        crop_start_yx=(1, 2),
        crop_shape_yx=(2, 2),
    )

    np.testing.assert_array_equal(actual, public[:, 1:3, 2:4])


def test_matlab_dataset_runs_numpy_solver_and_rytov_initializer(tmp_path: Path) -> None:
    path = tmp_path / "paper.mat"
    _classic_dataset(path)
    dataset = load_msbp_matlab_dataset(path, max_shots=1)
    dataset = replace(
        dataset,
        amplitude_syx=np.ones_like(dataset.amplitude_syx),
        phase_syx=np.zeros_like(dataset.phase_syx),
        focus_offsets_slices=(0.0,),
        max_iterations=1,
        tv_weight=0.0,
    )

    result = solve_msbp_matlab_dataset(dataset)
    initial = rytov_initial_guess(dataset)

    assert result.refractive_index_zyx.shape == (3, 4, 4)
    assert np.all(np.isfinite(result.refractive_index_zyx))
    assert initial.shape == (3, 4, 4)
    assert initial.dtype == np.float32
    assert np.all(np.isfinite(initial))
    assert np.all(initial >= 0)
