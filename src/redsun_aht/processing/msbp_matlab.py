"""MATLAB dataset bridge and Rytov initializer for the 2025/2026 MSBP port."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from redsun_aht.domain import MultiSliceAcquisitionMode
from redsun_aht.processing.multilayer import (
    MultiLayerConfig,
    MultiSliceModel,
)
from redsun_aht.processing.multilayer_iterative import (
    MultiLayerSolveConfig,
    MultiLayerSolveResult,
    solve_multislice,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from redsun_aht.processing.multilayer_job import MultiLayerReconstructionConfig


@dataclass(frozen=True, slots=True)
class MSBPMatlabDataset:
    """Cropped measurements and reconstruction parameters from one MAT file."""

    source_path: Path
    amplitude_syx: np.ndarray[Any, np.dtype[np.float32]]
    phase_syx: np.ndarray[Any, np.dtype[np.float32]]
    illumination_fxy: np.ndarray[Any, np.dtype[np.float64]]
    wavelength_um: float
    numerical_aperture: float
    refractive_index_medium: float
    pixel_size_yx_um: float
    slice_spacing_um: float
    depth_layers: int
    padding_yx: tuple[int, int]
    center_plane_um: float
    focus_offsets_slices: tuple[float, ...]
    skip_shots: tuple[int, ...]
    max_iterations: int
    step_size: float
    tv_weight: float
    roi_start_yx: tuple[int, int]
    original_fov_shape_yx: tuple[int, int]

    @property
    def complex_fields_syx(self) -> np.ndarray[Any, np.dtype[np.complex64]]:
        """Combine the retained amplitude and unwrapped phase measurements."""
        return np.asarray(
            self.amplitude_syx * np.exp(1j * self.phase_syx),
            dtype=np.complex64,
        )

    def model_config(self) -> MultiLayerConfig:
        """Build the exact ZYX optical configuration for this crop."""
        _, ny, nx = self.amplitude_syx.shape
        return MultiLayerConfig(
            (self.depth_layers, ny, nx),
            (self.slice_spacing_um, self.pixel_size_yx_um, self.pixel_size_yx_um),
            self.wavelength_um,
            self.numerical_aperture,
            self.refractive_index_medium,
            padding_yx=self.padding_yx,
            defocus_um=self.center_plane_um,
            allow_evanescent_pupil=True,
        )

    def solve_config(
        self,
        *,
        use_field: bool = False,
        rytov_initialization: bool = False,
    ) -> MultiLayerSolveConfig:
        """Build the MATLAB example's optimizer settings without UNLocBoX."""
        return MultiLayerSolveConfig(
            max_iterations=self.max_iterations,
            step_size=self.step_size,
            optimizer="fista",
            restart_on_loss_increase=True,
            random_order=True,
            seed=0,
            measurement_domain="field" if use_field else "amplitude",
            tv_weight=self.tv_weight,
            tv_iterations=15,
            enforce_physical_sign=True,
            early_stopping_relative=1e-4,
            subtract_first_slice_mean=True,
        )

    def reconstruction_config(
        self,
        *,
        use_field: bool = False,
        memory_budget_bytes: int = 512 * 1024 * 1024,
    ) -> MultiLayerReconstructionConfig:
        """Translate paper metadata into the offline worker's typed config."""
        from redsun_aht.processing.multilayer_job import (
            MultiLayerReconstructionConfig,
        )

        illumination_na = float(
            self.wavelength_um
            * np.max(np.hypot(self.illumination_fxy[:, 0], self.illumination_fxy[:, 1]))
        )
        return MultiLayerReconstructionConfig(
            model="multislice",
            depth_layers=self.depth_layers,
            voxel_size_zyx_um=(
                self.slice_spacing_um,
                self.pixel_size_yx_um,
                self.pixel_size_yx_um,
            ),
            wavelength_um=self.wavelength_um,
            numerical_aperture=self.numerical_aperture,
            refractive_index_medium=self.refractive_index_medium,
            illumination_na=illumination_na,
            acquisition_mode=MultiSliceAcquisitionMode.INTERFEROMETRIC_COMPLEX_FIELD,
            padding_yx=self.padding_yx,
            defocus_um=self.center_plane_um,
            focus_offsets_slices=self.focus_offsets_slices,
            illumination_fxy=tuple(
                (float(fx), float(fy)) for fx, fy in self.illumination_fxy
            ),
            skip_shots=self.skip_shots,
            normalization="none",
            solve=self.solve_config(use_field=use_field),
            memory_budget_bytes=memory_budget_bytes,
        )


def load_msbp_matlab_dataset(
    path: str | Path,
    *,
    crop_shape_yx: tuple[int, int] | None = None,
    crop_start_yx: tuple[int, int] | None = None,
    max_shots: int | None = None,
) -> MSBPMatlabDataset:
    """Load one paper dataset, optionally retaining a small ROI and shot prefix.

    ``crop_start_yx`` is zero-based in the paper's configured FOV, not in the
    full camera frame. Omitting it centers ``crop_shape_yx`` in that FOV.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"MSBP MATLAB input does not exist: {source}")
    if max_shots is not None and (type(max_shots) is not int or max_shots <= 0):
        raise ValueError("MSBP shot limit must be a positive integer")
    with source.open("rb") as stream:
        header = stream.read(128)
    if b"MATLAB 7.3 MAT-file" in header or header.startswith(b"\x89HDF\r\n\x1a\n"):
        values, array_reader = _read_hdf5_metadata(source)
    else:
        values, array_reader = _read_classic_metadata(source)

    fov_size = _positive_integer(values, "FOV_size")
    full_start = (
        _positive_integer(values, "x_start") - 1,
        _positive_integer(values, "y_start") - 1,
    )
    if crop_shape_yx is None:
        crop_shape_yx = (fov_size, fov_size)
    if len(crop_shape_yx) != 2 or any(
        type(value) is not int or value <= 0 or value > fov_size
        for value in crop_shape_yx
    ):
        raise ValueError("MSBP crop shape must fit the configured FOV")
    if crop_start_yx is None:
        resolved_crop_start = (
            (fov_size - crop_shape_yx[0]) // 2,
            (fov_size - crop_shape_yx[1]) // 2,
        )
    else:
        resolved_crop_start = crop_start_yx
    if len(resolved_crop_start) != 2 or any(
        type(value) is not int or value < 0 for value in resolved_crop_start
    ):
        raise ValueError("MSBP crop start must contain non-negative YX indices")
    if any(
        start + shape > fov_size
        for start, shape in zip(resolved_crop_start, crop_shape_yx, strict=True)
    ):
        raise ValueError("MSBP crop lies outside the configured FOV")

    array_start = tuple(
        base + offset
        for base, offset in zip(full_start, resolved_crop_start, strict=True)
    )
    amplitude, phase = array_reader(array_start, crop_shape_yx, max_shots)
    shots = amplitude.shape[0]
    frequencies = np.column_stack(
        (
            np.ravel(values["fx_illum_ref"])[:shots],
            np.ravel(values["fy_illum_ref"])[:shots],
        )
    ).astype(np.float64)
    skip_values = np.ravel(values["SkipList"])
    skip_shots = tuple(
        sorted(
            {
                int(value) - 1
                for value in skip_values
                if np.isfinite(value) and 1 <= int(value) <= shots
            }
        )
    )
    focus = tuple(float(value) for value in np.ravel(values["FocalPlane"]))
    if not focus:
        focus = (0.0,)
    padding = _nonnegative_integer(values, "pdar")
    return MSBPMatlabDataset(
        source,
        amplitude,
        phase,
        frequencies,
        _positive_float(values, "lambda"),
        _positive_float(values, "NA"),
        _positive_float(values, "n_imm"),
        _positive_float(values, "ps"),
        _positive_float(values, "psz"),
        _positive_integer(values, "O"),
        (padding, padding),
        _finite_float(values, "z_plane"),
        focus,
        skip_shots,
        _positive_integer(values, "maxiter"),
        _positive_float(values, "step_size"),
        _nonnegative_float(values, "regParam"),
        resolved_crop_start,
        (fov_size, fov_size),
    )


def solve_msbp_matlab_dataset(
    dataset: MSBPMatlabDataset,
    *,
    xp: Any = np,
    use_field: bool = False,
    rytov_initialization: bool = False,
    check_cancelled: Callable[[], None] = lambda: None,
) -> MultiLayerSolveResult:
    """Reconstruct one loaded paper dataset on NumPy or a CuPy namespace."""
    model = MultiSliceModel(dataset.model_config(), xp=xp)
    initial = None
    if rytov_initialization:
        initial = rytov_initial_guess(dataset, check_cancelled=check_cancelled)
        initial = xp.asarray(initial)
    return solve_multislice(
        model,
        xp.asarray(dataset.complex_fields_syx),
        dataset.illumination_fxy,
        dataset.focus_offsets_slices,
        dataset.skip_shots,
        dataset.solve_config(
            use_field=use_field, rytov_initialization=rytov_initialization
        ),
        initial_native_zyx=initial,
        check_cancelled=check_cancelled,
    )


def rytov_initial_guess(
    dataset: MSBPMatlabDataset,
    *,
    check_cancelled: Callable[[], None] = lambda: None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Port ``Rytov_recon_init.m`` and return an unpadded ZYX delta-RI guess."""
    amplitude = np.asarray(dataset.amplitude_syx, dtype=np.float32)
    phase = np.asarray(dataset.phase_syx, dtype=np.float32)
    shots, ny, nx = amplitude.shape
    if ny != nx:
        raise ValueError("the paper's Rytov initializer requires a square XY crop")
    size = ny
    pixel_size = dataset.pixel_size_yx_um
    spacing = 1.0 / (size * pixel_size)
    frequency = spacing * np.arange(-size // 2, size - size // 2)
    fxx, fyy = np.meshgrid(frequency, frequency, indexing="xy")
    f0 = 1.0 / dataset.wavelength_um
    axial = np.real(
        np.sqrt(
            ((dataset.refractive_index_medium * f0) ** 2 - fxx**2 - fyy**2).astype(
                np.complex64
            )
        )
    )
    support = (
        fxx**2 + fyy**2 < (dataset.numerical_aperture / dataset.wavelength_um) ** 2
    )
    spectrum_zyx = np.zeros((size, size, size), dtype=np.complex64)
    counts_zyx = np.zeros((size, size, size), dtype=np.float32)
    for shot in range(shots):
        check_cancelled()
        field = np.log(np.maximum(amplitude[shot], np.finfo(np.float32).eps))
        field = field + 1j * phase[shot]
        field_spectrum = _fft_centered_2d(field) / (size * size * spacing**2)
        fx, fy = dataset.illumination_fxy[shot]
        field_spectrum = _subpixel_shift(field_spectrum, -fy / spacing, -fx / spacing)
        field_spectrum = np.where(support, field_spectrum, 0)
        valid_axial = np.where(support, axial, 0)
        shifted_x = fxx - fx
        shifted_y = fyy - fy
        incident_z = np.sqrt(
            (dataset.refractive_index_medium * f0) ** 2 - fx**2 - fy**2
        )
        shifted_z = valid_axial - incident_z
        scattering = 1j * 4.0 * np.pi * valid_axial * field_spectrum
        valid = valid_axial > 0
        iz = np.rint(shifted_z[valid] / spacing + size / 2).astype(int)
        iy = np.rint(shifted_y[valid] / spacing + size / 2).astype(int)
        ix = np.rint(shifted_x[valid] / spacing + size / 2).astype(int)
        inside = (
            (iz >= 0) & (iz < size) & (iy >= 0) & (iy < size) & (ix >= 0) & (ix < size)
        )
        index = (iz[inside], iy[inside], ix[inside])
        np.add.at(spectrum_zyx, index, scattering[valid][inside])
        np.add.at(counts_zyx, index, 1)
    occupied = counts_zyx > 0
    spectrum_zyx[occupied] /= counts_zyx[occupied]
    potential = _ifft_centered_nd(spectrum_zyx) / pixel_size**3
    refractive_index = np.real(
        dataset.refractive_index_medium
        * np.sqrt(
            1
            - potential
            * (dataset.wavelength_um / (dataset.refractive_index_medium * 2.0 * np.pi))
            ** 2
        )
    )
    target_before_resize = round(
        dataset.depth_layers * dataset.slice_spacing_um / pixel_size
    )
    if target_before_resize > refractive_index.shape[0]:
        total = target_before_resize - refractive_index.shape[0]
        refractive_index = np.pad(
            refractive_index,
            ((total // 2, total - total // 2), (0, 0), (0, 0)),
            mode="constant",
            constant_values=dataset.refractive_index_medium,
        )
    center = refractive_index.shape[0] // 2
    start = max(center - target_before_resize // 2, 0)
    refractive_index = refractive_index[
        start : min(start + target_before_resize, refractive_index.shape[0])
    ]
    refractive_index = _resize_z(refractive_index, dataset.depth_layers)
    return np.maximum(refractive_index - dataset.refractive_index_medium, 0).astype(
        np.float32
    )


def load_msbp_matlab_reference(
    path: str | Path,
    *,
    padding_yx: tuple[int, int],
    crop_start_yx: tuple[int, int] = (0, 0),
    crop_shape_yx: tuple[int, int] | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Read and crop the supplied v7.3 ``gpuArray`` delta-RI reference."""
    source = Path(path).resolve()
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - optional bridge dependency
        raise RuntimeError("reading MATLAB v7.3 references requires h5py") from error
    py, px = padding_yx
    if any(type(value) is not int or value < 0 for value in padding_yx):
        raise ValueError("MSBP reference padding must be non-negative YX integers")
    with h5py.File(source, "r") as handle:
        candidates = [
            value
            for value in handle["#refs#"].values()
            if getattr(value, "ndim", 0) == 3
            and (
                np.issubdtype(value.dtype, np.floating)
                or (value.dtype.fields is not None and "real" in value.dtype.fields)
            )
        ]
        if len(candidates) != 1:
            raise ValueError("could not resolve one MATLAB gpuArray volume")
        volume = candidates[0]
        # MATLAB v7.3 reverses dimensions: stored ZXY becomes public ZYX.
        public_y = volume.shape[2] - 2 * py
        public_x = volume.shape[1] - 2 * px
        if public_y <= 0 or public_x <= 0:
            raise ValueError("MSBP reference padding removes the entire volume")
        if crop_shape_yx is None:
            y, x = 0, 0
            ny, nx = public_y, public_x
        else:
            y, x = crop_start_yx
            ny, nx = crop_shape_yx
        if any(type(value) is not int or value < 0 for value in (y, x)) or any(
            type(value) is not int or value <= 0 for value in (ny, nx)
        ):
            raise ValueError("MSBP reference crop must contain valid YX integers")
        if y + ny > public_y or x + nx > public_x:
            raise ValueError("MSBP reference crop lies outside the unpadded volume")
        selection = (
            slice(None),
            slice(px + x, px + x + nx),
            slice(py + y, py + y + ny),
        )
        stored = volume[selection]
    real = stored if stored.dtype.fields is None else stored["real"]
    return np.asarray(real.transpose(0, 2, 1), dtype=np.float32)


def _read_hdf5_metadata(path: Path) -> tuple[dict[str, Any], Any]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - optional bridge dependency
        raise RuntimeError("reading MATLAB v7.3 datasets requires h5py") from error
    names = _metadata_names()
    with h5py.File(path, "r") as handle:
        values = {name: handle[name][()] for name in names}
        if handle["SkipList"].attrs.get("MATLAB_empty", 0):
            values["SkipList"] = np.empty(0)

    def read_arrays(
        start_yx: tuple[int, int],
        shape_yx: tuple[int, int],
        max_shots: int | None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        with h5py.File(path, "r") as handle:
            shots = handle["Efield_amplitude"].shape[0]
            stop = shots if max_shots is None else min(max_shots, shots)
            if stop <= 0:
                raise ValueError("MSBP shot limit must retain at least one shot")
            y, x = start_yx
            ny, nx = shape_yx
            # HDF5 stores reversed MATLAB axes as SXY.
            selection = (slice(0, stop), slice(x, x + nx), slice(y, y + ny))
            amplitude = handle["Efield_amplitude"][selection].transpose(0, 2, 1)
            phase = handle["Efield_phase"][selection].transpose(0, 2, 1)
        return np.asarray(amplitude, np.float32), np.asarray(phase, np.float32)

    return values, read_arrays


def _read_classic_metadata(path: Path) -> tuple[dict[str, Any], Any]:
    try:
        from scipy.io import loadmat
    except ImportError as error:  # pragma: no cover - optional bridge dependency
        raise RuntimeError("reading classic MATLAB datasets requires scipy") from error
    values = loadmat(path, variable_names=_metadata_names(), squeeze_me=True)

    def read_arrays(
        start_yx: tuple[int, int],
        shape_yx: tuple[int, int],
        max_shots: int | None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        arrays = loadmat(
            path,
            variable_names=("Efield_amplitude", "Efield_phase"),
            squeeze_me=False,
        )
        y, x = start_yx
        ny, nx = shape_yx
        amplitude = arrays["Efield_amplitude"][y : y + ny, x : x + nx]
        phase = arrays["Efield_phase"][y : y + ny, x : x + nx]
        stop = (
            amplitude.shape[2]
            if max_shots is None
            else min(max_shots, amplitude.shape[2])
        )
        if stop <= 0:
            raise ValueError("MSBP shot limit must retain at least one shot")
        return (
            np.asarray(amplitude[:, :, :stop].transpose(2, 0, 1), np.float32),
            np.asarray(phase[:, :, :stop].transpose(2, 0, 1), np.float32),
        )

    return values, read_arrays


def _metadata_names() -> tuple[str, ...]:
    return (
        "FOV_size",
        "FocalPlane",
        "NA",
        "O",
        "SkipList",
        "fx_illum_ref",
        "fy_illum_ref",
        "lambda",
        "maxiter",
        "n_imm",
        "pdar",
        "ps",
        "psz",
        "regParam",
        "step_size",
        "x_start",
        "y_start",
        "z_plane",
    )


def _finite_float(values: dict[str, Any], name: str) -> float:
    value = float(np.ravel(values[name])[0])
    if not np.isfinite(value):
        raise ValueError(f"MSBP MATLAB parameter {name} must be finite")
    return value


def _positive_float(values: dict[str, Any], name: str) -> float:
    value = _finite_float(values, name)
    if value <= 0:
        raise ValueError(f"MSBP MATLAB parameter {name} must be positive")
    return value


def _nonnegative_float(values: dict[str, Any], name: str) -> float:
    value = _finite_float(values, name)
    if value < 0:
        raise ValueError(f"MSBP MATLAB parameter {name} cannot be negative")
    return value


def _positive_integer(values: dict[str, Any], name: str) -> int:
    value = _positive_float(values, name)
    if not value.is_integer():
        raise ValueError(f"MSBP MATLAB parameter {name} must be an integer")
    return int(value)


def _nonnegative_integer(values: dict[str, Any], name: str) -> int:
    value = _nonnegative_float(values, name)
    if not value.is_integer():
        raise ValueError(f"MSBP MATLAB parameter {name} must be an integer")
    return int(value)


def _fft_centered_2d(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(value)))


def _ifft_centered_nd(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(value)))


def _subpixel_shift(
    image: np.ndarray[Any, Any], delta_row: float, delta_column: float
) -> np.ndarray[Any, Any]:
    rows, columns = image.shape
    row_index = np.fft.ifftshift(np.arange(-rows // 2, rows - rows // 2))
    column_index = np.fft.ifftshift(np.arange(-columns // 2, columns - columns // 2))
    columns_yx, rows_yx = np.meshgrid(column_index, row_index, indexing="xy")
    ramp = np.exp(
        1j
        * 2.0
        * np.pi
        * (delta_row * rows_yx / rows + delta_column * columns_yx / columns)
    )
    return np.fft.ifft2(np.fft.fft2(image) * ramp)


def _resize_z(volume_zyx: np.ndarray[Any, Any], depth: int) -> np.ndarray[Any, Any]:
    if volume_zyx.shape[0] == depth:
        return volume_zyx
    positions = np.linspace(0, volume_zyx.shape[0] - 1, depth)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, volume_zyx.shape[0] - 1)
    fraction = (positions - lower)[:, None, None]
    return np.asarray(
        volume_zyx[lower] * (1.0 - fraction) + volume_zyx[upper] * fraction
    )


__all__ = [
    "MSBPMatlabDataset",
    "load_msbp_matlab_dataset",
    "load_msbp_matlab_reference",
    "rytov_initial_guess",
    "solve_msbp_matlab_dataset",
]
