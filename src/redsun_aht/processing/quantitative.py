"""CPU reference DPCT and qOBT tile adapters for offline characterization."""

from __future__ import annotations

import importlib
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class QuantitativeTileConfig:
    """Frozen optical and regularization inputs for one quantitative model."""

    model: Literal["dpct", "qobt"]
    wavelength_um: float
    numerical_aperture: float
    pixel_size_um: float
    pixel_size_z_um: float
    refractive_index_medium: float = 1.0
    refractive_index_sample: float = 1.4
    source_azimuth_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    source_elevation_deg: float = 45.0
    source_sigma: float = 0.5
    background_division: Literal["background_over_sample", "sample_over_background"] = (
        "background_over_sample"
    )
    regularization_real: float = 5e-5
    regularization_imaginary: float = 5e-5
    regularization_scalar: float = 1e-2
    axial_window: Literal["hamming", "rectangular", "none"] = "hamming"

    def __post_init__(self) -> None:
        """Reject incomplete optics and unsupported shot geometry."""
        if self.model not in {"dpct", "qobt"}:
            raise ValueError("quantitative tile model must be DPCT or qOBT")
        positive = (
            self.wavelength_um,
            self.numerical_aperture,
            self.pixel_size_um,
            self.pixel_size_z_um,
            self.refractive_index_medium,
            self.refractive_index_sample,
            self.source_sigma,
            self.regularization_real,
            self.regularization_imaginary,
            self.regularization_scalar,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError(
                "quantitative optical and regularization values must be positive"
            )
        if self.numerical_aperture > self.refractive_index_medium:
            raise ValueError("numerical aperture cannot exceed the medium index")
        if len(self.source_azimuth_deg) != 4 or not all(
            np.isfinite(value) for value in self.source_azimuth_deg
        ):
            raise ValueError("DPCT/qOBT requires four finite source azimuths")
        if not np.isfinite(self.source_elevation_deg):
            raise ValueError("source elevation must be finite")
        if self.background_division not in {
            "background_over_sample",
            "sample_over_background",
        }:
            raise ValueError("background division convention is unsupported")
        if self.axial_window not in {"hamming", "rectangular", "none"}:
            raise ValueError("qOBT axial window is unsupported")


class CupyQuantitativeRuntime:
    """Worker-owned CUDA context, stream, allocator, and telemetry."""

    def __init__(self, gpu_id: int, *, cache_directory: str | None = None) -> None:
        if gpu_id < 0:
            raise ValueError("quantitative GPU device ID must be non-negative")
        if cache_directory is not None:
            cache_path = Path(cache_directory)
            if not cache_path.is_absolute():
                raise ValueError("CuPy cache directory must be absolute")
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ["CUPY_CACHE_DIR"] = str(cache_path)
        elif "CUPY_CACHE_DIR" not in os.environ:
            cache_path = Path(tempfile.gettempdir()) / "redsun-aht-cupy-cache"
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ["CUPY_CACHE_DIR"] = str(cache_path)
        try:
            cupy = importlib.import_module("cupy")
        except ImportError as error:
            raise RuntimeError(
                "CuPy GPU runtime is unavailable; install the gpu-cupy extra"
            ) from error
        if gpu_id >= int(cupy.cuda.runtime.getDeviceCount()):
            raise ValueError(f"quantitative GPU device does not exist: {gpu_id}")
        self.xp = cupy
        self.gpu_id = gpu_id
        self.device = cupy.cuda.Device(gpu_id)
        with self.device:
            self.stream = cupy.cuda.Stream(non_blocking=True)
            self.memory_pool = cupy.get_default_memory_pool()
            self.runtime_version = int(cupy.cuda.runtime.runtimeGetVersion())
            self.driver_version = int(cupy.cuda.runtime.driverGetVersion())
            started_ns = time.perf_counter_ns()
            with self.stream:
                warmup = cupy.fft.fftn(cupy.zeros((2, 2, 2), dtype=cupy.float32))
                cupy.asnumpy(warmup, stream=self.stream)
            self.stream.synchronize()
            self.startup_context_ns = time.perf_counter_ns() - started_ns
            del warmup
            self.memory_pool.free_all_blocks()
        self.observed_memory_bytes = 0
        self.closed = False

    @property
    def allocated_memory_bytes(self) -> int:
        """Return current CuPy allocator reservation."""
        with self.device:
            return int(self.memory_pool.total_bytes())

    def observe_memory(self) -> None:
        """Update the worker-owned allocator high-water mark."""
        with self.device:
            current = int(self.memory_pool.used_bytes())
        self.observed_memory_bytes = max(self.observed_memory_bytes, current)

    @property
    def telemetry(self) -> Mapping[str, Any]:
        """Return immutable CUDA placement and allocator telemetry."""
        return MappingProxyType(
            {
                "backend": "cupy",
                "device_id": self.gpu_id,
                "allocated_memory_bytes": self.allocated_memory_bytes,
                "observed_memory_bytes": self.observed_memory_bytes,
                "cuda_runtime_version": self.runtime_version,
                "cuda_driver_version": self.driver_version,
                "context_startup_ns": self.startup_context_ns,
            }
        )

    def close(self) -> None:
        """Synchronize and release cached CUDA allocations idempotently."""
        if self.closed:
            return
        with self.device:
            self.stream.synchronize()
            self.memory_pool.free_all_blocks()
        self.closed = True


@dataclass(slots=True)
class NumpyQuantitativeTileSolver:
    """Persistent NumPy Tikhonov adapter matching the committed donor path."""

    config: QuantitativeTileConfig
    _prepared_shape: tuple[int, int, int] | None = field(default=None, init=False)
    _operators: tuple[npt.NDArray[Any], ...] = field(default=(), init=False)
    _closed: bool = field(default=False, init=False)

    @property
    def model(self) -> Literal["dpct", "qobt"]:
        """Return the quantitative model selected by the frozen config."""
        return self.config.model

    @property
    def shots_num(self) -> int:
        """Return the exact four-shot donor contract."""
        return len(self.config.source_azimuth_deg)

    @property
    def gpu_telemetry(self) -> Mapping[str, Any]:
        """Report explicitly that this characterization adapter is CPU-only."""
        return MappingProxyType(
            {
                "backend": "numpy",
                "device_id": None,
                "allocated_memory_bytes": 0,
                "observed_memory_bytes": 0,
            }
        )

    def prepare(self, shape_zyx: tuple[int, int, int]) -> None:
        """Build and retain shape-dependent transfer functions."""
        if self._closed:
            raise RuntimeError("quantitative tile solver is closed")
        if any(value <= 0 for value in shape_zyx):
            raise ValueError("quantitative prepare shape must be positive ZYX")
        if self.config.model == "dpct":
            self._operators = _prepare_dpct(self.config, shape_zyx)
        else:
            self._operators = _prepare_qobt(self.config, shape_zyx)
        self._prepared_shape = shape_zyx

    def reconstruct_tile(
        self,
        sample_zsyx: npt.NDArray[Any],
        background_syx: npt.NDArray[Any] | None,
    ) -> npt.NDArray[np.float32]:
        """Run the donor Tikhonov equations and return canonical ZYX RI."""
        sample = np.asarray(sample_zsyx, dtype=np.float32)
        if sample.ndim != 4:
            raise ValueError("quantitative tile sample must have ZSYX axes")
        nz, shots, ny, nx = sample.shape
        if shots != self.shots_num:
            raise ValueError("quantitative tile sample has the wrong shot count")
        if self._prepared_shape != (nz, ny, nx):
            raise ValueError("quantitative tile does not match the prepared shape")
        normalized = _apply_background(sample, background_syx, self.config)
        if self.config.model == "dpct":
            result = _run_dpct(normalized, self.config, self._operators)
        else:
            result = _run_qobt(normalized, self.config, self._operators)
        return np.asarray(result, dtype=np.float32)

    def release_inputs(self) -> None:
        """Retain only shape-dependent operators; inputs are call-local."""

    def close(self) -> None:
        """Release persistent transfer functions idempotently."""
        if self._closed:
            return
        self._operators = ()
        self._prepared_shape = None
        self._closed = True


@dataclass(slots=True)
class CupyQuantitativeTileSolver:
    """Persistent CuPy DPCT/qOBT tile solver on one worker-owned runtime."""

    config: QuantitativeTileConfig
    runtime: CupyQuantitativeRuntime
    _prepared_shape: tuple[int, int, int] | None = field(default=None, init=False)
    _operators: tuple[Any, ...] = field(default=(), init=False)
    _closed: bool = field(default=False, init=False)

    @property
    def model(self) -> Literal["dpct", "qobt"]:
        """Return the configured quantitative model."""
        return self.config.model

    @property
    def shots_num(self) -> int:
        """Return the exact four-shot contract."""
        return len(self.config.source_azimuth_deg)

    @property
    def gpu_telemetry(self) -> Mapping[str, Any]:
        """Return current worker-owned CUDA telemetry."""
        return self.runtime.telemetry

    def prepare(self, shape_zyx: tuple[int, int, int]) -> None:
        """Build and retain shape-dependent transfer functions on CUDA."""
        if self._closed or self.runtime.closed:
            raise RuntimeError("quantitative GPU tile solver is closed")
        if any(value <= 0 for value in shape_zyx):
            raise ValueError("quantitative prepare shape must be positive ZYX")
        xp = self.runtime.xp
        with self.runtime.device, self.runtime.stream:
            if self.config.model == "dpct":
                self._operators = _prepare_dpct(self.config, shape_zyx, xp=xp)
            else:
                self._operators = _prepare_qobt(self.config, shape_zyx, xp=xp)
        self.runtime.stream.synchronize()
        self.runtime.observe_memory()
        self._prepared_shape = shape_zyx

    def reconstruct_tile(
        self,
        sample_zsyx: npt.NDArray[Any],
        background_syx: npt.NDArray[Any] | None,
    ) -> npt.NDArray[np.float32]:
        """Transfer one tile, reconstruct on CUDA, and return canonical ZYX."""
        sample = np.asarray(sample_zsyx, dtype=np.float32)
        if sample.ndim != 4:
            raise ValueError("quantitative tile sample must have ZSYX axes")
        nz, shots, ny, nx = sample.shape
        if shots != self.shots_num:
            raise ValueError("quantitative tile sample has the wrong shot count")
        if self._prepared_shape != (nz, ny, nx):
            raise ValueError("quantitative tile does not match the prepared shape")
        if self._closed or self.runtime.closed:
            raise RuntimeError("quantitative GPU tile solver is closed")
        xp = self.runtime.xp
        with self.runtime.device, self.runtime.stream:
            sample_gpu = xp.asarray(sample)
            background_gpu = (
                None
                if background_syx is None
                else xp.asarray(background_syx, dtype=xp.float32)
            )
            normalized = _apply_background(
                sample_gpu, background_gpu, self.config, xp=xp
            )
            if self.config.model == "dpct":
                result_gpu = _run_dpct(normalized, self.config, self._operators, xp=xp)
            else:
                result_gpu = _run_qobt(normalized, self.config, self._operators, xp=xp)
            result = xp.asnumpy(result_gpu, stream=self.runtime.stream)
        self.runtime.stream.synchronize()
        self.runtime.observe_memory()
        return np.asarray(result, dtype=np.float32)

    def release_inputs(self) -> None:
        """Retain CUDA operators; tile inputs are call-local."""

    def close(self) -> None:
        """Release persistent CUDA transfer functions idempotently."""
        if self._closed:
            return
        self._operators = ()
        self._prepared_shape = None
        with self.runtime.device:
            self.runtime.stream.synchronize()
            self.runtime.memory_pool.free_all_blocks()
        self._closed = True


def _frequency_axis(size: int, pixel_size_um: float, *, xp: Any = np) -> Any:
    spacing = 1.0 / (size * pixel_size_um)
    centered = (xp.arange(size, dtype=xp.complex64) - size // 2) * spacing
    return xp.fft.ifftshift(centered)


def _pupil(
    x_frequency: npt.NDArray[Any],
    y_frequency: npt.NDArray[Any],
    wavelength_um: float,
    numerical_aperture: float,
    *,
    xp: Any = np,
) -> Any:
    squared = x_frequency[xp.newaxis, :] ** 2 + y_frequency[:, xp.newaxis] ** 2
    cutoff = (numerical_aperture / wavelength_um) ** 2
    return xp.asarray(squared <= cutoff, dtype=xp.float32)


def _dpct_sources(
    config: QuantitativeTileConfig,
    x_frequency: npt.NDArray[Any],
    y_frequency: npt.NDArray[Any],
    pupil: npt.NDArray[Any],
    *,
    xp: Any = np,
) -> Any:
    sources: list[npt.NDArray[np.float32]] = []
    for angle in config.source_azimuth_deg:
        source = xp.zeros(pupil.shape, dtype=xp.float32)
        coordinate = y_frequency[:, xp.newaxis] * xp.cos(xp.deg2rad(angle)) + 1e-15
        threshold = x_frequency[xp.newaxis, :] * xp.sin(xp.deg2rad(angle))
        if angle < 180:
            source[coordinate >= threshold] = 1.0
            source *= pupil
        else:
            source[coordinate < threshold] = -1.0
            source *= pupil
            source += pupil
        sources.append(source)
    return xp.asarray(sources, dtype=xp.float32)


def _flip_source(source: npt.NDArray[Any], *, xp: Any = np) -> Any:
    flipped = xp.fft.fftshift(source)[::-1, ::-1]
    if flipped.shape[0] % 2 == 0:
        flipped = xp.roll(flipped, 1, axis=0)
    if flipped.shape[1] % 2 == 0:
        flipped = xp.roll(flipped, 1, axis=1)
    return xp.fft.ifftshift(flipped)


def _prepare_dpct(
    config: QuantitativeTileConfig,
    shape_zyx: tuple[int, int, int],
    *,
    xp: Any = np,
) -> tuple[npt.NDArray[Any], ...]:
    nz, ny, nx = shape_zyx
    fx = _frequency_axis(nx, config.pixel_size_um, xp=xp)
    fy = _frequency_axis(ny, config.pixel_size_um, xp=xp)
    pupil = _pupil(fx, fy, config.wavelength_um, config.numerical_aperture, xp=xp)
    sources = _dpct_sources(config, fx, fy, pupil, xp=xp)
    fz = xp.fft.ifftshift(
        (xp.arange(nz, dtype=xp.complex64) - nz // 2) * config.pixel_size_z_um
    )
    phase_defocus = (
        pupil
        * 2.0
        * xp.pi
        * xp.sqrt(
            (1.0 / config.wavelength_um) ** 2
            - fx[xp.newaxis, :] ** 2
            - fy[:, xp.newaxis] ** 2
        )
    )
    oblique = pupil / (
        4.0
        * xp.pi
        * xp.sqrt(
            (config.refractive_index_medium / config.wavelength_um) ** 2
            - fx[xp.newaxis, :] ** 2
            - fy[:, xp.newaxis] ** 2
        )
    )
    prop = xp.exp(1j * fz[xp.newaxis, xp.newaxis, :] * phase_defocus[:, :, None])
    window = xp.fft.ifftshift(xp.hamming(nz))
    sample_frequency_area = 1.0 / (ny * nx * config.pixel_size_um**2)
    real_transfer: list[npt.NDArray[Any]] = []
    imaginary_transfer: list[npt.NDArray[Any]] = []
    for source in sources:
        source_flipped = _flip_source(source, xp=xp)
        cross = (
            xp.fft.fft2(
                source_flipped[:, :, None] * pupil[:, :, None] * prop,
                axes=(0, 1),
            )
            * xp.fft.fft2(
                pupil[:, :, None] * prop * oblique[:, :, None], axes=(0, 1)
            ).conj()
        )
        real = 2.0 * xp.fft.ifft2(1j * cross.imag * sample_frequency_area, axes=(0, 1))
        imaginary = 2.0 * xp.fft.ifft2(cross.real * sample_frequency_area, axes=(0, 1))
        real *= window[None, None, :]
        imaginary *= window[None, None, :]
        real = xp.fft.fft(real, axis=2) * config.pixel_size_z_um
        imaginary = xp.fft.fft(imaginary, axis=2) * config.pixel_size_z_um
        total_source = xp.sum(source_flipped * pupil * pupil.conj())
        total_source *= sample_frequency_area
        real_transfer.append(real * 1j / total_source)
        imaginary_transfer.append(imaginary / total_source)
    return (
        xp.asarray(real_transfer, dtype=xp.complex64),
        xp.asarray(imaginary_transfer, dtype=xp.complex64),
    )


def _run_dpct(
    sample_zsyx: npt.NDArray[np.float32],
    config: QuantitativeTileConfig,
    operators: tuple[npt.NDArray[Any], ...],
    *,
    xp: Any = np,
) -> Any:
    real_transfer, imaginary_transfer = operators
    dpc = xp.transpose(sample_zsyx, (2, 3, 0, 1)).copy()
    mean_intensity = xp.mean(dpc, axis=(0, 1, 2), keepdims=True)
    if bool(xp.any(mean_intensity == 0)):
        raise ValueError("DPCT shot normalization encountered a zero mean")
    dpc /= mean_intensity
    dpc -= 1.0
    intensity = xp.fft.fftn(dpc, axes=(0, 1, 2)).transpose(3, 0, 1, 2)
    aha_00 = xp.sum(imaginary_transfer.conj() * imaginary_transfer, axis=0)
    aha_01 = xp.sum(imaginary_transfer.conj() * real_transfer, axis=0)
    aha_10 = xp.sum(real_transfer.conj() * imaginary_transfer, axis=0)
    aha_11 = xp.sum(real_transfer.conj() * real_transfer, axis=0)
    aha_00 += config.regularization_imaginary
    aha_11 += config.regularization_real
    ahy_0 = xp.sum(imaginary_transfer.conj() * intensity, axis=0)
    ahy_1 = xp.sum(real_transfer.conj() * intensity, axis=0)
    determinant = aha_00 * aha_11 - aha_01 * aha_10
    real = xp.fft.ifftn(
        (aha_00 * ahy_1 - aha_10 * ahy_0) / determinant, axes=(0, 1, 2)
    ).real
    imaginary = xp.fft.ifftn(
        (aha_11 * ahy_0 - aha_01 * ahy_1) / determinant, axes=(0, 1, 2)
    ).real
    wave_number = 2 * xp.pi / config.wavelength_um
    b_value = -(config.refractive_index_medium**2 - real / wave_number**2)
    c_value = -((-imaginary / wave_number**2 / 2) ** 2)
    refractive_index = xp.sqrt((-b_value + xp.sqrt(b_value**2 - 4 * c_value)) / 2)
    canonical = xp.rot90(xp.flip(refractive_index.T, axis=1), k=-1, axes=(1, 2))
    result = xp.asarray(canonical, dtype=xp.float32)
    if not bool(xp.all(xp.isfinite(result))):
        raise ValueError("DPCT reconstruction produced non-finite values")
    return result


def _illumination_pairs(
    angles: tuple[float, ...],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if abs(angles[0] - angles[1]) / 2 == 90:
        return (angles[0], angles[1]), (angles[2], angles[3])
    if abs(angles[0] - angles[2]) / 2 == 90:
        return (angles[0], angles[2]), (angles[1], angles[3])
    raise ValueError("qOBT source azimuths do not form two opposed pairs")


def _gaussian_source(
    config: QuantitativeTileConfig,
    angle: float,
    fx: npt.NDArray[Any],
    fy: npt.NDArray[Any],
    *,
    xp: Any = np,
) -> Any:
    elevation = xp.deg2rad(config.source_elevation_deg)
    azimuth = xp.deg2rad(angle)
    u_center = xp.sin(elevation) * xp.cos(azimuth) / config.wavelength_um
    v_center = xp.sin(elevation) * xp.sin(azimuth) / config.wavelength_um
    sigma = config.source_sigma * (config.numerical_aperture / config.wavelength_um)
    u_grid = fx[xp.newaxis, :]
    v_grid = fy[:, xp.newaxis]
    source = xp.exp(
        -((u_grid - u_center) ** 2 + (v_grid - v_center) ** 2) / (2 * sigma**2)
    ).real
    source[(u_grid**2 + v_grid**2).real > (1.0 / config.wavelength_um) ** 2] = 0
    total = xp.sum(source)
    if bool(total <= 0):
        raise ValueError("qOBT Gaussian source has zero support")
    return xp.asarray(source / total, dtype=xp.float32)


def _prepare_qobt(
    config: QuantitativeTileConfig,
    shape_zyx: tuple[int, int, int],
    *,
    xp: Any = np,
) -> tuple[npt.NDArray[Any], ...]:
    nz, ny, nx = shape_zyx
    fx = _frequency_axis(nx, config.pixel_size_um, xp=xp).real.astype(xp.float32)
    fy = _frequency_axis(ny, config.pixel_size_um, xp=xp).real.astype(xp.float32)
    pupil = _pupil(fx, fy, config.wavelength_um, config.numerical_aperture, xp=xp)
    pairs = _illumination_pairs(config.source_azimuth_deg)
    z_lags = (xp.arange(nz, dtype=xp.float32) - nz // 2) * config.pixel_size_z_um
    u_grid, v_grid = xp.meshgrid(fx, fy)
    radial_squared = u_grid**2 + v_grid**2
    medium_frequency = config.refractive_index_medium / config.wavelength_um
    axial_frequency = xp.sqrt(
        xp.maximum(medium_frequency**2 - radial_squared, 0.0)
    ).astype(xp.float32)
    negative_y = (-xp.arange(ny)) % ny
    negative_x = (-xp.arange(nx)) % nx
    negative_z = (-xp.arange(nz)) % nz
    wavelength_medium = config.wavelength_um / config.refractive_index_medium
    transfers: list[npt.NDArray[Any]] = []
    for first, second in pairs:
        source_difference = _gaussian_source(
            config, first, fx, fy, xp=xp
        ) - _gaussian_source(config, second, fx, fy, xp=xp)
        transfer_z = xp.empty((nz, ny, nx), dtype=xp.complex64)
        for z_index, z_lag in enumerate(z_lags):
            propagated = pupil * xp.exp(2j * xp.pi * axial_frequency * z_lag)
            correlation = xp.fft.ifft2(
                xp.fft.fft2(source_difference * propagated)
                * xp.fft.fft2(propagated).conj()
            )
            conjugate_negative = xp.conj(correlation[negative_y][:, negative_x])
            phase_transfer = 0.5j * (correlation - conjugate_negative)
            transfer_z[z_index] = -wavelength_medium / (4.0 * np.pi) * phase_transfer
        if config.axial_window == "hamming":
            transfer_z *= xp.hamming(nz).astype(xp.float32)[:, None, None]
        transform = (
            xp.fft.fft(xp.fft.ifftshift(transfer_z, axes=0), axis=0)
            * config.pixel_size_z_um
        )
        imaginary = transform.imag
        imaginary = 0.5 * (
            imaginary - imaginary[negative_z][:, negative_y][:, :, negative_x]
        )
        transfers.append(xp.asarray(1j * imaginary, dtype=xp.complex64))
    angle_to_index = {
        angle: index for index, angle in enumerate(config.source_azimuth_deg)
    }
    pair_indices = np.asarray(
        [[angle_to_index[first], angle_to_index[second]] for first, second in pairs],
        dtype=np.int64,
    )
    return xp.asarray(transfers), pair_indices


def _run_qobt(
    sample_zsyx: npt.NDArray[np.float32],
    config: QuantitativeTileConfig,
    operators: tuple[npt.NDArray[Any], ...],
    *,
    xp: Any = np,
) -> Any:
    transfer, pair_indices = operators
    images = xp.transpose(sample_zsyx, (1, 0, 2, 3))
    dpc_pairs = []
    for first, second in pair_indices:
        first_image = images[int(first)]
        second_image = images[int(second)]
        dpc_pairs.append(
            (first_image - second_image) / (first_image + second_image + 1e-6)
        )
    dpc = xp.asarray(dpc_pairs, dtype=xp.float32)
    measured = xp.fft.fftn(dpc, axes=(-3, -2, -1))
    numerator = xp.sum(measured * transfer.conj(), axis=0)
    denominator = xp.sum(xp.abs(transfer) ** 2, axis=0)
    denominator += config.regularization_scalar
    potential = xp.fft.ifftn(numerator / (denominator + 1e-12), axes=(-3, -2, -1)).real
    wave_number_medium = (
        2 * xp.pi * config.refractive_index_medium / config.wavelength_um
    )
    relative_squared = xp.maximum(1.0 - potential / wave_number_medium**2, 0.0)
    result = xp.asarray(
        config.refractive_index_medium * xp.sqrt(relative_squared),
        dtype=xp.float32,
    )
    if not bool(xp.all(xp.isfinite(result))):
        raise ValueError("qOBT reconstruction produced non-finite values")
    return result


def _apply_background(
    sample: npt.NDArray[np.float32],
    background_syx: npt.NDArray[Any] | None,
    config: QuantitativeTileConfig,
    *,
    xp: Any = np,
) -> Any:
    if background_syx is None:
        return sample
    background = xp.asarray(background_syx, dtype=xp.float32)
    if background.shape != sample.shape[1:]:
        raise ValueError("quantitative background must have matching SYX shape")
    if bool(xp.any(sample == 0)) or bool(xp.any(background == 0)):
        raise ValueError("quantitative background division cannot use zero pixels")
    if config.background_division == "background_over_sample":
        return xp.asarray(background[None, ...] / sample, dtype=xp.float32)
    return xp.asarray(sample / background[None, ...], dtype=xp.float32)


__all__ = [
    "CupyQuantitativeRuntime",
    "CupyQuantitativeTileSolver",
    "NumpyQuantitativeTileSolver",
    "QuantitativeTileConfig",
]
