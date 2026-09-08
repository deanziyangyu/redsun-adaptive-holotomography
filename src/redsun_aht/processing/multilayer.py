"""Backend-neutral nonlinear multi-layer Born and multi-slice IDT models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ModelName = Literal["multi_born", "multislice"]
MeasurementDomain = Literal["amplitude", "intensity", "field"]


def _scalar(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


@dataclass(frozen=True, slots=True)
class MultiLayerConfig:
    """Frozen ZYX geometry and optical inputs for a nonlinear IDT model."""

    shape_zyx: tuple[int, int, int]
    voxel_size_zyx_um: tuple[float, float, float]
    wavelength_um: float
    numerical_aperture: float
    refractive_index_medium: float
    padding_yx: tuple[int, int] = (0, 0)
    defocus_um: float = 0.0
    slice_binning_factor: int = 1
    allow_evanescent_pupil: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous axes, nonphysical optics, and unsupported binning."""
        if len(self.shape_zyx) != 3 or any(
            type(value) is not int or value <= 0 for value in self.shape_zyx
        ):
            raise ValueError("multi-layer shape must contain three positive ZYX values")
        if len(self.voxel_size_zyx_um) != 3 or any(
            not np.isfinite(value) or value <= 0 for value in self.voxel_size_zyx_um
        ):
            raise ValueError("multi-layer voxel size must contain positive ZYX values")
        optics = (
            self.wavelength_um,
            self.numerical_aperture,
            self.refractive_index_medium,
        )
        if not all(np.isfinite(value) and value > 0 for value in optics):
            raise ValueError("multi-layer optical values must be positive")
        if (
            self.numerical_aperture > self.refractive_index_medium
            and not self.allow_evanescent_pupil
        ):
            raise ValueError("numerical aperture cannot exceed the medium index")
        if not isinstance(self.allow_evanescent_pupil, bool):
            raise ValueError("evanescent-pupil selection must be boolean")
        if len(self.padding_yx) != 2 or any(
            type(value) is not int or value < 0 for value in self.padding_yx
        ):
            raise ValueError("multi-layer padding must contain non-negative YX values")
        if not np.isfinite(self.defocus_um):
            raise ValueError("multi-layer defocus must be finite")
        if type(self.slice_binning_factor) is not int:
            raise ValueError("multi-layer binning factor must be an integer")
        if self.slice_binning_factor != 1:
            raise NotImplementedError("multi-layer v1 preserves every physical Z layer")


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """One illumination field and the optional private adjoint cache."""

    field_yx: Any
    cache: ForwardCache | None = None


@dataclass(frozen=True, slots=True)
class ForwardCache:
    """Intermediate fields required by one exact model adjoint."""

    native_object_zyx: Any
    incident_fields_zyx: Any
    pre_pupil_spectrum_yx: Any


@dataclass(frozen=True, slots=True)
class MSBPForwardCache:
    """Fields retained by the 2025 MSBP reverse pass for one view/focus."""

    native_object_zyx: Any
    layer_fields_real_zyx: Any
    layer_fields_imag_zyx: Any
    incident_compensation_yx: Any


@dataclass(frozen=True, slots=True)
class MSBPForwardResult:
    """One digitally refocused 2025 MSBP prediction and reverse cache."""

    field_yx: Any
    cache: MSBPForwardCache | None = None


class BaseMultiLayerModel:
    """Shared propagation, pupil, image-plane, and loss operations."""

    model_name: ModelName

    def __init__(self, config: MultiLayerConfig, *, xp: Any = np) -> None:
        self.config = config
        self.xp = xp
        self.real_dtype = xp.float32
        self.complex_dtype = xp.complex64
        nz, ny, nx = config.shape_zyx
        py, px = config.padding_yx
        self.shape_zyx = (nz, ny, nx)
        self.padding_yx = (py, px)
        self.computational_shape_yx = (ny + 2 * py, nx + 2 * px)
        self._crop_y = slice(py, py + ny)
        self._crop_x = slice(px, px + nx)
        self.dz_um, self.dy_um, self.dx_um = config.voxel_size_zyx_um
        self.wavelength_um = config.wavelength_um
        self.na = config.numerical_aperture
        self.n0 = config.refractive_index_medium
        self.k0 = 2.0 * np.pi / self.wavelength_um

        cy, cx = self.computational_shape_yx
        fy = xp.asarray(np.fft.fftfreq(cy, d=self.dy_um), dtype=self.real_dtype)
        fx = xp.asarray(np.fft.fftfreq(cx, d=self.dx_um), dtype=self.real_dtype)
        self.fy_yx, self.fx_yx = xp.meshgrid(fy, fx, indexing="ij")
        radial_squared = self.fx_yx**2 + self.fy_yx**2
        axial_squared = (self.n0 / self.wavelength_um) ** 2 - radial_squared
        self.kz_yx = xp.sqrt(axial_squared.astype(self.complex_dtype))
        self.propagation_phase_yx = (1j * 2.0 * np.pi * self.kz_yx).astype(
            self.complex_dtype
        )
        self.pupil_support_yx = (
            radial_squared <= (self.na / self.wavelength_um) ** 2
        ).astype(self.complex_dtype)
        self.pupil_yx = self.pupil_support_yx.copy()
        y = (xp.arange(cy, dtype=self.real_dtype) - cy // 2) * self.dy_um
        x = (xp.arange(cx, dtype=self.real_dtype) - cx // 2) * self.dx_um
        self.y_yx, self.x_yx = xp.meshgrid(y, x, indexing="ij")
        self.dfy = 1.0 / (cy * self.dy_um)
        self.dfx = 1.0 / (cx * self.dx_um)
        self.defocus_kernel_yx = self.pupil_support_yx * xp.exp(
            self.propagation_phase_yx * config.defocus_um
        )

    @property
    def pupil(self) -> Any:
        """Return the computational-grid pupil in unshifted FFT order."""
        return self.pupil_yx

    def set_pupil(self, pupil: Any) -> None:
        """Install one fixed pupil, padding centered spectra when required."""
        resolved = self.xp.asarray(pupil, dtype=self.complex_dtype)
        if resolved.shape == self.shape_zyx[-2:] and self.padding_yx != (0, 0):
            padded = self.xp.zeros(
                self.computational_shape_yx, dtype=self.complex_dtype
            )
            padded[self._crop_y, self._crop_x] = self.xp.fft.fftshift(resolved)
            resolved = self.xp.fft.ifftshift(padded)
        if resolved.shape != self.computational_shape_yx:
            raise ValueError(
                "multi-layer pupil must match the native or computational XY shape"
            )
        self.pupil_yx = resolved * self.pupil_support_yx

    def reset_pupil(self) -> None:
        """Restore the ideal unaberrated pupil."""
        self.pupil_yx = self.pupil_support_yx.copy()

    def update_pupil(
        self,
        gradient: Any,
        *,
        step_size: float,
        method: Literal["gradient", "gauss_newton"] = "gradient",
        hessian: Any | None = None,
        epsilon: float = 1e-3,
    ) -> None:
        """Apply a projected gradient or diagonal Gauss-Newton pupil update."""
        if not np.isfinite(step_size) or step_size < 0:
            raise ValueError("pupil step size must be finite and non-negative")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("pupil update epsilon must be finite and positive")
        resolved = self.xp.asarray(gradient, dtype=self.complex_dtype)
        if resolved.shape != self.computational_shape_yx:
            raise ValueError("pupil gradient shape does not match the model")
        if method == "gradient":
            update = resolved
        elif method == "gauss_newton":
            if hessian is None:
                raise ValueError("Gauss-Newton pupil update requires a Hessian")
            resolved_hessian = self.xp.asarray(hessian, dtype=self.real_dtype)
            if resolved_hessian.shape != self.computational_shape_yx:
                raise ValueError("pupil Hessian shape does not match the model")
            update = resolved / (resolved_hessian + epsilon)
        else:
            raise ValueError("pupil update method is unsupported")
        self.pupil_yx -= step_size * update * self.pupil_support_yx
        amplitude = self.xp.clip(self.xp.abs(self.pupil_yx), 0.0, 1.0)
        phase = self.xp.angle(self.pupil_yx)
        self.pupil_yx = (
            amplitude * self.xp.exp(1j * phase) * self.pupil_support_yx
        ).astype(self.complex_dtype)

    def _pad_native(self, native_zyx: Any) -> Any:
        native = self.xp.asarray(native_zyx, dtype=self.real_dtype)
        if native.shape != self.shape_zyx:
            raise ValueError("multi-layer object does not match configured ZYX shape")
        py, px = self.padding_yx
        if not py and not px:
            return native
        return self.xp.pad(native, ((0, 0), (py, py), (px, px)), mode="constant")

    def _crop_native(self, value: Any) -> Any:
        return value[:, self._crop_y, self._crop_x]

    def _pad_field(self, field_yx: Any) -> Any:
        field = self.xp.asarray(field_yx, dtype=self.complex_dtype)
        if field.shape != self.shape_zyx[-2:]:
            raise ValueError("multi-layer image field does not match native XY shape")
        output = self.xp.zeros(self.computational_shape_yx, dtype=self.complex_dtype)
        output[self._crop_y, self._crop_x] = field
        return output

    def _propagate(
        self, field_yx: Any, distance_um: float, *, adjoint: bool = False
    ) -> Any:
        kernel = self.xp.exp(self.propagation_phase_yx * distance_um)
        if adjoint:
            kernel = self.xp.conj(kernel)
        return self.xp.fft.ifft2(self.xp.fft.fft2(field_yx) * kernel).astype(
            self.complex_dtype
        )

    def _illumination(self, fx_value: float, fy_value: float) -> tuple[Any, complex]:
        fx = float(np.round(fx_value / self.dfx) * self.dfx)
        fy = float(np.round(fy_value / self.dfy) * self.dfy)
        fz = np.sqrt((self.n0 / self.wavelength_um) ** 2 - fx**2 - fy**2 + 0j)
        field = self.xp.exp(
            1j * 2.0 * np.pi * (fx * self.x_yx + fy * self.y_yx)
        ).astype(self.complex_dtype)
        return field, fz

    def _image_forward(self, field_yx: Any) -> tuple[Any, Any]:
        detector = self.xp.fft.ifft2(
            self.xp.fft.fft2(field_yx) * self.defocus_kernel_yx
        )
        pre_pupil = self.xp.fft.fft2(detector)
        image = self.xp.fft.ifft2(self.pupil_yx * pre_pupil)
        return image[self._crop_y, self._crop_x], pre_pupil

    def _image_adjoint(self, residual_yx: Any, pre_pupil: Any) -> tuple[Any, Any, Any]:
        spectrum = self.xp.fft.fft2(self._pad_field(residual_yx))
        pixel_count = int(np.prod(self.computational_shape_yx))
        pupil_gradient = self.xp.conj(pre_pupil) * spectrum / pixel_count
        pupil_hessian = self.xp.abs(pre_pupil) ** 2 / pixel_count
        detector_gradient = self.xp.fft.ifft2(self.xp.conj(self.pupil_yx) * spectrum)
        model_gradient = self.xp.fft.ifft2(
            self.xp.conj(self.defocus_kernel_yx) * self.xp.fft.fft2(detector_gradient)
        )
        return model_gradient, pupil_gradient, pupil_hessian

    def ri_to_native(self, ri_zyx: Any) -> Any:
        """Map refractive index to the model's native object variable."""
        raise NotImplementedError

    def native_to_ri(self, native_zyx: Any) -> Any:
        """Map the model's native object variable to refractive index."""
        raise NotImplementedError

    def project_native(self, native_zyx: Any) -> Any:
        """Project a native object onto the configured physical sign constraint."""
        raise NotImplementedError

    def _forward_scattering(
        self, native_zyx: Any, fx: float, fy: float
    ) -> tuple[Any, Any]:
        raise NotImplementedError

    def _adjoint_scattering(self, residual_yx: Any, cache: ForwardCache) -> Any:
        raise NotImplementedError

    def forward_native(
        self, native_zyx: Any, fx: float, fy: float, *, return_cache: bool = False
    ) -> ForwardResult:
        """Propagate one native object for one illumination frequency."""
        padded = self._pad_native(native_zyx)
        field, incident = self._forward_scattering(padded, fx, fy)
        image, pre_pupil = self._image_forward(field)
        cache = ForwardCache(padded, incident, pre_pupil) if return_cache else None
        return ForwardResult(image.astype(self.complex_dtype), cache)

    def forward(self, ri_zyx: Any, illumination_fxy: Any) -> Any:
        """Return complex SYX fields for all supplied illuminations."""
        frequencies = np.asarray(illumination_fxy, dtype=np.float64)
        if frequencies.ndim != 2 or frequencies.shape[1] != 2:
            raise ValueError("illumination frequencies must have shape S,2")
        if not np.all(np.isfinite(frequencies)):
            raise ValueError("illumination frequencies must be finite")
        native = self.ri_to_native(ri_zyx)
        return self.xp.stack(
            [self.forward_native(native, fx, fy).field_yx for fx, fy in frequencies]
        )

    def loss_and_gradient(
        self,
        native_zyx: Any,
        measurement_yx: Any,
        fx: float,
        fy: float,
        *,
        measurement_domain: MeasurementDomain = "amplitude",
    ) -> tuple[float, Any, Any, Any, Any]:
        """Return one-shot loss plus native-object and pupil derivatives."""
        result = self.forward_native(native_zyx, fx, fy, return_cache=True)
        if result.cache is None:  # pragma: no cover - construction invariant
            raise RuntimeError("multi-layer adjoint cache was not created")
        predicted = result.field_yx
        measurement = self.xp.asarray(measurement_yx)
        if measurement.shape != predicted.shape:
            raise ValueError("multi-layer measurement shape does not match prediction")
        if measurement_domain == "field":
            residual = predicted - measurement.astype(self.complex_dtype)
            cost = 0.5 * self.xp.sum(self.xp.abs(residual) ** 2)
            field_gradient = residual
        elif measurement_domain == "amplitude":
            amplitude = self.xp.abs(predicted)
            radial = amplitude - measurement.astype(self.real_dtype)
            cost = 0.5 * self.xp.sum(radial**2)
            field_gradient = radial * predicted / self.xp.maximum(amplitude, 1e-8)
        elif measurement_domain == "intensity":
            residual = self.xp.abs(predicted) ** 2 - measurement.astype(self.real_dtype)
            cost = 0.5 * self.xp.sum(residual**2)
            field_gradient = 2.0 * predicted * residual
        else:
            raise ValueError("multi-layer measurement domain is unsupported")
        scattering, pupil_gradient, pupil_hessian = self._image_adjoint(
            field_gradient, result.cache.pre_pupil_spectrum_yx
        )
        gradient = self._adjoint_scattering(scattering, result.cache)
        return (
            _scalar(cost),
            gradient.astype(self.real_dtype),
            pupil_gradient.astype(self.complex_dtype),
            pupil_hessian.astype(self.real_dtype),
            predicted,
        )

    def phase_change_per_slice(self, ri_zyx: Any) -> float:
        """Return maximum single-layer phase change in radians."""
        contrast = self.xp.max(self.xp.abs(self.xp.asarray(ri_zyx) - self.n0))
        return _scalar(2.0 * np.pi * contrast * self.dz_um / self.wavelength_um)

    def estimated_cache_bytes(self) -> int:
        """Estimate the exact incident-field cache size."""
        nz = self.shape_zyx[0]
        cy, cx = self.computational_shape_yx
        return nz * cy * cx * np.dtype(np.complex64).itemsize * 2

    def estimated_working_set_bytes(self, shots_num: int = 1) -> int:
        """Return the donor's conservative sequential-shot working-set estimate."""
        nz, ny, nx = self.shape_zyx
        cy, cx = self.computational_shape_yx
        volume_work = 7 * nz * cy * cx * np.dtype(np.float32).itemsize
        plane_work = (
            (14 + max(shots_num, 1)) * cy * cx * np.dtype(np.complex64).itemsize
        )
        public_output = nz * ny * nx * np.dtype(np.float32).itemsize
        return self.estimated_cache_bytes() + volume_work + plane_work + public_output


class MultiSliceModel(BaseMultiLayerModel):
    """Defocus-diverse MSBP model from Kim and Chowdhury's 2025/2026 code."""

    model_name: ModelName = "multislice"

    def ri_to_native(self, ri_zyx: Any) -> Any:
        """Return refractive-index contrast relative to the medium."""
        return self.xp.asarray(ri_zyx, dtype=self.real_dtype) - self.n0

    def native_to_ri(self, native_zyx: Any) -> Any:
        """Restore absolute refractive index from contrast."""
        return self.xp.asarray(native_zyx.real, dtype=self.real_dtype) + self.n0

    def project_native(self, native_zyx: Any) -> Any:
        """Apply the donor's real, non-negative delta-RI constraint."""
        return self.xp.maximum(native_zyx.real, 0.0).astype(self.real_dtype)

    def forward_native(
        self, native_zyx: Any, fx: float, fy: float, *, return_cache: bool = False
    ) -> ForwardResult:
        """Evaluate the replacement MSBP model at its zero-focus plane.

        The public multi-layer interface has no focus-plane argument.  Keep
        that interface useful by selecting the donor's zero-focus plane;
        focus-diverse reconstruction uses :meth:`forward_refocused_native`.
        """
        result = self.forward_refocused_native(native_zyx, fx, fy, 0.0)
        return ForwardResult(result.field_yx)

    def loss_and_gradient(
        self,
        native_zyx: Any,
        measurement_yx: Any,
        fx: float,
        fy: float,
        *,
        measurement_domain: MeasurementDomain = "amplitude",
    ) -> tuple[float, Any, Any, Any, Any]:
        """Evaluate the replacement MSBP loss at its zero-focus plane."""
        loss, gradient, predicted = self.loss_and_gradient_refocused(
            native_zyx,
            measurement_yx,
            fx,
            fy,
            0.0,
            measurement_domain=measurement_domain,
            compensate_field=False,
        )
        pupil_gradient = self.xp.zeros_like(self.pupil_yx)
        pupil_hessian = self.xp.zeros_like(self.pupil_yx.real)
        return loss, gradient, pupil_gradient, pupil_hessian, predicted

    def illumination_obliquity(self, fx: float, fy: float) -> float:
        """Return the angle-dependent phase-delay obliquity factor."""
        sine = self.wavelength_um * np.hypot(fx, fy) / self.n0
        if sine >= 1.0:
            raise ValueError("MSBP illumination is evanescent in the medium")
        return float(1.0 / np.sqrt(1.0 - sine**2))

    def _truncated_illumination(self, fx: float, fy: float) -> Any:
        """Create the donor's FFT-grid-snapped incident plane wave."""
        snapped_fx = float(np.trunc(fx / self.dfx) * self.dfx)
        snapped_fy = float(np.trunc(fy / self.dfy) * self.dfy)
        return self.xp.exp(
            1j * 2.0 * np.pi * (snapped_fx * self.x_yx + snapped_fy * self.y_yx)
        ).astype(self.complex_dtype)

    def _signed_focus_kernel(self, offset_slices: float, *, adjoint: bool) -> Any:
        distance = abs(float(offset_slices)) * self.dz_um
        # The MATLAB reference back-propagates positive focus offsets and
        # forward-propagates negative ones. The adjoint reverses that choice.
        forward = offset_slices <= 0
        if adjoint:
            forward = not forward
        kernel = self.xp.exp(self.propagation_phase_yx * distance)
        return kernel if forward else self.xp.conj(kernel)

    def _center_plane_kernel(self, *, adjoint: bool) -> Any:
        distance = abs(self.config.defocus_um)
        kernel = self.xp.exp(self.propagation_phase_yx * distance)
        forward = self.config.defocus_um >= 0
        if adjoint:
            forward = not forward
        return kernel if forward else self.xp.conj(kernel)

    def refocus_measurements(
        self,
        fields_syx: Any,
        illumination_fxy: Any,
        focus_offsets_slices: tuple[float, ...],
    ) -> Any:
        """Digitally refocus raw complex fields and divide out illumination."""
        fields = self.xp.asarray(fields_syx, dtype=self.complex_dtype)
        frequencies = np.asarray(illumination_fxy, dtype=np.float64)
        if fields.ndim != 3 or fields.shape[1:] != self.shape_zyx[-2:]:
            raise ValueError("MSBP raw fields must have SYX axes")
        if frequencies.shape != (fields.shape[0], 2):
            raise ValueError("MSBP illuminations must match the shot axis")
        if not focus_offsets_slices:
            raise ValueError("MSBP requires at least one focus offset")
        py, px = self.padding_yx
        padded = self.xp.pad(
            fields,
            ((0, 0), (py, py), (px, px)),
            mode="constant",
            constant_values=1,
        )
        refocused = self.xp.empty(
            (fields.shape[0], len(focus_offsets_slices), *fields.shape[1:]),
            dtype=self.complex_dtype,
        )
        epsilon = np.finfo(np.float32).eps
        for shot, (fx, fy) in enumerate(frequencies):
            incident = self._truncated_illumination(float(fx), float(fy))
            modulated_spectrum = self.xp.fft.fft2(padded[shot] * incident)
            incident_spectrum = self.xp.fft.fft2(incident)
            for focus, offset in enumerate(focus_offsets_slices):
                kernel = self._signed_focus_kernel(offset, adjoint=False)
                numerator = self.xp.fft.ifft2(modulated_spectrum * kernel)
                denominator = self.xp.fft.ifft2(incident_spectrum * kernel)
                safe = self.xp.where(
                    self.xp.abs(denominator) > epsilon, denominator, 1.0 + 0.0j
                )
                normalized = numerator / safe
                refocused[shot, focus] = normalized[self._crop_y, self._crop_x]
        return refocused

    def forward_refocused_native(
        self,
        native_zyx: Any,
        fx: float,
        fy: float,
        focus_offset_slices: float,
        *,
        return_cache: bool = False,
    ) -> MSBPForwardResult:
        """Evaluate one angle and one digital-focus plane."""
        native = self._pad_native(native_zyx)
        nz = native.shape[0]
        incident = self._truncated_illumination(fx, fy)
        spectrum = self.xp.fft.fft2(incident)
        propagation = self.xp.exp(self.propagation_phase_yx * self.dz_um)
        obliquity = self.illumination_obliquity(fx, fy)
        transmission_phase = (
            2.0 * np.pi * native * self.dz_um * obliquity / self.wavelength_um
        )
        transmission_real = self.xp.cos(transmission_phase).astype(self.real_dtype)
        transmission_imag = self.xp.sin(transmission_phase).astype(self.real_dtype)
        layer_fields_real = self.xp.empty_like(native, dtype=self.real_dtype)
        layer_fields_imag = self.xp.empty_like(native, dtype=self.real_dtype)
        for layer in range(nz):
            field = self.xp.fft.ifft2(spectrum * propagation)
            layer_fields_real[layer] = field.real
            layer_fields_imag[layer] = field.imag
            transmitted = (
                field.real * transmission_real[layer]
                - field.imag * transmission_imag[layer]
            ).astype(self.complex_dtype)
            transmitted += 1j * (
                field.real * transmission_imag[layer]
                + field.imag * transmission_real[layer]
            )
            spectrum = self.xp.fft.fft2(transmitted)

        spectrum *= self.xp.conj(
            self.xp.exp(0.5 * nz * self.propagation_phase_yx * self.dz_um)
        )
        center_kernel = self._center_plane_kernel(adjoint=False) * self.pupil_yx
        center_field = self.xp.fft.ifft2(spectrum * center_kernel)
        focus_kernel = self._signed_focus_kernel(focus_offset_slices, adjoint=False)
        focused = self.xp.fft.ifft2(self.xp.fft.fft2(center_field) * focus_kernel)

        entrance_spectrum = self.xp.fft.fft2(incident)
        incident_to_center = self.xp.exp(
            self.propagation_phase_yx * (0.5 * nz * self.dz_um + self.config.defocus_um)
        )
        center_incident = self.xp.fft.ifft2(entrance_spectrum * incident_to_center)
        focused_incident = self.xp.fft.ifft2(
            self.xp.fft.fft2(center_incident) * focus_kernel
        )
        compensation = focused_incident[self._crop_y, self._crop_x]
        cache = (
            MSBPForwardCache(
                native,
                layer_fields_real,
                layer_fields_imag,
                compensation,
            )
            if return_cache
            else None
        )
        return MSBPForwardResult(
            focused[self._crop_y, self._crop_x].astype(self.complex_dtype), cache
        )

    def loss_and_gradient_refocused(
        self,
        native_zyx: Any,
        measurement_yx: Any,
        fx: float,
        fy: float,
        focus_offset_slices: float,
        *,
        measurement_domain: MeasurementDomain,
        compensate_field: bool = True,
    ) -> tuple[float, Any, Any]:
        """Return the donor-compatible loss, real delta-RI gradient, and field."""
        result = self.forward_refocused_native(
            native_zyx,
            fx,
            fy,
            focus_offset_slices,
            return_cache=True,
        )
        if result.cache is None:  # pragma: no cover - construction invariant
            raise RuntimeError("MSBP reverse cache was not created")
        predicted = result.field_yx
        measurement = self.xp.asarray(measurement_yx)
        if measurement.shape != predicted.shape:
            raise ValueError("MSBP measurement shape does not match prediction")
        if measurement_domain == "field":
            target = measurement.astype(self.complex_dtype)
            if compensate_field:
                target *= result.cache.incident_compensation_yx
            residual = predicted - target
        elif measurement_domain in {"amplitude", "intensity"}:
            target_amplitude = measurement.astype(self.real_dtype)
            if measurement_domain == "intensity":
                target_amplitude = self.xp.sqrt(self.xp.maximum(target_amplitude, 0))
            amplitude = self.xp.abs(predicted)
            residual = predicted - target_amplitude * predicted / self.xp.maximum(
                amplitude, 1e-8
            )
        else:  # pragma: no cover - closed Literal validated by solve config
            raise ValueError("MSBP measurement domain is unsupported")
        cost = self.xp.sum(self.xp.abs(residual) ** 2)

        padded_residual = self._pad_field(residual)
        focus_adjoint = self._signed_focus_kernel(focus_offset_slices, adjoint=True)
        back = self.xp.fft.ifft2(self.xp.fft.fft2(padded_residual) * focus_adjoint)
        center_adjoint = self._center_plane_kernel(adjoint=True) * self.xp.conj(
            self.pupil_yx
        )
        back = self.xp.fft.ifft2(self.xp.fft.fft2(back) * center_adjoint)
        back = self.xp.fft.ifft2(
            self.xp.fft.fft2(back)
            * self.xp.exp(
                0.5 * self.shape_zyx[0] * self.propagation_phase_yx * self.dz_um
            )
        )

        native = result.cache.native_object_zyx
        obliquity = self.illumination_obliquity(fx, fy)
        transmission_phase = (
            2.0 * np.pi * native * self.dz_um * obliquity / self.wavelength_um
        )
        transmission_real = self.xp.cos(transmission_phase).astype(self.real_dtype)
        transmission_imag = self.xp.sin(transmission_phase).astype(self.real_dtype)
        propagation_adjoint = self.xp.conj(
            self.xp.exp(self.propagation_phase_yx * self.dz_um)
        )
        gradient = self.xp.empty_like(native, dtype=self.real_dtype)
        sigma = 2.0 * np.pi * self.dz_um / self.wavelength_um
        for layer in range(native.shape[0] - 1, -1, -1):
            field_real = result.cache.layer_fields_real_zyx[layer]
            field_imag = result.cache.layer_fields_imag_zyx[layer]
            field_back_real = field_real * back.real + field_imag * back.imag
            field_back_imag = field_real * back.imag - field_imag * back.real
            update_imag = (
                transmission_real[layer] * field_back_imag
                - transmission_imag[layer] * field_back_real
            )
            # BPM_update accumulates a complex update but every regularization
            # and published RI path consumes its real component.
            gradient[layer] = sigma * update_imag
            back_real = (
                back.real * transmission_real[layer]
                + back.imag * transmission_imag[layer]
            )
            back_imag = (
                back.imag * transmission_real[layer]
                - back.real * transmission_imag[layer]
            )
            if layer > 0:
                back = back_real.astype(self.complex_dtype)
                back += 1j * back_imag
                back = self.xp.fft.ifft2(self.xp.fft.fft2(back) * propagation_adjoint)
        return _scalar(cost), self._crop_native(gradient), predicted

    def _forward_scattering(
        self, native_zyx: Any, fx: float, fy: float
    ) -> tuple[Any, Any]:
        result = self.forward_refocused_native(
            self._crop_native(native_zyx), fx, fy, 0.0, return_cache=True
        )
        if result.cache is None:  # pragma: no cover - construction invariant
            raise RuntimeError("MSBP reverse cache was not created")
        # BaseMultiLayerModel would apply its image-plane model a second time,
        # so generic forward_native is deliberately unsupported for MSBP.
        raise NotImplementedError(
            "use forward_refocused_native for the defocus-diverse MSBP model"
        )

    def _adjoint_scattering(self, residual_yx: Any, cache: ForwardCache) -> Any:
        raise NotImplementedError(
            "use loss_and_gradient_refocused for the defocus-diverse MSBP model"
        )

    def estimated_working_set_bytes(self, shots_num: int = 1) -> int:
        """Estimate the sequential MSBP solve excluding retained input fields."""
        nz, ny, nx = self.shape_zyx
        cy, cx = self.computational_shape_yx
        volume = (
            nz
            * cy
            * cx
            * (2 * np.dtype(np.complex64).itemsize + 3 * np.dtype(np.float32).itemsize)
        )
        planes = (18 + max(shots_num, 1)) * cy * cx * np.dtype(np.complex64).itemsize
        output = nz * ny * nx * np.dtype(np.float32).itemsize
        return volume + planes + output


class MultiLayerBornModel(BaseMultiLayerModel):
    """Recursive multi-layer Born multiple-scattering model."""

    model_name: ModelName = "multi_born"

    def __init__(self, config: MultiLayerConfig, *, xp: Any = np) -> None:
        super().__init__(config, xp=xp)
        radial_squared = self.fx_yx**2 + self.fy_yx**2
        mask = (self.n0 / self.wavelength_um) ** 2 > 1.01 * radial_squared
        with np.errstate(divide="ignore", invalid="ignore"):
            green = (
                -0.25j
                * xp.exp(2j * np.pi * self.kz_yx * self.dz_um)
                / (np.pi * self.kz_yx)
            )
        self.green_kernel_yx = xp.where(mask, green, 0).astype(self.complex_dtype)

    def ri_to_native(self, ri_zyx: Any) -> Any:
        """Convert refractive index to the donor scattering potential."""
        ri = self.xp.asarray(ri_zyx, dtype=self.real_dtype)
        return (self.k0**2 * (self.n0**2 - ri**2)).astype(self.real_dtype)

    def native_to_ri(self, native_zyx: Any) -> Any:
        """Convert scattering potential back to refractive index."""
        native = self.xp.asarray(native_zyx, dtype=self.real_dtype)
        return self.xp.sqrt(
            self.xp.maximum(self.n0**2 - native / self.k0**2, 0)
        ).astype(self.real_dtype)

    def project_native(self, native_zyx: Any) -> Any:
        """Enforce the donor's non-positive scattering-potential convention."""
        return self.xp.minimum(native_zyx.real, 0).astype(self.real_dtype)

    def _forward_scattering(
        self, native_zyx: Any, fx: float, fy: float
    ) -> tuple[Any, Any]:
        nz = self.shape_zyx[0]
        field, fz = self._illumination(fx, fy)
        field *= np.exp(-0.5j * 2.0 * np.pi * fz * (nz - 1) * self.dz_um)
        incident = self.xp.empty(
            (nz, *self.computational_shape_yx), dtype=self.complex_dtype
        )
        for layer in range(nz):
            incident[layer] = field
            before = field
            field = self._propagate(field, self.dz_um)
            source = before * native_zyx[layer]
            scattered = (
                self.xp.fft.ifft2(self.xp.fft.fft2(source) * self.green_kernel_yx)
                * self.dz_um
            )
            field += scattered
        field = self._propagate(field, 0.5 * (nz + 1) * self.dz_um, adjoint=True)
        return field, incident

    def _adjoint_scattering(self, residual_yx: Any, cache: ForwardCache) -> Any:
        native = cache.native_object_zyx
        nz = native.shape[0]
        field = self._propagate(residual_yx, 0.5 * (nz + 1) * self.dz_um)
        gradient = self.xp.empty_like(native, dtype=self.real_dtype)
        green_adjoint = self.xp.conj(self.green_kernel_yx)
        for layer in range(nz - 1, -1, -1):
            scattered = (
                self.xp.fft.ifft2(self.xp.fft.fft2(field) * green_adjoint) * self.dz_um
            )
            gradient[layer] = self.xp.real(
                scattered * self.xp.conj(cache.incident_fields_zyx[layer])
            )
            if layer > 0:
                field = self._propagate(field, self.dz_um, adjoint=True)
                field += scattered * native[layer]
        return self._crop_native(gradient)


def create_multilayer_model(
    model: ModelName | str, config: MultiLayerConfig, *, xp: Any = np
) -> BaseMultiLayerModel:
    """Construct one explicitly allowlisted nonlinear model."""
    normalized = str(model).strip().lower().replace("-", "_")
    if normalized in {"multi_born", "multiborn", "mlb"}:
        return MultiLayerBornModel(config, xp=xp)
    if normalized in {"multislice", "multi_slice", "ms", "msbp"}:
        return MultiSliceModel(config, xp=xp)
    raise ValueError(f"unknown multi-layer model: {model}")


def ring_illumination_frequencies(
    shots_num: int, *, wavelength_um: float, illumination_na: float
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Generate a deterministic evenly spaced illumination ring in cycles/um."""
    if (
        type(shots_num) is not int
        or shots_num <= 0
        or not np.isfinite(wavelength_um)
        or wavelength_um <= 0
        or not np.isfinite(illumination_na)
        or illumination_na < 0
    ):
        raise ValueError("illumination ring inputs are invalid")
    angles = np.linspace(0, 2 * np.pi, shots_num, endpoint=False)
    radius = illumination_na / wavelength_um
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


__all__ = [
    "BaseMultiLayerModel",
    "ForwardResult",
    "MSBPForwardCache",
    "MSBPForwardResult",
    "MeasurementDomain",
    "ModelName",
    "MultiLayerBornModel",
    "MultiLayerConfig",
    "MultiSliceModel",
    "create_multilayer_model",
    "ring_illumination_frequencies",
]
