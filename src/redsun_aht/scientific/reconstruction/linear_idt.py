"""Offline linear IDT kernel, based on the pyhololab Fourier reconstruction.

The inputs are camera-grid intensity contrast and ordered (fx, fy) frequencies.
No instrument, application session, or donor import is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def illumination_frequencies(
    path: Path,
    shots: int,
    wavelength: float,
    *,
    geometry_na: float = 0.9,
    legacy_rotation: bool = False,
    channel_labels: tuple[str, ...] | None = None,
) -> Any:
    """Load ordered illumination frequencies in cycles per micrometre."""
    suffix = path.suffix.lower()
    led_indices = None
    order_key = "firmware_led_index_zero_based"
    if suffix == ".npy":
        frequencies = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            frequencies = archive["illumination_fxy"]
            led_indices = archive.get(order_key)
    elif suffix == ".mat":
        from scipy.io import loadmat

        calibration = loadmat(path)
        led_indices = calibration.get(order_key)
        if "na_list" in calibration:
            frequencies = calibration["na_list"] / wavelength
        elif "illumination_fxy" in calibration:
            frequencies = calibration["illumination_fxy"]
        else:
            frequencies = np.column_stack(
                (calibration["kx"].ravel(), calibration["ky"].ravel())
            )
    elif suffix in {".h5", ".hdf5"}:
        import h5py

        if not 0 < geometry_na < 1:
            raise ValueError("HDF5 geometry NA must lie in (0, 1)")
        with h5py.File(path, "r") as calibration:
            positions = np.asarray(calibration["Sorted_Pos"])
            if order_key in calibration:
                led_indices = calibration[order_key][:]
        if positions.shape == (2, shots):
            positions = positions.T
        if positions.shape != (shots, 2):
            raise ValueError("Sorted_Pos must hold one XY position per ordered shot")
        distance_in_radii = 1 / np.tan(np.arcsin(geometry_na))
        frequencies = positions / (
            wavelength
            * np.sqrt(
                np.sum(positions**2, axis=1, keepdims=True) + distance_in_radii**2
            )
        )
        if legacy_rotation:
            frequencies = np.column_stack((-frequencies[:, 1], frequencies[:, 0]))
    else:
        raise ValueError("IDT illumination requires .npy, .npz, .mat, or .h5")
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.shape != (shots, 2) or not np.isfinite(frequencies).all():
        raise ValueError(
            "illumination calibration must have one finite (fx, fy) pair per shot"
        )
    if np.any(np.linalg.norm(frequencies, axis=1) * wavelength >= 1):
        raise ValueError("illumination directions must be propagating")
    if led_indices is not None and channel_labels is not None:
        declared = tuple(str(int(led)) for led in np.asarray(led_indices).ravel())
        if declared != channel_labels:
            raise ValueError(
                "IDT TIFF channels disagree with the calibration LED order"
            )
    return frequencies


def reconstruct_linear_idt(
    contrast: Any,
    frequencies: Any,
    depths: Any,
    *,
    pixel_um: float,
    step_um: float,
    wavelength: float,
    pupil_na: float,
    medium_index: float,
    reg_phase: float,
    reg_absorption: float,
    xp: Any = np,
) -> Any:
    """Solve the two-potential weak-scattering system on each output depth."""
    shots, height, width = contrast.shape
    fy = (xp.arange(height) - height // 2)[:, None] / (height * pixel_um)
    fx = (xp.arange(width) - width // 2)[None, :] / (width * pixel_um)
    pupil = xp.hypot(fx, fy) <= pupil_na / wavelength
    wave = 2 * np.pi / wavelength
    wave_medium = wave * medium_index
    spectra = xp.fft.fft2(xp.asarray(contrast, dtype=xp.float32), axes=(-2, -1)).astype(
        xp.complex64
    )
    result = xp.empty((len(depths), height, width), dtype=xp.float32)
    scale = 0.5 * step_um * wave**2
    for index, depth in enumerate(depths):
        phase_power = xp.zeros((height, width), dtype=xp.float32)
        absorption_power = xp.zeros_like(phase_power)
        cross = xp.zeros((height, width), dtype=xp.complex64)
        phase_data = xp.zeros_like(cross)
        absorption_data = xp.zeros_like(cross)
        for shot in range(shots):
            ux, vy = frequencies[shot]
            lateral_offset = (
                round(vy * pixel_um * height),
                round(ux * pixel_um * width),
            )
            axial_source = np.sqrt(1 - wavelength**2 * (ux**2 + vy**2))
            fields = []
            for sign in (1, -1):
                axial_term = 1 - wavelength**2 * (
                    (fx - sign * ux) ** 2 + (fy - sign * vy) ** 2
                )
                axial = xp.sqrt(xp.maximum(axial_term, 0))
                propagating = axial_term > 0
                denominator = xp.where(propagating, wave_medium * axial, 1.0)
                green = xp.where(propagating, 1.0 / denominator, 0.0)
                shifted = xp.roll(
                    pupil, tuple(sign * shift for shift in lateral_offset), axis=(0, 1)
                )
                fields.append(
                    shifted
                    * green
                    * xp.exp(1j * wave_medium * depth * (axial - axial_source))
                )
            forward, backward = fields
            phase_transfer = xp.fft.fftshift(1j * (forward.conj() - backward)) * scale
            absorption_transfer = xp.fft.fftshift(-(forward.conj() + backward)) * scale
            # Donor PTF = sin(+) + sin(-) + i[cos(+) - cos(-)].
            # Absorption = -cos(+) - cos(-) + i[sin(+) - sin(-)].
            phase_power += xp.abs(phase_transfer) ** 2
            absorption_power += xp.abs(absorption_transfer) ** 2
            cross += phase_transfer.conj() * absorption_transfer
            phase_data += phase_transfer.conj() * spectra[shot]
            absorption_data += absorption_transfer.conj() * spectra[shot]
        phase_power += reg_phase
        absorption_power += reg_absorption
        determinant = phase_power * absorption_power - xp.abs(cross) ** 2
        phase = xp.fft.ifft2(
            (absorption_power * phase_data - cross * absorption_data) / determinant
        ).real
        absorption = xp.fft.ifft2(
            (phase_power * absorption_data - cross.conj() * phase_data) / determinant
        ).real
        squared = medium_index**2 + phase
        result[index] = xp.sqrt((squared + xp.hypot(squared, absorption)) / 2)
    if not bool(xp.isfinite(result).all()):
        raise ValueError("IDT produced non-finite RI")
    return result
