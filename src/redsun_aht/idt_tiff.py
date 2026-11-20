"""Headless linear IDT preparation and output from declared-axis TIFFs."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import yaml

from .dpct_tiff import _digest, _mapped_tiff, _validate_channel_labels
from .scientific.reconstruction.linear_idt import (
    illumination_frequencies,
    reconstruct_linear_idt,
)
from .scientific.reconstruction.quantitative import CupyQuantitativeRuntime


def reconstruct(
    sample_path: Path,
    reference_path: Path | None,
    config_path: Path,
    output: Path,
    roi: str | None,
    zero_policy: str,
    *,
    backend: str,
    gpu_id: int,
    median_output: Path | None = None,
) -> None:
    """Reconstruct the optical depth stack from a single-focus IDT capture."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != 1:
        raise ValueError("expected reconstruction schema version 1")
    optics, model = config["optics"], config["idt"]
    wavelength, xy, dz, pupil_na, index = (
        float(optics[key])
        for key in (
            "wavelength_um",
            "pixel_size_um",
            "pixel_size_z_um",
            "numerical_aperture",
            "refractive_index_medium",
        )
    )
    reg_phase = float(model.get("regularization_phase", 2))
    reg_absorption = float(model.get("regularization_absorption", 100))
    if not all(
        np.isfinite(value) and value > 0
        for value in (wavelength, xy, dz, pupil_na, index, reg_phase, reg_absorption)
    ):
        raise ValueError("IDT optics and regularization must be positive and finite")
    if pupil_na > index:
        raise ValueError("IDT pupil NA cannot exceed the medium RI")
    start, stop = map(float, model["depth_range_um"])
    count_float = (stop - start) / dz
    if (
        not np.isfinite(count_float)
        or count_float < 0
        or not np.isclose(count_float, round(count_float), atol=1e-8, rtol=0)
    ):
        raise ValueError("IDT depth range must increase in whole Z steps")
    depths = start + np.arange(round(count_float) + 1) * dz
    calibration_path = (config_path.parent / model["illumination_file"]).resolve()
    if reference_path is None:
        raise ValueError("IDT requires a matched reference TIFF")
    cutoff = float(model.get("background_footprint_fraction", 0))
    taper = float(model.get("background_taper_px", 0))
    if not 0 <= cutoff < 1 or not np.isfinite(taper) or taper < 0:
        raise ValueError("invalid IDT background footprint or taper")
    sidecar = output.with_suffix(".provenance.json")
    outputs = [output, sidecar] + ([median_output] if median_output else [])
    if len({path.resolve() for path in outputs}) != len(outputs) or any(
        path.exists() for path in outputs
    ):
        raise FileExistsError(
            "IDT output, provenance and optional median must be distinct new files"
        )
    if (
        median_output is not None
        and median_output.parent.resolve() != output.parent.resolve()
    ):
        raise ValueError("place the IDT median beside the reconstruction output")
    paths = {
        "sample": sample_path,
        "reference": reference_path,
        "config": config_path,
        "illumination": calibration_path,
    }
    inputs = {
        key: {"path": str(path.resolve()), "sha256": _digest(path)}
        for key, path in paths.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix=".idt-", dir=output.parent) as folder,
        ExitStack() as stack,
    ):
        scratch = Path(folder)
        sample, _, sample_tags = stack.enter_context(
            _mapped_tiff(sample_path, scratch / "sample.raw", {"CYX"})
        )
        ref, ref_axes, ref_tags = stack.enter_context(
            _mapped_tiff(reference_path, scratch / "ref.raw", {"CYX", "TCYX"})
        )
        _validate_channel_labels(sample_tags, ref_tags)
        if ref.shape[-3:] != sample.shape or any(size == 0 for size in sample.shape):
            raise ValueError(
                "IDT reference and sample must match in channel and spatial dimensions"
            )
        shots, height, width = sample.shape
        frequencies = illumination_frequencies(
            calibration_path,
            shots,
            wavelength,
            geometry_na=float(model.get("illumination_geometry_na", 0.9)),
            legacy_rotation=bool(model.get("legacy_led_rotation", False)),
            channel_labels=sample_tags["channel_labels"] or ref_tags["channel_labels"],
        )
        region: tuple[slice, slice]
        if roi is None:
            region = np.s_[:, :]
        else:
            y, x, h, w = map(int, roi.split(","))
            if min(y, x) < 0 or min(h, w) <= 0 or y + h > height or x + w > width:
                raise ValueError("ROI lies outside the TIFF field")
            region = np.s_[y : y + h, x : x + w]
        ref_view = ref[(..., *region)]
        if ref_axes == "TCYX":
            median = np.median(ref if median_output else ref_view, axis=0).astype(
                np.float32
            )
        else:
            median = np.asarray(ref if median_output else ref_view, dtype=np.float32)
        reference = median[(..., *region)] if median_output else median
        chosen = sample[(..., *region)]
        usable = np.ones(reference.shape[-2:], dtype=bool)
        if cutoff:
            usable = (
                reference > np.median(reference, axis=(1, 2), keepdims=True) * cutoff
            ).all(axis=0)
        if not usable.any():
            raise ValueError("IDT background footprint leaves no usable pixels")
        invalid = ((chosen == 0) | (reference == 0)) & usable[None]
        invalid_count = int(invalid.sum())
        if invalid_count and zero_policy == "error":
            raise ValueError(
                "IDT normalization encountered zero pixels. "
                "Choose a valid ROI or set --zero-policy neutral."
            )
        contrast = np.ones(chosen.shape, dtype=np.float32)
        np.divide(chosen, reference, out=contrast, where=(reference != 0) & ~invalid)
        outside = ~usable
        for shot in contrast:
            shot[outside] = np.median(shot[usable])
        contrast -= np.median(contrast, axis=(1, 2), keepdims=True)
        if taper:
            from scipy.ndimage import gaussian_filter

            contrast *= gaussian_filter(usable.astype(np.float32), taper)[None]
        parameters: dict[str, Any] = dict(
            pixel_um=xy,
            step_um=dz,
            wavelength=wavelength,
            pupil_na=pupil_na,
            medium_index=index,
            reg_phase=reg_phase,
            reg_absorption=reg_absorption,
        )
        if backend == "cupy":
            runtime = CupyQuantitativeRuntime(
                gpu_id, cache_directory=str((scratch / "cupy-cache").absolute())
            )
            stack.callback(runtime.close)
            with runtime.device, runtime.stream:
                device_volume = reconstruct_linear_idt(
                    contrast, frequencies, depths, **parameters, xp=runtime.xp
                )
                volume = device_volume.get(stream=runtime.stream)
                del device_volume
        elif backend == "numpy":
            volume = reconstruct_linear_idt(contrast, frequencies, depths, **parameters)
        else:
            raise ValueError(f"unsupported reconstruction backend {backend!r}")
        resolution = (1 / xy, 1 / xy)
        temporary_tiff, temporary_sidecar = scratch / "ri.tif", scratch / "ri.json"
        tifffile.imwrite(
            temporary_tiff,
            volume,
            imagej=True,
            resolution=resolution,
            photometric="minisblack",
            metadata={
                "axes": "ZYX",
                "unit": "um",
                "spacing": dz,
                "Info": f"Absolute RI. Model defocus {start:g} to {stop:g} um.",
            },
        )
        pending = [(temporary_tiff, output), (temporary_sidecar, sidecar)]
        median_info = None
        if median_output is not None:
            temporary_median = scratch / "median.tif"
            labels = ref_tags["channel_labels"] or sample_tags["channel_labels"]
            metadata = {"axes": "CYX", "unit": "um"}
            tifffile.imwrite(
                temporary_median,
                median,
                imagej=True,
                resolution=resolution,
                photometric="minisblack",
                metadata={**metadata, "Labels": list(labels)} if labels else metadata,
            )
            median_info = {
                "path": median_output.name,
                "sha256": _digest(temporary_median),
                "dtype": "float32",
            }
            pending.append((temporary_median, median_output))
        report = dict(
            solver="idt",
            method="linear",
            backend=backend,
            gpu_id=gpu_id if backend == "cupy" else None,
            inputs=inputs,
            config=config,
            sample_axes="CYX",
            reference_axes=ref_axes,
            sample_channel_labels=sample_tags["channel_labels"],
            reference_channel_labels=ref_tags["channel_labels"],
            reference_median_repeats=ref.shape[0] if ref_axes == "TCYX" else None,
            roi_yxhw=roi,
            zero_policy=zero_policy,
            neutral_replacements=invalid_count,
            masked_background_pixels=int(outside.sum()),
            illumination_fxy_cycles_per_um=frequencies.tolist(),
            defocus_um=depths.tolist(),
            median_output=median_info,
            output={
                "axes": "ZYX",
                "shape": volume.shape,
                "dtype": "float32",
                "encoding": "absolute RI",
                "sha256": _digest(temporary_tiff),
            },
        )
        temporary_sidecar.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        linked = []
        try:
            for staged, destination in pending:
                os.link(staged, destination)
                linked.append(destination)
        except OSError:
            for destination in linked:
                destination.unlink()
            raise
