"""Direct legacy TIFF DPCT, adapted from AHT cbc0b4ed (Apache-2.0)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tifffile
import yaml

from .scientific.reconstruction.quantitative import (
    CupyQuantitativeRuntime,
    CupyQuantitativeTileSolver,
    NumpyQuantitativeTileSolver,
    QuantitativeTileConfig,
)
from .scientific.reconstruction.tiling import (
    TiledVolumeReconstructor,
    TilingConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from .scientific.reconstruction.tiling import TileSolver


def migrate_config(source: Path) -> dict[str, Any]:
    """Translate the legacy DPM DPCT YAML without changing the source file."""
    legacy = yaml.safe_load(source.read_text(encoding="utf-8"))
    optics = legacy["optical_setup"]
    model = legacy["solvers"]["dpct"]
    background = model.get("background_removal_params", {})
    params = model.get("properties", {})
    regularization = params.get("reg_params", {})
    solver_params = params.get("solver_params", {})
    boundary = solver_params.get("boundary_constraint", {})
    pixel_size = optics.get("pixel_size_um")
    if pixel_size is None:
        pixel_size = optics["pixel_size_detector_um"] / (
            optics["mag"] * optics.get("mag_4f", 1.0)
        )
    normalization = "none"
    if model.get("background_removal", False):
        normalization = (
            "reference_over_sample"
            if background.get("inv_bg_divide", False)
            else "sample_over_reference"
        )
    return {
        "schema_version": 1,
        "optics": {
            "wavelength_um": model.get("wavelength_um", optics["wavelength_um"]),
            "numerical_aperture": optics["numerical_aperture"],
            "refractive_index_medium": optics["refractive_index_medium"],
            "pixel_size_um": pixel_size,
        },
        "input": {
            "normalization": normalization,
            "allow_reference_broadcast": True,
            "z_spacing_um_override": optics.get("pixel_size_z_um"),
            "shots_order": background.get("shots_order", [0, 1, 2, 3]),
            "reverse_z_groups": model.get("loader_params", {}).get(
                "reverse_z_groups", False
            ),
        },
        "dpct": {
            "source_azimuth_deg": model["src_azimuth_agl_deg"],
            "regularization_real": regularization.get("reg_real", 5e-5),
            "regularization_imaginary": regularization.get("reg_imag", 5e-5),
            "axial_window": "hamming",
            "dpct_method": solver_params.get("method", "tikhonov"),
            # The legacy DPCT sets tau_u from YAML but its 3D ADMM reads tau.
            "dpct_tv_weight": 5e-5,
            "dpct_admm_rho": regularization.get("rho", 5e-5),
            "dpct_tv_iterations": solver_params.get("tv_iter", 20),
            "dpct_real_constraint": boundary.get("real", "negative"),
            "dpct_imag_constraint": boundary.get("imag", "negative"),
            "tiling": model.get("tiling", {}),
        },
    }


@contextmanager
def _mapped_tiff(
    path: Path, scratch: Path, expected_axes: set[str]
) -> Iterator[tuple[npt.NDArray[Any], str, dict[str, Any]]]:
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        axes = series.axes
        if axes not in expected_axes:
            raise ValueError(
                f"{path.name} declares {axes}; "
                f"expected {', '.join(sorted(expected_axes))}"
            )
        frames: npt.NDArray[Any]
        try:
            frames = tifffile.memmap(path, mode="r")
        except ValueError:
            frames = series.asarray(out=scratch)
        try:
            metadata = tiff.imagej_metadata or {}
            labels = metadata.get("Labels")
            if tiff.ome_metadata:
                try:
                    image = ET.fromstring(tiff.ome_metadata).find("{*}Image")
                except ET.ParseError as error:
                    raise ValueError(f"invalid OME metadata in {path.name}") from error
                if image is not None:
                    labels = [
                        node.get("Name")
                        for node in image.findall("{*}Pixels/{*}Channel")
                    ]
            shot_count = series.shape[axes.index("C")]
            channel_labels = (
                tuple(labels)
                if labels and len(labels) == shot_count and all(labels)
                else None
            )
            yield (
                frames,
                axes,
                {
                    "spacing": metadata.get("spacing"),
                    "unit": metadata.get("unit"),
                    "channel_labels": channel_labels,
                },
            )
        finally:
            _close_mapping(frames)


def _close_mapping(array: npt.NDArray[Any]) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _validate_channel_labels(sample: dict[str, Any], reference: dict[str, Any]) -> None:
    sample_labels, reference_labels = (
        sample["channel_labels"],
        reference["channel_labels"],
    )
    if (
        sample_labels is not None
        and reference_labels is not None
        and sample_labels != reference_labels
    ):
        raise ValueError(
            "sample and reference TIFF illumination channel labels disagree"
        )


def _normalize_planes(
    target: npt.NDArray[Any],
    sample: npt.NDArray[Any],
    reference: npt.NDArray[Any] | None,
    order: list[int],
    reverse: bool,
    mode: str,
    zero_policy: str,
) -> int:
    replaced = 0
    for z in range(len(sample)):
        source_z = len(sample) - z - 1 if reverse else z
        values = sample[source_z, order].astype(np.float32)
        if reference is None:
            target[z] = values
            continue
        background = (reference if reference.ndim == 3 else reference[source_z])[order]
        invalid = (values == 0) | (background == 0)
        count = int(np.count_nonzero(invalid))
        if count and zero_policy == "error":
            raise ValueError(
                f"background normalization encountered {count} zero pixels "
                f"at input Z {source_z}; "
                "choose a valid ROI or explicitly set --zero-policy neutral"
            )
        numerator, denominator = (
            (values, background)
            if mode == "sample_over_reference"
            else (background, values)
        )
        target[z].fill(1)
        np.divide(numerator, denominator, out=target[z], where=~invalid)
        replaced += count
    return replaced


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reconstruct(
    sample_path: Path,
    reference_path: Path | None,
    config_path: Path,
    output: Path,
    roi: str | None = None,
    zero_policy: str = "error",
    *,
    backend: str = "numpy",
    gpu_id: int = 0,
    method: str | None = None,
    tile_size: str | None = None,
    output_dtype: str = "uint16",
) -> None:
    """Reconstruct DPCT offline with disk-backed normalization and XY tiling."""
    paths = {"sample": sample_path, "reference": reference_path, "config": config_path}
    inputs = {
        key: {"path": str(path.resolve()), "sha256": _digest(path)}
        for key, path in paths.items()
        if path is not None
    }
    requested = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = requested if "schema_version" in requested else migrate_config(config_path)
    if config["schema_version"] != 1:
        raise ValueError("expected reconstruction schema version 1")
    options = config.get("input", {})
    mode = options.get("normalization", "none")
    if mode not in {"none", "sample_over_reference", "reference_over_sample"}:
        raise ValueError("unsupported background normalization")
    if (reference_path is None) != (mode == "none"):
        raise ValueError(
            "supply a reference exactly when background normalization is enabled"
        )
    order = options.get("shots_order", [0, 1, 2, 3])
    if sorted(order) != [0, 1, 2, 3] or any(type(index) is not int for index in order):
        raise ValueError("shots_order must be a permutation of [0, 1, 2, 3]")
    sidecar = output.with_suffix(".provenance.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("output TIFF or provenance already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = config["dpct"]
    if method is not None:
        model["dpct_method"] = method
    optics = config["optics"]
    with (
        tempfile.TemporaryDirectory(prefix=".dpct-", dir=output.parent) as folder,
        ExitStack() as stack,
    ):
        scratch = Path(folder)
        sample, _, spacing_tags = stack.enter_context(
            _mapped_tiff(sample_path, scratch / "sample.raw", {"ZCYX"})
        )
        source_shape = sample.shape
        if source_shape[1] != 4 or any(size == 0 for size in source_shape):
            raise ValueError("a DPCT sample must have four illumination channels")
        reference = None
        reference_repeats = None
        if reference_path is not None:
            reference, axes, reference_tags = stack.enter_context(
                _mapped_tiff(
                    reference_path, scratch / "reference.raw", {"CYX", "ZCYX", "TCYX"}
                )
            )
            _validate_channel_labels(spacing_tags, reference_tags)
            if axes != "ZCYX" and not options.get("allow_reference_broadcast", False):
                raise ValueError(
                    "a broadcast reference requires allow_reference_broadcast"
                )
            if axes == "TCYX" and reference.shape[1:] == sample.shape[1:]:
                reference_repeats = reference.shape[0]
            elif axes == "TCYX" or reference.shape not in (
                sample.shape,
                sample.shape[1:],
            ):
                raise ValueError(
                    "reference shape must match the sample or one CYX plane"
                )
        if roi is not None:
            y, x, height, width = map(int, roi.split(","))
            if (
                min(y, x) < 0
                or min(height, width) <= 0
                or y + height > source_shape[-2]
                or x + width > source_shape[-1]
            ):
                raise ValueError("ROI lies outside the TIFF field")
            sample = sample[..., y : y + height, x : x + width]
            if reference is not None:
                reference = reference[..., y : y + height, x : x + width]
        if reference_repeats is not None:
            assert reference is not None
            reference = np.median(reference, axis=0).astype(np.float32)
        spacing = options.get("z_spacing_um_override")
        if spacing is None and spacing_tags["unit"] in {
            "um",
            "µm",
            "micron",
            "micrometer",
        }:
            spacing = spacing_tags["spacing"]
        if spacing is None or not math.isfinite(float(spacing)) or float(spacing) <= 0:
            raise ValueError("supply positive Z spacing in micrometres")
        resolved = QuantitativeTileConfig(
            model="dpct",
            pixel_size_z_um=float(spacing),
            source_azimuth_deg=tuple(model["source_azimuth_deg"]),
            regularization_real=float(model.get("regularization_real", 5e-5)),
            regularization_imaginary=float(model.get("regularization_imaginary", 5e-5)),
            axial_window=model.get("axial_window", "hamming"),
            dpct_method=model.get("dpct_method", "tikhonov"),
            dpct_tv_weight=float(model.get("dpct_tv_weight", 5e-5)),
            dpct_admm_rho=float(model.get("dpct_admm_rho", 5e-5)),
            dpct_tv_iterations=model.get("dpct_tv_iterations", 20),
            dpct_real_constraint=model.get("dpct_real_constraint", "negative"),
            dpct_imag_constraint=model.get("dpct_imag_constraint", "negative"),
            **optics,
        )
        tiling = dict(model.get("tiling", {}))
        for option, target in (
            ("shape_yx", "explicit_shape_yx"),
            ("overlap_yx", "explicit_overlap_yx"),
        ):
            if option in tiling:
                tiling[target] = tiling.pop(option)
        if tile_size is not None:
            height, width = map(int, tile_size.split(","))
            tiling.update(mode="auto", explicit_shape_yx=(height, width))
        normalized = np.memmap(
            scratch / "normalized.raw", mode="w+", dtype=np.float32, shape=sample.shape
        )
        stack.callback(_close_mapping, normalized)
        replaced = _normalize_planes(
            normalized,
            sample,
            reference,
            order,
            options.get("reverse_z_groups", False),
            mode,
            zero_policy,
        )
        solver: TileSolver
        if backend == "cupy":
            runtime = CupyQuantitativeRuntime(
                gpu_id, cache_directory=str((scratch / "cupy-cache").absolute())
            )
            stack.callback(runtime.close)
            solver = CupyQuantitativeTileSolver(resolved, runtime)
        elif backend == "numpy":
            solver = NumpyQuantitativeTileSolver(resolved)
        else:
            raise ValueError(f"unsupported reconstruction backend {backend!r}")
        reconstructor = TiledVolumeReconstructor(solver, TilingConfig(**tiling))
        stack.callback(reconstructor.close)
        result = reconstructor.run(normalized)
        encoded = result.volume_zyx
        encoding = "absolute RI"
        if output_dtype == "uint16":
            encoded = np.memmap(
                scratch / "encoded.raw", mode="w+", dtype=np.uint16, shape=encoded.shape
            )
            stack.callback(_close_mapping, encoded)
            for index, plane in enumerate(result.volume_zyx):
                encoded[index] = (plane * 1e4).astype(np.uint16)
            encoding = "RI * 1e4"
        elif output_dtype != "float32":
            raise ValueError("DPCT output dtype must be uint16 or float32")
        temporary_tiff = scratch / output.name
        tifffile.imwrite(
            temporary_tiff,
            encoded,
            imagej=True,
            resolution=(1 / resolved.pixel_size_um, 1 / resolved.pixel_size_um),
            metadata={"axes": "ZYX", "spacing": float(spacing), "unit": "um"},
        )
        tile_plan = dict(result.plan.as_provenance())
        tile_plan["placements"] = [dict(item) for item in tile_plan["placements"]]
        temporary_sidecar = scratch / sidecar.name
        temporary_sidecar.write_text(
            json.dumps(
                {
                    "solver": "dpct",
                    "backend": backend,
                    "gpu_id": gpu_id if backend == "cupy" else None,
                    "reference_median_repeats": reference_repeats,
                    "inputs": inputs,
                    "sample_axes": "ZCYX",
                    "sample_shape": source_shape,
                    "roi_yxhw": roi,
                    "zero_policy": zero_policy,
                    "neutral_replacements": replaced,
                    "input_spacing_metadata": spacing_tags,
                    "config": config,
                    "optics": asdict(resolved),
                    "tiling": tile_plan,
                    "output": {
                        "axes": "ZYX",
                        "shape": encoded.shape,
                        "dtype": output_dtype,
                        "encoding": encoding,
                        "sha256": _digest(temporary_tiff),
                    },
                },
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.link(temporary_tiff, output)
        try:
            os.link(temporary_sidecar, sidecar)
        except OSError:
            output.unlink()
            raise
