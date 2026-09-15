"""Prepare cached 5 µm label-grid RI samples from the four biofilm TIFF groups.

This is the one-time CPU analysis step for presentation figures. It writes a
compact, provenance-carrying NPZ cache; use ``plot_ri_grid_label_violins.py``
with ``--group-cache`` to restyle or re-render the plot without reprocessing
the source TIFF stacks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from redsun_aht.ri_analysis.pipeline import analyze_volume, load_config, load_ri_tiff

if TYPE_CHECKING:
    from redsun_aht.ri_analysis.config import RiAnalysisConfig


GROUP_ORDER = ("cuce", "cuce_pmb", "cuonly", "cuonly_pmb")
VOXEL_SIZE_ZYX_UM = (1.0, 0.112, 0.112)
CACHE_SCHEMA_VERSION = 2


def prepare_group_grid_samples(
    group_root: Path,
    output_path: Path,
    *,
    config: RiAnalysisConfig,
) -> Path:
    """Analyze ordered TIFF groups once and atomically write their RI means."""
    group_root = group_root.resolve()
    if not group_root.is_dir():
        raise ValueError(f"group root is not a directory: {group_root}")
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"group grid-sample cache already exists: {output_path}")
    group_indices: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    label_group_indices: list[np.ndarray] = []
    label_source_indices: list[np.ndarray] = []
    label_volumes: list[np.ndarray] = []
    label_voxel_counts: list[np.ndarray] = []
    group_metadata: list[dict[str, object]] = []
    source_index = 0
    for group_index, directory_name in enumerate(GROUP_ORDER):
        directory = group_root / directory_name
        label, color = _primary_label(directory)
        sources: list[dict[str, object]] = []
        group_point_count = 0
        group_label_count = 0
        for tiff_path in _group_tiffs(directory):
            volume = load_ri_tiff(
                tiff_path,
                voxel_size_zyx_um=VOXEL_SIZE_ZYX_UM,
            )
            artifacts = analyze_volume(volume, config)
            ri_mean = np.asarray(
                artifacts.label_grid_table.columns["ri_mean"], dtype=np.float32
            )
            label_volume_um3 = np.asarray(
                artifacts.label_table.columns["roi_volume_um3"], dtype=np.float64
            )
            label_voxel_count = np.asarray(
                artifacts.label_table.columns["voxel_count"], dtype=np.int64
            )
            sources.append(
                {
                    "path": str(tiff_path.resolve()),
                    "sha256": volume.source_checksum,
                    "label_cube_count": int(ri_mean.size),
                    "label_count": int(label_volume_um3.size),
                }
            )
            if ri_mean.size:
                values.append(ri_mean)
                group_indices.append(np.full(ri_mean.size, group_index, dtype=np.int8))
                source_indices.append(
                    np.full(ri_mean.size, source_index, dtype=np.int16)
                )
                group_point_count += int(ri_mean.size)
            if label_volume_um3.size:
                label_volumes.append(label_volume_um3)
                label_voxel_counts.append(label_voxel_count)
                label_group_indices.append(
                    np.full(label_volume_um3.size, group_index, dtype=np.int8)
                )
                label_source_indices.append(
                    np.full(label_volume_um3.size, source_index, dtype=np.int16)
                )
                group_label_count += int(label_volume_um3.size)
            source_index += 1
        if group_point_count == 0:
            raise ValueError(f"group has no retained label-grid samples: {directory}")
        if group_label_count == 0:
            raise ValueError(f"group has no retained morphology labels: {directory}")
        group_metadata.append(
            {
                "directory_name": directory_name,
                "label": label,
                "color_rgb": list(color),
                "source_files": sources,
            }
        )
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "group_root": str(group_root),
        "groups": group_metadata,
        "configuration": config.to_mapping(),
        "voxel_size_zyx_um": list(VOXEL_SIZE_ZYX_UM),
        "voxel_size_source": "explicit_override",
        "value_definition": "mean RI of one retained label in one grid cell",
        "morphology_value_definition": "physical volume of one retained label",
    }
    _write_cache(
        output_path,
        metadata,
        group_index=np.concatenate(group_indices),
        source_index=np.concatenate(source_indices),
        ri_mean=np.concatenate(values),
        label_group_index=np.concatenate(label_group_indices),
        label_source_index=np.concatenate(label_source_indices),
        label_volume_um3=np.concatenate(label_volumes),
        label_voxel_count=np.concatenate(label_voxel_counts),
    )
    return output_path


def _primary_label(directory: Path) -> tuple[str, tuple[int, int, int]]:
    """Read the first display name and RGB color supplied for a group."""
    label_path = directory / "label.txt"
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(
            f"group label file is missing or unreadable: {label_path}"
        ) from error
    fields: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() not in fields:
            fields[key.strip().lower()] = value.strip()
    name = fields.get("name")
    color_text = fields.get("rgb")
    if not name or not color_text:
        raise ValueError(f"group label file needs primary name and rgb: {label_path}")
    try:
        color = tuple(int(item.strip()) for item in color_text.split(","))
    except ValueError as error:
        raise ValueError(f"group label RGB is invalid: {label_path}") from error
    if len(color) != 3 or any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(
            f"group label RGB must contain three values from 0 to 255: {label_path}"
        )
    return name, color


def _group_tiffs(directory: Path) -> list[Path]:
    """Return all direct donor TIFF inputs in a group in stable name order."""
    if not directory.is_dir():
        raise ValueError(f"required group directory is missing: {directory}")
    tiffs = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not tiffs:
        raise ValueError(f"group has no TIFF inputs: {directory}")
    return tiffs


def _write_cache(
    output_path: Path,
    metadata: dict[str, object],
    *,
    group_index: np.ndarray,
    source_index: np.ndarray,
    ri_mean: np.ndarray,
    label_group_index: np.ndarray,
    label_source_index: np.ndarray,
    label_volume_um3: np.ndarray,
    label_voxel_count: np.ndarray,
) -> None:
    """Write the complete cache to a sibling temporary file before exposure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                group_index=group_index,
                source_index=source_index,
                ri_mean=ri_mean,
                label_group_index=label_group_index,
                label_source_index=label_source_index,
                label_volume_um3=label_volume_um3,
                label_voxel_count=label_voxel_count,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the one-time group TIFF-to-grid-sample cache preparation."""
    args = _parser().parse_args(argv)
    try:
        output = prepare_group_grid_samples(
            args.group_root,
            args.output,
            config=load_config(args.config),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR RI group-grid preparation: {error}", file=sys.stderr)
        return 2
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
