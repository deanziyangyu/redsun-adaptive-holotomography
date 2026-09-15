"""Create standalone RI-distribution violin plots from verified analysis bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr

from redsun_aht.ri_analysis.publication import BUNDLE_NAME, verify_analysis_bundle


def plot_violins(
    input_path: Path,
    output_path: Path,
    *,
    group_by: Literal["labels", "grid"],
    max_groups: int,
    title: str | None = None,
) -> Path:
    """Render the largest label or grid distributions into one PNG or PDF."""
    if max_groups < 1:
        raise ValueError("max_groups must be positive")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "install the AHT 'analysis' extra to create violin plots"
        ) from error
    manifest = verify_analysis_bundle(input_path)
    root = input_path.resolve()
    if root.name == BUNDLE_NAME:
        root = root.parent
    bundle = zarr.open_group(root / BUNDLE_NAME, mode="r")
    arrays = _distributions(bundle, manifest, group_by, max_groups)
    if not arrays:
        raise ValueError("the selected analysis result has no non-empty distributions")
    labels = [item[0] for item in arrays]
    values = [item[1] for item in arrays]
    figure_width = max(8.0, 0.5 * len(values))
    figure, axis = plt.subplots(figsize=(figure_width, 5.0), layout="constrained")
    axis.violinplot(values, showmeans=True, showmedians=True, showextrema=False)
    axis.set_xticks(range(1, len(labels) + 1), labels, rotation=75, ha="right")
    axis.set_ylabel("Refractive index")
    axis.set_title(title or f"RI distributions by {group_by}")
    axis.grid(axis="y", alpha=0.25)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _distributions(
    bundle: Any,
    manifest: dict[str, Any],
    group_by: Literal["labels", "grid"],
    max_groups: int,
) -> list[tuple[str, np.ndarray]]:
    ri = np.asarray(bundle["images/ri/0"][:])
    roi = np.asarray(bundle["masks/selected_roi/0"][:], dtype=bool)
    if group_by == "labels":
        labels = np.asarray(bundle["masks/labels/0"][:])
        table = bundle["tables/labels"]
        ids = np.asarray(table["label_id"][:], dtype=np.int64)
        counts = np.asarray(table["voxel_count"][:], dtype=np.int64)
        chosen = ids[np.argsort(counts)[::-1][:max_groups]]
        return [
            (f"label {label_id}", ri[labels == label_id])
            for label_id in chosen
            if np.any(labels == label_id)
        ]
    table = bundle["tables/grid"]
    counts = np.asarray(table["voxel_count"][:], dtype=np.int64)
    chosen = np.argsort(counts)[::-1][:max_groups]
    spacing = _spacing(manifest)
    distributions: list[tuple[str, np.ndarray]] = []
    for row in chosen:
        starts = tuple(
            round(float(table[f"{axis}_start_um"][row]) / spacing[index])
            for index, axis in enumerate("zyx")
        )
        stops = tuple(
            round(float(table[f"{axis}_stop_um"][row]) / spacing[index])
            for index, axis in enumerate("zyx")
        )
        slices = tuple(
            slice(start, stop) for start, stop in zip(starts, stops, strict=True)
        )
        values = ri[slices][roi[slices]]
        if values.size:
            grid_id = int(table["grid_id"][row])
            distributions.append((f"grid {grid_id}", values))
    return distributions


def _spacing(manifest: dict[str, Any]) -> tuple[float, float, float]:
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("RI analysis manifest source is invalid")
    values = source.get("voxel_size_zyx_um")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("RI analysis manifest voxel size is invalid")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-by", choices=("labels", "grid"), default="labels")
    parser.add_argument("--max-groups", type=int, default=40)
    parser.add_argument("--title")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the standalone plotting command."""
    args = _parser().parse_args(argv)
    try:
        output = plot_violins(
            args.input,
            args.output,
            group_by=args.group_by,
            max_groups=args.max_groups,
            title=args.title,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR RI violin plot: {error}", file=sys.stderr)
        return 2
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
