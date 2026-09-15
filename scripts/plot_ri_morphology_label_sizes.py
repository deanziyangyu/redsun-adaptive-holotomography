"""Plot retained morphology-label volumes from a prepared biofilm cache.

The plot reads the schema-2 cache made by ``prepare_ri_group_grid_samples.py``
and never opens or reprocesses source TIFF stacks. Each point is one connected
component's physical volume; violin and quartile summaries use all labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np


@dataclass(frozen=True, slots=True)
class MorphologyGroup:
    """One condition's primary display metadata and retained label volumes."""

    label: str
    color_rgb: tuple[int, int, int]
    volumes_um3: np.ndarray


def plot_morphology_label_sizes(
    cache_path: Path,
    output_path: Path,
    *,
    max_display_points: int = 1_500,
    title: str | None = None,
) -> Path:
    """Render log-volume morphology violins for the four cached conditions."""
    if max_display_points < 1:
        raise ValueError("max_display_points must be positive")
    groups = _load_morphology_groups(cache_path)
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import FuncFormatter
    except ImportError as error:
        raise RuntimeError(
            "install the AHT 'analysis' extra to create morphology plots"
        ) from error
    log_volumes = [np.log10(group.volumes_um3) for group in groups]
    figure, axis = plt.subplots(figsize=(11.5, 6.5), layout="constrained")
    violins = axis.violinplot(
        log_volumes,
        positions=range(1, len(groups) + 1),
        widths=0.65,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    bodies = cast("list[Any]", violins["bodies"])
    for position, (group, body, values) in enumerate(
        zip(groups, bodies, log_volumes, strict=True), start=1
    ):
        color = tuple(channel / 255.0 for channel in group.color_rgb)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
        body.set_zorder(4)
        display_values = _display_points(values, max_display_points)
        axis.scatter(
            position
            + _density_jitter(
                display_values,
                values,
                width=0.65,
                seed=position,
            ),
            display_values,
            color=color,
            edgecolors="white",
            linewidths=0.15,
            s=10,
            alpha=0.78,
            rasterized=True,
            zorder=3,
        )
        q1, q3 = np.quantile(values, (0.25, 0.75))
        axis.add_patch(
            Rectangle(
                (position - 0.11, q1),
                0.22,
                q3 - q1,
                fill=False,
                edgecolor="#242424",
                linewidth=1.15,
                zorder=6,
            )
        )
    median = violins.get("cmedians")
    if median is not None:
        median.set_color("#242424")
        median.set_linewidth(1.25)
        median.set_zorder(5)
    axis.set_xticks(
        range(1, len(groups) + 1),
        [f"{group.label}\nn = {len(group.volumes_um3):,} labels" for group in groups],
    )
    axis.set_xlabel("Treatment group")
    axis.set_ylabel("Retained label volume (µm³; log10 scale)")
    axis.yaxis.set_major_formatter(FuncFormatter(_format_log_volume_tick))
    axis.set_title(title or "Biofilm morphology: retained label volumes")
    axis.grid(axis="y", alpha=0.25, zorder=0)
    axis.margins(x=0.05)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _load_morphology_groups(cache_path: Path) -> list[MorphologyGroup]:
    """Read primary labels, colors, and retained volumes from a schema-2 cache."""
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            schema_version = int(cache["schema_version"])
            metadata = json.loads(str(cache["metadata_json"].item()))
            group_indices = np.asarray(cache["label_group_index"], dtype=np.int8)
            volumes = np.asarray(cache["label_volume_um3"], dtype=np.float64)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"morphology cache is invalid: {cache_path}") from error
    if schema_version != 2 or group_indices.shape != volumes.shape or volumes.ndim != 1:
        raise ValueError(f"morphology cache requires schema 2: {cache_path}")
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError(f"morphology cache volumes must be positive: {cache_path}")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("groups"), list):
        raise ValueError(f"morphology cache metadata is invalid: {cache_path}")
    groups: list[MorphologyGroup] = []
    for index, group in enumerate(metadata["groups"]):
        if not isinstance(group, dict):
            raise ValueError(
                f"morphology cache group metadata is invalid: {cache_path}"
            )
        label = group.get("label")
        color = group.get("color_rgb")
        if not isinstance(label, str) or not isinstance(color, list) or len(color) != 3:
            raise ValueError(
                f"morphology cache group metadata is invalid: {cache_path}"
            )
        group_volumes = volumes[group_indices == index]
        if group_volumes.size == 0:
            raise ValueError(f"morphology cache has no labels for {label}")
        groups.append(
            MorphologyGroup(
                label=label,
                color_rgb=cast(
                    "tuple[int, int, int]", tuple(int(item) for item in color)
                ),
                volumes_um3=group_volumes,
            )
        )
    return groups


def _display_points(values: np.ndarray, maximum: int) -> np.ndarray:
    """Keep evenly spaced order statistics for a compact but representative swarm."""
    if values.size <= maximum:
        return values
    ordered = np.sort(values)
    indices = np.rint(np.linspace(0, values.size - 1, maximum)).astype(np.int64)
    return ordered[indices]


def _density_jitter(
    display_values: np.ndarray,
    all_values: np.ndarray,
    *,
    width: float,
    seed: int,
) -> np.ndarray:
    """Shape jitter by smoothed log-volume density, like a compact beeswarm."""
    if display_values.size <= 1 or np.ptp(all_values) == 0:
        return np.zeros(display_values.size)
    bin_count = min(256, max(48, int(np.sqrt(all_values.size))))
    counts, edges = np.histogram(all_values, bins=bin_count)
    density = np.convolve(counts, np.asarray((1, 4, 6, 4, 1)) / 16, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    local_density = np.interp(display_values, centers, density, left=0, right=0)
    half_width = width * 0.43 * (local_density / density.max())
    return np.asarray(
        np.random.default_rng(seed).uniform(-half_width, half_width),
        dtype=np.float64,
    )


def _format_log_volume_tick(value: float, _: int) -> str:
    """Format the log10-coordinate y-axis as a physical cubic-micrometre value."""
    physical = 10**value
    return f"{physical:.0f}" if physical >= 1 else f"{physical:.2g}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-display-points", type=int, default=1_500)
    parser.add_argument("--title")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the standalone cached morphology-label size plotter."""
    args = _parser().parse_args(argv)
    try:
        output = plot_morphology_label_sizes(
            args.group_cache,
            args.output,
            max_display_points=args.max_display_points,
            title=args.title,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR morphology label-size plot: {error}", file=sys.stderr)
        return 2
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
