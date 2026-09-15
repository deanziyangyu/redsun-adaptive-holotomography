"""Plot label-wise RI distributions from physical grid samples in RI bundles.

Each dot is the mean RI of one retained label within one grid cell.  Each
violin therefore describes spatial subvolume measurements rather than every
voxel in the component.  Multiple verified analysis bundles are displayed as
clusters: one x-axis tick per sample, with its retained labels as violins.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import zarr

from redsun_aht.ri_analysis.publication import BUNDLE_NAME, verify_analysis_bundle

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class GroupDistribution:
    """One treatment group's label-derived presentation metadata and samples."""

    directory_name: str
    label: str
    color_rgb: tuple[int, int, int]
    values: np.ndarray
    source_count: int


@dataclass(frozen=True, slots=True)
class PairwiseTTest:
    """One two-sided Welch t-test performed on TIFF-level RI means."""

    group_a: GroupDistribution
    group_b: GroupDistribution
    t_statistic: float
    degrees_of_freedom: float
    p_value: float
    p_value_bonferroni: float
    mean_ri_a: float
    mean_ri_b: float


T_TEST_GROUP_PAIRS = ((0, 1), (0, 2), (1, 3), (2, 3))


def plot_group_cache_violins(
    cache_path: Path,
    output_path: Path,
    *,
    max_display_points: int = 1_500,
    statistics_path: Path | None = None,
    title: str | None = None,
) -> Path:
    """Plot cached treatment-group cube means without reopening TIFF stacks.

    The violin and quartile summaries use every cached cube. Dots are a
    deterministic, stratified display subset spread by local density so very
    large groups remain legible without changing their statistical summaries.
    """
    if max_display_points < 1:
        raise ValueError("max_display_points must be positive")
    output_path = output_path.resolve()
    resolved_statistics_path = (
        statistics_path.resolve()
        if statistics_path is not None
        else output_path.with_name(f"{output_path.stem}.welch-t-tests.csv")
    )
    if resolved_statistics_path.exists():
        raise FileExistsError(
            f"Welch t-test output already exists: {resolved_statistics_path}"
        )
    groups, cell_size_zyx_um, tiff_means = _load_group_cache(cache_path)
    results = _pairwise_welch_t_tests(groups, tiff_means)
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as error:
        raise RuntimeError(
            "install the AHT 'analysis' extra to create violin plots"
        ) from error
    values = [group.values for group in groups]
    figure, axis = plt.subplots(figsize=(11.5, 7.0), layout="constrained")
    violins = axis.violinplot(
        values,
        positions=range(1, len(groups) + 1),
        widths=0.65,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    bodies = cast("list[Any]", violins["bodies"])
    for position, (group, body) in enumerate(zip(groups, bodies, strict=True), start=1):
        color = tuple(channel / 255.0 for channel in group.color_rgb)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
        body.set_zorder(4)
        display_values = _display_points(group.values, max_display_points)
        axis.scatter(
            position
            + _density_jitter(
                display_values,
                group.values,
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
        q1, q3 = np.quantile(group.values, (0.25, 0.75))
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
    grid_name = " x ".join(f"{value:g}" for value in cell_size_zyx_um)
    cell_volume_um3 = float(np.prod(cell_size_zyx_um))
    axis.set_xticks(
        range(1, len(groups) + 1),
        [
            f"{group.label}\n"
            f"n = {len(group.values):,} subvolumes\n"
            f"≤{cell_volume_um3:g} µm³ per subvolume\n"
            f"mean ± SD = {np.mean(group.values):.4f} ± "
            f"{np.std(group.values, ddof=1):.4f}"
            for group in groups
        ],
    )
    axis.set_xlabel("Treatment group")
    axis.set_ylabel("Mean Refractive Index per label-grid subvolume (a.u.)")
    axis.set_title(f"Biofilm Refractive Index by {grid_name} µm grid subvolume")
    axis.grid(axis="y", alpha=0.25, zorder=0)
    axis.margins(x=0.05)
    _annotate_planned_t_tests(axis, results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    _write_t_tests(
        resolved_statistics_path,
        results,
    )
    return output_path


def _load_group_cache(
    cache_path: Path,
) -> tuple[list[GroupDistribution], tuple[float, float, float], list[np.ndarray]]:
    """Load a prepared label-cube cache and validate its plotting fields."""
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            schema_version = int(cache["schema_version"])
            metadata = json.loads(str(cache["metadata_json"].item()))
            group_indices = np.asarray(cache["group_index"], dtype=np.int8)
            source_indices = np.asarray(cache["source_index"], dtype=np.int16)
            values = np.asarray(cache["ri_mean"], dtype=np.float32)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"group grid-sample cache is invalid: {cache_path}") from error
    if (
        schema_version not in {1, 2}
        or values.ndim != 1
        or group_indices.shape != values.shape
        or source_indices.shape != values.shape
    ):
        raise ValueError(f"group grid-sample cache schema is unsupported: {cache_path}")
    if not isinstance(metadata, dict):
        raise ValueError(f"group grid-sample cache metadata is invalid: {cache_path}")
    groups_data = metadata.get("groups")
    configuration = metadata.get("configuration")
    if not isinstance(groups_data, list) or not isinstance(configuration, dict):
        raise ValueError(
            f"group grid-sample cache metadata is incomplete: {cache_path}"
        )
    grid = configuration.get("grid")
    cell_size = grid.get("cell_size_zyx_um") if isinstance(grid, dict) else None
    if not isinstance(cell_size, list) or len(cell_size) != 3:
        raise ValueError(f"group grid-sample cache grid size is invalid: {cache_path}")
    distributions: list[GroupDistribution] = []
    tiff_means: list[np.ndarray] = []
    for index, group in enumerate(groups_data):
        if not isinstance(group, dict):
            raise ValueError(
                f"group grid-sample cache group metadata is invalid: {cache_path}"
            )
        name = group.get("label")
        directory_name = group.get("directory_name")
        color = group.get("color_rgb")
        sources = group.get("source_files")
        if (
            not isinstance(name, str)
            or not isinstance(directory_name, str)
            or not isinstance(color, list)
            or len(color) != 3
            or not isinstance(sources, list)
        ):
            raise ValueError(
                f"group grid-sample cache group metadata is invalid: {cache_path}"
            )
        group_values = values[group_indices == index]
        group_source_indices = source_indices[group_indices == index]
        if group_values.size == 0:
            raise ValueError(f"group grid-sample cache contains an empty group: {name}")
        source_means = np.asarray(
            [
                group_values[group_source_indices == source_index].mean()
                for source_index in np.unique(group_source_indices)
            ],
            dtype=np.float64,
        )
        distributions.append(
            GroupDistribution(
                directory_name=directory_name,
                label=name,
                color_rgb=cast(
                    "tuple[int, int, int]", tuple(int(value) for value in color)
                ),
                values=group_values,
                source_count=len(sources),
            )
        )
        tiff_means.append(source_means)
    return (
        distributions,
        tuple(float(value) for value in cell_size),  # type: ignore[return-value]
        tiff_means,
    )


def _pairwise_welch_t_tests(
    groups: list[GroupDistribution], tiff_means: list[np.ndarray]
) -> list[PairwiseTTest]:
    """Run the four planned TIFF-level Welch comparisons with correction."""
    if len(groups) != 4 or len(tiff_means) != 4:
        raise ValueError("planned Welch t-tests require exactly four groups")
    if any(values.size < 2 for values in tiff_means):
        raise ValueError("each planned Welch t-test group needs at least two TIFFs")
    try:
        from scipy.stats import ttest_ind
    except ImportError as error:  # pragma: no cover - SciPy is a core dependency
        raise RuntimeError("SciPy is required for Welch t-tests") from error
    uncorrected: list[
        tuple[GroupDistribution, GroupDistribution, float, float, float, float, float]
    ] = []
    for group_a_index, group_b_index in T_TEST_GROUP_PAIRS:
        group_a = groups[group_a_index]
        group_b = groups[group_b_index]
        values_a = tiff_means[group_a_index]
        values_b = tiff_means[group_b_index]
        result = ttest_ind(values_a, values_b, equal_var=False)
        variance_a = values_a.var(ddof=1)
        variance_b = values_b.var(ddof=1)
        numerator = (variance_a / len(values_a) + variance_b / len(values_b)) ** 2
        denominator = (variance_a / len(values_a)) ** 2 / (len(values_a) - 1) + (
            variance_b / len(values_b)
        ) ** 2 / (len(values_b) - 1)
        uncorrected.append(
            (
                group_a,
                group_b,
                float(result.statistic),
                numerator / denominator,
                float(result.pvalue),
                float(values_a.mean()),
                float(values_b.mean()),
            )
        )
    comparison_count = len(uncorrected)
    return [
        PairwiseTTest(
            group_a=group_a,
            group_b=group_b,
            t_statistic=t_statistic,
            degrees_of_freedom=degrees_of_freedom,
            p_value=p_value,
            p_value_bonferroni=min(1.0, p_value * comparison_count),
            mean_ri_a=mean_ri_a,
            mean_ri_b=mean_ri_b,
        )
        for (
            group_a,
            group_b,
            t_statistic,
            degrees_of_freedom,
            p_value,
            mean_ri_a,
            mean_ri_b,
        ) in uncorrected
    ]


def _write_t_tests(path: Path, results: list[PairwiseTTest]) -> Path:
    """Write reproducible source-level pairwise Welch test results as CSV."""
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Welch t-test output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "group_a",
                "group_b",
                "n_tiffs_a",
                "n_tiffs_b",
                "mean_ri_a",
                "mean_ri_b",
                "t_statistic",
                "welch_degrees_of_freedom",
                "p_value_two_sided",
                "p_value_bonferroni_4_planned_pairs",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "group_a": result.group_a.label,
                    "group_b": result.group_b.label,
                    "n_tiffs_a": result.group_a.source_count,
                    "n_tiffs_b": result.group_b.source_count,
                    "mean_ri_a": result.mean_ri_a,
                    "mean_ri_b": result.mean_ri_b,
                    "t_statistic": result.t_statistic,
                    "welch_degrees_of_freedom": result.degrees_of_freedom,
                    "p_value_two_sided": result.p_value,
                    "p_value_bonferroni_4_planned_pairs": result.p_value_bonferroni,
                }
            )
    return path


def _annotate_planned_t_tests(axis: Any, results: list[PairwiseTTest]) -> None:
    """Draw compact brackets for the four Bonferroni-corrected planned tests."""
    bracket_levels = (0, 1, 2, 0)
    lower, upper = axis.get_ylim()
    span = upper - lower
    bracket_height = span * 0.018
    base = upper + span * 0.035
    step = span * 0.105
    for result, (left, right), level in zip(
        results, T_TEST_GROUP_PAIRS, bracket_levels, strict=True
    ):
        height = base + level * step
        x_left = left + 1
        x_right = right + 1
        axis.plot(
            (x_left, x_left, x_right, x_right),
            (height, height + bracket_height, height + bracket_height, height),
            color="#242424",
            linewidth=1.0,
            clip_on=False,
            zorder=7,
        )
        axis.text(
            (x_left + x_right) / 2,
            height + bracket_height * 1.2,
            _bracket_label(result.p_value_bonferroni),
            ha="center",
            va="bottom",
            color="#242424",
            fontsize="medium",
            clip_on=False,
            zorder=7,
        )
    axis.set_ylim(lower, base + (max(bracket_levels) + 1.85) * step)


def _bracket_label(p_value: float) -> str:
    """Format an adjusted p-value compactly for an annotated bracket."""
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.0001:
        return "p < 0.0001"
    return f"p = {p_value:.4f}"


def plot_grid_sampled_label_violins(
    input_paths: Sequence[Path],
    output_path: Path,
    *,
    sample_names: Sequence[str] | None = None,
    cell_size_zyx_um: tuple[float, float, float] = (5.0, 5.0, 5.0),
    max_labels_per_sample: int = 8,
    min_subvolumes: int = 2,
    title: str | None = None,
) -> Path:
    """Render one clustered label-violin group for every supplied sample.

    Grid cells truncated at the outer edge still contribute one dot when they
    contain the label.  All other cells must use the requested physical grid
    size, which defaults to the V1 5 x 5 x 5 micrometre grid.
    """
    if not input_paths:
        raise ValueError("at least one analysis bundle is required")
    if max_labels_per_sample < 1:
        raise ValueError("max_labels_per_sample must be positive")
    if min_subvolumes < 2:
        raise ValueError("min_subvolumes must be at least two for a violin")
    if any(value <= 0 for value in cell_size_zyx_um):
        raise ValueError("cell_size_zyx_um values must be positive")
    names = _sample_names(input_paths, sample_names)
    samples = [
        _load_grid_label_distributions(
            path,
            cell_size_zyx_um=cell_size_zyx_um,
            max_labels=max_labels_per_sample,
            min_subvolumes=min_subvolumes,
        )
        for path in input_paths
    ]
    if not any(sample for sample in samples):
        raise ValueError("no labels contain enough intersecting grid cells")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as error:
        raise RuntimeError(
            "install the AHT 'analysis' extra to create violin plots"
        ) from error

    total_violins = sum(len(sample) for sample in samples)
    figure_width = max(7.0, 1.6 * len(samples), 0.55 * total_violins)
    figure, axis = plt.subplots(figsize=(figure_width, 5.5), layout="constrained")
    color_map = plt.get_cmap("tab20")
    legend: list[Patch] = []
    for sample_index, distributions in enumerate(samples, start=1):
        positions = _cluster_positions(sample_index, len(distributions))
        if not positions:
            continue
        values = [item[1] for item in distributions]
        violins = axis.violinplot(
            values,
            positions=positions,
            widths=_violin_width(len(distributions)),
            showmeans=True,
            showmedians=True,
            showextrema=False,
        )
        bodies = cast("list[Any]", violins["bodies"])
        for label_index, ((label_id, points), position) in enumerate(
            zip(distributions, positions, strict=True)
        ):
            body = bodies[label_index]
            color = color_map(label_index % color_map.N)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
            jitter = _dot_jitter(len(points), _violin_width(len(distributions)))
            axis.scatter(
                np.asarray(position + jitter),
                points,
                color=color,
                edgecolors="black",
                linewidths=0.25,
                s=18,
                alpha=0.85,
                zorder=3,
            )
            legend.append(
                Patch(
                    facecolor=color,
                    alpha=0.55,
                    label=f"{names[sample_index - 1]} · label {label_id}",
                )
            )
    axis.set_xticks(range(1, len(names) + 1), names)
    axis.set_xlabel("Sample")
    axis.set_ylabel("Mean Refractive Index per label-grid subvolume (a.u.)")
    grid_name = " x ".join(f"{value:g}" for value in cell_size_zyx_um)
    axis.set_title(f"Biofilm Refractive Index by {grid_name} µm grid subvolume")
    axis.grid(axis="y", alpha=0.25)
    if legend:
        axis.legend(handles=legend, loc="best", fontsize="small", frameon=False)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _load_grid_label_distributions(
    input_path: Path,
    *,
    cell_size_zyx_um: tuple[float, float, float],
    max_labels: int,
    min_subvolumes: int,
) -> list[tuple[int, np.ndarray]]:
    """Return largest labels as means of their intersecting physical cells."""
    manifest = verify_analysis_bundle(input_path)
    _require_grid_size(manifest, cell_size_zyx_um)
    root = input_path.resolve()
    if root.name == BUNDLE_NAME:
        root = root.parent
    bundle = zarr.open_group(root / BUNDLE_NAME, mode="r")
    tables: Any = bundle["tables"]
    if "label_grid" not in tables:
        raise ValueError(
            "bundle lacks canonical label-grid sampling; rerun analyze-ri with "
            "the current RI analysis module"
        )
    table: Any = tables["label_grid"]
    samples: defaultdict[int, list[float]] = defaultdict(list)
    for label_id, ri_mean in zip(
        table["label_id"][:], table["ri_mean"][:], strict=True
    ):
        samples[int(label_id)].append(float(ri_mean))
    ranked = sorted(samples.items(), key=lambda item: (-len(item[1]), item[0]))
    return [
        (label_id, np.asarray(values, dtype=np.float32))
        for label_id, values in ranked
        if len(values) >= min_subvolumes
    ][:max_labels]


def _require_grid_size(
    manifest: dict[str, Any], expected: tuple[float, float, float]
) -> None:
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("RI analysis manifest configuration is invalid")
    grid = configuration.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("RI analysis manifest grid configuration is invalid")
    actual = grid.get("cell_size_zyx_um")
    if not isinstance(actual, list) or len(actual) != 3:
        raise ValueError("RI analysis manifest grid cell size is invalid")
    if not np.allclose(np.asarray(actual, dtype=float), expected, rtol=0.0, atol=1e-9):
        wanted = " x ".join(f"{value:g}" for value in expected)
        found = " x ".join(f"{float(value):g}" for value in actual)
        raise ValueError(f"bundle uses {found} µm grid cells, not {wanted} µm")


def _sample_names(paths: Sequence[Path], supplied: Sequence[str] | None) -> list[str]:
    if supplied is None:
        return [
            path.resolve().parent.name if path.name == BUNDLE_NAME else path.name
            for path in paths
        ]
    if len(supplied) != len(paths):
        raise ValueError("sample_names must contain one name for each input bundle")
    if any(not name.strip() for name in supplied):
        raise ValueError("sample_names cannot contain empty names")
    return list(supplied)


def _cluster_positions(sample_index: int, count: int) -> list[float]:
    if count == 0:
        return []
    if count == 1:
        return [float(sample_index)]
    return list(np.linspace(sample_index - 0.32, sample_index + 0.32, count))


def _violin_width(count: int) -> float:
    return min(0.24, 0.7 / max(1, count))


def _dot_jitter(count: int, width: float, *, seed: int = 0) -> np.ndarray:
    """Use reproducible non-linear jitter to avoid overplotting stripes."""
    if count == 1:
        return np.zeros(1)
    return np.random.default_rng(seed).uniform(-width * 0.31, width * 0.31, count)


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
    """Shape jitter by a smoothed one-dimensional density, like a beeswarm."""
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help=(
            "one or more verified RI analysis directories or analysis.ome.zarr bundles"
        ),
    )
    source.add_argument(
        "--group-cache",
        type=Path,
        help="prepared .npz group grid-sample cache; never reads TIFF stacks",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=1_500,
        help="maximum dot count per group-cache violin (default: 1500)",
    )
    parser.add_argument(
        "--statistics-output",
        type=Path,
        help=(
            "CSV destination for four planned Welch t-tests; default is beside the plot"
        ),
    )
    parser.add_argument(
        "--sample-name",
        action="append",
        help="display name; repeat once for each --input value",
    )
    parser.add_argument("--max-labels-per-sample", type=int, default=8)
    parser.add_argument("--min-subvolumes", type=int, default=2)
    parser.add_argument(
        "--grid-cell-size-um",
        type=float,
        default=5.0,
        help="required cubic grid side length in micrometres (default: 5)",
    )
    parser.add_argument("--title")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the standalone clustered grid-sampled violin plot script."""
    args = _parser().parse_args(argv)
    try:
        if args.group_cache is not None:
            if args.sample_name:
                raise ValueError("--sample-name is only valid with --input")
            output = plot_group_cache_violins(
                args.group_cache,
                args.output,
                max_display_points=args.max_display_points,
                statistics_path=args.statistics_output,
                title=args.title,
            )
        else:
            grid_side = float(args.grid_cell_size_um)
            output = plot_grid_sampled_label_violins(
                args.input,
                args.output,
                sample_names=args.sample_name,
                cell_size_zyx_um=(grid_side, grid_side, grid_side),
                max_labels_per_sample=args.max_labels_per_sample,
                min_subvolumes=args.min_subvolumes,
                title=args.title,
            )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR RI grid-label violin plot: {error}", file=sys.stderr)
        return 2
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
