"""Deterministic multi-axis list, grid, and snake scan construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MAX_SCAN_POINTS = 1_000_000


@dataclass(frozen=True, slots=True)
class ScanPoint:
    """One ordered multi-axis scan position."""

    index: int
    positions: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, float]:
        """Return a mutable mapping suitable for a stage move request."""
        return dict(self.positions)


def linear_points(start: float, stop: float, intervals: int) -> tuple[float, ...]:
    """Return both endpoints split into an exact number of intervals."""
    if not isfinite(start) or not isfinite(stop):
        raise ValueError("scan endpoints must be finite")
    if intervals < 1:
        raise ValueError("intervals must be at least 1")
    if intervals + 1 > MAX_SCAN_POINTS:
        raise ValueError(f"scan exceeds {MAX_SCAN_POINTS} points")
    step = (stop - start) / intervals
    return (*(start + step * index for index in range(intervals)), stop)


def list_scan(axes: Mapping[str, Sequence[float]]) -> tuple[ScanPoint, ...]:
    """Zip equal-length axis lists into ordered scan points."""
    normalized = _normalize_axes(axes)
    lengths = {len(values) for _, values in normalized}
    if len(lengths) != 1:
        raise ValueError("list-scan axes must contain the same number of points")
    point_count = lengths.pop()
    if point_count > MAX_SCAN_POINTS:
        raise ValueError(f"scan exceeds {MAX_SCAN_POINTS} points")
    return tuple(
        ScanPoint(
            index,
            tuple((name, values[index]) for name, values in normalized),
        )
        for index in range(point_count)
    )


def grid_scan(
    axes: Mapping[str, Sequence[float]],
    *,
    snake_axes: frozenset[str] = frozenset(),
) -> tuple[ScanPoint, ...]:
    """Build an N-D product with the first axis slowest and optional snaking."""
    normalized = _normalize_axes(axes)
    names = tuple(name for name, _ in normalized)
    unknown_snake_axes = snake_axes.difference(names)
    if unknown_snake_axes:
        unknown = ", ".join(sorted(unknown_snake_axes))
        raise ValueError(f"snake axes are not present in scan: {unknown}")
    point_count = 1
    for _, values in normalized:
        point_count *= len(values)
        if point_count > MAX_SCAN_POINTS:
            raise ValueError(f"scan exceeds {MAX_SCAN_POINTS} points")

    points: list[ScanPoint] = []

    def append_level(
        level: int,
        outer_indices: tuple[int, ...],
        positions: tuple[tuple[str, float], ...],
    ) -> None:
        if level == len(normalized):
            points.append(ScanPoint(len(points), positions))
            return
        name, values = normalized[level]
        indices: Sequence[int] = range(len(values))
        if name in snake_axes and sum(outer_indices) % 2:
            indices = range(len(values) - 1, -1, -1)
        for value_index in indices:
            append_level(
                level + 1,
                (*outer_indices, value_index),
                (*positions, (name, values[value_index])),
            )

    append_level(0, (), ())
    return tuple(points)


def _normalize_axes(
    axes: Mapping[str, Sequence[float]],
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if not axes:
        raise ValueError("scan must contain at least one axis")
    normalized: list[tuple[str, tuple[float, ...]]] = []
    for name, raw_values in axes.items():
        if not name:
            raise ValueError("axis name cannot be empty")
        values = tuple(float(value) for value in raw_values)
        if not values:
            raise ValueError(f"axis {name!r} has no scan points")
        if not all(isfinite(value) for value in values):
            raise ValueError(f"axis {name!r} contains a non-finite point")
        normalized.append((name, values))
    return tuple(normalized)
