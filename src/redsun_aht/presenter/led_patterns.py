"""Panel geometry and state for previewing LED pattern arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import cos, sin, tau
from pathlib import Path

from psygnal import Signal
from redsun import slot


@dataclass(frozen=True)
class PanelGeometry:
    name: str
    counts: tuple[int, ...]
    radii_mm: tuple[float, ...]
    starts: tuple[int, ...]

    @property
    def pixels(self) -> int:
        return sum(self.counts)

    def coordinates(self) -> tuple[tuple[float, float], ...]:
        """Return coordinates in hardware index order."""
        positions = [(0.0, 0.0)] * self.pixels
        for count, radius, start in zip(self.counts, self.radii_mm, self.starts, strict=True):
            for local in range(count):
                angle = tau * local / count
                positions[start + local] = radius * sin(angle), radius * cos(angle)
        return tuple(positions)


PANELS = {
    "arh": PanelGeometry("ARH", (1, 6, 12, 18, 24), (0, 12, 21, 30, 39), (60, 54, 42, 24, 0)),
    "holospim": PanelGeometry("HoloSPIM", (1, 6, 12, 18, 24, 30),
                              (0, 2.7, 5.4, 8.1, 10.8, 13.5), (0, 1, 7, 19, 37, 61)),
}
DIAGNOSTIC = "Diagnostic · graded right semicircle"


def graded_semicircle(panel: PanelGeometry) -> tuple[int, ...]:
    """Blue-only test weights, increasing away from the vertical diameter."""
    limit = panel.radii_mm[-1]
    return tuple(round(48 + 207 * x / limit) if x > 1e-9 else 0
                 for x, _ in panel.coordinates())


def pattern_weights(panel: PanelGeometry, values: object) -> tuple[int, ...]:
    if (not isinstance(values, (list, tuple)) or len(values) != panel.pixels
            or any(type(value) is not int or not 0 <= value <= 255 for value in values)):
        raise ValueError(f"Expected {panel.pixels} integer weights from 0 to 255")
    return tuple(values)


class LedPatternPresenter:
    """Keep imported patterns and report the selected array to the plot view."""

    sig_patterns = Signal(tuple, str)
    sig_pattern = Signal(object, str, tuple)
    sig_error = Signal(str)

    def __init__(self, name: str, *, panel: str, sources: tuple[str, ...] = ()) -> None:
        self.name = name
        self.panel = PANELS[panel]
        self.sources = tuple(sources)
        self.patterns = {DIAGNOSTIC: graded_semicircle(self.panel)}

    def contents(self, selected: str = DIAGNOSTIC) -> tuple[tuple[str, ...], PanelGeometry, str, tuple[int, ...]]:
        return tuple(self.patterns), self.panel, selected, self.patterns[selected]

    def _publish(self, label: str) -> None:
        self.sig_patterns.emit(tuple(self.patterns), label)
        self.select(label)

    def read(self, filename: str) -> str:
        path = Path(filename)
        with path.open("rb") as stream:
            raw = stream.read(8193)
        if len(raw) > 8192:
            raise ValueError("Pattern file is too large")
        label = str(path.resolve())
        self.patterns[label] = pattern_weights(self.panel, json.loads(raw))
        return label

    @slot(thread="main")
    def load(self, filename: str) -> None:
        try:
            label = self.read(filename)
        except (OSError, ValueError) as exc:
            self.sig_error.emit(str(exc))
            return
        self._publish(label)

    @slot(thread="main")
    def show_uploaded(self, label: str, values: tuple) -> None:
        try:
            self.patterns[label] = pattern_weights(self.panel, values)
        except ValueError as exc:
            self.sig_error.emit(str(exc))
            return
        self._publish(label)

    @slot(thread="main")
    def select(self, label: str) -> None:
        if label in self.patterns:
            self.sig_pattern.emit(self.panel, label, self.patterns[label])


def draw_pattern(figure, panel: PanelGeometry, label: str, weights: tuple[int, ...]) -> None:
    """Render blue-only weights with hardware indices and physical axes."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle

    weights = pattern_weights(panel, weights)
    figure.clear()
    ax = figure.add_subplot()
    points = panel.coordinates()
    radius = panel.radii_mm[-1]
    cmap = LinearSegmentedColormap.from_list("led-blue", ["#f4f8ff", "#83b0e9", "#084d9a"])
    plot = ax.scatter(*zip(*points), c=weights, cmap=cmap, vmin=0, vmax=255, s=130,
                      edgecolors="#5075a2", linewidths=.6, zorder=3)
    for r in panel.radii_mm[1:]:
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor="#b4c5db", linewidth=.65))
    ax.axhline(0, color="#b4c5db", linewidth=.65)
    ax.axvline(0, color="#b4c5db", linewidth=.65)
    for index, ((x, y), weight) in enumerate(zip(points, weights, strict=True)):
        ax.annotate(str(index), (x, y), ha="center", va="center", fontsize=6.5,
                    color="white" if weight >= 165 else "#0e2746", zorder=4)
    ax.set(aspect="equal", xlim=(-1.16 * radius, 1.16 * radius),
           ylim=(-1.16 * radius, 1.16 * radius), xlabel="X (mm)", ylabel="Y (mm)")
    ax.set_title(f"{panel.name} · {label}", fontsize=10, wrap=True)
    figure.colorbar(plot, ax=ax, orientation="vertical", fraction=.06, pad=.04,
                    ticks=(0, 64, 128, 192, 255), label="8-bit weight")
