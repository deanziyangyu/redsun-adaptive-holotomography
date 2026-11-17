"""Verify LED coordinates, validation and Matplotlib preview without Qt."""

import json
from collections import Counter
from math import hypot

import pytest

from redsun_aht.configurations.profile import read_instrument_profile
from redsun_aht.presenter.led_patterns import (
    DIAGNOSTIC, PANELS, LedPatternPresenter, draw_pattern, graded_semicircle,
)


@pytest.mark.parametrize("name, center, counts, maximum", [
    ("arh", 60, (1, 6, 12, 18, 24), 39),
    ("holospim", 0, (1, 6, 12, 18, 24, 30), 13.5),
])
def test_semicircle_follows_panel_coordinates(name, center, counts, maximum):
    panel = PANELS[name]
    points = panel.coordinates()
    assert len(set(points)) == sum(counts)
    assert points[center] == (0, 0)
    assert Counter(round(hypot(x, y), 4) for x, y in points) == dict(zip(panel.radii_mm, counts))
    assert panel.radii_mm[-1] == maximum
    weights = graded_semicircle(panel)
    assert all((weight > 0) == (x > 1e-9) for weight, (x, _) in zip(weights, points, strict=True))


def test_import_uses_hardware_order_and_rejects_wrong_panel(tmp_path):
    pattern = tmp_path / "center.json"
    pattern.write_text(json.dumps([0] * 60 + [255]), encoding="utf-8")
    presenter = LedPatternPresenter("patterns", panel="arh")
    label = presenter.read(str(pattern))
    _, panel, selected, weights = presenter.contents(label)
    assert selected == str(pattern)
    assert weights[60] == 255 and panel.coordinates()[60] == (0, 0)
    with pytest.raises(ValueError, match="91 integer"):
        LedPatternPresenter("patterns", panel="holospim").read(str(pattern))


def test_holospim_plot_uses_one_marker_and_label_per_index(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    panel = PANELS["holospim"]
    weights = graded_semicircle(panel)
    figure = Figure()
    FigureCanvasAgg(figure)
    draw_pattern(figure, panel, DIAGNOSTIC, weights)
    axes = figure.axes[0]
    assert list(axes.collections[0].get_array()) == list(weights)
    assert [t.get_text() for t in axes.texts] == list(map(str, range(91)))
    figure.savefig(tmp_path / "preview.png")
    assert (tmp_path / "preview.png").stat().st_size > 1000


def test_profile_sources_are_relative_to_profile(tmp_path):
    config = tmp_path / "instrument.yaml"
    config.write_text("presenters:\n  led_pattern_ctrl:\n    sources: [patterns/center.json]\n")
    sources = read_instrument_profile(config)["presenters"]["led_pattern_ctrl"]["sources"]
    assert sources == [str(tmp_path / "patterns" / "center.json")]
