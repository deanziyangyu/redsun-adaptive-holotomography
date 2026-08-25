from __future__ import annotations

from collections.abc import Callable

import pytest

from redsun_aht.motion import grid_scan, linear_points, list_scan


def test_linear_points_include_exact_endpoints() -> None:
    assert linear_points(-1.0, 1.0, 4) == (-1.0, -0.5, 0.0, 0.5, 1.0)
    assert linear_points(2.0, -1.0, 3) == (2.0, 1.0, 0.0, -1.0)


def test_list_scan_zips_axes_in_declared_order() -> None:
    points = list_scan({"x": (0, 1), "z": (10, 20)})

    assert [point.as_dict() for point in points] == [
        {"x": 0.0, "z": 10.0},
        {"x": 1.0, "z": 20.0},
    ]


def test_grid_scan_snakes_selected_inner_axis() -> None:
    points = grid_scan(
        {"x": (0, 1), "y": (10, 20), "z": (100,)},
        snake_axes=frozenset({"y"}),
    )

    assert [point.as_dict() for point in points] == [
        {"x": 0.0, "y": 10.0, "z": 100.0},
        {"x": 0.0, "y": 20.0, "z": 100.0},
        {"x": 1.0, "y": 20.0, "z": 100.0},
        {"x": 1.0, "y": 10.0, "z": 100.0},
    ]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: list_scan({}),
        lambda: list_scan({"x": (0, 1), "y": (0,)}),
        lambda: grid_scan({"x": (0,)}, snake_axes=frozenset({"y"})),
        lambda: linear_points(0, 1, 0),
        lambda: linear_points(float("nan"), 1, 1),
        lambda: linear_points(0, 1, 1_000_000),
        lambda: list_scan({"": (0,)}),
        lambda: list_scan({"x": ()}),
        lambda: list_scan({"x": (float("inf"),)}),
    ],
)
def test_scan_validation_rejects_ambiguous_inputs(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        factory()
