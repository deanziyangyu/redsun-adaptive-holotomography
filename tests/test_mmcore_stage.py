"""Tests for ``MMStage`` and ``MMAxis`` over the stage service's signal set.

The stage is connected in mock mode with the axes the service would publish, so
what is checked is the device's own logic: named axes, the limits it refuses a
target against, the tolerance a move waits with, the stop request that reaches
the service, and what the plan-facing ``redsun_aht.contracts`` adapter makes of
the published signals. The service behind the signals is qualified in
``test_mmcore_services.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from mmcore_fakes import mock_stage
from ophyd_async.core import get_mock_put, set_mock_value

from redsun_aht.contracts import QualityLabel, Stage
from redsun_aht.device.mmcore import MOVE_TIMEOUT, POSITION_TOLERANCE, MMStage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.fixture
async def stage() -> AsyncGenerator[MMStage, None]:
    """Return a stage with the axes ``x`` and ``y`` and mock signals."""
    yield await mock_stage("PREFIX:XY:")


async def test_the_axes_are_the_ones_the_service_serves(stage: MMStage) -> None:
    """Name the axes as the service does, and read them by that name."""
    assert set(stage.axis) == {"x", "y"}

    readings = await stage.read()

    assert set(readings) == {"XY-axis-x", "XY-axis-y"}
    assert set(await stage.describe()) == set(readings)
    assert stage.axis["x"].parent is stage.axis


async def test_an_axis_publishes_the_limits_the_service_enforces(
    stage: MMStage,
) -> None:
    """Report each axis's travel bounds and precision as configuration."""
    configuration = await stage.axis["x"].read_configuration()

    assert list(configuration) == ["XY-axis-x-limits", "XY-axis-x-precision"]
    assert np.array_equal(
        configuration["XY-axis-x-limits"]["value"], np.array([-1000.0, 1000.0])
    )


async def test_a_target_inside_the_limits_is_moved_to(stage: MMStage) -> None:
    """Move an axis, and read back where it stopped."""
    axis = stage.axis["x"]

    await axis.set(10.0)

    assert (await axis.locate())["readback"] == pytest.approx(10.0)
    assert (await stage.read())["XY-axis-x"]["value"] == pytest.approx(10.0)


async def test_a_target_outside_the_limits_is_refused(stage: MMStage) -> None:
    """Refuse an out-of-range target before it is put to the service."""
    axis = stage.axis["x"]
    put = get_mock_put(axis.position)

    with pytest.raises(ValueError, match="outside the axis limits"):
        await axis.set(5000.0)

    put.assert_not_called()


async def test_a_stop_request_reaches_the_service(stage: MMStage) -> None:
    """Stop the whole stage through the service's stop request."""
    set_mock_value(stage.stop_request, False)

    await stage.stop()

    get_mock_put(stage.stop_request).assert_called_once_with(True)


async def test_an_axis_stop_request_reaches_the_service(stage: MMStage) -> None:
    """Stop one axis through its own stop request."""
    axis = stage.axis["y"]
    set_mock_value(axis.stop_request, False)

    # ophyd-async leaves StandardMovable.stop unannotated, so the call is
    # untyped however this device subclasses it
    await axis.stop()  # type: ignore[no-untyped-call]

    get_mock_put(axis.stop_request).assert_called_once_with(True)


async def test_a_refused_move_leaves_the_axis_usable(stage: MMStage) -> None:
    """Leave the axis movable after a target the service would refuse."""
    axis = stage.axis["x"]

    with pytest.raises(ValueError, match="outside the axis limits"):
        await axis.set(-5000.0)

    await axis.set(5.0)

    assert (await axis.locate())["readback"] == pytest.approx(5.0)


def test_move_bounds_cover_the_measured_stage_behaviour() -> None:
    """Wait longer than a measured traverse, and settle no tighter than it lands."""
    assert POSITION_TOLERANCE == 1.5
    assert MOVE_TIMEOUT == 60.0


async def test_axes_are_the_declared_axes(stage: MMStage) -> None:
    """Turn the published signals into the contract's declared axes."""
    specs = dict(stage.axes)

    assert set(specs) == {"x", "y"}
    assert specs["x"].units == "um"
    assert specs["x"].limits == (-1000.0, 1000.0)
    assert specs["x"].precision == pytest.approx(0.001)
    assert isinstance(stage, Stage)


async def test_axes_refresh_from_the_service(stage: MMStage) -> None:
    """Re-read the declared axes when the service publishes new bounds."""
    set_mock_value(stage.axis["y"].limits, np.array([-5.0, 5.0]))

    await stage.refresh_axes()

    assert stage.axes["y"].limits == (-5.0, 5.0)


async def test_move_and_read_position_agree_with_the_axis_readback(
    stage: MMStage,
) -> None:
    """Move through the contract surface, and read back what the axes report."""
    reading = await stage.move({"x": 12.5, "y": -3.0})

    assert reading.quality is QualityLabel.MEASURED
    assert reading.positions == {"x": 12.5, "y": -3.0}
    assert (await stage.axis["x"].locate())["readback"] == pytest.approx(12.5)

    again = await stage.read_position()

    assert again.positions == reading.positions
    assert again.monotonic_ns >= reading.monotonic_ns


async def test_move_refuses_an_axis_the_stage_does_not_serve(stage: MMStage) -> None:
    """Refuse an unknown axis before any axis moves."""
    with pytest.raises(ValueError, match="no axis named 'z'"):
        await stage.move({"x": 1.0, "z": 2.0})

    assert (await stage.axis["x"].locate())["readback"] == pytest.approx(0.0)


async def test_an_axis_the_service_cannot_describe_is_left_out(
    stage: MMStage,
) -> None:
    """Leave out an unbounded axis without failing the stage or its other axes."""
    set_mock_value(stage.axis["y"].limits, np.array([-np.inf, np.inf]))

    await stage.refresh_axes()

    assert set(stage.axes) == {"x"}
    with pytest.raises(ValueError, match="publishes no travel limits"):
        await stage.axis["y"].declared_spec("y")
    # it is still movable: the service is the authority on its bounds
    await stage.move({"y": 2.5})
    assert (await stage.axis["y"].locate())["readback"] == pytest.approx(2.5)


def test_an_unusable_published_precision_falls_back() -> None:
    """A zero or negative published step must not hide an otherwise movable axis."""
    from redsun_aht.device.mmcore.stage import (
        FALLBACK_PRECISION_UM,
        _published_precision,
    )

    assert _published_precision(0.0, "x") == FALLBACK_PRECISION_UM
    assert _published_precision(-2.0, "y") == FALLBACK_PRECISION_UM
    assert _published_precision(float("nan"), "z") == FALLBACK_PRECISION_UM
    assert _published_precision(0.05, "x") == 0.05


async def test_simulated_stage_provenance_is_not_upgraded(stage: MMStage) -> None:
    """A simulator readback remains simulated, even when it is readable."""
    set_mock_value(stage.provenance, "simulated")
    reading = await stage.read_position()
    assert reading.quality is QualityLabel.SIMULATED
    assert reading.axes and not any(axis.measured for axis in reading.axes)


async def test_unknown_stage_provenance_cannot_claim_positions(stage: MMStage) -> None:
    """An unspecified source must not be presented as a measured position."""
    set_mock_value(stage.provenance, "unknown")
    reading = await stage.read_position()
    assert reading.quality is QualityLabel.UNAVAILABLE
    assert not reading.axes


async def test_multi_axis_move_prevalidates_all_targets(stage: MMStage) -> None:
    set_mock_value(stage.axis["y"].limits, np.array([-5.0, 5.0]))
    with pytest.raises(ValueError, match="outside the axis limits"):
        await stage.move({"x": 1.0, "y": 6.0})
    get_mock_put(stage.axis["x"].position).assert_not_called()
    get_mock_put(stage.axis["y"].position).assert_not_called()
    assert (await stage.axis["x"].locate())["readback"] == 0.0


async def test_multi_axis_move_rejects_nonfinite_before_any_put(stage: MMStage) -> None:
    with pytest.raises(ValueError, match="finite"):
        await stage.move({"x": 1.0, "y": float("nan")})
    get_mock_put(stage.axis["x"].position).assert_not_called()
    get_mock_put(stage.axis["y"].position).assert_not_called()
