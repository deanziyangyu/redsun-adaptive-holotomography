from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from redsun_aht.device.stage import (
    AxisSpec,
    CompositeStageDevice,
    SimulatedAxisBackend,
    StageSettleError,
)
from redsun_aht.domain import PoseSample


def _build_stage(
    *, settle_polls: int = 0, stalled: bool = False
) -> tuple[CompositeStageDevice, dict[str, SimulatedAxisBackend]]:
    backends = {
        "x": SimulatedAxisBackend(settle_polls=settle_polls, stalled=stalled),
        "z": SimulatedAxisBackend(settle_polls=settle_polls, stalled=stalled),
    }
    stage = CompositeStageDevice(
        specs=(
            AxisSpec("x", "mm", -5.0, 5.0, 0.001, uncertainty=0.0005),
            AxisSpec("z", "mm", 0.0, 10.0, 0.001, uncertainty=0.0005),
        ),
        backends=backends,
        settle_timeout=0.02,
        settle_poll_interval=0,
    )
    return stage, backends


def test_composite_stage_moves_settles_and_reports_pose() -> None:
    stage, backends = _build_stage(settle_polls=2)

    async def exercise() -> tuple[PoseSample, PoseSample]:
        commanded = await stage.move({"x": 1.25, "z": 4.5})
        measured = await stage.settle()
        return commanded, measured

    commanded, measured = asyncio.run(exercise())

    assert stage.axes == {"x": "mm", "z": "mm"}
    assert commanded.sample_kind == "commanded"
    assert commanded.joints == (1.25, 4.5)
    assert measured.sample_kind == "measured"
    assert measured.joints == (1.25, 4.5)
    assert measured.coordinate_frame == "aht-sample-stage"
    assert backends["x"].commands == [1.25]


def test_composite_stage_skips_unchanged_axis_commands_and_polls() -> None:
    class CountingBackend(SimulatedAxisBackend):
        motion_checks: int = 0
        position_reads: int = 0

        async def motion_complete(self) -> bool:
            self.motion_checks += 1
            return await super().motion_complete()

        async def read_position(self) -> float:
            self.position_reads += 1
            return await super().read_position()

    x_backend = CountingBackend()
    z_backend = CountingBackend()
    stage = CompositeStageDevice(
        specs=(AxisSpec("x", "mm", -1, 1, 0.001), AxisSpec("z", "mm", -1, 1, 0.001)),
        backends={"x": x_backend, "z": z_backend},
        settle_poll_interval=0,
    )

    async def exercise() -> None:
        await stage.move({"x": 0.1, "z": 0.1})
        await stage.settle()
        x_backend.motion_checks = x_backend.position_reads = 0
        z_backend.motion_checks = z_backend.position_reads = 0
        await stage.move({"x": 0.1, "z": 0.2})
        measured = await stage.settle()
        assert measured.joints == (0.1, 0.2)

    asyncio.run(exercise())

    assert x_backend.commands == [0.1]
    assert x_backend.motion_checks == x_backend.position_reads == 0
    assert z_backend.commands == [0.1, 0.2]
    assert z_backend.motion_checks == z_backend.position_reads == 1


def test_move_validation_is_atomic_across_axes() -> None:
    stage, backends = _build_stage()

    async def exercise() -> None:
        with pytest.raises(ValueError, match="outside"):
            await stage.move({"x": 1.0, "z": 11.0})

    asyncio.run(exercise())

    assert backends["x"].commands == []
    assert backends["z"].commands == []


def test_stalled_stage_times_out_and_stops_all_axes() -> None:
    stage, backends = _build_stage(stalled=True)

    async def exercise() -> None:
        await stage.move({"x": 1.0})
        with pytest.raises(StageSettleError, match="did not settle"):
            await stage.settle()

    asyncio.run(exercise())

    assert all(backend.stopped for backend in backends.values())


@pytest.mark.parametrize(
    "spec",
    [
        lambda: AxisSpec("", "mm", 0, 1, 0.1),
        lambda: AxisSpec("x", "mm", 0, float("inf"), 0.1),
        lambda: AxisSpec("x", "mm", 1, 1, 0.1),
        lambda: AxisSpec("x", "mm", 0, 1, 0),
        lambda: AxisSpec("x", "mm", 0, 1, 0.1, uncertainty=-1),
    ],
)
def test_axis_spec_rejects_invalid_metadata(spec: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        spec()


def test_composite_rejects_incomplete_or_ambiguous_topology() -> None:
    backend = SimulatedAxisBackend()
    spec = AxisSpec("x", "mm", 0, 1, 0.1)

    with pytest.raises(ValueError, match="at least one"):
        CompositeStageDevice((), {})
    with pytest.raises(ValueError, match="unique"):
        CompositeStageDevice((spec, spec), {"x": backend})
    with pytest.raises(ValueError, match="exactly one"):
        CompositeStageDevice((spec,), {})
    with pytest.raises(ValueError, match="settle timing"):
        CompositeStageDevice((spec,), {"x": backend}, settle_timeout=0)


def test_move_rejects_empty_unknown_and_nonfinite_targets() -> None:
    stage, _ = _build_stage()

    async def exercise() -> None:
        for target in ({}, {"bad": 1.0}, {"x": float("nan")}):
            with pytest.raises(ValueError):
                await stage.move(target)

    asyncio.run(exercise())


def test_partial_backend_start_failure_stops_every_axis() -> None:
    class FailingBackend(SimulatedAxisBackend):
        async def move_to(self, position: float) -> None:
            raise OSError("injected start failure")

    x_backend = SimulatedAxisBackend()
    z_backend = FailingBackend()
    stage = CompositeStageDevice(
        specs=(
            AxisSpec("x", "mm", 0, 5, 0.1),
            AxisSpec("z", "mm", 0, 5, 0.1),
        ),
        backends={"x": x_backend, "z": z_backend},
    )

    async def exercise() -> None:
        with pytest.raises(OSError, match="injected"):
            await stage.move({"x": 1, "z": 1})

    asyncio.run(exercise())
    assert x_backend.stopped
    assert z_backend.stopped
