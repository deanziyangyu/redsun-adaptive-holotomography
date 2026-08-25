from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pytest

from redsun_aht.processing import (
    ProcessingCancelled,
    TiledVolumeReconstructor,
    TilePlacement,
    TilingConfig,
    resolve_tile_plan,
    tile_weight,
)


@dataclass
class _TileSolver:
    model: Literal["dpct", "qobt"] = "dpct"
    shots_num: int = 4
    fail_at: Literal["prepare", "reconstruct"] | None = None
    invalid_result: bool = False
    prepared: list[tuple[int, int, int]] = field(default_factory=list)
    sample_shapes: list[tuple[int, ...]] = field(default_factory=list)
    background_shapes: list[tuple[int, ...] | None] = field(default_factory=list)
    release_count: int = 0
    close_count: int = 0

    @property
    def gpu_telemetry(self) -> dict[str, Any]:
        return {"device_id": 1, "peak_observed_used_bytes": 4096}

    def prepare(self, shape_zyx: tuple[int, int, int]) -> None:
        self.prepared.append(shape_zyx)
        if self.fail_at == "prepare":
            raise MemoryError("representative tile")

    def reconstruct_tile(
        self,
        sample_zsyx: np.ndarray[Any, Any],
        background_syx: np.ndarray[Any, Any] | None,
    ) -> np.ndarray[Any, Any]:
        self.sample_shapes.append(sample_zsyx.shape)
        self.background_shapes.append(
            None if background_syx is None else background_syx.shape
        )
        if self.fail_at == "reconstruct":
            raise MemoryError("tile allocation")
        result = np.mean(sample_zsyx, axis=1, dtype=np.float64).astype(np.float32)
        if self.invalid_result:
            return result[:, :-1, :]
        return result

    def release_inputs(self) -> None:
        self.release_count += 1

    def close(self) -> None:
        self.close_count += 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mode": "bad"}, "mode"),
        ({"voxel_budget": 0}, "voxel budget"),
        ({"alignment_px": 0}, "alignment"),
        ({"overlap_fraction": 0.5}, "overlap fraction"),
        ({"blend": "linear"}, "raised-cosine"),
        ({"pad_mode": "wrap"}, "pad mode"),
        ({"max_shape_yx": (0, 8)}, "maximum shape"),
        (
            {"min_shape_yx": (9, 8), "max_shape_yx": (8, 8)},
            "minimum shape",
        ),
        ({"explicit_shape_yx": (0, 8)}, "tile dimensions"),
        ({"explicit_overlap_yx": (-1, 0)}, "overlap dimensions"),
        (
            {"mode": "off", "explicit_shape_yx": (8, 8)},
            "cannot include explicit",
        ),
    ],
)
def test_tiling_config_rejects_invalid_geometry(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TilingConfig(**changes)


def test_tile_plan_is_deterministic_row_major_and_weights_are_positive() -> None:
    config = TilingConfig(
        explicit_shape_yx=(4, 5),
        explicit_overlap_yx=(2, 2),
        min_shape_yx=(1, 1),
        max_shape_yx=(8, 8),
        alignment_px=1,
    )

    plan = resolve_tile_plan((3, 7, 9), config)

    assert plan.full_shape_zyx == (3, 7, 9)
    assert plan.tile_shape_yx == (4, 5)
    assert plan.overlap_yx == (2, 2)
    assert [(item.row, item.column) for item in plan.placements] == [
        (row, column) for row in range(3) for column in range(3)
    ]
    assert [(item.y_start, item.x_start) for item in plan.placements] == [
        (0, 0),
        (0, 3),
        (0, 4),
        (2, 0),
        (2, 3),
        (2, 4),
        (3, 0),
        (3, 3),
        (3, 4),
    ]
    assert plan.tiled
    assert all(np.all(tile_weight(item) > 0) for item in plan.placements)
    assert plan.as_provenance()["order"] == "row-major"


def test_one_tile_and_overlapped_results_match_for_pointwise_solver() -> None:
    sample = np.random.default_rng(2).normal(size=(3, 4, 7, 9)).astype(np.float32)
    one_solver = _TileSolver()
    tiled_solver = _TileSolver()
    untiled = TiledVolumeReconstructor(one_solver, TilingConfig(mode="off"))
    tiled = TiledVolumeReconstructor(
        tiled_solver,
        TilingConfig(
            explicit_shape_yx=(4, 5),
            explicit_overlap_yx=(2, 2),
            min_shape_yx=(1, 1),
            max_shape_yx=(8, 8),
            alignment_px=1,
        ),
    )

    baseline = untiled.run(sample)
    blended = tiled.run(sample)

    np.testing.assert_allclose(
        baseline.volume_zyx,
        np.mean(sample, axis=1, dtype=np.float64).astype(np.float32),
    )
    np.testing.assert_allclose(blended.volume_zyx, baseline.volume_zyx, rtol=1e-6)
    assert one_solver.prepared == [(3, 7, 9)]
    assert tiled_solver.prepared == [(3, 4, 5)]
    assert all(shape == (3, 4, 4, 5) for shape in tiled_solver.sample_shapes)
    assert tiled_solver.release_count == len(blended.plan.placements)
    assert len(blended.tile_timing_ns) == len(blended.plan.placements)
    provenance = blended.as_provenance()
    assert provenance["model"] == "dpct"
    assert provenance["input_shape_zsyx"] == (3, 4, 7, 9)
    assert provenance["gpu"]["device_id"] == 1


def test_tiler_retains_full_z_shots_background_and_reports_progress() -> None:
    sample = np.ones((2, 4, 5, 6), dtype=np.float32)
    background = np.ones((3, 4, 5, 6), dtype=np.float32)
    solver = _TileSolver(model="qobt")
    completed: list[tuple[int, int, int]] = []
    reconstructor = TiledVolumeReconstructor(
        solver,
        TilingConfig(
            explicit_shape_yx=(4, 4),
            explicit_overlap_yx=(1, 1),
            min_shape_yx=(1, 1),
            max_shape_yx=(8, 8),
            alignment_px=1,
        ),
    )

    result = reconstructor.run(
        sample,
        background,
        progress=lambda done, total, placement: completed.append(
            (done, total, placement.index)
        ),
    )

    assert result.model == "qobt"
    assert all(shape[:2] == (2, 4) for shape in solver.sample_shapes)
    assert all(shape == (4, 4, 4) for shape in solver.background_shapes)
    assert completed[-1] == (
        len(result.plan.placements),
        len(result.plan.placements),
        result.plan.placements[-1].index,
    )


def test_tiler_cancels_between_tiles_and_closes_idempotently() -> None:
    solver = _TileSolver()
    reconstructor = TiledVolumeReconstructor(
        solver,
        TilingConfig(
            explicit_shape_yx=(3, 3),
            explicit_overlap_yx=(1, 1),
            min_shape_yx=(1, 1),
            max_shape_yx=(4, 4),
            alignment_px=1,
        ),
    )
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    with pytest.raises(ProcessingCancelled, match="cancelled"):
        reconstructor.run(np.ones((2, 4, 5, 5), dtype=np.float32), cancelled=cancelled)

    assert solver.release_count == 2
    reconstructor.close()
    reconstructor.close()
    assert solver.close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        reconstructor.run(np.ones((2, 4, 5, 5), dtype=np.float32))


@pytest.mark.parametrize("failure", ["prepare", "reconstruct"])
def test_tiler_turns_oom_into_stable_smaller_tile_diagnostic(
    failure: Literal["prepare", "reconstruct"],
) -> None:
    solver = _TileSolver(fail_at=failure)
    reconstructor = TiledVolumeReconstructor(
        solver,
        TilingConfig(
            explicit_shape_yx=(64, 64),
            explicit_overlap_yx=(8, 8),
            min_shape_yx=(16, 16),
            max_shape_yx=(64, 64),
            alignment_px=8,
        ),
    )

    with pytest.raises(RuntimeError, match=r"explicit tile shape of \(48, 48\)"):
        reconstructor.run(np.ones((2, 4, 96, 96), dtype=np.float32))


def test_tiler_rejects_invalid_inputs_plans_and_solver_outputs() -> None:
    with pytest.raises(ValueError, match="supports only"):
        TiledVolumeReconstructor(_TileSolver(model="other"), TilingConfig())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shot count"):
        TiledVolumeReconstructor(_TileSolver(shots_num=0), TilingConfig())
    with pytest.raises(ValueError, match="volume dimensions"):
        resolve_tile_plan((0, 2, 2), TilingConfig(mode="off"))
    with pytest.raises(ValueError, match="overlap"):
        resolve_tile_plan(
            (1, 4, 4),
            TilingConfig(
                explicit_shape_yx=(2, 2),
                explicit_overlap_yx=(2, 0),
                min_shape_yx=(1, 1),
                max_shape_yx=(4, 4),
                alignment_px=1,
            ),
        )
    with pytest.raises(ValueError, match="positive XY"):
        TilePlacement(0, 0, 0, 1, 1, 0, 1)

    reconstructor = TiledVolumeReconstructor(_TileSolver(), TilingConfig(mode="off"))
    with pytest.raises(ValueError, match="ZSYX"):
        reconstructor.run(np.ones((4, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="contains 3 shots"):
        reconstructor.run(np.ones((2, 3, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="SYX or CSYX"):
        reconstructor.run(
            np.ones((2, 4, 3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="dimensions differ"):
        reconstructor.run(
            np.ones((2, 4, 3, 3), dtype=np.float32),
            np.ones((4, 2, 3), dtype=np.float32),
        )

    invalid = TiledVolumeReconstructor(
        _TileSolver(invalid_result=True), TilingConfig(mode="off")
    )
    with pytest.raises(ValueError, match="returned"):
        invalid.run(np.ones((2, 4, 3, 3), dtype=np.float32))


def test_auto_plan_respects_budget_alignment_and_no_z_tiling() -> None:
    plan = resolve_tile_plan(
        (4, 130, 135),
        TilingConfig(
            voxel_budget=4 * 64 * 64,
            min_shape_yx=(32, 32),
            max_shape_yx=(96, 96),
            alignment_px=16,
            overlap_fraction=0.25,
        ),
    )

    assert plan.tile_shape_yx == (64, 64)
    assert plan.overlap_yx == (16, 16)

    with pytest.raises(ValueError, match="Z tiling is unsupported"):
        resolve_tile_plan(
            (100, 512, 512),
            TilingConfig(
                voxel_budget=100,
                min_shape_yx=(16, 16),
                max_shape_yx=(32, 32),
                alignment_px=8,
            ),
        )
