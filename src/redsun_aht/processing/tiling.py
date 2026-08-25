"""Deterministic offline XY tiling with complementary overlap blending."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from redsun_aht.processing.solvers import ProcessingCancelled

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy.typing as npt


DEFAULT_VOXEL_BUDGET = 512 * 512 * 100


@dataclass(frozen=True, slots=True)
class TilingConfig:
    """Resolved-input-independent limits for one deterministic XY tile plan."""

    mode: Literal["auto", "off"] = "auto"
    voxel_budget: int = DEFAULT_VOXEL_BUDGET
    max_shape_yx: tuple[int, int] = (768, 768)
    min_shape_yx: tuple[int, int] = (256, 256)
    alignment_px: int = 32
    overlap_fraction: float = 0.20
    blend: Literal["raised_cosine"] = "raised_cosine"
    pad_mode: Literal["reflect", "edge"] = "reflect"
    explicit_shape_yx: tuple[int, int] | None = None
    explicit_overlap_yx: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or uncovered geometry before worker admission."""
        if self.mode not in {"auto", "off"}:
            raise ValueError("tiling mode must be 'auto' or 'off'")
        if self.voxel_budget <= 0:
            raise ValueError("tiling voxel budget must be positive")
        if self.alignment_px <= 0:
            raise ValueError("tiling alignment must be positive")
        if not 0.0 <= self.overlap_fraction < 0.5:
            raise ValueError("tiling overlap fraction must be in [0, 0.5)")
        if self.blend != "raised_cosine":
            raise ValueError("only raised-cosine tile blending is supported")
        if self.pad_mode not in {"reflect", "edge"}:
            raise ValueError("tiling pad mode must be 'reflect' or 'edge'")
        for name, shape in (
            ("maximum", self.max_shape_yx),
            ("minimum", self.min_shape_yx),
        ):
            if any(value <= 0 for value in shape):
                raise ValueError(f"tiling {name} shape values must be positive")
        if any(
            minimum > maximum
            for minimum, maximum in zip(
                self.min_shape_yx, self.max_shape_yx, strict=True
            )
        ):
            raise ValueError("tiling minimum shape cannot exceed maximum shape")
        if self.explicit_shape_yx is not None and any(
            value <= 0 for value in self.explicit_shape_yx
        ):
            raise ValueError("explicit tile dimensions must be positive")
        if self.explicit_overlap_yx is not None and any(
            value < 0 for value in self.explicit_overlap_yx
        ):
            raise ValueError("explicit overlap dimensions cannot be negative")
        if self.mode == "off" and (
            self.explicit_shape_yx is not None or self.explicit_overlap_yx is not None
        ):
            raise ValueError("tiling-off mode cannot include explicit tile geometry")


@dataclass(frozen=True, slots=True)
class TilePlacement:
    """Translation-only placement in the source and output pixel grid."""

    index: int
    row: int
    column: int
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    pad_bottom: int = 0
    pad_right: int = 0
    top_overlap: int = 0
    bottom_overlap: int = 0
    left_overlap: int = 0
    right_overlap: int = 0

    def __post_init__(self) -> None:
        """Validate one bounded positive placement."""
        if min(self.index, self.row, self.column, self.y_start, self.x_start) < 0:
            raise ValueError("tile indices and starts cannot be negative")
        if self.y_stop <= self.y_start or self.x_stop <= self.x_start:
            raise ValueError("tile placement must contain positive XY extents")
        if (
            min(
                self.pad_bottom,
                self.pad_right,
                self.top_overlap,
                self.bottom_overlap,
                self.left_overlap,
                self.right_overlap,
            )
            < 0
        ):
            raise ValueError("tile padding and overlap cannot be negative")

    @property
    def shape_yx(self) -> tuple[int, int]:
        """Return the valid, unpadded source extent."""
        return self.y_stop - self.y_start, self.x_stop - self.x_start

    def as_provenance(self) -> Mapping[str, int]:
        """Return the complete placement as immutable JSON-compatible values."""
        return MappingProxyType(
            {
                "index": self.index,
                "row": self.row,
                "column": self.column,
                "y_start": self.y_start,
                "y_stop": self.y_stop,
                "x_start": self.x_start,
                "x_stop": self.x_stop,
                "pad_bottom": self.pad_bottom,
                "pad_right": self.pad_right,
                "top_overlap": self.top_overlap,
                "bottom_overlap": self.bottom_overlap,
                "left_overlap": self.left_overlap,
                "right_overlap": self.right_overlap,
            }
        )


@dataclass(frozen=True, slots=True)
class TilePlan:
    """Fully resolved row-major tiling geometry for one aligned volume."""

    full_shape_zyx: tuple[int, int, int]
    tile_shape_yx: tuple[int, int]
    overlap_yx: tuple[int, int]
    placements: tuple[TilePlacement, ...]
    config: TilingConfig

    def __post_init__(self) -> None:
        """Require at least one tile and positive resolved dimensions."""
        if any(value <= 0 for value in (*self.full_shape_zyx, *self.tile_shape_yx)):
            raise ValueError("resolved tile plan dimensions must be positive")
        if not self.placements:
            raise ValueError("resolved tile plan requires at least one placement")

    @property
    def tiled(self) -> bool:
        """Return whether this plan subdivides the source XY plane."""
        return len(self.placements) > 1

    def as_provenance(self) -> Mapping[str, Any]:
        """Return immutable resolved geometry suitable for result metadata."""
        return MappingProxyType(
            {
                "mode": self.config.mode,
                "voxel_budget": self.config.voxel_budget,
                "requested_tile_shape_yx": self.config.explicit_shape_yx,
                "requested_overlap_yx": self.config.explicit_overlap_yx,
                "max_shape_yx": self.config.max_shape_yx,
                "min_shape_yx": self.config.min_shape_yx,
                "alignment_px": self.config.alignment_px,
                "overlap_fraction": self.config.overlap_fraction,
                "resolved_tile_shape_yx": self.tile_shape_yx,
                "resolved_overlap_yx": self.overlap_yx,
                "blend": self.config.blend,
                "pad_mode": self.config.pad_mode,
                "tile_count": len(self.placements),
                "order": "row-major",
                "placements": tuple(
                    placement.as_provenance() for placement in self.placements
                ),
            }
        )


class TileSolver(Protocol):
    """Persistent DPCT/qOBT tile solver consumed by the offline tiler."""

    @property
    def model(self) -> Literal["dpct", "qobt"]:
        """Return the admitted quantitative model identity."""

    @property
    def shots_num(self) -> int:
        """Return the exact illumination-shot count required per Z plane."""

    @property
    def gpu_telemetry(self) -> Mapping[str, Any]:
        """Return bounded current worker-owned GPU telemetry."""

    def prepare(self, shape_zyx: tuple[int, int, int]) -> None:
        """Prepare shape-dependent operators for one full-Z XY tile."""

    def reconstruct_tile(
        self,
        sample_zsyx: npt.NDArray[Any],
        background_syx: npt.NDArray[Any] | None,
    ) -> npt.NDArray[Any]:
        """Reconstruct one full-Z/full-shot tile into a ZYX volume."""

    def release_inputs(self) -> None:
        """Release per-tile arrays while retaining prepared operators."""

    def close(self) -> None:
        """Release persistent solver resources idempotently."""


@dataclass(frozen=True, slots=True)
class TiledReconstructionResult:
    """Completed blended volume with resolved plan and execution provenance."""

    volume_zyx: npt.NDArray[np.float32]
    plan: TilePlan
    model: Literal["dpct", "qobt"]
    shots_num: int
    tile_timing_ns: tuple[int, ...]
    total_timing_ns: int
    gpu_telemetry: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze telemetry and verify timing/output completeness."""
        if self.volume_zyx.shape != self.plan.full_shape_zyx:
            raise ValueError("tiled result shape does not match its resolved plan")
        if self.shots_num <= 0:
            raise ValueError("tiled result shot count must be positive")
        if len(self.tile_timing_ns) != len(self.plan.placements):
            raise ValueError("tiled result requires one timing per placement")
        if self.total_timing_ns < 0 or any(value < 0 for value in self.tile_timing_ns):
            raise ValueError("tiled result timings cannot be negative")
        object.__setattr__(
            self, "gpu_telemetry", MappingProxyType(dict(self.gpu_telemetry))
        )

    def as_provenance(self) -> Mapping[str, Any]:
        """Return complete JSON-compatible execution provenance."""
        return MappingProxyType(
            {
                "schema_version": 1,
                "axes": {
                    "sample": "ZSYX",
                    "background": "SYX|CSYX",
                    "output": "ZYX",
                },
                "model": self.model,
                "input_shape_zsyx": (
                    self.plan.full_shape_zyx[0],
                    self.shots_num,
                    self.plan.full_shape_zyx[1],
                    self.plan.full_shape_zyx[2],
                ),
                "output_shape_zyx": self.plan.full_shape_zyx,
                "tiling": self.plan.as_provenance(),
                "timing_ns": {
                    "total": self.total_timing_ns,
                    "tiles": self.tile_timing_ns,
                },
                "gpu": self.gpu_telemetry,
            }
        )


class TiledVolumeReconstructor:
    """Run one persistent DPCT/qOBT solver over a deterministic serial plan."""

    def __init__(self, solver: TileSolver, config: TilingConfig) -> None:
        if solver.model not in {"dpct", "qobt"}:
            raise ValueError("tiled reconstruction supports only DPCT and qOBT")
        if solver.shots_num <= 0:
            raise ValueError("tiled solver shot count must be positive")
        self.solver = solver
        self.config = config
        self._prepared_shape: tuple[int, int, int] | None = None
        self._closed = False

    def run(
        self,
        sample_zsyx: npt.ArrayLike,
        background: npt.ArrayLike | None = None,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[int, int, TilePlacement], None] | None = None,
    ) -> TiledReconstructionResult:
        """Reconstruct all tiles, normalize coverage, and retain provenance."""
        if self._closed:
            raise RuntimeError("tiled reconstructor is closed")
        sample = np.asarray(sample_zsyx)
        if sample.ndim != 4:
            raise ValueError(
                f"sample input must have ZSYX axes; received {sample.shape}"
            )
        nz, shots, ny, nx = sample.shape
        if shots != self.solver.shots_num:
            raise ValueError(
                f"sample contains {shots} shots but {self.solver.model} requires "
                f"{self.solver.shots_num}"
            )
        background_array = _validate_background(background, shots, ny, nx)
        plan = resolve_tile_plan((nz, ny, nx), self.config)
        prepared_shape = (nz, *plan.tile_shape_yx)
        if self._prepared_shape != prepared_shape:
            try:
                self.solver.prepare(prepared_shape)
            except Exception as error:
                _raise_tile_error(error, plan.tile_shape_yx, self.config, "preparing")
                raise
            self._prepared_shape = prepared_shape

        accumulator = np.zeros((nz, ny, nx), dtype=np.float32)
        weight_sum = np.zeros((ny, nx), dtype=np.float32)
        timings: list[int] = []
        started_ns = time.perf_counter_ns()
        try:
            for position, placement in enumerate(plan.placements, start=1):
                _raise_if_cancelled(cancelled)
                tile_started_ns = time.perf_counter_ns()
                sample_tile = _extract_tile(sample, placement, self.config.pad_mode)
                background_tile = (
                    None
                    if background_array is None
                    else _extract_tile(
                        background_array, placement, self.config.pad_mode
                    )
                )
                if background_tile is not None and background_tile.ndim == 4:
                    background_tile = np.median(background_tile, axis=0).astype(
                        np.float32
                    )
                try:
                    reconstructed = np.asarray(
                        self.solver.reconstruct_tile(sample_tile, background_tile),
                        dtype=np.float32,
                    )
                except Exception as error:
                    _raise_tile_error(
                        error, plan.tile_shape_yx, self.config, "reconstructing"
                    )
                    raise
                valid_y, valid_x = placement.shape_yx
                reconstructed = reconstructed[:, :valid_y, :valid_x]
                expected = (nz, valid_y, valid_x)
                if reconstructed.shape != expected:
                    raise ValueError(
                        f"tile {placement.index} returned {reconstructed.shape}; "
                        f"expected {expected}"
                    )
                weight = tile_weight(placement)
                target = (
                    slice(None),
                    slice(placement.y_start, placement.y_stop),
                    slice(placement.x_start, placement.x_stop),
                )
                accumulator[target] += reconstructed * weight[None, ...]
                weight_sum[target[1:]] += weight
                self.solver.release_inputs()
                timings.append(time.perf_counter_ns() - tile_started_ns)
                if progress is not None:
                    progress(position, len(plan.placements), placement)
        except Exception:
            self.solver.release_inputs()
            raise

        if not np.all(np.isfinite(weight_sum)) or np.any(weight_sum <= 0):
            raise RuntimeError("tile plan left uncovered or invalid output pixels")
        accumulator /= weight_sum[None, ...]
        if not np.all(np.isfinite(accumulator)):
            raise RuntimeError(
                "blended tiled reconstruction contains non-finite values"
            )
        return TiledReconstructionResult(
            accumulator,
            plan,
            self.solver.model,
            self.solver.shots_num,
            tuple(timings),
            time.perf_counter_ns() - started_ns,
            self.solver.gpu_telemetry,
        )

    def close(self) -> None:
        """Release persistent solver state idempotently."""
        if self._closed:
            return
        self.solver.close()
        self._closed = True


def resolve_tile_plan(
    full_shape_zyx: tuple[int, int, int], config: TilingConfig
) -> TilePlan:
    """Resolve a deterministic row-major XY plan from a post-selection shape."""
    nz, ny, nx = full_shape_zyx
    if min(nz, ny, nx) <= 0:
        raise ValueError("volume dimensions must be positive")
    if config.mode == "off":
        tile_y, tile_x = ny, nx
    elif config.explicit_shape_yx is not None:
        tile_y = min(ny, config.explicit_shape_yx[0])
        tile_x = min(nx, config.explicit_shape_yx[1])
    else:
        edge = _round_down(math.sqrt(config.voxel_budget / nz), config.alignment_px)
        needs_tiling = ny > config.min_shape_yx[0] or nx > config.min_shape_yx[1]
        if needs_tiling and edge < min(config.min_shape_yx):
            raise ValueError(
                "requested Z depth leaves fewer than the minimum XY pixels under "
                "the configured voxel budget; Z tiling is unsupported"
            )
        edge = max(edge, min(config.min_shape_yx))
        tile_y = min(ny, edge, config.max_shape_yx[0])
        tile_x = min(nx, edge, config.max_shape_yx[1])

    if config.mode == "off":
        overlap_y = overlap_x = 0
    elif config.explicit_overlap_yx is not None:
        overlap_y, overlap_x = config.explicit_overlap_yx
    else:
        overlap_y = (
            _round_up(tile_y * config.overlap_fraction, 16) if ny > tile_y else 0
        )
        overlap_x = (
            _round_up(tile_x * config.overlap_fraction, 16) if nx > tile_x else 0
        )
    if ny <= tile_y:
        overlap_y = 0
    if nx <= tile_x:
        overlap_x = 0
    if overlap_y >= tile_y or overlap_x >= tile_x:
        raise ValueError("tile overlap must be smaller than its tile dimension")

    y_starts = _axis_starts(ny, tile_y, overlap_y)
    x_starts = _axis_starts(nx, tile_x, overlap_x)
    placements: list[TilePlacement] = []
    for row, y_start in enumerate(y_starts):
        y_stop = min(y_start + tile_y, ny)
        for column, x_start in enumerate(x_starts):
            x_stop = min(x_start + tile_x, nx)
            placements.append(
                TilePlacement(
                    index=len(placements),
                    row=row,
                    column=column,
                    y_start=y_start,
                    y_stop=y_stop,
                    x_start=x_start,
                    x_stop=x_stop,
                    pad_bottom=tile_y - (y_stop - y_start),
                    pad_right=tile_x - (x_stop - x_start),
                    top_overlap=(
                        0 if row == 0 else y_starts[row - 1] + tile_y - y_start
                    ),
                    bottom_overlap=(
                        0 if row == len(y_starts) - 1 else y_stop - y_starts[row + 1]
                    ),
                    left_overlap=(
                        0 if column == 0 else x_starts[column - 1] + tile_x - x_start
                    ),
                    right_overlap=(
                        0
                        if column == len(x_starts) - 1
                        else x_stop - x_starts[column + 1]
                    ),
                )
            )
    return TilePlan(
        (nz, ny, nx),
        (tile_y, tile_x),
        (overlap_y, overlap_x),
        tuple(placements),
        config,
    )


def tile_weight(placement: TilePlacement) -> npt.NDArray[np.float32]:
    """Return separable complementary raised-cosine weights for one tile."""
    height, width = placement.shape_yx
    y_weight = np.ones(height, dtype=np.float32)
    x_weight = np.ones(width, dtype=np.float32)
    if placement.top_overlap:
        y_weight[: placement.top_overlap] = _cosine_ramp(placement.top_overlap)
    if placement.bottom_overlap:
        y_weight[-placement.bottom_overlap :] = 1.0 - _cosine_ramp(
            placement.bottom_overlap
        )
    if placement.left_overlap:
        x_weight[: placement.left_overlap] = _cosine_ramp(placement.left_overlap)
    if placement.right_overlap:
        x_weight[-placement.right_overlap :] = 1.0 - _cosine_ramp(
            placement.right_overlap
        )
    return y_weight[:, None] * x_weight[None, :]


def _axis_starts(size: int, tile: int, overlap: int) -> tuple[int, ...]:
    if size <= tile:
        return (0,)
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile overlap must leave a positive stride")
    starts = list(range(0, size - tile + 1, stride))
    final_start = size - tile
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _round_down(value: float, alignment: int) -> int:
    return int(value) // alignment * alignment


def _round_up(value: float, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _cosine_ramp(length: int) -> npt.NDArray[np.float32]:
    positions = (np.arange(length, dtype=np.float32) + 0.5) / length
    return (0.5 - 0.5 * np.cos(np.pi * positions)).astype(np.float32)


def _validate_background(
    background: npt.ArrayLike | None, shots: int, ny: int, nx: int
) -> npt.NDArray[Any] | None:
    if background is None:
        return None
    array = np.asarray(background)
    if array.ndim not in {3, 4}:
        raise ValueError("background input must have SYX or CSYX axes")
    if array.shape[-3:] != (shots, ny, nx):
        raise ValueError("background and sample shot/spatial dimensions differ")
    return array


def _extract_tile(
    array: npt.NDArray[Any],
    placement: TilePlacement,
    pad_mode: Literal["reflect", "edge"],
) -> npt.NDArray[Any]:
    tile = np.asarray(
        array[
            ...,
            placement.y_start : placement.y_stop,
            placement.x_start : placement.x_stop,
        ]
    )
    if not placement.pad_bottom and not placement.pad_right:
        return tile
    padding = [(0, 0)] * tile.ndim
    padding[-2] = (0, placement.pad_bottom)
    padding[-1] = (0, placement.pad_right)
    effective_mode = (
        "edge"
        if pad_mode == "reflect" and (tile.shape[-2] == 1 or tile.shape[-1] == 1)
        else pad_mode
    )
    return np.pad(tile, padding, mode=effective_mode)


def _raise_if_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise ProcessingCancelled("tiled reconstruction was cancelled")


def _raise_tile_error(
    error: Exception,
    tile_shape_yx: tuple[int, int],
    config: TilingConfig,
    operation: str,
) -> None:
    if (
        not isinstance(error, MemoryError)
        and type(error).__name__ != "OutOfMemoryError"
    ):
        return
    recommended = tuple(
        max(minimum, _round_down(size * 0.8, config.alignment_px))
        for size, minimum in zip(tile_shape_yx, config.min_shape_yx, strict=True)
    )
    raise RuntimeError(
        f"GPU memory was exhausted while {operation} a tile; retry with an "
        f"explicit tile shape of {recommended}; geometry is not changed automatically"
    ) from error


__all__ = [
    "DEFAULT_VOXEL_BUDGET",
    "TilePlacement",
    "TilePlan",
    "TileSolver",
    "TiledReconstructionResult",
    "TiledVolumeReconstructor",
    "TilingConfig",
    "resolve_tile_plan",
    "tile_weight",
]
