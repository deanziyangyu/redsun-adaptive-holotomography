"""Validated configuration for offline RI analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SobelConfig:
    """Physical Sobel edge-evidence controls."""

    pre_smoothing_sigma_um: float = 0.0
    edge_percentile: float = 90.0
    edge_close_radius_um: float = 0.25

    def __post_init__(self) -> None:
        """Validate physical Sobel controls."""
        if self.pre_smoothing_sigma_um < 0 or self.edge_close_radius_um < 0:
            raise ValueError("Sobel radii must be non-negative")
        if not 0.0 < self.edge_percentile < 100.0:
            raise ValueError("edge_percentile must be strictly between 0 and 100")


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """Candidate generation and donor-derived morphology controls."""

    strategy: Literal["ri_only", "edge_only", "hybrid"] = "hybrid"
    seed_mode: Literal["otsu_high", "manual_window"] = "otsu_high"
    ri_lower: float | None = None
    ri_upper: float | None = None
    close_radius_um: float = 0.25
    open_radius_um: float = 0.0
    dilation_radius_um: float = 0.0
    min_component_volume_um3: float = 1.0
    connectivity: Literal[6, 18, 26] = 26
    refinement_max_component_voxels: int | None = None
    refinement_window_um: float = 1.0
    refinement_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Validate segmentation controls and dependent options."""
        if self.strategy not in {"ri_only", "edge_only", "hybrid"}:
            raise ValueError("unsupported segmentation strategy")
        if self.seed_mode not in {"otsu_high", "manual_window"}:
            raise ValueError("unsupported segmentation seed mode")
        if self.seed_mode == "manual_window":
            if self.ri_lower is None or self.ri_upper is None:
                raise ValueError("manual_window requires ri_lower and ri_upper")
            if self.ri_upper < self.ri_lower:
                raise ValueError("ri_upper must be >= ri_lower")
        if (
            min(
                self.close_radius_um,
                self.open_radius_um,
                self.dilation_radius_um,
                self.min_component_volume_um3,
            )
            < 0
        ):
            raise ValueError("morphology radii and minimum volume must be non-negative")
        if self.refinement_max_component_voxels is not None:
            if self.refinement_max_component_voxels < 1:
                raise ValueError("refinement_max_component_voxels must be positive")
            if self.refinement_window_um <= 0 or self.refinement_multiplier <= 0:
                raise ValueError("refinement controls must be positive")


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Physical subvolume grid controls in canonical Z/Y/X order."""

    cell_size_zyx_um: tuple[float, float, float] = (5.0, 5.0, 5.0)

    def __post_init__(self) -> None:
        """Validate a positive physical grid cell."""
        if len(self.cell_size_zyx_um) != 3 or any(
            value <= 0 for value in self.cell_size_zyx_um
        ):
            raise ValueError("cell_size_zyx_um must contain three positive values")


@dataclass(frozen=True, slots=True)
class HistogramConfig:
    """One shared RI histogram domain for all reported populations."""

    ri_min: float = 1.0
    ri_max: float = 2.0
    bin_count: int = 128

    def __post_init__(self) -> None:
        """Validate the common histogram domain."""
        if self.ri_max <= self.ri_min:
            raise ValueError("histogram ri_max must be greater than ri_min")
        if self.bin_count < 2:
            raise ValueError("histogram bin_count must be at least two")


@dataclass(frozen=True, slots=True)
class LabelFilterConfig:
    """Inclusive voxel-count bounds applied after component labeling."""

    min_voxel_count: int = 0
    max_voxel_count: int | None = None

    def __post_init__(self) -> None:
        """Require non-negative, ordered component-size bounds."""
        if self.min_voxel_count < 0:
            raise ValueError("min_voxel_count must be non-negative")
        if (
            self.max_voxel_count is not None
            and self.max_voxel_count < self.min_voxel_count
        ):
            raise ValueError("max_voxel_count must be >= min_voxel_count")


@dataclass(frozen=True, slots=True)
class RiAnalysisConfig:
    """Complete immutable configuration for the TIFF-to-analysis workflow."""

    schema_version: int = SCHEMA_VERSION
    sobel: SobelConfig = field(default_factory=SobelConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    label_filter: LabelFilterConfig = field(default_factory=LabelFilterConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    histogram: HistogramConfig = field(default_factory=HistogramConfig)

    def __post_init__(self) -> None:
        """Reject unsupported schema revisions."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported RI analysis schema {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )

    @classmethod
    def from_mapping(cls, value: object) -> RiAnalysisConfig:
        """Construct a strictly validated configuration from JSON data."""
        if not isinstance(value, dict):
            raise ValueError("RI analysis config must be a JSON object")
        allowed = {
            "schema_version",
            "sobel",
            "segmentation",
            "label_filter",
            "grid",
            "histogram",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown RI analysis config keys: {', '.join(unknown)}")
        sobel = _section(value.get("sobel"), SobelConfig, "sobel")
        segmentation = _section(
            value.get("segmentation"), SegmentationConfig, "segmentation"
        )
        label_filter = _section(
            value.get("label_filter"), LabelFilterConfig, "label_filter"
        )
        grid_data = _strict_section(value.get("grid"), GridConfig, "grid")
        if "cell_size_zyx_um" in grid_data:
            grid_data["cell_size_zyx_um"] = _triple(
                grid_data["cell_size_zyx_um"], "grid"
            )
        grid = GridConfig(**grid_data)
        histogram = _section(value.get("histogram"), HistogramConfig, "histogram")
        return cls(
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
            sobel=sobel,
            segmentation=segmentation,
            label_filter=label_filter,
            grid=grid,
            histogram=histogram,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""
        return asdict(self)


def _section(value: object, cls: type[Any], name: str) -> Any:
    """Construct a strict dataclass section from an optional mapping."""
    return cls(**_strict_section(value, cls, name))


def _strict_section(value: object, cls: type[Any], name: str) -> dict[str, Any]:
    data = _mapping(value, name)
    unknown = sorted(set(data) - {item.name for item in fields(cls)})
    if unknown:
        raise ValueError(f"unknown {name} config keys: {', '.join(unknown)}")
    return data


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{name}.cell_size_zyx_um must contain exactly three values")
    converted = tuple(_number(item, name) for item in value)
    return converted[0], converted[1], converted[2]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name}.cell_size_zyx_um values must be numeric")
    return float(value)
