"""Immutable values that cross AHT component boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | Sequence[JsonValue] | Mapping[str, JsonValue]
type FrozenJsonValue = (
    JsonScalar | tuple[FrozenJsonValue, ...] | Mapping[str, FrozenJsonValue]
)


def freeze_json(value: JsonValue) -> FrozenJsonValue:
    """Recursively copy JSON-compatible input into immutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(freeze_json(item) for item in value)
    return value


def freeze_json_mapping(
    value: Mapping[str, JsonValue],
) -> Mapping[str, FrozenJsonValue]:
    """Recursively copy a JSON object into an immutable mapping."""
    frozen = freeze_json(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - type invariant
        raise TypeError("expected a JSON object")
    return frozen


def thaw_json(value: JsonValue) -> object:
    """Convert an immutable JSON value to standard serializable containers."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [thaw_json(item) for item in value]
    return value


class BackendKind(StrEnum):
    """Supported per-device staging backends."""

    REAL = "real"
    MOCK = "mock"
    REPLAY = "replay"
    DISABLED = "disabled"


class CalibrationKind(StrEnum):
    """Immutable calibration artifact categories."""

    DARK = "dark"
    FLAT = "flat"
    MEDIAN = "median"
    REGISTRATION = "registration"
    OPTICAL = "optical"


class RunState(StrEnum):
    """Deterministic acquisition lifecycle states."""

    NEW = "new"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    RECOVERING = "recovering"
    COMPLETED = "completed"


class RunEventKind(StrEnum):
    """Events accepted by the run-state reducer."""

    STARTED = "started"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"
    RECOVERY_STARTED = "recovery_started"
    RECOVERED = "recovered"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ArrayReference:
    """Reference to an immutable externally stored array."""

    uri: str
    checksum: str
    shape: tuple[int, ...]
    dtype: str
    byte_order: str = "="


@dataclass(frozen=True, slots=True)
class DetectorCapabilities:
    """A detector's discoverable acquisition constraints."""

    trigger_modes: frozenset[str]
    exposure_range_s: tuple[float, float]
    pixel_formats: frozenset[str]
    supports_hardware_timestamps: bool
    supports_roi: bool
    binning: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        minimum, maximum = self.exposure_range_s
        if minimum < 0 or maximum < minimum:
            raise ValueError("exposure_range_s must be ordered and non-negative")


@dataclass(frozen=True, slots=True)
class Frame:
    """Traceable acquired-frame metadata and payload reference."""

    run_id: str
    frame_id: str
    detector_id: str
    channel_id: str
    sequence: int
    exposure_started_ns: int
    exposure_ended_ns: int
    array: ArrayReference
    configuration_revision: str
    quality_flags: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PoseSample:
    """Measured or commanded pose in an explicitly named frame."""

    monotonic_ns: int
    sample_kind: str
    joints: tuple[float, ...]
    joint_units: tuple[str, ...]
    coordinate_frame: str
    transform_to_parent: tuple[float, ...]
    transform_units: str
    uncertainty: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        if len(self.transform_to_parent) != 16:
            raise ValueError("transform_to_parent must contain a row-major 4x4 matrix")
        if len(self.joints) != len(self.joint_units):
            raise ValueError("every joint must have an explicit unit")


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """Immutable calibration selection and its validity scope."""

    artifact_id: str
    kind: CalibrationKind
    validity_scope: Mapping[str, JsonValue]
    source_run_id: str
    settings_fingerprint: str
    created_ns: int
    payload: ArrayReference

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "validity_scope", freeze_json_mapping(self.validity_scope)
        )


@dataclass(frozen=True, slots=True)
class ProcessingJob:
    """Immutable request for a detached local processing worker."""

    job_id: str
    solver_id: str
    solver_version: str
    input_selectors: tuple[str, ...]
    configuration: Mapping[str, JsonValue]
    priority: int
    latency_class: str
    resource_request: Mapping[str, JsonValue]
    output_destination: str
    cancellation_policy: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "configuration", freeze_json_mapping(self.configuration)
        )
        object.__setattr__(
            self, "resource_request", freeze_json_mapping(self.resource_request)
        )


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One ordered append-only lifecycle event."""

    run_id: str
    sequence: int
    kind: RunEventKind
    timestamp_ns: int
    detail: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", freeze_json_mapping(self.detail))


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Replayable materialized run state."""

    run_id: str
    state: RunState = RunState.NEW
    last_sequence: int = -1
    failure: str | None = None
