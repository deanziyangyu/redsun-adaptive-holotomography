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


class BufferUsage(StrEnum):
    """Shared-memory traffic classes with distinct loss guarantees."""

    PREVIEW = "preview"
    ACQUISITION = "acquisition"


class ServiceConnectionState(StrEnum):
    """Connection lifecycle reported by an isolated hardware service."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    FAULTED = "faulted"


class AcquisitionState(StrEnum):
    """Detector acquisition lifecycle reported by a hardware service."""

    IDLE = "idle"
    ARMED = "armed"
    ACQUIRING = "acquiring"


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


class ProcessingExecutionMode(StrEnum):
    """Hardware-free or run-coupled processing composition mode."""

    LIVE = "live"
    OFFLINE = "offline"
    REPLAY = "replay"


class MultiSliceAcquisitionMode(StrEnum):
    """Input-domain contract for multi-slice reconstruction measurements.

    The AHT instrument acquires real-valued amplitudes at several physical
    focal positions.  The donor interferometric setup instead acquires a
    complex field (real amplitude and imaginary phase) and can numerically
    refocus it.  Keeping this distinction in the data contract prevents the
    solver from silently applying interferometric preprocessing to AHT data.
    """

    AHT_REAL_AMPLITUDE_FOCAL_STACK = "aht_real_amplitude_focal_stack"
    INTERFEROMETRIC_COMPLEX_FIELD = "interferometric_complex_field"

    @property
    def uses_numerical_refocusing(self) -> bool:
        """Whether the acquisition requires digital refocusing before solve."""
        return self is MultiSliceAcquisitionMode.INTERFEROMETRIC_COMPLEX_FIELD

    @property
    def requires_complex_field(self) -> bool:
        """Whether both real and imaginary field components are required."""
        return self is MultiSliceAcquisitionMode.INTERFEROMETRIC_COMPLEX_FIELD


class SolverState(StrEnum):
    """Explicit detached solver lifecycle states."""

    DISABLED = "disabled"
    STARTING = "starting"
    READY = "ready"
    WAITING_INPUT = "waiting_input"
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    RUNNING = "running"
    PUBLISHING = "publishing"
    PAUSING = "pausing"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"


class QueueStrategy(StrEnum):
    """Per-solver observation retention policy."""

    LOSSLESS = "lossless"
    BOUNDED_SAMPLE = "bounded_sample"
    LATEST_ONLY = "latest_only"


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
class FrameBufferDescriptor:
    """Generation-aware reference to one committed shared-memory frame."""

    service_id: str
    run_id: str
    frame_id: str
    sequence: int
    generation: int
    shared_memory_name: str
    slot: int
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    timestamp_ns: int
    checksum: str
    usage: BufferUsage
    lease_timeout_ns: int

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.generation < 0 or self.slot < 0:
            raise ValueError(
                "buffer sequence, generation, and slot must be non-negative"
            )
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("buffer shape dimensions must be positive")
        if self.lease_timeout_ns <= 0:
            raise ValueError("lease_timeout_ns must be positive")


@dataclass(frozen=True, slots=True)
class FrameLease:
    """Bounded consumer authority over one frame descriptor."""

    lease_id: str
    descriptor: FrameBufferDescriptor
    expires_ns: int


@dataclass(frozen=True, slots=True)
class HardwareServiceStatus:
    """Inspectable health snapshot for one isolated detector service."""

    service_id: str
    epics_prefix: str
    schema_generation: int
    heartbeat_ns: int
    connection_state: ServiceConnectionState
    acquisition_state: AcquisitionState
    command_state: str
    frames_published: int
    frames_dropped: int
    last_error: str | None = None


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
        if not all(
            (
                self.job_id,
                self.solver_id,
                self.solver_version,
                self.input_selectors,
                self.latency_class,
                self.output_destination,
                self.cancellation_policy,
            )
        ):
            raise ValueError("processing job identities and selectors cannot be empty")
        if self.priority < 0:
            raise ValueError("processing job priority must be non-negative")
        object.__setattr__(
            self, "configuration", freeze_json_mapping(self.configuration)
        )
        object.__setattr__(
            self, "resource_request", freeze_json_mapping(self.resource_request)
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """Immutable committed input reference shared by live and offline modes."""

    run_id: str
    observation_id: str
    detector_id: str
    channel_id: str
    sequence: int
    array: ArrayReference
    committed: bool
    configuration_revision: str
    calibration_ids: tuple[str, ...]
    replay_key: str
    latency_class: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(
            (
                self.run_id,
                self.observation_id,
                self.detector_id,
                self.channel_id,
                self.configuration_revision,
                self.replay_key,
                self.latency_class,
            )
        ):
            raise ValueError("observation identities cannot be empty")
        if self.sequence < 0:
            raise ValueError("observation sequence must be non-negative")
        if not self.committed:
            raise ValueError("detached processing accepts only committed observations")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SolverCapabilities:
    """Inspectable admission contract for one solver plugin version."""

    solver_id: str
    solver_version: str
    dimensionality: tuple[int, ...]
    required_channels: int
    required_shots: int
    streaming_supported: bool
    batch_supported: bool
    incremental_supported: bool
    live_supported: bool
    output_types: tuple[str, ...]
    requires_gpu: bool = False
    gpu_runtime: str | None = None

    def __post_init__(self) -> None:
        if not self.solver_id or not self.solver_version or not self.output_types:
            raise ValueError("solver identity, version, and outputs are required")
        if (
            not self.dimensionality
            or any(dimension <= 0 for dimension in self.dimensionality)
            or self.required_channels <= 0
            or self.required_shots <= 0
        ):
            raise ValueError("solver input requirements must be positive")
        if self.requires_gpu != (self.gpu_runtime is not None):
            raise ValueError("GPU solver capability and runtime must agree")


@dataclass(frozen=True, slots=True)
class SolverStatus:
    """One detached worker health and progress snapshot."""

    solver_id: str
    state: SolverState
    heartbeat_ns: int
    progress: float
    input_cursor: int
    output_cursor: int
    queue_depth: int
    worker_pid: int | None
    gpu_id: int | None
    allocated_memory_bytes: int
    observed_memory_bytes: int
    restart_count: int
    cancellation_requested: bool
    last_error: str | None = None

    def __post_init__(self) -> None:
        if not self.solver_id:
            raise ValueError("solver status identity cannot be empty")
        if not 0 <= self.progress <= 1:
            raise ValueError("solver progress must be between zero and one")
        if any(
            value < 0
            for value in (
                self.heartbeat_ns,
                self.input_cursor,
                self.output_cursor,
                self.queue_depth,
                self.allocated_memory_bytes,
                self.observed_memory_bytes,
                self.restart_count,
            )
        ):
            raise ValueError("solver counters and resource values cannot be negative")


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Immutable published output from one solver worker."""

    result_id: str
    job_id: str
    run_id: str
    solver_id: str
    solver_version: str
    output: ArrayReference
    metrics: Mapping[str, JsonValue]
    input_checksums: tuple[str, ...]
    configuration_fingerprint: str
    started_ns: int
    completed_ns: int
    worker_pid: int
    cached: bool = False

    def __post_init__(self) -> None:
        if not all(
            (
                self.result_id,
                self.job_id,
                self.run_id,
                self.solver_id,
                self.solver_version,
                self.input_checksums,
                self.configuration_fingerprint,
            )
        ):
            raise ValueError("processing result identities and inputs cannot be empty")
        if self.started_ns < 0 or self.completed_ns < self.started_ns:
            raise ValueError("processing result timestamps are not ordered")
        if self.worker_pid <= 0:
            raise ValueError("processing result worker PID must be positive")
        object.__setattr__(self, "metrics", freeze_json_mapping(self.metrics))


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
