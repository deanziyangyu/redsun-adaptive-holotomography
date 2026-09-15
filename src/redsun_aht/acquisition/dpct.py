"""Traceable simulation-first DPCT acquisition orchestration."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

import numpy as np

from redsun_aht.motion import list_scan

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from redsun_aht.domain import Frame, PoseSample
    from redsun_aht.domain.models import JsonValue
    from redsun_aht.motion import ScanPoint
    from redsun_aht.protocols import CompositeStage


class DpctDetector(Protocol):
    """Lossless detector lifecycle required by the DPCT runner."""

    @property
    def service_id(self) -> str:
        """Return the detector's stable identity."""

    async def connect(self) -> None:
        """Connect the isolated detector service."""

    async def configure(self, settings: Mapping[str, JsonValue]) -> str:
        """Apply bounded settings and return a configuration revision."""

    async def arm(self) -> None:
        """Arm acquisition."""

    async def trigger(self) -> None:
        """Acquire one losslessly retained frame."""

    async def read(self) -> Frame:
        """Read metadata for the latest committed frame."""

    async def copy(self, sequence: int) -> npt.NDArray[Any]:
        """Copy the exact retained frame without acknowledging it."""

    async def acknowledge(self, sequence: int) -> None:
        """Acknowledge the exact copied frame sequence."""

    async def stop(self) -> None:
        """Stop ordinary acquisition."""

    async def disconnect(self) -> None:
        """Disconnect and release service-owned resources."""


class PatternController(Protocol):
    """Synchronous pattern selection boundary owned by the MCU host client."""

    def show_pattern(self, pattern_id: int) -> None:
        """Select one committed illumination pattern."""

    def all_off(self) -> None:
        """Clear ordinary illumination output."""


class DpctFrameSink(Protocol):
    """Durable destination that must commit payloads before acknowledgement."""

    @property
    def uri(self) -> str:
        """Return the stable run-bundle URI."""

    def begin(self, recipe: DpctRecipe, detector_ids: tuple[str, ...]) -> None:
        """Create an incomplete run bundle before the first frame."""

    def write_shot(
        self,
        shot: DpctShotRecord,
        arrays: tuple[npt.NDArray[Any], ...],
    ) -> None:
        """Durably commit one ordered detector group."""

    def complete(self, result: DpctRunResult) -> None:
        """Publish the immutable completion manifest."""

    def fail(self, error: BaseException) -> None:
        """Mark an incomplete bundle without publishing completion."""


@dataclass(frozen=True, slots=True)
class ExternalAssetInfo:
    """Backend-neutral location used to compose one Bluesky Resource."""

    spec: str
    root: str
    resource_path: str
    resource_kwargs: Mapping[str, JsonValue]
    path_semantics: Literal["posix", "windows"]


@runtime_checkable
class DpctExternalAssetSink(Protocol):
    """Optional sink capability for standard Bluesky external-asset documents."""

    def external_asset_info(self, detector_id: str) -> ExternalAssetInfo:
        """Describe the durable asset containing one detector's frames."""


@dataclass(frozen=True, slots=True)
class DpctRecipe:
    """Bounded scan/pattern product and common detector configuration."""

    run_id: str
    scan_points: tuple[ScanPoint, ...]
    pattern_ids: tuple[int, ...]
    detector_settings: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        """Detach caller-owned settings and validate acquisition axes."""
        if not self.run_id:
            raise ValueError("DPCT run_id cannot be empty")
        if not self.scan_points or not self.pattern_ids:
            raise ValueError("DPCT requires scan points and illumination patterns")
        if len(set(self.pattern_ids)) != len(self.pattern_ids):
            raise ValueError("DPCT pattern IDs must be unique")
        if any(not 0 <= pattern_id <= 0xFFFF for pattern_id in self.pattern_ids):
            raise ValueError("DPCT pattern IDs must fit unsigned 16-bit integers")
        object.__setattr__(
            self, "detector_settings", MappingProxyType(dict(self.detector_settings))
        )


@dataclass(frozen=True, slots=True)
class DpctShotRecord:
    """One settled pose, displayed pattern, and simultaneous detector group."""

    shot_index: int
    scan_index: int
    pattern_id: int
    pose: PoseSample
    frames: tuple[Frame, ...]


@dataclass(frozen=True, slots=True)
class DpctRunResult:
    """Completed immutable shot sequence for one run."""

    run_id: str
    detector_ids: tuple[str, ...]
    shots: tuple[DpctShotRecord, ...]
    storage_uri: str | None = None


@dataclass(frozen=True, slots=True)
class TimingDistribution:
    """Descriptive statistics for one non-empty sequence of intervals."""

    count: int
    minimum_ns: int
    maximum_ns: int
    mean_ns: float
    median_ns: float
    p95_ns: float
    standard_deviation_ns: float

    def __post_init__(self) -> None:
        """Reject invalid or incomplete interval summaries."""
        values = (
            self.mean_ns,
            self.median_ns,
            self.p95_ns,
            self.standard_deviation_ns,
        )
        if self.count <= 0 or self.minimum_ns <= 0 or self.maximum_ns <= 0:
            raise ValueError("timing distribution requires positive intervals")
        if self.minimum_ns > self.maximum_ns or any(
            not np.isfinite(value) or value < 0 for value in values
        ):
            raise ValueError("timing distribution is invalid")

    def as_dict(self) -> dict[str, float | int]:
        """Return JSON-compatible interval statistics in nanoseconds."""
        return {
            "count": self.count,
            "minimum_ns": self.minimum_ns,
            "maximum_ns": self.maximum_ns,
            "mean_ns": self.mean_ns,
            "median_ns": self.median_ns,
            "p95_ns": self.p95_ns,
            "standard_deviation_ns": self.standard_deviation_ns,
        }


@dataclass(frozen=True, slots=True)
class MultishotTimingProfile:
    """Measured acquisition cadence for one completed finite multishot plan."""

    elapsed_ns: int
    shot_count: int
    detector_frame_rates_hz: Mapping[str, float]
    detector_capture_duration_ns: Mapping[str, int]
    detector_frame_intervals: Mapping[str, TimingDistribution]
    detector_pattern_intervals: Mapping[str, TimingDistribution]
    detector_scan_transition_intervals: Mapping[str, TimingDistribution]

    def __post_init__(self) -> None:
        """Reject incomplete timing summaries."""
        if self.elapsed_ns < 0 or self.shot_count <= 0:
            raise ValueError("multishot timing must describe completed shots")
        if any(
            rate <= 0 or not np.isfinite(rate)
            for rate in self.detector_frame_rates_hz.values()
        ):
            raise ValueError("multishot frame rates must be finite and positive")
        object.__setattr__(
            self,
            "detector_frame_rates_hz",
            MappingProxyType(dict(self.detector_frame_rates_hz)),
        )
        if set(self.detector_frame_rates_hz) != set(self.detector_capture_duration_ns):
            raise ValueError("timing profile detectors do not match")
        if set(self.detector_frame_rates_hz) != set(self.detector_frame_intervals):
            raise ValueError("timing profile frame intervals do not match")
        if set(self.detector_frame_rates_hz) != set(self.detector_pattern_intervals):
            raise ValueError("timing profile pattern intervals do not match")
        if set(self.detector_frame_rates_hz) != set(
            self.detector_scan_transition_intervals
        ):
            raise ValueError("timing profile scan transitions do not match")
        if any(value <= 0 for value in self.detector_capture_duration_ns.values()):
            raise ValueError("timing profile capture duration must be positive")
        object.__setattr__(
            self,
            "detector_capture_duration_ns",
            MappingProxyType(dict(self.detector_capture_duration_ns)),
        )
        object.__setattr__(
            self,
            "detector_frame_intervals",
            MappingProxyType(dict(self.detector_frame_intervals)),
        )
        object.__setattr__(
            self,
            "detector_pattern_intervals",
            MappingProxyType(dict(self.detector_pattern_intervals)),
        )
        object.__setattr__(
            self,
            "detector_scan_transition_intervals",
            MappingProxyType(dict(self.detector_scan_transition_intervals)),
        )

    def as_dict(self) -> dict[str, object]:
        """Return the full throughput profile in JSON-compatible units."""
        return {
            "full_plan_elapsed_ns": self.elapsed_ns,
            "shot_count": self.shot_count,
            "detectors": {
                detector_id: {
                    "capture_duration_ns": self.detector_capture_duration_ns[
                        detector_id
                    ],
                    "capture_rate_hz": self.detector_frame_rates_hz[detector_id],
                    "frame_intervals": self.detector_frame_intervals[
                        detector_id
                    ].as_dict(),
                    "within_scan_point_pattern_intervals": (
                        self.detector_pattern_intervals[detector_id].as_dict()
                    ),
                    "scan_point_transition_intervals": (
                        self.detector_scan_transition_intervals[detector_id].as_dict()
                    ),
                }
                for detector_id in self.detector_frame_rates_hz
            },
        }


@dataclass(slots=True)
class MultishotListScanPlan:
    """Coordinate an explicit finite list scan and its four-shot acquisition."""

    stage: CompositeStage
    illumination: PatternController
    detectors: tuple[DpctDetector, ...]
    sink: DpctFrameSink | None = None

    def __post_init__(self) -> None:
        """Require at least one uniquely identified detector."""
        detector_ids = tuple(detector.service_id for detector in self.detectors)
        if not detector_ids or len(set(detector_ids)) != len(detector_ids):
            raise ValueError("DPCT detectors must be non-empty and uniquely identified")

    @staticmethod
    def four_shot_fixed_step_recipe(
        run_id: str,
        *,
        fixed_positions: Mapping[str, float],
        z_start_mm: float,
        z_step_um: float,
        z_positions: int,
        pattern_ids: tuple[int, int, int, int],
        detector_settings: Mapping[str, JsonValue],
    ) -> DpctRecipe:
        """Build the finite four-shot Z list scan used for fixed-median DPCT."""
        if len(pattern_ids) != 4 or len(set(pattern_ids)) != 4:
            raise ValueError("fixed-median DPCT requires exactly four patterns")
        if z_positions <= 0:
            raise ValueError("fixed-median DPCT requires positive Z positions")
        if not np.isfinite(z_start_mm) or not np.isfinite(z_step_um):
            raise ValueError("fixed-median DPCT Z positions must be finite")
        step_mm = z_step_um / 1000.0
        axes = {
            name: tuple(float(value) for _ in range(z_positions))
            for name, value in fixed_positions.items()
        }
        if "z" in axes:
            raise ValueError("fixed_positions must not include the scanned z axis")
        axes["z"] = tuple(
            z_start_mm + position * step_mm for position in range(z_positions)
        )
        return DpctRecipe(
            run_id=run_id,
            scan_points=list_scan(axes),
            pattern_ids=pattern_ids,
            detector_settings=detector_settings,
        )

    @staticmethod
    def four_shot_parked_recipe(
        run_id: str,
        *,
        parked_positions: Mapping[str, float],
        grouped_sequences: int,
        pattern_ids: tuple[int, int, int, int],
        detector_settings: Mapping[str, JsonValue],
    ) -> DpctRecipe:
        """Build repeated four-shot groups while retaining one parked stage pose."""
        if len(pattern_ids) != 4 or len(set(pattern_ids)) != 4:
            raise ValueError("fixed-median DPCT requires exactly four patterns")
        if grouped_sequences <= 0:
            raise ValueError("parked multishot requires positive grouped sequences")
        if not parked_positions:
            raise ValueError("parked multishot requires an explicit stage pose")
        axes = {
            name: tuple(float(value) for _ in range(grouped_sequences))
            for name, value in parked_positions.items()
        }
        return DpctRecipe(
            run_id=run_id,
            scan_points=list_scan(axes),
            pattern_ids=pattern_ids,
            detector_settings=detector_settings,
        )

    async def run(self, recipe: DpctRecipe) -> DpctRunResult:
        """Execute pose outer, pattern inner ordering with bounded cleanup."""
        shots: list[DpctShotRecord] = []
        sink_started = False
        try:
            for detector in self.detectors:
                await detector.connect()
                await detector.configure(recipe.detector_settings)
                await detector.arm()

            if self.sink is not None:
                detector_ids = tuple(detector.service_id for detector in self.detectors)
                sink_started = True
                await asyncio.to_thread(self.sink.begin, recipe, detector_ids)

            for scan_point in recipe.scan_points:
                await self.stage.move(scan_point.as_dict())
                pose = await self.stage.settle()
                for pattern_id in recipe.pattern_ids:
                    await asyncio.to_thread(self.illumination.show_pattern, pattern_id)
                    await asyncio.gather(
                        *(detector.trigger() for detector in self.detectors)
                    )
                    frame_list = [await detector.read() for detector in self.detectors]
                    frames = tuple(frame_list)
                    self._validate_frames(recipe.run_id, frames)
                    arrays = tuple(
                        await asyncio.gather(
                            *(
                                detector.copy(frame.sequence)
                                for detector, frame in zip(
                                    self.detectors, frames, strict=True
                                )
                            )
                        )
                    )
                    self._validate_arrays(frames, arrays)
                    shot = DpctShotRecord(
                        shot_index=len(shots),
                        scan_index=scan_point.index,
                        pattern_id=pattern_id,
                        pose=pose,
                        frames=frames,
                    )
                    if self.sink is not None:
                        await asyncio.to_thread(self.sink.write_shot, shot, arrays)
                    await asyncio.gather(
                        *(
                            detector.acknowledge(frame.sequence)
                            for detector, frame in zip(
                                self.detectors, frames, strict=True
                            )
                        )
                    )
                    shots.append(shot)
            result = DpctRunResult(
                run_id=recipe.run_id,
                detector_ids=tuple(detector.service_id for detector in self.detectors),
                shots=tuple(shots),
                storage_uri=None if self.sink is None else self.sink.uri,
            )
            if self.sink is not None:
                await asyncio.to_thread(self.sink.complete, result)
            return result
        except BaseException as run_error:
            if self.sink is not None and sink_started:
                try:
                    await asyncio.to_thread(self.sink.fail, run_error)
                except BaseException as sink_error:
                    raise BaseExceptionGroup(
                        "DPCT run and storage failure reporting both failed",
                        [run_error, sink_error],
                    ) from None
            raise
        finally:
            await self._cleanup()

    async def run_bare(self, recipe: DpctRecipe) -> DpctRunResult:
        """Profile trigger throughput without copying or durably storing payloads.

        This intentionally records frame metadata only and acknowledges each
        frame immediately. It must not be used for reconstruction input.
        """
        if self.sink is not None:
            raise ValueError("bare multishot profiling cannot use a frame sink")
        shots: list[DpctShotRecord] = []
        previous_positions: dict[str, float] | None = None
        pose: PoseSample | None = None
        try:
            for detector in self.detectors:
                await detector.connect()
                await detector.configure(recipe.detector_settings)
                await detector.arm()

            for scan_point in recipe.scan_points:
                positions = scan_point.as_dict()
                if positions != previous_positions:
                    await self.stage.move(positions)
                    pose = await self.stage.settle()
                    previous_positions = positions
                if pose is None:  # pragma: no cover - first-stage-move invariant
                    raise RuntimeError("bare multishot has no settled stage pose")
                for pattern_id in recipe.pattern_ids:
                    await asyncio.to_thread(self.illumination.show_pattern, pattern_id)
                    await asyncio.gather(
                        *(detector.trigger() for detector in self.detectors)
                    )
                    frame_list = [await detector.read() for detector in self.detectors]
                    frames = tuple(frame_list)
                    self._validate_frames(recipe.run_id, frames)
                    shot = DpctShotRecord(
                        shot_index=len(shots),
                        scan_index=scan_point.index,
                        pattern_id=pattern_id,
                        pose=pose,
                        frames=frames,
                    )
                    await asyncio.gather(
                        *(
                            detector.acknowledge(frame.sequence)
                            for detector, frame in zip(
                                self.detectors, frames, strict=True
                            )
                        )
                    )
                    shots.append(shot)
            return DpctRunResult(
                run_id=recipe.run_id,
                detector_ids=tuple(detector.service_id for detector in self.detectors),
                shots=tuple(shots),
            )
        finally:
            await self._cleanup()

    async def run_profiled(
        self, recipe: DpctRecipe
    ) -> tuple[DpctRunResult, MultishotTimingProfile]:
        """Run one finite plan and derive cadence from committed frame timestamps."""
        started_ns = asyncio.get_running_loop().time()
        result = await self.run(recipe)
        elapsed_ns = int(
            (asyncio.get_running_loop().time() - started_ns) * 1_000_000_000
        )
        return result, self._timing_profile(result, elapsed_ns)

    async def run_bare_profiled(
        self, recipe: DpctRecipe
    ) -> tuple[DpctRunResult, MultishotTimingProfile]:
        """Run one metadata-only plan and return its acquisition throughput."""
        started_ns = asyncio.get_running_loop().time()
        result = await self.run_bare(recipe)
        elapsed_ns = int(
            (asyncio.get_running_loop().time() - started_ns) * 1_000_000_000
        )
        return result, self._timing_profile(result, elapsed_ns)

    @staticmethod
    def _timing_profile(
        result: DpctRunResult, elapsed_ns: int
    ) -> MultishotTimingProfile:
        """Derive frame, within-point, and scan-transition timing statistics."""
        timestamps: dict[str, list[int]] = {
            detector_id: [] for detector_id in result.detector_ids
        }
        scan_indices: list[int] = []
        for shot in result.shots:
            scan_indices.append(shot.scan_index)
            for frame in shot.frames:
                timestamps[frame.detector_id].append(frame.exposure_ended_ns)
        rates: dict[str, float] = {}
        durations: dict[str, int] = {}
        frame_intervals: dict[str, TimingDistribution] = {}
        pattern_intervals: dict[str, TimingDistribution] = {}
        scan_transition_intervals: dict[str, TimingDistribution] = {}
        for detector_id, values in timestamps.items():
            intervals = np.asarray(
                np.diff(np.asarray(values, dtype=np.int64)), dtype=np.int64
            )
            valid = intervals[intervals > 0]
            if valid.size != intervals.size or valid.size == 0:
                raise RuntimeError(
                    "committed frame timestamps are not strictly ordered"
                )
            transition_mask = np.diff(np.asarray(scan_indices, dtype=np.int64)) != 0
            patterns = valid[~transition_mask]
            transitions = valid[transition_mask]
            if patterns.size == 0 or transitions.size == 0:
                raise RuntimeError(
                    "finite multishot timing requires patterns and scans"
                )
            duration_ns = values[-1] - values[0]
            if duration_ns <= 0:
                raise RuntimeError("committed frame capture duration is not positive")
            rates[detector_id] = (len(values) - 1) * 1_000_000_000 / duration_ns
            durations[detector_id] = duration_ns
            frame_intervals[detector_id] = _timing_distribution(valid)
            pattern_intervals[detector_id] = _timing_distribution(patterns)
            scan_transition_intervals[detector_id] = _timing_distribution(transitions)
        return MultishotTimingProfile(
            elapsed_ns,
            len(result.shots),
            rates,
            durations,
            frame_intervals,
            pattern_intervals,
            scan_transition_intervals,
        )

    async def _cleanup(self) -> None:
        try:
            await asyncio.gather(
                *(detector.stop() for detector in self.detectors),
                return_exceptions=True,
            )
        finally:
            try:
                await asyncio.to_thread(self.illumination.all_off)
            finally:
                try:
                    await self.stage.stop()
                finally:
                    await asyncio.gather(
                        *(detector.disconnect() for detector in self.detectors),
                        return_exceptions=True,
                    )

    def _validate_frames(self, run_id: str, frames: tuple[Frame, ...]) -> None:
        detector_ids = tuple(frame.detector_id for frame in frames)
        expected_ids = tuple(detector.service_id for detector in self.detectors)
        if detector_ids != expected_ids:
            raise RuntimeError(
                "DPCT frame detector ordering does not match composition"
            )
        if any(frame.run_id != run_id for frame in frames):
            raise RuntimeError("DPCT frame run identity does not match recipe")

    @staticmethod
    def _validate_arrays(
        frames: tuple[Frame, ...], arrays: tuple[npt.NDArray[Any], ...]
    ) -> None:
        if len(frames) != len(arrays):  # pragma: no cover - gather invariant
            raise RuntimeError("DPCT detector payload count does not match frames")
        for frame, array in zip(frames, arrays, strict=True):
            payload = np.asarray(array)
            if payload.shape != frame.array.shape:
                raise RuntimeError(
                    f"DPCT payload shape does not match frame {frame.frame_id}"
                )
            if frame.array.byte_order not in {"=", "<", ">", "|"}:
                raise RuntimeError(
                    f"DPCT payload byte order is invalid for frame {frame.frame_id}"
                )
            byte_order = cast("Literal['=', '<', '>', '|']", frame.array.byte_order)
            expected_dtype = np.dtype(frame.array.dtype).newbyteorder(byte_order)
            if payload.dtype != expected_dtype:
                raise RuntimeError(
                    f"DPCT payload dtype does not match frame {frame.frame_id}"
                )
            checksum = hashlib.sha256(payload.tobytes(order="C")).hexdigest()
            if checksum != frame.array.checksum:
                raise RuntimeError(
                    f"DPCT payload checksum does not match frame {frame.frame_id}"
                )


# Compatibility spelling retained for callers introduced before the finite plan
# was named explicitly. New compositions should use MultishotListScanPlan.
DpctRunner = MultishotListScanPlan


def _timing_distribution(intervals: npt.NDArray[np.int64]) -> TimingDistribution:
    """Summarize an already-validated non-empty interval vector."""
    return TimingDistribution(
        count=int(intervals.size),
        minimum_ns=int(np.min(intervals)),
        maximum_ns=int(np.max(intervals)),
        mean_ns=float(np.mean(intervals)),
        median_ns=float(np.median(intervals)),
        p95_ns=float(np.percentile(intervals, 95)),
        standard_deviation_ns=float(np.std(intervals)),
    )
