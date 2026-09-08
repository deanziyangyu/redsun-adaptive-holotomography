"""RedSun-declarable ophyd facades for DPCT acquisition backends."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ophyd_async.core import AsyncStatus, Device

from redsun_aht.acquisition.dpct import DpctRunResult, DpctShotRecord

if TYPE_CHECKING:
    from bluesky.protocols import Reading
    from event_model import DataKey

    from redsun_aht.acquisition.dpct import (
        DpctDetector,
        DpctFrameSink,
        DpctRecipe,
        PatternController,
    )
    from redsun_aht.domain import PoseSample
    from redsun_aht.motion import ScanPoint
    from redsun_aht.protocols import CompositeStage


class DpctStageDevice(Device):
    """Expose the validated composite stage as a RedSun device."""

    def __init__(self, name: str, /, *, backend: CompositeStage) -> None:
        super().__init__(name=name)
        self.backend = backend
        self.pose: PoseSample | None = None
        self.scan_index: int | None = None

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Stage without moving the hardware."""

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Stop outstanding motion during plan teardown."""
        await self.backend.stop()

    @AsyncStatus.wrap
    async def set(self, value: ScanPoint) -> None:
        """Move to and settle at one validated scan point."""
        await self.backend.move(value.as_dict())
        self.pose = await self.backend.settle()
        self.scan_index = value.index

    @AsyncStatus.wrap
    async def stop(self, success: bool = False) -> None:
        """Stop ordinary motion regardless of run outcome."""
        del success
        await self.backend.stop()

    async def read(self) -> dict[str, Reading[Any]]:
        """Read the latest settled pose as scalar Bluesky fields."""
        pose = self.pose or await self.backend.read()
        timestamp = time.time()
        readings: dict[str, Reading[Any]] = {
            f"{self.name}_scan_index": {
                "value": -1 if self.scan_index is None else self.scan_index,
                "timestamp": timestamp,
            }
        }
        for axis, value in zip(self.backend.axes, pose.joints, strict=True):
            readings[f"{self.name}_{axis}"] = {
                "value": value,
                "timestamp": timestamp,
            }
        return readings

    async def describe(self) -> dict[str, DataKey]:
        """Describe stage coordinates and their engineering units."""
        description: dict[str, DataKey] = {
            f"{self.name}_scan_index": {
                "source": f"AHT:{self.name}:scan-index",
                "dtype": "integer",
                "shape": [],
            }
        }
        for axis, unit in self.backend.axes.items():
            description[f"{self.name}_{axis}"] = {
                "source": f"AHT:{self.name}:{axis}",
                "dtype": "number",
                "shape": [],
                "units": unit,
            }
        return description


class DpctPatternDevice(Device):
    """Expose synchronous illumination pattern selection as a RedSun device."""

    def __init__(self, name: str, /, *, backend: PatternController) -> None:
        super().__init__(name=name)
        self.backend = backend
        self.pattern_id: int | None = None

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Stage without changing illumination output."""

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Clear ordinary illumination output."""
        await asyncio.to_thread(self.backend.all_off)
        self.pattern_id = None

    @AsyncStatus.wrap
    async def set(self, value: int) -> None:
        """Display one committed illumination pattern."""
        if not 0 <= value <= 0xFFFF:
            raise ValueError("DPCT pattern ID must fit unsigned 16-bit")
        await asyncio.to_thread(self.backend.show_pattern, value)
        self.pattern_id = value

    @AsyncStatus.wrap
    async def stop(self, success: bool = False) -> None:
        """Clear illumination during interruption handling."""
        del success
        await asyncio.to_thread(self.backend.all_off)
        self.pattern_id = None

    async def read(self) -> dict[str, Reading[int]]:
        """Read the currently displayed pattern identity."""
        return {
            f"{self.name}_pattern_id": {
                "value": -1 if self.pattern_id is None else self.pattern_id,
                "timestamp": time.time(),
            }
        }

    async def describe(self) -> dict[str, DataKey]:
        """Describe the illumination pattern field."""
        return {
            f"{self.name}_pattern_id": {
                "source": f"AHT:{self.name}:pattern-id",
                "dtype": "integer",
                "shape": [],
            }
        }


class DpctDetectorGroupDevice(Device):
    """Coordinate one lossless multi-detector shot behind an ophyd facade."""

    def __init__(
        self,
        name: str,
        /,
        *,
        detectors: tuple[DpctDetector, ...],
        recipe: DpctRecipe,
        sink: DpctFrameSink | None = None,
    ) -> None:
        super().__init__(name=name)
        detector_ids = tuple(detector.service_id for detector in detectors)
        if not detector_ids or len(set(detector_ids)) != len(detector_ids):
            raise ValueError("DPCT detectors must be non-empty and uniquely identified")
        self.detectors = detectors
        self.recipe = recipe
        self.sink = sink
        self._sink_started = False
        self._complete = False
        self._context: tuple[int, int, PoseSample] | None = None
        self._shots: list[DpctShotRecord] = []
        self._last_shot: DpctShotRecord | None = None
        self._result: DpctRunResult | None = None
        self._failure: BaseException | None = None

    @property
    def result(self) -> DpctRunResult | None:
        """Return the finalized run result."""
        return self._result

    def set_context(self, scan_index: int, pattern_id: int, pose: PoseSample) -> None:
        """Bind the settled pose and pattern used by the next trigger."""
        self._context = (scan_index, pattern_id, pose)

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Connect, configure, and arm every detector before the run."""
        try:
            for detector in self.detectors:
                await detector.connect()
                await detector.configure(self.recipe.detector_settings)
                await detector.arm()
            if self.sink is not None:
                await asyncio.to_thread(
                    self.sink.begin,
                    self.recipe,
                    tuple(detector.service_id for detector in self.detectors),
                )
                self._sink_started = True
        except BaseException as error:
            self._failure = error
            await self._cleanup()
            raise

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Report incomplete storage and clean every detector."""
        if self.sink is not None and self._sink_started and not self._complete:
            failure = self._failure or RuntimeError("DPCT plan did not complete")
            await asyncio.to_thread(self.sink.fail, failure)
            self._sink_started = False
        await self._cleanup()

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        """Acquire, persist, and acknowledge one detector group."""
        if self._context is None:
            raise RuntimeError("DPCT trigger requires settled scan/pattern context")
        scan_index, pattern_id, pose = self._context
        try:
            await asyncio.gather(*(detector.trigger() for detector in self.detectors))
            frames = tuple([await detector.read() for detector in self.detectors])
            self._validate_frames(frames)
            arrays = tuple(
                await asyncio.gather(
                    *(
                        detector.copy(frame.sequence)
                        for detector, frame in zip(self.detectors, frames, strict=True)
                    )
                )
            )
            self._validate_arrays(frames, arrays)
            shot = DpctShotRecord(
                shot_index=len(self._shots),
                scan_index=scan_index,
                pattern_id=pattern_id,
                pose=pose,
                frames=frames,
            )
            if self.sink is not None:
                await asyncio.to_thread(self.sink.write_shot, shot, arrays)
            await asyncio.gather(
                *(
                    detector.acknowledge(frame.sequence)
                    for detector, frame in zip(self.detectors, frames, strict=True)
                )
            )
            self._shots.append(shot)
            self._last_shot = shot
        except BaseException as error:
            self._failure = error
            raise

    async def finalize(self) -> DpctRunResult:
        """Validate coverage and publish the storage completion marker."""
        result = DpctRunResult(
            run_id=self.recipe.run_id,
            detector_ids=tuple(detector.service_id for detector in self.detectors),
            shots=tuple(self._shots),
            storage_uri=None if self.sink is None else self.sink.uri,
        )
        if self.sink is not None:
            await asyncio.to_thread(self.sink.complete, result)
        self._complete = True
        self._sink_started = False
        self._result = result
        return result

    @AsyncStatus.wrap
    async def stop(self, success: bool = False) -> None:
        """Stop detectors without acknowledging an uncommitted frame."""
        del success
        await asyncio.gather(
            *(detector.stop() for detector in self.detectors),
            return_exceptions=True,
        )

    async def read(self) -> dict[str, Reading[Any]]:
        """Describe the latest committed shot without embedding image arrays."""
        if self._last_shot is None:
            raise RuntimeError("DPCT detector group has not produced a shot")
        timestamp = time.time()
        readings: dict[str, Reading[Any]] = {}
        for frame in self._last_shot.frames:
            prefix = f"{self.name}_{frame.detector_id}"
            readings[f"{prefix}_sequence"] = {
                "value": frame.sequence,
                "timestamp": timestamp,
            }
            readings[f"{prefix}_frame_id"] = {
                "value": frame.frame_id,
                "timestamp": timestamp,
            }
            readings[f"{prefix}_checksum"] = {
                "value": frame.array.checksum,
                "timestamp": timestamp,
            }
        return readings

    async def describe(self) -> dict[str, DataKey]:
        """Describe lightweight per-detector shot metadata."""
        description: dict[str, DataKey] = {}
        for detector in self.detectors:
            prefix = f"{self.name}_{detector.service_id}"
            description[f"{prefix}_sequence"] = {
                "source": f"AHT:{detector.service_id}:sequence",
                "dtype": "integer",
                "shape": [],
            }
            for suffix in ("frame_id", "checksum"):
                description[f"{prefix}_{suffix}"] = {
                    "source": f"AHT:{detector.service_id}:{suffix}",
                    "dtype": "string",
                    "shape": [],
                }
        return description

    async def _cleanup(self) -> None:
        try:
            await asyncio.gather(
                *(detector.stop() for detector in self.detectors),
                return_exceptions=True,
            )
        finally:
            await asyncio.gather(
                *(detector.disconnect() for detector in self.detectors),
                return_exceptions=True,
            )

    def _validate_frames(self, frames: tuple[Any, ...]) -> None:
        expected = tuple(detector.service_id for detector in self.detectors)
        if tuple(frame.detector_id for frame in frames) != expected:
            raise RuntimeError(
                "DPCT frame detector ordering does not match composition"
            )
        if any(frame.run_id != self.recipe.run_id for frame in frames):
            raise RuntimeError("DPCT frame run identity does not match recipe")

    @staticmethod
    def _validate_arrays(
        frames: tuple[Any, ...], arrays: tuple[np.ndarray[Any, Any], ...]
    ) -> None:
        for frame, array in zip(frames, arrays, strict=True):
            payload = np.asarray(array)
            byte_order = cast("str", frame.array.byte_order)
            expected_dtype = np.dtype(frame.array.dtype).newbyteorder(byte_order)
            if payload.shape != frame.array.shape or payload.dtype != expected_dtype:
                raise RuntimeError(
                    f"DPCT payload layout does not match frame {frame.frame_id}"
                )
            checksum = hashlib.sha256(payload.tobytes(order="C")).hexdigest()
            if checksum != frame.array.checksum:
                raise RuntimeError(
                    f"DPCT payload checksum does not match frame {frame.frame_id}"
                )


__all__ = ["DpctDetectorGroupDevice", "DpctPatternDevice", "DpctStageDevice"]
