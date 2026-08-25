"""Public structural contracts for AHT components and extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from bluesky.protocols import Status
    from event_model import PartialEvent

    from redsun_aht.domain import (
        CalibrationArtifact,
        DetectorCapabilities,
        Frame,
        HardwareServiceStatus,
        Observation,
        PoseSample,
        ProcessingJob,
    )
    from redsun_aht.domain.models import JsonValue


@runtime_checkable
class Detector(Protocol):
    """Lifecycle contract for independently staged detectors."""

    async def discover(self) -> DetectorCapabilities: ...

    async def connect(self) -> None: ...

    async def configure(self, settings: Mapping[str, JsonValue]) -> str: ...

    async def arm(self) -> None: ...

    async def trigger(self) -> None: ...

    async def read(self) -> Frame: ...

    async def stop(self) -> None: ...

    async def disconnect(self) -> None: ...


@runtime_checkable
class SupervisedDetector(Detector, Protocol):
    """Detector service surface used by discovery and reconnect supervision."""

    @property
    def service_id(self) -> str: ...

    @property
    def epics_prefix(self) -> str: ...

    @property
    def status(self) -> HardwareServiceStatus: ...


@runtime_checkable
class CompositeStage(Protocol):
    """Named-axis motion interface with explicit units and cleanup."""

    @property
    def axes(self) -> Mapping[str, str]: ...

    async def move(self, positions: Mapping[str, float]) -> PoseSample: ...

    async def settle(self) -> PoseSample: ...

    async def read(self) -> PoseSample: ...

    async def stop(self) -> None: ...


@runtime_checkable
class MotionValidator(Protocol):
    """Public seam implemented by a future robotic safety extension."""

    async def authorize(
        self, current: PoseSample, requested: PoseSample
    ) -> tuple[bool, str]: ...


@runtime_checkable
class RemoteReconstructionProvider(Protocol):
    """Loose remote-processing seam; never a staged base-package device."""

    async def submit(self, job: ProcessingJob) -> str: ...

    async def status(self, remote_job_id: str) -> Mapping[str, JsonValue]: ...

    async def cancel(self, remote_job_id: str) -> None: ...


@runtime_checkable
class SolverPlugin(Protocol):
    """Detached numerical solver contract."""

    @property
    def plugin_id(self) -> str: ...

    async def prepare(self, calibrations: Sequence[CalibrationArtifact]) -> None: ...

    async def run(
        self, job: ProcessingJob
    ) -> AsyncIterator[Mapping[str, JsonValue]]: ...

    async def cancel(self, job_id: str) -> None: ...


@runtime_checkable
class ProcessingFlyer(Protocol):
    """Run-coupled lifecycle adapter; kernels execute in detached workers."""

    def prepare(self, job: ProcessingJob) -> Status: ...

    def kickoff(self) -> Status: ...

    async def accept(self, observation: Observation) -> None: ...

    def complete(self) -> Status: ...

    def collect(self) -> AsyncIterator[PartialEvent]: ...

    def stop(self) -> Status: ...


__all__ = [
    "CompositeStage",
    "Detector",
    "MotionValidator",
    "ProcessingFlyer",
    "RemoteReconstructionProvider",
    "SolverPlugin",
    "SupervisedDetector",
]
