"""Bluesky lifecycle adapter for one detached processing worker."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, cast

from ophyd_async.core import AsyncStatus, Device

from redsun_aht.domain import SolverState
from redsun_aht.streams import (
    PROCESSING_PROGRESS_STREAM,
    PROCESSING_RESULT_STREAM,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from event_model import DataKey, PartialEvent

    from redsun_aht.domain import Observation, ProcessingJob, ProcessingResult
    from redsun_aht.domain.models import JsonValue
    from redsun_aht.processing import SolverWorker


class ProcessingFlyerDevice(Device):
    """Expose one persistent solver through a Bluesky flyer lifecycle.

    The adapter only owns job/reference orchestration. Numerical work remains
    inside the worker process supervised by :class:`SolverWorker`.
    """

    def __init__(self, worker: SolverWorker, *, name: str) -> None:
        if not name:
            raise ValueError("processing flyer name cannot be empty")
        super().__init__(name=name)
        self.worker = worker
        self._job: ProcessingJob | None = None
        self._observations: dict[str, Observation] = {}
        self._replay_keys: set[str] = set()
        self._events: list[PartialEvent] = []
        self._collect_cursor = 0
        self._state = SolverState.DISABLED
        self._accepting = False
        self._result: ProcessingResult | None = None
        self._failure: str | None = None

    @property
    def result(self) -> ProcessingResult | None:
        """Return the immutable result retained after completion."""
        return self._result

    @property
    def failure(self) -> str | None:
        """Return an isolated processing failure retained after completion."""
        return self._failure

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Reset per-run state without constructing hardware."""
        await self._stop_worker()
        self._job = None
        self._observations.clear()
        self._replay_keys.clear()
        self._events.clear()
        self._collect_cursor = 0
        self._state = SolverState.DISABLED
        self._accepting = False
        self._result = None
        self._failure = None

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Release the detached worker after document collection."""
        await self._stop_worker()

    @AsyncStatus.wrap
    async def prepare(self, job: ProcessingJob) -> None:
        """Validate one job and start its persistent solver worker."""
        if self._job is not None or self._state is not SolverState.DISABLED:
            raise RuntimeError(f"{self.name}: flyer is already prepared")
        if job.solver_id != self.worker.solver_id:
            raise ValueError("processing job solver does not match flyer worker")
        capabilities = await asyncio.to_thread(self.worker.start)
        try:
            if capabilities.solver_version != job.solver_version:
                raise ValueError("processing job solver version is not admitted")
            if not capabilities.batch_supported:
                raise ValueError("processing flyer requires a batch-capable solver")
        except BaseException:
            await self._stop_worker()
            raise
        self._job = job
        self._state = SolverState.READY

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        """Begin accepting committed observation references for the job."""
        if self._job is None or self._state is not SolverState.READY:
            raise RuntimeError(f"{self.name}: prepare() must precede kickoff()")
        self._accepting = True
        self._state = SolverState.WAITING_INPUT
        self._append_progress(progress=0.0)

    async def accept(self, observation: Observation) -> None:
        """Accept one committed, selected observation without copying its array."""
        job = self._require_job()
        if not self._accepting or self._state is not SolverState.WAITING_INPUT:
            raise RuntimeError(f"{self.name}: flyer is not accepting observations")
        if observation.observation_id not in job.input_selectors:
            raise ValueError("observation is not selected by the processing job")
        if (
            self._observations
            and observation.run_id != next(iter(self._observations.values())).run_id
        ):
            raise ValueError("processing flyer observations span multiple runs")
        if (
            observation.observation_id in self._observations
            or observation.replay_key in self._replay_keys
        ):
            raise ValueError("processing flyer observation is duplicated")
        self._observations[observation.observation_id] = observation
        self._replay_keys.add(observation.replay_key)

    @AsyncStatus.wrap
    async def complete(self) -> None:
        """Close input, drain the detached job, and retain its published result."""
        job = self._require_job()
        if not self._accepting or self._state is not SolverState.WAITING_INPUT:
            raise RuntimeError(f"{self.name}: kickoff() must precede complete()")
        self._accepting = False
        selected = set(job.input_selectors)
        if set(self._observations) != selected:
            missing = sorted(selected.difference(self._observations))
            extra = sorted(set(self._observations).difference(selected))
            self._record_failure(
                "processing flyer input coverage differs; "
                f"missing={missing}, extra={extra}"
            )
            return
        observations = tuple(
            self._observations[selector] for selector in job.input_selectors
        )
        self._state = SolverState.QUEUED
        self._append_progress(progress=0.0)
        try:
            self.worker.submit(job, observations)
            result = await asyncio.to_thread(self.worker.wait)
        except asyncio.CancelledError as error:
            self.worker.cancel()
            self._record_failure(str(error) or type(error).__qualname__)
            raise
        except Exception as error:
            self._record_failure(str(error) or type(error).__qualname__)
            return
        self._result = result
        self._state = SolverState.READY
        self._append_progress(progress=1.0)
        self._append_result(result)

    async def describe_collect(self) -> dict[str, dict[str, DataKey]]:
        """Describe progress and external-reference result streams."""
        return {
            PROCESSING_PROGRESS_STREAM: {
                "progress_flyer_name": _data_key("progress_flyer_name", "string"),
                "progress_solver_id": _data_key("progress_solver_id", "string"),
                "solver_state": _data_key("solver_state", "string"),
                "progress": _data_key("progress", "number"),
                "input_cursor": _data_key("input_cursor", "integer"),
                "output_cursor": _data_key("output_cursor", "integer"),
                "last_error": _data_key("last_error", "string"),
            },
            PROCESSING_RESULT_STREAM: {
                "result_flyer_name": _data_key("result_flyer_name", "string"),
                "result_id": _data_key("result_id", "string"),
                "job_id": _data_key("job_id", "string"),
                "result_solver_id": _data_key("result_solver_id", "string"),
                "solver_version": _data_key("solver_version", "string"),
                "output_uri": _data_key("output_uri", "string"),
                "output_checksum": _data_key("output_checksum", "string"),
                "cached": _data_key("cached", "boolean"),
                "completed_ns": _data_key("completed_ns", "integer"),
            },
        }

    async def collect(self) -> AsyncIterator[PartialEvent]:
        """Yield each lightweight progress/result event exactly once."""
        while self._collect_cursor < len(self._events):
            event = self._events[self._collect_cursor]
            self._collect_cursor += 1
            yield event

    @AsyncStatus.wrap
    async def stop(self) -> None:
        """Cancel and close the detached worker idempotently."""
        await self._stop_worker()

    async def describe_configuration(self) -> Mapping[str, JsonValue]:
        """Return the run-visible detached-worker configuration."""
        job = self._job
        return {
            "name": self.name,
            "solver_id": self.worker.solver_id,
            "job_id": None if job is None else job.job_id,
            "solver_version": None if job is None else job.solver_version,
            "gpu_id": self.worker.gpu_id,
            "gpu_memory_reservation_bytes": self.worker.memory_reservation_bytes,
        }

    def _require_job(self) -> ProcessingJob:
        if self._job is None:
            raise RuntimeError(f"{self.name}: processing job is not prepared")
        return self._job

    async def _stop_worker(self) -> None:
        self._accepting = False
        self.worker.cancel()
        await asyncio.to_thread(self.worker.close)
        self._state = SolverState.STOPPED

    def _record_failure(self, message: str) -> None:
        self._failure = message
        self._state = SolverState.FAILED
        self._append_progress(progress=1.0, last_error=message)

    def _append_progress(self, *, progress: float, last_error: str = "") -> None:
        status = self.worker.status
        self._events.append(
            _event(
                {
                    "progress_flyer_name": self.name,
                    "progress_solver_id": self.worker.solver_id,
                    "solver_state": self._state.value,
                    "progress": progress,
                    "input_cursor": len(self._observations),
                    "output_cursor": status.output_cursor,
                    "last_error": last_error,
                }
            )
        )

    def _append_result(self, result: ProcessingResult) -> None:
        self._events.append(
            _event(
                {
                    "result_flyer_name": self.name,
                    "result_id": result.result_id,
                    "job_id": result.job_id,
                    "result_solver_id": result.solver_id,
                    "solver_version": result.solver_version,
                    "output_uri": result.output.uri,
                    "output_checksum": result.output.checksum,
                    "cached": result.cached,
                    "completed_ns": result.completed_ns,
                }
            )
        )


def _data_key(source: str, dtype: str) -> DataKey:
    return cast(
        "DataKey",
        {"source": f"AHT:processing:{source}", "dtype": dtype, "shape": []},
    )


def _event(data: Mapping[str, JsonValue]) -> PartialEvent:
    timestamp = time.time()
    return cast(
        "PartialEvent",
        {
            "time": timestamp,
            "data": dict(data),
            "timestamps": dict.fromkeys(data, timestamp),
        },
    )


__all__ = ["ProcessingFlyerDevice"]
