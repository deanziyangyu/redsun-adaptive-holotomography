"""Independent spawned-worker supervision for detached solver pipelines."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from redsun_aht.domain import (
    Observation,
    ProcessingJob,
    ProcessingResult,
    SolverCapabilities,
    SolverState,
    SolverStatus,
)
from redsun_aht.processing.messages import (
    capabilities_from_payload,
    job_to_payload,
    observation_to_payload,
    pack,
    result_from_payload,
    status_from_payload,
    unpack,
)
from redsun_aht.processing.stdio_transport import start_gpu_stdio_worker
from redsun_aht.processing.worker import processing_worker_main

if TYPE_CHECKING:
    from collections.abc import Mapping

    from redsun_aht.processing.resources import GpuLease


class WorkerUnavailableError(RuntimeError):
    """Raised when a worker fails startup or exits before a response."""


class WorkerJobError(RuntimeError):
    """Structured local worker job failure."""

    def __init__(self, solver_id: str, job_id: str, error_type: str, message: str):
        super().__init__(f"{solver_id}/{job_id} failed ({error_type}): {message}")
        self.solver_id = solver_id
        self.job_id = job_id
        self.error_type = error_type


@dataclass(frozen=True, slots=True)
class ProcessingBatchResult:
    """Independent successes and failures after every submitted worker responds."""

    results: Mapping[str, ProcessingResult]
    failures: Mapping[str, str]
    statuses: Mapping[str, SolverStatus]

    def __post_init__(self) -> None:
        """Detach batch mappings from supervisor-owned mutable state."""
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))
        object.__setattr__(self, "statuses", MappingProxyType(dict(self.statuses)))


@dataclass(slots=True)
class SolverWorker:
    """Supervise one persistent spawned process and one solver plugin."""

    solver_id: str
    startup_timeout_s: float = 10.0
    result_timeout_s: float = 30.0
    shutdown_timeout_s: float = 5.0
    gpu_id: int | None = None
    memory_reservation_bytes: int = 0
    gpu_cache_directory: str | None = None
    _context: Any = field(init=False, repr=False)
    _command_queue: Any = field(init=False, default=None, repr=False)
    _response_queue: Any = field(init=False, default=None, repr=False)
    _cancel_event: Any = field(init=False, default=None, repr=False)
    _process: Any = field(init=False, default=None, repr=False)
    _pending_job_id: str | None = field(init=False, default=None)
    _restart_count: int = field(init=False, default=0)
    _capabilities: SolverCapabilities | None = field(init=False, default=None)
    _status: SolverStatus = field(init=False)

    def __post_init__(self) -> None:
        """Validate bounded timing and select Windows-safe spawn semantics."""
        if not self.solver_id:
            raise ValueError("solver worker identity cannot be empty")
        if (
            min(
                self.startup_timeout_s,
                self.result_timeout_s,
                self.shutdown_timeout_s,
            )
            <= 0
        ):
            raise ValueError("solver worker timeouts must be positive")
        if self.gpu_id is None and self.memory_reservation_bytes != 0:
            raise ValueError("CPU worker cannot reserve GPU memory")
        if self.gpu_id is not None and (
            self.gpu_id < 0 or self.memory_reservation_bytes <= 0
        ):
            raise ValueError("GPU worker placement and reservation are invalid")
        if self.gpu_cache_directory is not None and (
            self.gpu_id is None or not Path(self.gpu_cache_directory).is_absolute()
        ):
            raise ValueError("GPU cache directory requires an absolute GPU placement")
        self._context = mp.get_context("spawn")
        self._status = _local_status(
            self.solver_id,
            SolverState.DISABLED,
            gpu_id=self.gpu_id,
            allocated_memory_bytes=self.memory_reservation_bytes,
        )

    @property
    def capabilities(self) -> SolverCapabilities:
        """Return worker-reported capabilities after startup."""
        if self._capabilities is None:
            raise RuntimeError("solver worker has not started")
        return self._capabilities

    @property
    def status(self) -> SolverStatus:
        """Return the latest immutable status snapshot."""
        return self._status

    @property
    def process_id(self) -> int | None:
        """Return the active operating-system process identity."""
        return None if self._process is None else self._process.pid

    def start(self) -> SolverCapabilities:
        """Spawn and admit one persistent worker process."""
        if self._process is not None:
            if self._process.is_alive():
                return self.capabilities
            raise WorkerUnavailableError("solver worker process exited unexpectedly")
        self._status = _local_status(
            self.solver_id,
            SolverState.STARTING,
            restart_count=self._restart_count,
            gpu_id=self.gpu_id,
            allocated_memory_bytes=self.memory_reservation_bytes,
        )
        if self.gpu_id is not None:
            if self.gpu_cache_directory is None:
                raise ValueError("GPU worker requires an explicit cache directory")
            (
                self._command_queue,
                self._response_queue,
                self._cancel_event,
                self._process,
            ) = start_gpu_stdio_worker(
                self.solver_id,
                self.gpu_id,
                self.memory_reservation_bytes,
                self.gpu_cache_directory,
                self._restart_count,
            )
        else:
            self._command_queue = self._context.Queue()
            self._response_queue = self._context.Queue()
            self._cancel_event = self._context.Event()
            self._process = self._context.Process(
                target=processing_worker_main,
                args=(
                    self.solver_id,
                    self._command_queue,
                    self._response_queue,
                    self._cancel_event,
                    self._restart_count,
                ),
                name=f"aht-solver-{self.solver_id}",
                daemon=False,
            )
            self._process.start()
        try:
            kind, body = self._receive(self.startup_timeout_s)
            if kind != "ready":
                raise WorkerUnavailableError(
                    f"solver worker returned {kind!r} during startup"
                )
            capabilities = capabilities_from_payload(body["capabilities"])
            if capabilities.solver_id != self.solver_id:
                raise WorkerUnavailableError("worker capability identity mismatch")
            self._capabilities = capabilities
            self._status = status_from_payload(body["status"])
            return capabilities
        except BaseException:
            self.close()
            raise

    def submit(self, job: ProcessingJob, observations: tuple[Observation, ...]) -> None:
        """Queue one reference-only job without waiting for solver completion."""
        self._require_available()
        if self._pending_job_id is not None:
            raise RuntimeError("solver worker already has a pending job")
        if job.solver_id != self.solver_id:
            raise ValueError("job solver identity does not match worker")
        self._pending_job_id = job.job_id
        self._cancel_event.clear()
        self._status = _local_status(
            self.solver_id,
            SolverState.QUEUED,
            worker_pid=self.process_id,
            restart_count=self._restart_count,
            gpu_id=self.gpu_id,
            allocated_memory_bytes=self.memory_reservation_bytes,
        )
        self._command_queue.put(
            pack(
                "run",
                {
                    "job": job_to_payload(job),
                    "observations": [
                        observation_to_payload(item) for item in observations
                    ],
                },
            )
        )

    def wait(self, timeout_s: float | None = None) -> ProcessingResult:
        """Wait boundedly for the pending result while consuming status updates."""
        if self._pending_job_id is None:
            raise RuntimeError("solver worker has no pending job")
        deadline = time.monotonic() + (
            self.result_timeout_s if timeout_s is None else timeout_s
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"solver worker timed out: {self.solver_id}")
            kind, body = self._receive(remaining)
            if "status" in body:
                self._status = status_from_payload(body["status"])
            if kind == "status":
                continue
            pending_job_id = self._pending_job_id
            self._pending_job_id = None
            if kind == "result":
                result = result_from_payload(body["result"])
                if result.job_id != pending_job_id:
                    raise WorkerUnavailableError("worker result job identity mismatch")
                return result
            if kind == "failure":
                raise WorkerJobError(
                    self.solver_id,
                    str(body["job_id"]),
                    str(body["error_type"]),
                    str(body["message"]),
                )
            raise WorkerUnavailableError(f"unexpected solver worker response: {kind}")

    def cancel(self) -> None:
        """Request cooperative cancellation of the currently running job."""
        if self._cancel_event is not None and self._pending_job_id is not None:
            self._cancel_event.set()

    def restart(self) -> SolverCapabilities:
        """Replace a failed/stopped process while preserving durable outputs."""
        self.close()
        self._restart_count += 1
        return self.start()

    def close(self) -> None:
        """Stop the worker idempotently and release local queue resources."""
        process = self._process
        if process is not None:
            if process.is_alive() and self._command_queue is not None:
                self._command_queue.put(pack("shutdown"))
                process.join(self.shutdown_timeout_s)
            if process.is_alive():
                process.terminate()
                process.join(self.shutdown_timeout_s)
            process.close()
        for owned_queue in (self._command_queue, self._response_queue):
            if owned_queue is not None:
                owned_queue.close()
                owned_queue.join_thread()
        if self._cancel_event is not None and hasattr(self._cancel_event, "close"):
            self._cancel_event.close()
        self._process = None
        self._command_queue = None
        self._response_queue = None
        self._cancel_event = None
        self._pending_job_id = None
        self._capabilities = None
        self._status = _local_status(
            self.solver_id,
            SolverState.STOPPED,
            restart_count=self._restart_count,
            gpu_id=self.gpu_id,
            allocated_memory_bytes=self.memory_reservation_bytes,
        )

    def _receive(self, timeout_s: float) -> tuple[str, dict[str, Any]]:
        if self._response_queue is None:
            raise WorkerUnavailableError("solver response queue is unavailable")
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"solver worker response timed out: {self.solver_id}"
                )
            try:
                payload = self._response_queue.get(timeout=min(remaining, 0.1))
                break
            except queue.Empty:
                if self._process is None or not self._process.is_alive():
                    raise WorkerUnavailableError(
                        f"solver worker exited without response: {self.solver_id}"
                    ) from None
        if not isinstance(payload, bytes):
            raise WorkerUnavailableError("solver worker response is not bytes")
        return unpack(payload)

    def _require_available(self) -> None:
        if self._process is None or not self._process.is_alive():
            raise WorkerUnavailableError("solver worker is not running")


class ProcessingSupervisor:
    """Coordinate independent solver queues without serializing their kernels."""

    def __init__(
        self,
        solver_ids: tuple[str, ...],
        *,
        gpu_leases: Mapping[str, GpuLease] | None = None,
        gpu_cache_directories: Mapping[str, str] | None = None,
    ) -> None:
        if not solver_ids or len(set(solver_ids)) != len(solver_ids):
            raise ValueError("processing supervisor solver IDs must be unique")
        leases = dict(gpu_leases or {})
        cache_directories = dict(gpu_cache_directories or {})
        if not set(leases).issubset(solver_ids) or any(
            solver_id != lease.solver_id for solver_id, lease in leases.items()
        ):
            raise ValueError("GPU leases must match supervised solver identities")
        if set(cache_directories) != set(leases):
            raise ValueError("every GPU lease requires exactly one cache directory")
        self.workers = {
            solver_id: SolverWorker(
                solver_id,
                startup_timeout_s=60.0 if solver_id in leases else 10.0,
                result_timeout_s=120.0 if solver_id in leases else 30.0,
                gpu_id=None if solver_id not in leases else leases[solver_id].device_id,
                memory_reservation_bytes=(
                    0
                    if solver_id not in leases
                    else leases[solver_id].reservation_bytes
                ),
                gpu_cache_directory=cache_directories.get(solver_id),
            )
            for solver_id in solver_ids
        }

    def start(self) -> Mapping[str, SolverCapabilities]:
        """Start every solver independently and return admitted capabilities."""
        admitted: dict[str, SolverCapabilities] = {}
        try:
            for solver_id, worker in self.workers.items():
                admitted[solver_id] = worker.start()
        except BaseException:
            self.close()
            raise
        return MappingProxyType(admitted)

    def run_batch(
        self,
        requests: Mapping[str, tuple[ProcessingJob, tuple[Observation, ...]]],
    ) -> ProcessingBatchResult:
        """Submit every pipeline first, then independently collect all outcomes."""
        if set(requests) != set(self.workers):
            raise ValueError("processing batch must address every supervised solver")
        for solver_id, (job, observations) in requests.items():
            self.workers[solver_id].submit(job, observations)
        results: dict[str, ProcessingResult] = {}
        failures: dict[str, str] = {}
        for solver_id, worker in self.workers.items():
            try:
                results[solver_id] = worker.wait()
            except (WorkerJobError, WorkerUnavailableError, TimeoutError) as error:
                failures[solver_id] = str(error)
        return ProcessingBatchResult(
            results,
            failures,
            {solver_id: worker.status for solver_id, worker in self.workers.items()},
        )

    def close(self) -> None:
        """Stop every worker even when another worker cleanup fails."""
        errors: list[Exception] = []
        for worker in self.workers.values():
            try:
                worker.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("processing worker cleanup failed", errors)

    def __enter__(self) -> ProcessingSupervisor:
        """Start all workers for bounded context-managed use."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Release all worker processes."""
        self.close()


def _local_status(
    solver_id: str,
    state: SolverState,
    *,
    worker_pid: int | None = None,
    restart_count: int = 0,
    gpu_id: int | None = None,
    allocated_memory_bytes: int = 0,
) -> SolverStatus:
    return SolverStatus(
        solver_id=solver_id,
        state=state,
        heartbeat_ns=time.monotonic_ns(),
        progress=0.0,
        input_cursor=0,
        output_cursor=0,
        queue_depth=0,
        worker_pid=worker_pid,
        gpu_id=gpu_id,
        allocated_memory_bytes=allocated_memory_bytes,
        observed_memory_bytes=0,
        restart_count=restart_count,
        cancellation_requested=False,
    )


__all__ = [
    "ProcessingBatchResult",
    "ProcessingSupervisor",
    "SolverWorker",
    "WorkerJobError",
    "WorkerUnavailableError",
]
