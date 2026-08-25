"""Versioned MsgPack payloads for local detached processing workers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import msgspec

from redsun_aht.domain import (
    ArrayReference,
    Observation,
    ProcessingJob,
    ProcessingResult,
    SolverCapabilities,
    SolverState,
    SolverStatus,
)
from redsun_aht.domain.models import JsonValue, thaw_json

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_VERSION = 1


def pack(kind: str, body: Mapping[str, object] | None = None) -> bytes:
    """Encode one bounded schema-v1 worker envelope."""
    if not kind:
        raise ValueError("processing message kind cannot be empty")
    return msgspec.msgpack.encode(
        {"schema_version": SCHEMA_VERSION, "kind": kind, "body": dict(body or {})}
    )


def unpack(payload: bytes) -> tuple[str, dict[str, Any]]:
    """Decode and structurally validate one worker envelope."""
    decoded = msgspec.msgpack.decode(payload)
    if (
        not isinstance(decoded, dict)
        or set(decoded) != {"schema_version", "kind", "body"}
        or decoded["schema_version"] != SCHEMA_VERSION
        or not isinstance(decoded["kind"], str)
        or not isinstance(decoded["body"], dict)
    ):
        raise ValueError("invalid processing worker envelope")
    return decoded["kind"], cast("dict[str, Any]", decoded["body"])


def array_reference_to_payload(reference: ArrayReference) -> dict[str, object]:
    """Serialize one external array reference."""
    return {
        "uri": reference.uri,
        "checksum": reference.checksum,
        "shape": list(reference.shape),
        "dtype": reference.dtype,
        "byte_order": reference.byte_order,
    }


def array_reference_from_payload(payload: Mapping[str, Any]) -> ArrayReference:
    """Reconstruct one external array reference."""
    return ArrayReference(
        uri=str(payload["uri"]),
        checksum=str(payload["checksum"]),
        shape=tuple(int(value) for value in payload["shape"]),
        dtype=str(payload["dtype"]),
        byte_order=str(payload["byte_order"]),
    )


def job_to_payload(job: ProcessingJob) -> dict[str, object]:
    """Serialize one immutable processing job."""
    return {
        "job_id": job.job_id,
        "solver_id": job.solver_id,
        "solver_version": job.solver_version,
        "input_selectors": list(job.input_selectors),
        "configuration": thaw_json(job.configuration),
        "priority": job.priority,
        "latency_class": job.latency_class,
        "resource_request": thaw_json(job.resource_request),
        "output_destination": job.output_destination,
        "cancellation_policy": job.cancellation_policy,
    }


def job_from_payload(payload: Mapping[str, Any]) -> ProcessingJob:
    """Reconstruct one immutable processing job."""
    return ProcessingJob(
        job_id=str(payload["job_id"]),
        solver_id=str(payload["solver_id"]),
        solver_version=str(payload["solver_version"]),
        input_selectors=tuple(str(value) for value in payload["input_selectors"]),
        configuration=cast("Mapping[str, JsonValue]", payload["configuration"]),
        priority=int(payload["priority"]),
        latency_class=str(payload["latency_class"]),
        resource_request=cast("Mapping[str, JsonValue]", payload["resource_request"]),
        output_destination=str(payload["output_destination"]),
        cancellation_policy=str(payload["cancellation_policy"]),
    )


def observation_to_payload(observation: Observation) -> dict[str, object]:
    """Serialize one committed observation reference."""
    return {
        "run_id": observation.run_id,
        "observation_id": observation.observation_id,
        "detector_id": observation.detector_id,
        "channel_id": observation.channel_id,
        "sequence": observation.sequence,
        "array": array_reference_to_payload(observation.array),
        "committed": observation.committed,
        "configuration_revision": observation.configuration_revision,
        "calibration_ids": list(observation.calibration_ids),
        "replay_key": observation.replay_key,
        "latency_class": observation.latency_class,
        "metadata": thaw_json(observation.metadata),
    }


def observation_from_payload(payload: Mapping[str, Any]) -> Observation:
    """Reconstruct one committed observation reference."""
    return Observation(
        run_id=str(payload["run_id"]),
        observation_id=str(payload["observation_id"]),
        detector_id=str(payload["detector_id"]),
        channel_id=str(payload["channel_id"]),
        sequence=int(payload["sequence"]),
        array=array_reference_from_payload(payload["array"]),
        committed=bool(payload["committed"]),
        configuration_revision=str(payload["configuration_revision"]),
        calibration_ids=tuple(str(value) for value in payload["calibration_ids"]),
        replay_key=str(payload["replay_key"]),
        latency_class=str(payload["latency_class"]),
        metadata=cast("Mapping[str, JsonValue]", payload["metadata"]),
    )


def capabilities_to_payload(capabilities: SolverCapabilities) -> dict[str, object]:
    """Serialize one solver capability declaration."""
    return {
        "solver_id": capabilities.solver_id,
        "solver_version": capabilities.solver_version,
        "dimensionality": list(capabilities.dimensionality),
        "required_channels": capabilities.required_channels,
        "required_shots": capabilities.required_shots,
        "streaming_supported": capabilities.streaming_supported,
        "batch_supported": capabilities.batch_supported,
        "incremental_supported": capabilities.incremental_supported,
        "live_supported": capabilities.live_supported,
        "output_types": list(capabilities.output_types),
        "requires_gpu": capabilities.requires_gpu,
        "gpu_runtime": capabilities.gpu_runtime,
    }


def capabilities_from_payload(payload: Mapping[str, Any]) -> SolverCapabilities:
    """Reconstruct one solver capability declaration."""
    gpu_runtime = payload.get("gpu_runtime")
    return SolverCapabilities(
        solver_id=str(payload["solver_id"]),
        solver_version=str(payload["solver_version"]),
        dimensionality=tuple(int(value) for value in payload["dimensionality"]),
        required_channels=int(payload["required_channels"]),
        required_shots=int(payload["required_shots"]),
        streaming_supported=bool(payload["streaming_supported"]),
        batch_supported=bool(payload["batch_supported"]),
        incremental_supported=bool(payload["incremental_supported"]),
        live_supported=bool(payload["live_supported"]),
        output_types=tuple(str(value) for value in payload["output_types"]),
        requires_gpu=bool(payload.get("requires_gpu", False)),
        gpu_runtime=None if gpu_runtime is None else str(gpu_runtime),
    )


def status_to_payload(status: SolverStatus) -> dict[str, object]:
    """Serialize one worker status snapshot."""
    return {
        "solver_id": status.solver_id,
        "state": status.state.value,
        "heartbeat_ns": status.heartbeat_ns,
        "progress": status.progress,
        "input_cursor": status.input_cursor,
        "output_cursor": status.output_cursor,
        "queue_depth": status.queue_depth,
        "worker_pid": status.worker_pid,
        "gpu_id": status.gpu_id,
        "allocated_memory_bytes": status.allocated_memory_bytes,
        "observed_memory_bytes": status.observed_memory_bytes,
        "restart_count": status.restart_count,
        "cancellation_requested": status.cancellation_requested,
        "last_error": status.last_error,
    }


def status_from_payload(payload: Mapping[str, Any]) -> SolverStatus:
    """Reconstruct one worker status snapshot."""
    worker_pid = payload["worker_pid"]
    gpu_id = payload["gpu_id"]
    return SolverStatus(
        solver_id=str(payload["solver_id"]),
        state=SolverState(str(payload["state"])),
        heartbeat_ns=int(payload["heartbeat_ns"]),
        progress=float(payload["progress"]),
        input_cursor=int(payload["input_cursor"]),
        output_cursor=int(payload["output_cursor"]),
        queue_depth=int(payload["queue_depth"]),
        worker_pid=None if worker_pid is None else int(worker_pid),
        gpu_id=None if gpu_id is None else int(gpu_id),
        allocated_memory_bytes=int(payload["allocated_memory_bytes"]),
        observed_memory_bytes=int(payload["observed_memory_bytes"]),
        restart_count=int(payload["restart_count"]),
        cancellation_requested=bool(payload["cancellation_requested"]),
        last_error=(
            None if payload["last_error"] is None else str(payload["last_error"])
        ),
    )


def result_to_payload(result: ProcessingResult) -> dict[str, object]:
    """Serialize one immutable processing result."""
    return {
        "result_id": result.result_id,
        "job_id": result.job_id,
        "run_id": result.run_id,
        "solver_id": result.solver_id,
        "solver_version": result.solver_version,
        "output": array_reference_to_payload(result.output),
        "metrics": thaw_json(result.metrics),
        "input_checksums": list(result.input_checksums),
        "configuration_fingerprint": result.configuration_fingerprint,
        "started_ns": result.started_ns,
        "completed_ns": result.completed_ns,
        "worker_pid": result.worker_pid,
        "cached": result.cached,
    }


def result_from_payload(payload: Mapping[str, Any]) -> ProcessingResult:
    """Reconstruct one immutable processing result."""
    return ProcessingResult(
        result_id=str(payload["result_id"]),
        job_id=str(payload["job_id"]),
        run_id=str(payload["run_id"]),
        solver_id=str(payload["solver_id"]),
        solver_version=str(payload["solver_version"]),
        output=array_reference_from_payload(payload["output"]),
        metrics=cast("Mapping[str, JsonValue]", payload["metrics"]),
        input_checksums=tuple(str(value) for value in payload["input_checksums"]),
        configuration_fingerprint=str(payload["configuration_fingerprint"]),
        started_ns=int(payload["started_ns"]),
        completed_ns=int(payload["completed_ns"]),
        worker_pid=int(payload["worker_pid"]),
        cached=bool(payload["cached"]),
    )


__all__ = [
    "SCHEMA_VERSION",
    "array_reference_from_payload",
    "array_reference_to_payload",
    "capabilities_from_payload",
    "capabilities_to_payload",
    "job_from_payload",
    "job_to_payload",
    "observation_from_payload",
    "observation_to_payload",
    "pack",
    "result_from_payload",
    "result_to_payload",
    "status_from_payload",
    "status_to_payload",
    "unpack",
]
