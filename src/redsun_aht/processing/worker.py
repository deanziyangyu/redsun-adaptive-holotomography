"""Spawn-safe persistent local solver worker implementation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit

import numpy as np
import zarr

from redsun_aht.domain import (
    ArrayReference,
    ProcessingJob,
    ProcessingResult,
    SolverState,
    SolverStatus,
)
from redsun_aht.domain.models import thaw_json
from redsun_aht.processing.messages import (
    capabilities_to_payload,
    job_from_payload,
    observation_from_payload,
    pack,
    result_from_payload,
    result_to_payload,
    status_to_payload,
    unpack,
)
from redsun_aht.processing.solvers import KernelOutput, build_solver

if TYPE_CHECKING:
    from collections.abc import Mapping

    from redsun_aht.domain import Observation


def processing_worker_main(
    solver_id: str,
    command_queue: Any,
    response_queue: Any,
    cancel_event: Any,
    restart_count: int,
    gpu_id: int | None = None,
    memory_reservation_bytes: int = 0,
    gpu_cache_directory: str | None = None,
) -> None:
    """Own one solver instance and service versioned reference-only commands."""
    solver = build_solver(
        solver_id, gpu_id=gpu_id, gpu_cache_directory=gpu_cache_directory
    )
    input_cursor = 0
    output_cursor = 0
    response_queue.put(
        pack(
            "ready",
            {
                "capabilities": capabilities_to_payload(solver.capabilities),
                "status": status_to_payload(
                    _solver_status(
                        solver,
                        solver_id,
                        SolverState.WAITING_INPUT,
                        input_cursor,
                        output_cursor,
                        restart_count,
                        memory_reservation_bytes,
                    )
                ),
            },
        )
    )
    try:
        while True:
            kind, body = unpack(command_queue.get())
            if kind == "shutdown":
                break
            if kind != "run":
                raise ValueError(f"unsupported processing worker command: {kind}")
            job = job_from_payload(body["job"])
            observations = tuple(
                observation_from_payload(item) for item in body["observations"]
            )
            response_queue.put(
                pack(
                    "status",
                    {
                        "status": status_to_payload(
                            _solver_status(
                                solver,
                                solver_id,
                                SolverState.RUNNING,
                                input_cursor,
                                output_cursor,
                                restart_count,
                                memory_reservation_bytes,
                            )
                        )
                    },
                )
            )
            try:
                _validate_worker_resources(job, solver.gpu_id, memory_reservation_bytes)
                result = _execute_job(
                    solver.capabilities.solver_id,
                    solver.capabilities.solver_version,
                    solver.run,
                    job,
                    observations,
                    cancel_event.is_set,
                )
                input_cursor += len(observations)
                output_cursor += 1
                response_queue.put(
                    pack(
                        "result",
                        {
                            "result": result_to_payload(result),
                            "status": status_to_payload(
                                _solver_status(
                                    solver,
                                    solver_id,
                                    SolverState.WAITING_INPUT,
                                    input_cursor,
                                    output_cursor,
                                    restart_count,
                                    memory_reservation_bytes,
                                )
                            ),
                        },
                    )
                )
            except Exception as error:
                response_queue.put(
                    pack(
                        "failure",
                        {
                            "job_id": job.job_id,
                            "error_type": type(error).__qualname__,
                            "message": str(error),
                            "status": status_to_payload(
                                _solver_status(
                                    solver,
                                    solver_id,
                                    SolverState.FAILED,
                                    input_cursor,
                                    output_cursor,
                                    restart_count,
                                    memory_reservation_bytes,
                                    cancellation_requested=cancel_event.is_set(),
                                    last_error=str(error),
                                )
                            ),
                        },
                    )
                )
    finally:
        solver.close()
        response_queue.put(
            pack(
                "stopped",
                {
                    "status": status_to_payload(
                        _solver_status(
                            solver,
                            solver_id,
                            SolverState.STOPPED,
                            input_cursor,
                            output_cursor,
                            restart_count,
                            memory_reservation_bytes,
                        )
                    )
                },
            )
        )


def _execute_job(
    solver_id: str,
    solver_version: str,
    run_kernel: Any,
    job: ProcessingJob,
    observations: tuple[Observation, ...],
    cancelled: Any,
) -> ProcessingResult:
    if job.solver_id != solver_id or job.solver_version != solver_version:
        raise ValueError("processing job does not match worker solver identity")
    if tuple(observation.observation_id for observation in observations) != (
        job.input_selectors
    ):
        raise ValueError("processing observations do not match ordered selectors")
    run_ids = {observation.run_id for observation in observations}
    if len(run_ids) != 1:
        raise ValueError("processing job observations must belong to one run")
    output_root = Path(job.output_destination)
    if not output_root.is_absolute():
        raise ValueError("processing output destination must be an absolute path")
    fingerprint = _configuration_fingerprint(job, observations)
    manifest_path = output_root / "result-manifest.json"
    if manifest_path.is_file():
        cached = _read_result(manifest_path)
        if (
            cached.job_id != job.job_id
            or cached.solver_id != solver_id
            or cached.configuration_fingerprint != fingerprint
            or cached.input_checksums
            != tuple(observation.array.checksum for observation in observations)
        ):
            raise ValueError("existing processing result does not match submitted job")
        provenance_checksum = cached.metrics.get("provenance_sha256")
        _verify_product(
            cached.output,
            expected_provenance_checksum=(
                provenance_checksum if isinstance(provenance_checksum, str) else None
            ),
        )
        return replace(cached, cached=True)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"processing output is incomplete or occupied: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    started_ns = time.time_ns()
    output = cast("KernelOutput", run_kernel(job, observations, cancelled))
    if cancelled():
        raise RuntimeError("processing job was cancelled before publication")
    _, product_uri, stored = _write_product(output_root, solver_id, job.job_id, output)
    checksum = hashlib.sha256(stored.tobytes(order="C")).hexdigest()
    if not np.array_equal(stored, output.array, equal_nan=True):
        raise OSError("processing product readback differs from solver output")
    completed_ns = max(time.time_ns(), started_ns)
    metrics = dict(output.metrics)
    if output.provenance:
        metrics["provenance_sha256"] = _provenance_checksum(output.provenance)
    result = ProcessingResult(
        result_id=hashlib.sha256(
            f"{job.job_id}\0{fingerprint}\0{checksum}".encode()
        ).hexdigest(),
        job_id=job.job_id,
        run_id=run_ids.pop(),
        solver_id=solver_id,
        solver_version=solver_version,
        output=ArrayReference(
            uri=product_uri,
            checksum=checksum,
            shape=tuple(stored.shape),
            dtype=str(stored.dtype),
            byte_order=_canonical_byte_order(stored.dtype),
        ),
        metrics=cast("Mapping[str, Any]", metrics),
        input_checksums=tuple(
            observation.array.checksum for observation in observations
        ),
        configuration_fingerprint=fingerprint,
        started_ns=started_ns,
        completed_ns=completed_ns,
        worker_pid=os.getpid(),
    )
    _write_result(manifest_path, result)
    return result


def _validate_worker_resources(
    job: ProcessingJob,
    gpu_id: int | None,
    memory_reservation_bytes: int,
) -> None:
    if gpu_id is None:
        return
    gpu_ids = job.resource_request.get("gpu_ids")
    reservation = job.resource_request.get("gpu_memory_reservation_bytes")
    if (
        not isinstance(gpu_ids, tuple)
        or gpu_id not in gpu_ids
        or not isinstance(reservation, int)
        or isinstance(reservation, bool)
        or reservation != memory_reservation_bytes
    ):
        raise ValueError("GPU job resource request does not match worker placement")


def _configuration_fingerprint(
    job: ProcessingJob, observations: tuple[Observation, ...]
) -> str:
    payload = {
        "solver_id": job.solver_id,
        "solver_version": job.solver_version,
        "configuration": thaw_json(job.configuration),
        "resource_request": thaw_json(job.resource_request),
        "input_replay_keys": [item.replay_key for item in observations],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_result(path: Path, result: ProcessingResult) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                {"schema_version": 1, "result": result_to_payload(result)},
                stream,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_result(path: Path) -> ProcessingResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "result"}
            or payload["schema_version"] != 1
            or not isinstance(payload["result"], dict)
        ):
            raise TypeError("result manifest fields differ")
        return result_from_payload(payload["result"])
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid processing result manifest") from error


def _verify_product(
    reference: ArrayReference,
    *,
    expected_provenance_checksum: str | None = None,
) -> None:
    parsed = urlsplit(reference.uri)
    if parsed.scheme != "file":
        raise ValueError("cached processing product must be one local Zarr array")
    path_text = unquote(parsed.path)
    if os.name == "nt" and path_text.startswith("/"):
        path_text = path_text[1:]
    if parsed.netloc:
        path_text = f"//{parsed.netloc}{path_text}"
    product_path = Path(path_text)
    if parsed.fragment:
        group = zarr.open_group(product_path, mode="r")
        array = cast("Any", group[parsed.fragment])
        attributes = dict(group.attrs)
    else:
        array = zarr.open_array(product_path, mode="r")
        attributes = dict(array.attrs)
    stored = np.asarray(array[:])
    checksum = hashlib.sha256(stored.tobytes(order="C")).hexdigest()
    if (
        checksum != reference.checksum
        or tuple(stored.shape) != reference.shape
        or str(stored.dtype) != reference.dtype
        or _canonical_byte_order(stored.dtype) != reference.byte_order
    ):
        raise ValueError("cached processing product integrity check failed")
    if expected_provenance_checksum is not None:
        aht_attributes = attributes.get("aht")
        if not isinstance(aht_attributes, dict):
            raise ValueError("cached processing product provenance is missing")
        provenance = aht_attributes.get("provenance")
        stored_checksum = aht_attributes.get("provenanceSha256")
        if (
            not isinstance(provenance, dict)
            or stored_checksum != expected_provenance_checksum
            or _provenance_checksum(provenance) != expected_provenance_checksum
        ):
            raise ValueError("cached processing product provenance check failed")


def _write_product(
    output_root: Path,
    solver_id: str,
    job_id: str,
    output: KernelOutput,
) -> tuple[Path, str, Any]:
    array = np.asarray(output.array)
    if len(output.dimension_names) != array.ndim:
        raise ValueError("solver output dimensions do not match named axes")
    aht_attributes: dict[str, Any] = {
        "solverId": solver_id,
        "jobId": job_id,
    }
    if output.provenance:
        provenance = thaw_json(output.provenance)
        aht_attributes.update(
            provenance=provenance,
            provenanceSha256=_provenance_checksum(output.provenance),
        )
    if output.ome_image:
        product_path = output_root / "product.ome.zarr"
        group = zarr.open_group(product_path, mode="w", zarr_format=3)
        axes = [{"name": name, "type": "space"} for name in output.dimension_names]
        group.attrs.update(
            {
                "ome": {
                    "version": "0.5",
                    "multiscales": [
                        {
                            "name": solver_id,
                            "axes": axes,
                            "datasets": [
                                {
                                    "path": "0",
                                    "coordinateTransformations": [
                                        {
                                            "type": "scale",
                                            "scale": [1.0] * array.ndim,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
                "aht": aht_attributes,
            }
        )
        created = group.create_array(
            "0",
            data=array,
            chunks=tuple(array.shape),
            dimension_names=output.dimension_names,
        )
        stored = np.asarray(created[:])
        uri = f"{product_path.resolve().as_uri()}#0"
    else:
        product_path = output_root / "product.zarr"
        created = zarr.create_array(
            product_path,
            data=array,
            chunks=tuple(array.shape),
            dimension_names=output.dimension_names,
            attributes={"aht": aht_attributes},
            zarr_format=3,
        )
        stored = np.asarray(created[:])
        uri = product_path.resolve().as_uri()
    return product_path, uri, stored


def _provenance_checksum(provenance: Mapping[str, Any]) -> str:
    payload = json.dumps(
        thaw_json(provenance), separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _status(
    solver_id: str,
    state: SolverState,
    input_cursor: int,
    output_cursor: int,
    restart_count: int,
    *,
    gpu_id: int | None = None,
    allocated_memory_bytes: int = 0,
    observed_memory_bytes: int = 0,
    cancellation_requested: bool = False,
    last_error: str | None = None,
) -> SolverStatus:
    return SolverStatus(
        solver_id=solver_id,
        state=state,
        heartbeat_ns=time.monotonic_ns(),
        progress=1.0 if output_cursor else 0.0,
        input_cursor=input_cursor,
        output_cursor=output_cursor,
        queue_depth=0,
        worker_pid=os.getpid(),
        gpu_id=gpu_id,
        allocated_memory_bytes=allocated_memory_bytes,
        observed_memory_bytes=observed_memory_bytes,
        restart_count=restart_count,
        cancellation_requested=cancellation_requested,
        last_error=last_error,
    )


def _solver_status(
    solver: Any,
    solver_id: str,
    state: SolverState,
    input_cursor: int,
    output_cursor: int,
    restart_count: int,
    memory_reservation_bytes: int,
    *,
    cancellation_requested: bool = False,
    last_error: str | None = None,
) -> SolverStatus:
    return _status(
        solver_id,
        state,
        input_cursor,
        output_cursor,
        restart_count,
        gpu_id=solver.gpu_id,
        allocated_memory_bytes=max(
            memory_reservation_bytes, int(solver.allocated_memory_bytes)
        ),
        observed_memory_bytes=int(solver.observed_memory_bytes),
        cancellation_requested=cancellation_requested,
        last_error=last_error,
    )


def _canonical_byte_order(dtype: np.dtype[Any]) -> str:
    if dtype.itemsize == 1:
        return "|"
    if dtype.byteorder == "=":
        return "<" if np.little_endian else ">"
    return dtype.byteorder


__all__ = ["processing_worker_main"]
