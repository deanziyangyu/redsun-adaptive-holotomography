from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr

from redsun_aht.__main__ import main
from redsun_aht.configurations import build_dpct_simulation
from redsun_aht.domain import ArrayReference, Observation, ProcessingJob, SolverState
from redsun_aht.processing import (
    ProcessingSupervisor,
    SolverWorker,
    WorkerJobError,
    observations_from_dpct_bundle,
    read_observation_array,
)
from redsun_aht.processing.messages import (
    job_to_payload,
    observation_to_payload,
    pack,
    unpack,
)
from redsun_aht.processing.solvers import (
    KernelOutput,
    MeanProjectionSolver,
    ProcessingCancelled,
    build_solver,
)
from redsun_aht.processing.supervisor import WorkerUnavailableError
from redsun_aht.processing.worker import (
    _execute_job,
    _read_result,
    _verify_product,
    _write_product,
)


def _input_bundle(tmp_path: Path, run_id: str = "processing-run") -> Path:
    root = tmp_path / "input"
    asyncio.run(build_dpct_simulation(run_id, output_root=root).run())
    return root


def _jobs(
    observations: tuple[Observation, ...], output: Path
) -> dict[str, ProcessingJob]:
    selectors = tuple(item.observation_id for item in observations)
    run_id = observations[0].run_id
    return {
        solver_id: ProcessingJob(
            job_id=f"{run_id}-{solver_id}",
            solver_id=solver_id,
            solver_version="1.0.0",
            input_selectors=selectors,
            configuration={"simulation_delay_s": 0.1},
            priority=0,
            latency_class="offline",
            resource_request={"cpu_threads": 1, "gpu_ids": []},
            output_destination=str((output / solver_id).resolve()),
            cancellation_policy="cooperative",
        )
        for solver_id in ("mean-projection", "quality-metrics")
    }


def _requests(
    jobs: dict[str, ProcessingJob], observations: tuple[Observation, ...]
) -> dict[str, tuple[ProcessingJob, tuple[Observation, ...]]]:
    return {solver_id: (job, observations) for solver_id, job in jobs.items()}


def _array_observation(
    tmp_path: Path,
    shape: tuple[int, ...],
    *,
    name: str,
    run_id: str = "array-run",
) -> Observation:
    root = tmp_path / f"{name}.zarr"
    group = zarr.open_group(root, mode="w", zarr_format=3)
    payload = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    group.create_array("0", data=payload, chunks=shape)
    return Observation(
        run_id=run_id,
        observation_id=f"{run_id}:{name}",
        detector_id=name,
        channel_id="channel",
        sequence=0,
        array=ArrayReference(
            uri=f"{root.resolve().as_uri()}#0",
            checksum=hashlib.sha256(payload.tobytes(order="C")).hexdigest(),
            shape=shape,
            dtype="uint8",
            byte_order="|",
        ),
        committed=True,
        configuration_revision="revision",
        calibration_ids=(),
        replay_key=f"replay-{name}",
        latency_class="offline",
    )


def test_dpct_replay_builds_reference_only_verified_observations(
    tmp_path: Path,
) -> None:
    root = _input_bundle(tmp_path)

    observations = observations_from_dpct_bundle(root)
    payload = read_observation_array(observations[0])

    assert tuple(item.detector_id for item in observations) == (
        "dhm",
        "fluorescence",
    )
    assert payload.shape == (2, 2, 8, 8)
    assert observations[0].array.uri.startswith("file:///")
    assert observations[0].array.uri.endswith("#detectors/dhm/0")

    message = pack(
        "run",
        {
            "job": job_to_payload(
                _jobs(observations, tmp_path / "output")["mean-projection"]
            ),
            "observations": [observation_to_payload(item) for item in observations],
        },
    )
    kind, body = unpack(message)
    assert kind == "run"
    assert body["observations"][0]["array"]["uri"] == observations[0].array.uri
    assert "data" not in body["observations"][0]


def test_processing_rejects_invalid_messages_references_and_solver_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="kind cannot be empty"):
        pack("")
    with pytest.raises(ValueError, match="invalid processing worker envelope"):
        unpack(b"\x81\xaeschema_version\x02")
    with pytest.raises(ValueError, match="unknown solver plugin"):
        build_solver("unknown")

    valid = _array_observation(tmp_path, (1, 1, 2, 2), name="valid")
    jobs = _jobs((valid,), tmp_path / "solver-products")
    job = jobs["mean-projection"]
    solver = MeanProjectionSolver()

    with pytest.raises(ValueError, match="requires committed observations"):
        solver.run(job, (), lambda: False)
    with pytest.raises(ProcessingCancelled, match="cancelled"):
        solver.run(job, (valid,), lambda: True)
    with pytest.raises(ValueError, match="simulation_delay_s"):
        solver.run(
            replace(job, configuration={"simulation_delay_s": "bad"}),
            (valid,),
            lambda: False,
        )
    with pytest.raises(ValueError, match="version does not match"):
        solver.run(replace(job, solver_version="2.0.0"), (valid,), lambda: False)

    three_dimensional = _array_observation(
        tmp_path, (1, 2, 2), name="three-dimensional"
    )
    with pytest.raises(ValueError, match="require CZYX"):
        solver.run(
            replace(job, input_selectors=(three_dimensional.observation_id,)),
            (three_dimensional,),
            lambda: False,
        )
    different_shape = _array_observation(tmp_path, (1, 1, 3, 3), name="different-shape")
    with pytest.raises(ValueError, match="share one CZYX shape"):
        solver.run(
            replace(
                job,
                input_selectors=(valid.observation_id, different_shape.observation_id),
            ),
            (valid, different_shape),
            lambda: False,
        )

    invalid_uri = replace(valid, array=replace(valid.array, uri="https://invalid/0"))
    with pytest.raises(ValueError, match="file Zarr array"):
        read_observation_array(invalid_uri)
    invalid_checksum = replace(valid, array=replace(valid.array, checksum="0" * 64))
    with pytest.raises(ValueError, match="integrity check failed"):
        read_observation_array(invalid_checksum)


def test_processing_publication_and_supervisor_validation_paths(tmp_path: Path) -> None:
    observation = _array_observation(tmp_path, (1, 1, 2, 2), name="publication")
    job = _jobs((observation,), tmp_path / "publication-products")["mean-projection"]
    solver = MeanProjectionSolver()

    with pytest.raises(ValueError, match="worker solver identity"):
        _execute_job(
            "quality-metrics",
            "1.0.0",
            solver.run,
            job,
            (observation,),
            lambda: False,
        )
    with pytest.raises(ValueError, match="ordered selectors"):
        _execute_job(
            "mean-projection",
            "1.0.0",
            solver.run,
            replace(job, input_selectors=("other",)),
            (observation,),
            lambda: False,
        )
    with pytest.raises(ValueError, match="belong to one run"):
        second_run = replace(observation, run_id="second-run", observation_id="second")
        _execute_job(
            "mean-projection",
            "1.0.0",
            solver.run,
            replace(job, input_selectors=(observation.observation_id, "second")),
            (observation, second_run),
            lambda: False,
        )
    with pytest.raises(ValueError, match="absolute path"):
        _execute_job(
            "mean-projection",
            "1.0.0",
            solver.run,
            replace(job, output_destination="relative-output"),
            (observation,),
            lambda: False,
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "partial.txt").write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="incomplete or occupied"):
        _execute_job(
            "mean-projection",
            "1.0.0",
            solver.run,
            replace(job, output_destination=str(occupied.resolve())),
            (observation,),
            lambda: False,
        )

    cancelled_output = tmp_path / "cancelled-before-publication"
    with pytest.raises(RuntimeError, match="cancelled before publication"):
        _execute_job(
            "mean-projection",
            "1.0.0",
            lambda *_: KernelOutput(np.zeros((1,), dtype=np.float32), ("x",), {}),
            replace(job, output_destination=str(cancelled_output.resolve())),
            (observation,),
            lambda: True,
        )
    with pytest.raises(ValueError, match="dimensions do not match"):
        _write_product(
            tmp_path / "invalid-product",
            "mean-projection",
            "job",
            KernelOutput(np.zeros((1,), dtype=np.float32), ("x", "y"), {}),
        )
    with pytest.raises(ValueError, match="local Zarr array"):
        _verify_product(
            ArrayReference("https://invalid/product", "0" * 64, (1,), "uint8", "|")
        )

    invalid_manifest = tmp_path / "invalid-result.json"
    invalid_manifest.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid processing result manifest"):
        _read_result(invalid_manifest)

    with pytest.raises(ValueError, match="identity cannot be empty"):
        SolverWorker("")
    with pytest.raises(ValueError, match="timeouts must be positive"):
        SolverWorker("mean-projection", startup_timeout_s=0)
    idle = SolverWorker("mean-projection")
    with pytest.raises(RuntimeError, match="has not started"):
        _ = idle.capabilities
    with pytest.raises(RuntimeError, match="no pending job"):
        idle.wait()
    with pytest.raises(WorkerUnavailableError, match="not running"):
        idle.submit(job, (observation,))
    idle.cancel()
    idle.close()

    with pytest.raises(ValueError, match="IDs must be unique"):
        ProcessingSupervisor(())
    with pytest.raises(ValueError, match="IDs must be unique"):
        ProcessingSupervisor(("mean-projection", "mean-projection"))
    supervisor = ProcessingSupervisor(("mean-projection",))
    with pytest.raises(ValueError, match="address every supervised solver"):
        supervisor.run_batch({})


def test_two_spawned_solvers_overlap_restart_and_fail_independently(
    tmp_path: Path,
) -> None:
    observations = observations_from_dpct_bundle(_input_bundle(tmp_path))
    jobs = _jobs(observations, tmp_path / "products")
    supervisor = ProcessingSupervisor(tuple(jobs))
    try:
        capabilities = supervisor.start()
        first_pids = {
            solver_id: worker.process_id
            for solver_id, worker in supervisor.workers.items()
        }
        assert len(set(first_pids.values())) == 2
        assert all(pid is not None for pid in first_pids.values())
        assert all(item.batch_supported for item in capabilities.values())

        first = supervisor.run_batch(_requests(jobs, observations))

        assert not first.failures
        assert set(first.results) == {"mean-projection", "quality-metrics"}
        assert max(item.started_ns for item in first.results.values()) < min(
            item.completed_ns for item in first.results.values()
        )
        mean_array = cast(
            Any,
            zarr.open_group(
                Path(jobs["mean-projection"].output_destination) / "product.ome.zarr",
                mode="r",
            )["0"],
        )
        mean = np.asarray(mean_array[:])
        quality = np.asarray(
            zarr.open_array(
                Path(jobs["quality-metrics"].output_destination) / "product.zarr",
                mode="r",
            )[:]
        )
        assert mean.shape == (2, 8, 8)
        assert quality.shape == (2, 2, 2)

        old_mean_pid = supervisor.workers["mean-projection"].process_id
        quality_pid = supervisor.workers["quality-metrics"].process_id
        supervisor.workers["mean-projection"].restart()
        assert supervisor.workers["mean-projection"].process_id != old_mean_pid
        assert supervisor.workers["quality-metrics"].process_id == quality_pid

        replayed = supervisor.run_batch(_requests(jobs, observations))

        assert not replayed.failures
        assert all(result.cached for result in replayed.results.values())

        mean_array = cast(
            Any,
            zarr.open_group(
                Path(jobs["mean-projection"].output_destination) / "product.ome.zarr",
                mode="a",
            )["0"],
        )
        mean_array[0, 0, 0] = float(mean_array[0, 0, 0]) + 1

        isolated = supervisor.run_batch(_requests(jobs, observations))

        assert "mean-projection" in isolated.failures
        assert isolated.results["quality-metrics"].cached
        assert isolated.statuses["mean-projection"].state is SolverState.FAILED
        assert isolated.statuses["quality-metrics"].state is SolverState.WAITING_INPUT
    finally:
        supervisor.close()


def test_solver_worker_cancels_cooperatively(tmp_path: Path) -> None:
    observations = observations_from_dpct_bundle(_input_bundle(tmp_path, "cancel-run"))
    jobs = _jobs(observations, tmp_path / "cancel-products")
    worker = SolverWorker("quality-metrics")
    try:
        worker.start()
        worker.submit(jobs["quality-metrics"], observations)
        worker.cancel()

        with pytest.raises(WorkerJobError, match="cancelled"):
            worker.wait()

        assert worker.status.state is SolverState.FAILED
        assert worker.status.cancellation_requested
        assert not (
            Path(jobs["quality-metrics"].output_destination) / "result-manifest.json"
        ).exists()
    finally:
        worker.close()


def test_process_headless_cli_runs_without_hardware(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = _input_bundle(tmp_path, "cli-processing-run")
    output_root = tmp_path / "cli-products"

    assert (
        main(
            [
                "process-headless",
                "--input",
                str(input_root),
                "--output",
                str(output_root),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "mean-projection: computed" in output
    assert "quality-metrics: computed" in output
    assert (output_root / "mean-projection" / "result-manifest.json").is_file()
    assert (output_root / "quality-metrics" / "result-manifest.json").is_file()
