from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from redsun.engine import RunEngine

from redsun_aht.configurations import (
    build_dpct_simulation,
    processing_flyer_plan,
    run_processing_flyer_replay,
)
from redsun_aht.device.processing import (
    ProcessingDeviceAdapter,
    ProcessingFlyerDevice,
)
from redsun_aht.domain import ProcessingJob
from redsun_aht.processing import SolverWorker, observations_from_dpct_bundle
from redsun_aht.storage import DocumentJournal
from redsun_aht.streams import (
    PROCESSING_PROGRESS_STREAM,
    PROCESSING_RESULT_STREAM,
)

if TYPE_CHECKING:
    from event_model import PartialEvent

    from redsun_aht.domain import Observation


def _observations(tmp_path: Path) -> tuple[Observation, ...]:
    input_root = tmp_path / "input"
    asyncio.run(build_dpct_simulation("flyer-run", output_root=input_root).run())
    return observations_from_dpct_bundle(input_root)


def _job(
    solver_id: str, observations: tuple[Observation, ...], output: Path
) -> ProcessingJob:
    return ProcessingJob(
        job_id=f"flyer-run-{solver_id}",
        solver_id=solver_id,
        solver_version="1.0.0",
        input_selectors=tuple(item.observation_id for item in observations),
        configuration={"simulation_delay_s": 0.01},
        priority=0,
        latency_class="near-live",
        resource_request={"cpu_threads": 1, "gpu_ids": []},
        output_destination=str(output.resolve()),
        cancellation_policy="cooperative",
    )


async def _collect(flyer: ProcessingFlyerDevice) -> list[PartialEvent]:
    return [event async for event in flyer.collect()]


def test_processing_flyer_runs_detached_job_and_collects_references(
    tmp_path: Path,
) -> None:
    observations = _observations(tmp_path)
    job = _job("mean-projection", observations, tmp_path / "mean")
    flyer = ProcessingFlyerDevice(SolverWorker(job.solver_id), name="mean_flyer")

    async def exercise() -> None:
        await flyer.stage()
        await flyer.prepare(job)
        assert flyer.worker.process_id is not None
        await flyer.kickoff()
        for observation in reversed(observations):
            await flyer.accept(observation)
        await flyer.complete()

        descriptions = await flyer.describe_collect()
        assert set(descriptions) == {
            PROCESSING_PROGRESS_STREAM,
            PROCESSING_RESULT_STREAM,
        }
        events = await _collect(flyer)
        assert len(events) == 4
        assert not await _collect(flyer)
        result_event = events[-1]["data"]
        result = flyer.result
        assert result is not None
        assert result_event["result_id"] == result.result_id
        assert str(result_event["output_uri"]).endswith("product.ome.zarr#0")
        assert "array" not in result_event
        configuration = await flyer.describe_configuration()
        assert configuration["job_id"] == job.job_id
        await flyer.unstage()
        await flyer.stop()

    asyncio.run(exercise())

    assert isinstance(flyer, ProcessingDeviceAdapter)
    assert flyer.result is not None
    assert flyer.failure is None
    assert flyer.worker.process_id is None


def test_processing_flyer_validates_order_selection_and_coverage(
    tmp_path: Path,
) -> None:
    observations = _observations(tmp_path)
    job = _job("quality-metrics", observations, tmp_path / "quality")
    flyer = ProcessingFlyerDevice(SolverWorker(job.solver_id), name="quality_flyer")

    async def exercise() -> None:
        await flyer.stage()
        with pytest.raises(RuntimeError, match="not prepared"):
            await flyer.accept(observations[0])
        await flyer.prepare(job)
        with pytest.raises(RuntimeError, match="already prepared"):
            await flyer.prepare(job)
        await flyer.kickoff()
        await flyer.accept(observations[0])
        with pytest.raises(ValueError, match="duplicated"):
            await flyer.accept(observations[0])
        await flyer.complete()
        assert flyer.result is None
        assert flyer.failure is not None
        assert "missing=" in flyer.failure
        events = await _collect(flyer)
        assert events[-1]["data"]["solver_state"] == "failed"
        await flyer.unstage()

    asyncio.run(exercise())


def test_two_processing_flyers_run_in_one_bluesky_run(tmp_path: Path) -> None:
    observations = _observations(tmp_path)
    jobs = (
        _job("mean-projection", observations, tmp_path / "mean"),
        _job("quality-metrics", observations, tmp_path / "quality"),
    )
    flyers = tuple(
        ProcessingFlyerDevice(SolverWorker(job.solver_id), name=f"flyer_{index}")
        for index, job in enumerate(jobs)
    )
    documents: list[tuple[str, dict[str, Any]]] = []
    engine = RunEngine({})

    engine(
        processing_flyer_plan(flyers, jobs, observations, source_run_id="flyer-run"),
        lambda name, document: documents.append((name, document)),
    ).result()

    assert [name for name, _ in documents].count("start") == 1
    assert [name for name, _ in documents].count("stop") == 1
    assert {flyer.result.solver_id for flyer in flyers if flyer.result} == {
        "mean-projection",
        "quality-metrics",
    }
    assert len({flyer.result.worker_pid for flyer in flyers if flyer.result}) == 2
    descriptors = [document for name, document in documents if name == "descriptor"]
    assert {descriptor["name"] for descriptor in descriptors} == {
        PROCESSING_PROGRESS_STREAM,
        PROCESSING_RESULT_STREAM,
    }
    result_pages = [
        document
        for name, document in documents
        if name == "event_page" and "result_id" in document["data"]
    ]
    assert len(result_pages) == 2
    assert all(flyer.worker.process_id is None for flyer in flyers)


def test_processing_flyer_replay_journals_run_documents(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    asyncio.run(build_dpct_simulation("journal-run", output_root=input_root).run())
    output_root = tmp_path / "output"

    replay = run_processing_flyer_replay(input_root, output_root)

    assert replay.source_run_id == "journal-run"
    assert set(replay.results) == {"mean-projection", "quality-metrics"}
    assert not replay.failures
    assert replay.journal_path == output_root / "processing-documents.jsonl"
    assert DocumentJournal(replay.journal_path).read() == replay.documents
    assert replay.documents[0].name == "start"
    assert replay.documents[-1].name == "stop"
    assert replay.documents[0].document["aht_source_run_id"] == "journal-run"
    assert replay.documents[0].document["aht_source_bundle_uri"] == (
        input_root.resolve().as_uri()
    )
    with pytest.raises(FileExistsError, match="journal exists"):
        run_processing_flyer_replay(input_root, output_root)


def test_processing_flyer_plan_abort_unstages_every_worker(tmp_path: Path) -> None:
    observations = _observations(tmp_path)
    job = _job("mean-projection", observations[:1], tmp_path / "abort-output")
    flyer = ProcessingFlyerDevice(SolverWorker(job.solver_id), name="abort_flyer")
    engine = RunEngine({})

    with pytest.raises(ValueError, match="not selected"):
        engine(
            processing_flyer_plan(
                (flyer,), (job,), observations, source_run_id="flyer-run"
            )
        ).result()

    assert flyer.worker.process_id is None
    assert flyer.result is None
