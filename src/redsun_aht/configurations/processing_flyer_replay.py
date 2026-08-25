"""Hardware-free Bluesky replay through two processing flyers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from bluesky import RunEngine
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp

from redsun_aht.acquisition import DocumentRecord
from redsun_aht.device.processing import ProcessingFlyerDevice
from redsun_aht.domain import ProcessingJob
from redsun_aht.processing import SolverWorker, observations_from_dpct_bundle
from redsun_aht.storage import DocumentJournal

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

    from bluesky.utils import Msg

    from redsun_aht.domain import Observation, ProcessingResult
    from redsun_aht.domain.models import JsonValue


@dataclass(frozen=True, slots=True)
class ProcessingFlyerReplayResult:
    """Collected documents and independently resolved flyer outcomes."""

    source_run_id: str
    documents: tuple[DocumentRecord, ...]
    results: Mapping[str, ProcessingResult]
    failures: Mapping[str, str]
    journal_path: Path

    def __post_init__(self) -> None:
        """Detach outcome mappings from the mutable flyer composition."""
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))


@dataclass(slots=True)
class ProcessingFlyerReplay:
    """Inspectable two-flyer replay composition with no hardware devices."""

    source_run_id: str
    observations: tuple[Observation, ...]
    jobs: tuple[ProcessingJob, ...]
    flyers: tuple[ProcessingFlyerDevice, ...]
    input_root: Path
    output_root: Path

    def plan(self) -> Generator[Msg, Any, None]:
        """Return the standard staged Bluesky processing-flyer plan."""
        return processing_flyer_plan(
            self.flyers,
            self.jobs,
            self.observations,
            source_run_id=self.source_run_id,
            source_bundle_uri=self.input_root.as_uri(),
        )

    def run(self) -> ProcessingFlyerReplayResult:
        """Run the flyer plan and durably journal every emitted document."""
        journal_path = self.output_root / "processing-documents.jsonl"
        if journal_path.exists():
            raise FileExistsError(f"processing document journal exists: {journal_path}")
        journal = DocumentJournal(journal_path)
        documents: list[DocumentRecord] = []

        def record(name: str, document: Mapping[str, Any]) -> None:
            item = DocumentRecord(
                name,
                cast("Mapping[str, JsonValue]", document),
            )
            journal.append(item)
            documents.append(item)

        engine = RunEngine({})
        engine(self.plan(), record)
        results = {
            flyer.worker.solver_id: flyer.result
            for flyer in self.flyers
            if flyer.result is not None
        }
        failures = {
            flyer.worker.solver_id: flyer.failure
            for flyer in self.flyers
            if flyer.failure is not None
        }
        return ProcessingFlyerReplayResult(
            self.source_run_id,
            tuple(documents),
            results,
            failures,
            journal_path,
        )


def build_processing_flyer_replay(
    input_root: Path, output_root: Path
) -> ProcessingFlyerReplay:
    """Build a two-solver replay composition without starting worker processes."""
    observations = observations_from_dpct_bundle(input_root)
    if not observations:
        raise ValueError("processing flyer replay requires detector observations")
    source_run_id = observations[0].run_id
    if any(item.run_id != source_run_id for item in observations):
        raise ValueError("processing flyer replay observations span multiple runs")
    selectors = tuple(item.observation_id for item in observations)
    solver_ids = ("mean-projection", "quality-metrics")
    jobs = tuple(
        ProcessingJob(
            job_id=f"{source_run_id}-{solver_id}-flyer",
            solver_id=solver_id,
            solver_version="1.0.0",
            input_selectors=selectors,
            configuration={"execution_mode": "replay", "simulation_delay_s": 0.05},
            priority=0,
            latency_class="near-live",
            resource_request={"cpu_threads": 1, "gpu_ids": []},
            output_destination=str((output_root / solver_id).resolve()),
            cancellation_policy="cooperative",
        )
        for solver_id in solver_ids
    )
    flyers = tuple(
        ProcessingFlyerDevice(
            SolverWorker(job.solver_id), name=f"{job.solver_id}-flyer"
        )
        for job in jobs
    )
    return ProcessingFlyerReplay(
        source_run_id,
        observations,
        jobs,
        flyers,
        input_root.resolve(),
        output_root.resolve(),
    )


def run_processing_flyer_replay(
    input_root: Path, output_root: Path
) -> ProcessingFlyerReplayResult:
    """Build and execute the hardware-free two-flyer replay workflow."""
    return build_processing_flyer_replay(input_root, output_root).run()


def processing_flyer_plan(
    flyers: tuple[ProcessingFlyerDevice, ...],
    jobs: tuple[ProcessingJob, ...],
    observations: tuple[Observation, ...],
    *,
    source_run_id: str,
    source_bundle_uri: str | None = None,
) -> Generator[Msg, Any, None]:
    """Stage, run, drain, and collect independent processing flyers."""
    if not flyers or len(flyers) != len(jobs):
        raise ValueError("processing flyer plan requires one job per flyer")

    def body() -> Generator[Msg, Any, None]:
        for flyer, job in zip(flyers, jobs, strict=True):
            yield from bps.prepare(flyer, job, group="processing-prepare")
        yield from bps.wait(group="processing-prepare")
        yield from bps.kickoff_all(*flyers, group="processing-kickoff", wait=True)
        admission_tasks = yield from bps.wait_for(
            [
                lambda flyer=flyer, observation=observation: flyer.accept(observation)
                for flyer in flyers
                for observation in observations
            ]
        )
        admission_errors: list[BaseException] = []
        for task in admission_tasks:
            try:
                task.result()
            except BaseException as error:
                admission_errors.append(error)
        if len(admission_errors) == 1:
            raise admission_errors[0]
        if admission_errors:
            raise BaseExceptionGroup(
                "processing flyer observation admission failed", admission_errors
            )
        yield from bps.complete_all(*flyers, group="processing-complete", wait=True)
        for flyer in flyers:
            yield from bps.collect(flyer)

    staged = bpp.stage_wrapper(body(), flyers)  # type: ignore[no-untyped-call]
    metadata = {
        "aht_source_run_id": source_run_id,
        "aht_processing_mode": "replay",
    }
    if source_bundle_uri is not None:
        metadata["aht_source_bundle_uri"] = source_bundle_uri
    yield from bpp.run_wrapper(  # type: ignore[no-untyped-call]
        staged,
        md=metadata,
    )


__all__ = [
    "ProcessingFlyerReplay",
    "ProcessingFlyerReplayResult",
    "build_processing_flyer_replay",
    "processing_flyer_plan",
    "run_processing_flyer_replay",
]
