"""Hardware-free Phase 1 run-lifecycle composition."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from redsun_aht import __version__
from redsun_aht.acquisition import compose_simulation_documents
from redsun_aht.domain import RunEventKind, RunSnapshot, RunState
from redsun_aht.storage import (
    DocumentJournal,
    EventJournal,
    RunController,
    RunManifest,
    RunManifestStore,
    replay_events,
)
from redsun_aht.streams import PRIMARY_STREAM

from .profiles import PROFILES, ApplicationProfile
from .redsun_simulation import build_redsun_simulation_container

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Literal

    from redsun_aht.domain.models import JsonValue

    from .redsun_simulation import RedSunSimulationContainer


@dataclass(slots=True)
class SimulationApplication:
    """Minimal composition used to prove lifecycle and replay contracts."""

    profile: ApplicationProfile
    controller: RunController
    document_journal: DocumentJournal
    manifest: RunManifest
    manifest_store: RunManifestStore
    container: RedSunSimulationContainer

    def run(self, transitions: Sequence[RunEventKind] | None = None) -> RunSnapshot:
        """Execute and replay one deterministic no-hardware scenario."""
        scenario = transitions or (RunEventKind.STARTED, RunEventKind.COMPLETED)
        self.container.lifecycle.announce_started("simulate")
        try:
            snapshot = self.controller.snapshot
            for transition in scenario:
                detail: dict[str, JsonValue] | None = (
                    {"reason": "injected simulation failure"}
                    if transition is RunEventKind.FAILED
                    else None
                )
                snapshot = self.controller.record(transition, detail)
            terminal_states = {
                RunState.CANCELLED,
                RunState.FAILED,
                RunState.COMPLETED,
            }
            if snapshot.state not in terminal_states:
                raise RuntimeError("simulation scenario did not reach a terminal state")
            timestamp_ns = self.controller.journal.read()[0].timestamp_ns
            exit_status: Literal["success", "abort", "fail"]
            if snapshot.state is RunState.CANCELLED:
                exit_status = "abort"
            elif snapshot.state is RunState.FAILED:
                exit_status = "fail"
            else:
                exit_status = "success"
            documents = compose_simulation_documents(
                snapshot.run_id,
                timestamp_ns=timestamp_ns,
                state=snapshot.state.value,
                exit_status=exit_status,
                reason=snapshot.failure or "",
            )
            self.document_journal.extend(documents)
            replayed = replay_events(snapshot.run_id, self.controller.journal.read())
            if replayed != snapshot:
                raise RuntimeError(
                    "journal replay did not reproduce the terminal state"
                )
            if self.document_journal.read() != documents:
                raise RuntimeError(
                    "document replay did not reproduce the emitted documents"
                )
            if self.manifest_store.read() != self.manifest:
                raise RuntimeError("manifest replay did not reproduce the run snapshot")
            return replayed
        finally:
            self.container.lifecycle.announce_finished()
            self.container.shutdown()


def build_simulation(
    journal_path: Path, *, run_id: str | None = None
) -> SimulationApplication:
    """Build, but do not run, the hardware-free simulation composition."""
    profile = PROFILES["simulate"]
    if profile.constructs_hardware:
        raise AssertionError("simulation profile must never construct hardware")
    resolved_run_id = run_id or str(uuid4())
    controller = RunController(resolved_run_id, EventJournal(journal_path))
    if controller.snapshot.state is not RunState.NEW:
        raise ValueError("simulation journal must be empty for a new run")
    document_path = journal_path.with_name(f"{journal_path.stem}.documents.jsonl")
    document_journal = DocumentJournal(document_path)
    if document_journal.read():
        raise ValueError("simulation document journal must be empty for a new run")
    manifest_path = journal_path.with_name(f"{journal_path.stem}.manifest.json")
    if manifest_path.exists():
        raise ValueError("simulation run manifest must not already exist")
    manifest = RunManifest(
        schema_version=1,
        run_id=resolved_run_id,
        created_ns=time.time_ns(),
        software_version=__version__,
        hardware={
            "devices": [
                {
                    "id": selection.device_id,
                    "backend": selection.backend.value,
                }
                for selection in profile.devices
            ]
        },
        reconstruction={"plugins": []},
        experiment={"recipe": "simulation", "streams": [PRIMARY_STREAM]},
        deployment={"profile": profile.name, "frontend": "headless"},
    )
    manifest_store = RunManifestStore(manifest_path)
    manifest_store.write(manifest)
    container = build_redsun_simulation_container(journal_path.parent).build()
    return SimulationApplication(
        profile=profile,
        controller=controller,
        document_journal=document_journal,
        manifest=manifest,
        manifest_store=manifest_store,
        container=container,
    )


def run_simulation(journal_path: Path) -> RunSnapshot:
    """Build and run the hardware-free simulation composition."""
    return build_simulation(journal_path).run()
