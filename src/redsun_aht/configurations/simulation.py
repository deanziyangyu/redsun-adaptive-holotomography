"""Hardware-free Phase 1 run-lifecycle composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from redsun_aht.domain import RunEventKind, RunSnapshot, RunState
from redsun_aht.storage import EventJournal, RunController, replay_events

from .profiles import PROFILES, ApplicationProfile

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class SimulationApplication:
    """Minimal composition used to prove lifecycle and replay contracts."""

    profile: ApplicationProfile
    controller: RunController

    def run(self) -> RunSnapshot:
        """Complete one deterministic no-hardware simulated run."""
        self.controller.record(RunEventKind.STARTED)
        completed = self.controller.record(RunEventKind.COMPLETED)
        replayed = replay_events(completed.run_id, self.controller.journal.read())
        if replayed != completed:
            raise RuntimeError("journal replay did not reproduce the completed state")
        return replayed


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
    return SimulationApplication(profile=profile, controller=controller)


def run_simulation(journal_path: Path) -> RunSnapshot:
    """Build and run the hardware-free simulation composition."""
    return build_simulation(journal_path).run()
