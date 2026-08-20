from __future__ import annotations

import sys
from pathlib import Path

from redsun_aht.configurations import PROFILES, DeviceSelection, build_simulation
from redsun_aht.domain import BackendKind, RunState


def test_all_backend_selections_are_inert(
    backend_selection: DeviceSelection,
) -> None:
    assert backend_selection.backend in BackendKind


def test_build_simulation_is_inspectable_and_hardware_free(tmp_path: Path) -> None:
    application = build_simulation(tmp_path / "events.jsonl", run_id="run-1")

    assert application.profile is PROFILES["simulate"]
    assert not application.profile.constructs_hardware
    assert application.controller.snapshot.state is RunState.NEW
    assert "arh_sim" not in sys.modules
    assert not any(name.lower().startswith("slar") for name in sys.modules)


def test_simulation_completes_and_replays(tmp_path: Path) -> None:
    application = build_simulation(tmp_path / "events.jsonl", run_id="run-1")

    assert application.run().state is RunState.COMPLETED
    assert len(application.controller.journal.read()) == 2
