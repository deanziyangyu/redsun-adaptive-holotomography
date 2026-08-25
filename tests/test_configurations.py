from __future__ import annotations

import sys
from pathlib import Path

import pytest

from redsun_aht.configurations import (
    PROFILES,
    DeviceSelection,
    build_dual_detector_simulation,
    build_simulation,
)
from redsun_aht.domain import BackendKind, RunEventKind, RunState


def test_all_backend_selections_are_inert(
    backend_selection: DeviceSelection,
) -> None:
    assert backend_selection.backend in BackendKind


def test_build_simulation_is_inspectable_and_hardware_free(tmp_path: Path) -> None:
    application = build_simulation(tmp_path / "events.jsonl", run_id="run-1")

    assert application.profile is PROFILES["simulate"]
    assert not application.profile.constructs_hardware
    assert application.controller.snapshot.state is RunState.NEW
    assert application.manifest_store.read() == application.manifest
    assert application.manifest.hardware["devices"] == (
        {"id": "detector", "backend": "mock"},
    )
    assert application.container.is_built
    assert "arh_sim" not in sys.modules
    assert not any(name.lower().startswith("slar") for name in sys.modules)


def test_simulation_completes_and_replays(tmp_path: Path) -> None:
    application = build_simulation(tmp_path / "events.jsonl", run_id="run-1")

    assert application.run().state is RunState.COMPLETED
    assert len(application.controller.journal.read()) == 2
    assert len(application.document_journal.read()) == 4
    assert not application.container.is_built


@pytest.mark.parametrize(
    ("transitions", "expected"),
    [
        (
            (
                RunEventKind.STARTED,
                RunEventKind.CANCELLATION_REQUESTED,
                RunEventKind.CANCELLED,
            ),
            RunState.CANCELLED,
        ),
        (
            (RunEventKind.STARTED, RunEventKind.FAILED),
            RunState.FAILED,
        ),
        (
            (
                RunEventKind.STARTED,
                RunEventKind.FAILED,
                RunEventKind.RECOVERY_STARTED,
                RunEventKind.RECOVERED,
                RunEventKind.COMPLETED,
            ),
            RunState.COMPLETED,
        ),
    ],
)
def test_simulation_terminal_scenarios_replay(
    tmp_path: Path,
    transitions: tuple[RunEventKind, ...],
    expected: RunState,
) -> None:
    journal = tmp_path / f"{expected.value}.jsonl"
    application = build_simulation(journal, run_id=f"run-{expected.value}")

    assert application.run(transitions).state is expected
    stop = application.document_journal.read()[-1].document
    expected_exit = {
        RunState.CANCELLED: "abort",
        RunState.FAILED: "fail",
        RunState.COMPLETED: "success",
    }[expected]
    assert stop["exit_status"] == expected_exit


def test_build_simulation_rejects_stale_document_journal(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.with_name("events.documents.jsonl").write_text(
        '{"name":"start","document":{}}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="document journal must be empty"):
        build_simulation(path, run_id="run-1")


def test_build_simulation_rejects_existing_manifest(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.with_name("events.manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest must not already exist"):
        build_simulation(path, run_id="run-1")


def test_dual_detector_factory_is_hardware_free_and_inspectable() -> None:
    simulation = build_dual_detector_simulation()

    assert simulation.registry.services == {
        "fluorescence": simulation.fluorescence,
        "dhm": simulation.dhm,
    }
    assert PROFILES["detector-simulate"].devices == (
        DeviceSelection("fluorescence", BackendKind.MOCK),
        DeviceSelection("dhm", BackendKind.MOCK),
    )
    assert not PROFILES["detector-simulate"].constructs_hardware
