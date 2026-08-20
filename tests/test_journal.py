from __future__ import annotations

from pathlib import Path

import pytest

from redsun_aht.domain import RunEventKind, RunState
from redsun_aht.storage import EventJournal, InvalidTransition, RunController


def test_cancel_is_deterministic_after_replay(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "events.jsonl")
    controller = RunController("run-1", journal)
    controller.record(RunEventKind.STARTED)
    controller.record(RunEventKind.CANCELLATION_REQUESTED)
    cancelled = controller.record(RunEventKind.CANCELLED)

    replayed = RunController("run-1", journal).snapshot
    assert replayed == cancelled
    assert replayed.state is RunState.CANCELLED


def test_fail_recover_and_complete_after_replay(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "events.jsonl")
    controller = RunController("run-1", journal)
    controller.record(RunEventKind.STARTED)
    controller.record(RunEventKind.FAILED, {"reason": "injected fault"})
    controller.record(RunEventKind.RECOVERY_STARTED)
    recovered = controller.record(RunEventKind.RECOVERED)
    completed = controller.record(RunEventKind.COMPLETED)

    assert recovered.failure is None
    assert RunController("run-1", journal).snapshot == completed
    assert completed.state is RunState.COMPLETED


def test_invalid_transition_is_not_appended(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "events.jsonl")
    controller = RunController("run-1", journal)

    with pytest.raises(InvalidTransition, match="cannot apply completed"):
        controller.record(RunEventKind.COMPLETED)

    assert journal.read() == ()


def test_journal_reports_corrupt_record_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        EventJournal(path).read()
