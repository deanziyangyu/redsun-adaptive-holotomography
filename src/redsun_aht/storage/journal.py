"""Append-only run journal and deterministic lifecycle reducer."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from redsun_aht.domain import RunEvent, RunEventKind, RunSnapshot, RunState
from redsun_aht.domain.models import thaw_json

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from redsun_aht.domain.models import JsonValue


class InvalidTransition(RuntimeError):
    """Raised when an event is illegal for the materialized state."""


_TRANSITIONS: dict[tuple[RunState, RunEventKind], RunState] = {
    (RunState.NEW, RunEventKind.STARTED): RunState.RUNNING,
    (RunState.RUNNING, RunEventKind.CANCELLATION_REQUESTED): RunState.CANCELLING,
    (RunState.CANCELLING, RunEventKind.CANCELLED): RunState.CANCELLED,
    (RunState.RUNNING, RunEventKind.FAILED): RunState.FAILED,
    (RunState.CANCELLING, RunEventKind.FAILED): RunState.FAILED,
    (RunState.FAILED, RunEventKind.RECOVERY_STARTED): RunState.RECOVERING,
    (RunState.RECOVERING, RunEventKind.RECOVERED): RunState.RUNNING,
    (RunState.RUNNING, RunEventKind.COMPLETED): RunState.COMPLETED,
}


def apply_event(snapshot: RunSnapshot, event: RunEvent) -> RunSnapshot:
    """Apply one strictly ordered event to a run snapshot."""
    if event.run_id != snapshot.run_id:
        raise ValueError("event run_id does not match snapshot")
    expected_sequence = snapshot.last_sequence + 1
    if event.sequence != expected_sequence:
        raise ValueError(
            f"expected event sequence {expected_sequence}, got {event.sequence}"
        )
    try:
        next_state = _TRANSITIONS[(snapshot.state, event.kind)]
    except KeyError as error:
        raise InvalidTransition(
            f"cannot apply {event.kind.value} while run is {snapshot.state.value}"
        ) from error
    failure = snapshot.failure
    if event.kind is RunEventKind.FAILED:
        raw_failure = event.detail.get("reason", "unspecified failure")
        failure = str(raw_failure)
    elif event.kind is RunEventKind.RECOVERED:
        failure = None
    return replace(
        snapshot,
        state=next_state,
        last_sequence=event.sequence,
        failure=failure,
    )


def replay_events(run_id: str, events: Iterable[RunEvent]) -> RunSnapshot:
    """Materialize a deterministic snapshot from ordered events."""
    snapshot = RunSnapshot(run_id=run_id)
    for event in events:
        snapshot = apply_event(snapshot, event)
    return snapshot


class EventJournal:
    """Durable JSON-lines journal with flush and fsync on every append."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: RunEvent) -> None:
        """Append one event without rewriting previous records."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": event.run_id,
            "sequence": event.sequence,
            "kind": event.kind.value,
            "timestamp_ns": event.timestamp_ns,
            "detail": thaw_json(event.detail),
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def read(self) -> tuple[RunEvent, ...]:
        """Read all complete records in append order."""
        if not self.path.exists():
            return ()
        events: list[RunEvent] = []
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    record["kind"] = RunEventKind(record["kind"])
                    events.append(RunEvent(**record))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid event journal record at line {line_number}"
                    ) from error
        return tuple(events)


class RunController:
    """Stateful event producer whose state is always reconstructible."""

    def __init__(self, run_id: str, journal: EventJournal) -> None:
        self.journal = journal
        events = journal.read()
        self.snapshot = replay_events(run_id, events)

    def record(
        self, kind: RunEventKind, detail: dict[str, JsonValue] | None = None
    ) -> RunSnapshot:
        """Validate, durably append, and apply one transition."""
        event = RunEvent(
            run_id=self.snapshot.run_id,
            sequence=self.snapshot.last_sequence + 1,
            kind=kind,
            timestamp_ns=time.time_ns(),
            detail=detail or {},
        )
        next_snapshot = apply_event(self.snapshot, event)
        self.journal.append(event)
        self.snapshot = next_snapshot
        return next_snapshot
