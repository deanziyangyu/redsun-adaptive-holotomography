from __future__ import annotations

from pathlib import Path

import pytest

from redsun_aht.acquisition import compose_simulation_documents
from redsun_aht.storage import DocumentJournal
from redsun_aht.streams import PRIMARY_STREAM


def test_simulation_documents_use_canonical_stream_and_stable_ids() -> None:
    records = compose_simulation_documents(
        "run-1", timestamp_ns=1_000_000_000, state="completed"
    )

    assert tuple(record.name for record in records) == (
        "start",
        "descriptor",
        "event",
        "stop",
    )
    assert records[0].document["uid"] == "run-1-start"
    assert records[1].document["name"] == PRIMARY_STREAM
    assert records[2].document["data"] == {"simulation_state": "completed"}
    assert records[3].document["exit_status"] == "success"


def test_document_journal_round_trips_and_replays(tmp_path: Path) -> None:
    journal = DocumentJournal(tmp_path / "documents.jsonl")
    records = compose_simulation_documents(
        "run-1", timestamp_ns=1_000_000_000, state="completed"
    )
    replayed: list[tuple[str, object]] = []

    journal.extend(records)
    journal.replay(lambda name, document: replayed.append((name, document)))

    assert journal.read() == records
    assert replayed == [(record.name, record.document) for record in records]


@pytest.mark.parametrize(
    "line",
    [
        "not-json\n",
        '{"name":1,"document":{}}\n',
        '{"name":"start","document":[]}\n',
    ],
)
def test_document_journal_reports_corrupt_record_line(
    tmp_path: Path, line: str
) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(line, encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        DocumentJournal(path).read()
