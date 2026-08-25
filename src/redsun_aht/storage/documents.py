"""Append-only storage and replay for Bluesky documents."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, cast

from redsun_aht.acquisition import DocumentRecord
from redsun_aht.domain.models import JsonValue, thaw_json

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path


class DocumentJournal:
    """Durable JSON-lines journal for canonical Bluesky documents."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: DocumentRecord) -> None:
        """Append and fsync one complete document record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": record.name, "document": thaw_json(record.document)}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def extend(self, records: Iterable[DocumentRecord]) -> None:
        """Append records in their source order."""
        for record in records:
            self.append(record)

    def read(self) -> tuple[DocumentRecord, ...]:
        """Read every complete document in append order."""
        if not self.path.exists():
            return ()
        records: list[DocumentRecord] = []
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    name = payload["name"]
                    document = payload["document"]
                    if not isinstance(name, str) or not isinstance(document, dict):
                        raise TypeError("document record fields have invalid types")
                    records.append(
                        DocumentRecord(
                            name,
                            cast("Mapping[str, JsonValue]", document),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid document journal record at line {line_number}"
                    ) from error
        return tuple(records)

    def replay(self, callback: Callable[[str, Mapping[str, JsonValue]], None]) -> None:
        """Replay stored documents to a standard name/document callback."""
        for record in self.read():
            callback(record.name, record.document)


__all__ = ["DocumentJournal"]
