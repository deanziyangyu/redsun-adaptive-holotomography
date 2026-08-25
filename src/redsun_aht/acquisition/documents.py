"""Canonical deterministic Bluesky documents for Phase 1 simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from event_model import compose_run

from redsun_aht.domain.models import JsonValue, freeze_json_mapping
from redsun_aht.streams import PRIMARY_STREAM

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """One ordered, recursively immutable Bluesky document."""

    name: str
    document: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        """Detach the record from the document producer's mutable mapping."""
        object.__setattr__(self, "document", freeze_json_mapping(self.document))


def compose_simulation_documents(
    run_id: str,
    *,
    timestamp_ns: int,
    state: str,
    exit_status: Literal["success", "abort", "fail"] = "success",
    reason: str = "",
) -> tuple[DocumentRecord, ...]:
    """Compose a deterministic start/descriptor/event/stop sequence."""
    timestamp = timestamp_ns / 1_000_000_000
    run = compose_run(
        uid=f"{run_id}-start",
        time=timestamp,
        metadata={"aht_run_id": run_id, "profile": "simulate"},
    )
    descriptor = run.compose_descriptor(
        name=PRIMARY_STREAM,
        data_keys={
            "simulation_state": {
                "source": "SIM:redsun_aht",
                "dtype": "string",
                "shape": [],
            }
        },
        uid=f"{run_id}-descriptor",
        time=timestamp,
    )
    event = descriptor.compose_event(
        data={"simulation_state": state},
        timestamps={"simulation_state": timestamp},
        seq_num=1,
        uid=f"{run_id}-event-1",
        time=timestamp,
    )
    stop = run.compose_stop(
        exit_status=exit_status,
        reason=reason,
        uid=f"{run_id}-stop",
        time=timestamp,
    )

    raw_records = (
        ("start", run.start_doc),
        ("descriptor", descriptor.descriptor_doc),
        ("event", event),
        ("stop", stop),
    )
    return tuple(
        DocumentRecord(name, cast("Mapping[str, JsonValue]", document))
        for name, document in raw_records
    )


__all__ = ["DocumentRecord", "compose_simulation_documents"]
