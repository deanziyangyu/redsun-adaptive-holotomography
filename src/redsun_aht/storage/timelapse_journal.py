"""Optional host-local JSONL journal for time-lapse association events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TimeLapseJournal:
    """Persist run identity, store mapping and host-inferred events as JSONL.

    One journal belongs to one run and is created exclusively. A flushed line
    is not a guarantee against power loss. Camera stores may be on another host.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.error: Exception | None = None
        self.descriptors: dict[str, str] = {}

    def __call__(self, name: str, doc: dict[str, Any]) -> None:
        """Write only the documents needed to interpret the plane stores."""
        if self.error is not None:
            return
        try:
            if name == "start":
                keys = (
                    "uid",
                    "time",
                    "plan_name",
                    "plane_stores",
                    "frames_per_plane",
                    "channel_exposures_ms",
                    "channel_frame_patterns",
                    "axis_order",
                    "position_ids",
                    "channel_labels",
                    "stage_binding",
                    "auto_shutter",
                    "requested_stacks",
                    "confirmation_id",
                    "mcu_run_ids",
                    "association",
                )
                payload = {key: doc[key] for key in keys if key in doc}
            elif name == "descriptor":
                self.descriptors[doc["uid"]] = doc["name"]
                payload = {key: doc[key] for key in ("uid", "run_start", "name")}
            elif name == "event":
                if self.descriptors.get(doc["descriptor"]) not in (
                    "timelapse_frame",
                    "timelapse_plane",
                ):
                    return
                payload = {
                    key: doc[key]
                    for key in ("uid", "descriptor", "seq_num", "time", "data")
                }
            elif name == "stop":
                payload = {
                    key: doc[key]
                    for key in ("uid", "run_start", "time", "exit_status", "reason")
                    if key in doc
                }
            else:
                return
            self.stream.write(
                json.dumps({"document": name, "data": payload}, allow_nan=False) + "\n"
            )
            self.stream.flush()
        except Exception as exc:
            self.error = exc

    def close(self) -> None:
        """Close the stream and report any previously captured write error."""
        try:
            self.stream.close()
        except Exception as exc:
            if self.error is None:
                self.error = exc
        if self.error is not None:
            raise RuntimeError("time-lapse journal write failed") from self.error
