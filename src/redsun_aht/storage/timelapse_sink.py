"""Create-only pixel storage and host-order time-lapse frame index."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from .writer import FrameStamp, RunFrameStore, local_store_path

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from redsun_aht.plans.timelapse import TimeLapsePoint


class TimeLapseFrameSink:
    """Write ordered pixels and a partial-aware host timestamp sidecar.

    The caller must close on success and failure. Frame indices and timestamps
    are host associations, not proof of exposure position or sensor timing.
    """

    def __init__(self, uri: str, template: NDArray[Any], *, count: int) -> None:
        if type(count) is not int or not 1 <= count <= 10_000:
            raise ValueError("invalid planned frame count")
        self.store = RunFrameStore(
            uri, "timelapse", template, planned_frames=count, setting_revision=None
        )
        self.path = local_store_path(uri) / "timelapse-index.json"
        self.expected = count
        self.records: list[dict[str, Any]] = []
        self._failed = False
        self._closed = False

    def append(
        self,
        point: TimeLapsePoint,
        image: NDArray[Any],
        *,
        host_receipt_ns: int | None = None,
    ) -> int:
        """Store one frame in order, recording receipt before storage work."""
        if self._closed or self._failed:
            raise RuntimeError("time-lapse sink closed or uncertain")
        received_ns = (
            time.perf_counter_ns() if host_receipt_ns is None else host_receipt_ns
        )
        try:
            if (
                point.sequence != len(self.records)
                or not (
                    point.host_start_ns == -1 or 0 < point.host_start_ns <= received_ns
                )
                or (
                    self.records
                    and point.host_start_ns != -1
                    and self.records[-1]["host_start_ns"] != -1
                    and point.host_start_ns < self.records[-1]["host_start_ns"]
                )
                or (self.records and received_ns < self.records[-1]["host_receipt_ns"])
            ):
                raise ValueError("invalid or out-of-order frame identity")
            index = self.store.append(
                image,
                stamp=FrameStamp(
                    pattern_id=point.pattern_id if point.pattern_id is not None else 0,
                    scan_index=point.position,
                ),
            )
            self.records.append({**asdict(point), "host_receipt_ns": received_ns})
            return index
        except Exception:
            self._failed = True
            raise

    def close(self) -> None:
        """Close the pixel store and write a sidecar, retaining partial results."""
        if self._closed:
            raise RuntimeError("time-lapse sink already closed")
        self._closed = True
        self.store.close()
        starts = [record["host_start_ns"] for record in self.records]
        intervals = [(b - a) / 1e6 for a, b in pairwise(starts) if a > 0 and b > 0]
        receipts = [record["host_receipt_ns"] for record in self.records]
        receipt_intervals = [(b - a) / 1e6 for a, b in pairwise(receipts)]
        complete = not self._failed and len(self.records) == self.expected
        sidecar = {
            "schema": "redsun-aht-timelapse-index-v1",
            "clock": "python.time.perf_counter_ns",
            "start_semantics": "host_before_capture_callback; -1 when unavailable",
            "timepoint_start_semantics": (
                "host_timepoint_dispatch; -1 when unavailable"
            ),
            "timepoint_lateness_semantics": (
                "host_dispatch_after_requested_floor; -1 when unavailable"
            ),
            "receipt_semantics": "host_frame_received_or_sink_entry",
            "expected": self.expected,
            "recorded": len(self.records),
            "complete": complete,
            "median_start_interval_ms": statistics.median(intervals)
            if intervals
            else None,
            "median_receipt_interval_ms": statistics.median(receipt_intervals)
            if receipt_intervals
            else None,
            "frames": self.records,
        }
        with self.path.open("x", encoding="utf-8") as output:
            json.dump(sidecar, output, indent=2)
            output.write("\n")
        if not complete:
            raise RuntimeError("time-lapse sink incomplete; inspect partial store")
