"""Headless Bluesky rehearsal using the in-process simulated stack model.

The fake camera identities are derived from fake edges, so this exercise does
not establish physical camera association or real asynchronous acquisition.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from redsun_aht.storage.writer import RunFrameStore

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np

from .mcu_stack import Frame, SimulatedMcu, Stack, State, camera_frames, reconcile


class _PlaneRecord:
    """Read-only simulated event source for one fully reconciled plane."""

    name = "simulated_stack_plane"

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def read(self) -> dict[str, dict[str, Any]]:
        """Provide values and a host timestamp to Bluesky."""
        now = time.time()
        return {
            key: {"value": value, "timestamp": now} for key, value in self.data.items()
        }

    def describe(self) -> dict[str, dict[str, Any]]:
        """Declare the event schema without claiming measured hardware data."""
        return {
            key: {
                "source": "SIM:stack",
                "dtype": "string" if isinstance(value, str) else "number",
                "shape": [],
            }
            for key, value in self.data.items()
        }


def simulated_stack_plan(
    mcu: SimulatedMcu,
    stack: Stack,
    *,
    frame_source: Callable[..., tuple[Frame, ...]] = camera_frames,
    fail_before_plane: int | None = None,
    frame_store: RunFrameStore | None = None,
) -> Generator[Any, Any, None]:
    """Run and reconcile one fake stack, emitting a simulated event per plane."""
    owner = "simulated-stack-plan"

    @bpp.run_decorator(md={"simulation": True, "stack_id": stack.run_id})  # type: ignore[untyped-decorator]
    def run() -> Generator[Any, Any, None]:
        reserved = False
        try:
            mcu.reserve(owner)
            reserved = True
            mcu.commit(owner, stack)
            mcu.arm(owner, camera_ready=True, referenced=True)
            edges = mcu.start(owner, stack.run_id, fail_before_plane=fail_before_plane)
            edge_source = _PlaneRecord({})
            for edge in edges:
                edge_source.data = {
                    "stack_id": edge.run_id,
                    "edge_plane": edge.plane,
                    "edge_target_um": edge.target_um,
                    "edge_pattern": edge.pattern,
                    "edge_evidence": "simulated",
                }
                yield from bps.create(name="simulated_edges")
                yield from bps.read(edge_source)
                yield from bps.save()
            frames = frame_source(edges)
            if frame_store is not None:
                for frame in frames:
                    # A tiny synthetic image makes ordering visible; it is not
                    # camera output and its pixel value is the fake plane ID.
                    frame_store.append(np.full((4, 4), frame.plane, dtype=np.uint16))
            source = _PlaneRecord({})
            for index, (edge, frame) in enumerate(reconcile(edges, frames, stack)):
                source.data = {
                    "stack_id": edge.run_id,
                    "plane": edge.plane,
                    "target_um": edge.target_um,
                    "pattern": edge.pattern,
                    "frame_plane": frame.plane,
                    "store_index": index if frame_store is not None else -1,
                    "evidence": "simulated",
                }
                yield from bps.create(name="simulated_stack")
                yield from bps.read(source)
                yield from bps.save()
        finally:
            try:
                if frame_store is not None:
                    frame_store.close()
            finally:
                if reserved:
                    if mcu.state not in (State.COMPLETE, State.ABORTED):
                        mcu.cancel(owner)
                    mcu.release(owner)

    yield from run()
