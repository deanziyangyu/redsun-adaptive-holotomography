"""Headless Bluesky plan for the fake MCU stack PVA client."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np

from redsun_aht.storage.writer import FrameStamp, RunFrameStore

from .stack_plan import _PlaneRecord

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator

    from .mcu_stack import Stack
    from .pva_stack import PvaSimulatedStack


def pva_stack_plan(
    client: PvaSimulatedStack,
    stack: Stack,
    *,
    timeout_s: float = 15.0,
    store_uri: str | None = None,
) -> Generator[Any, Any, None]:
    """Submit one fake stack and emit reconciled synthetic frame events."""
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")

    @bpp.run_decorator(md={"simulation": True, "stack_id": stack.run_id})  # type: ignore[untyped-decorator]
    def run() -> Generator[Any, Any, None]:
        store = None
        try:
            if store_uri is not None:
                store = RunFrameStore(
                    store_uri,
                    "simulated",
                    np.zeros((4, 4), dtype=np.uint16),
                    planned_frames=len(stack.targets_um),
                )
            try:
                tasks = yield from bps.wait_for(
                    [lambda: asyncio.wait_for(client.execute(stack), timeout=timeout_s)]
                )
                evidence = tasks[0].result()
            except TimeoutError:
                # Scope the request to this run. An acknowledged abort does not
                # prove remote termination; a late start still needs recovery.
                try:
                    abort_tasks = yield from bps.wait_for(
                        [
                            lambda: asyncio.wait_for(
                                client.abort_stack(stack.run_id), timeout=2.0
                            )
                        ]
                    )
                    abort_tasks[0].result()
                    partial_tasks = yield from bps.wait_for(
                        [
                            lambda: asyncio.wait_for(
                                client.wait_for_aborted_result(stack.run_id),
                                timeout=2.0,
                            )
                        ]
                    )
                    partial = partial_tasks[0].result()
                    if not isinstance(partial, str):
                        raise TypeError("partial result must be a JSON string")
                    source = _PlaneRecord(
                        {
                            "stack_id": stack.run_id,
                            "partial_result_json": partial,
                            "association": "unassociated",
                            "evidence": "simulated",
                        }
                    )
                    yield from bps.create(name="simulated_stack_partial")
                    yield from bps.read(source)
                    yield from bps.save()
                except Exception:
                    logger.exception(
                        "Simulated abort or partial result was not confirmed"
                    )
                raise
            source = _PlaneRecord({})
            for edge, frame_number in zip(
                evidence.edges, evidence.frame_sequences, strict=True
            ):
                index = -1
                if store is not None:
                    index = store.append(
                        np.full((4, 4), frame_number, dtype=np.uint16),
                        stamp=FrameStamp(
                            pattern_id=edge.pattern,
                            scan_index=edge.plane,
                            producer_sequence=frame_number,
                        ),
                    )
                source.data = {
                    "stack_id": edge.run_id,
                    "plane": edge.plane,
                    "target_um": edge.target_um,
                    "pattern": edge.pattern,
                    "camera_frame_sequence": frame_number,
                    "store_index": index,
                    "association": "synthetic-order-only",
                    "evidence": "simulated",
                }
                yield from bps.create(name="simulated_stack")
                yield from bps.read(source)
                yield from bps.save()
        finally:
            if store is not None:
                store.close()

    yield from run()
