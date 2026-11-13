"""Experimental one-stack workflow retaining unassociated camera evidence.

Resource ownership, trigger configuration and illumination are caller assertions.
Camera sequence start is not a hardware-ready signal, and abort is not a stop.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from redsun_aht.device.pva_capture_evidence import (
        CaptureEvidence,
        PvaCaptureEvidence,
    )
    from redsun_aht.device.unified_mcu import AngularStack, StackProgress

logger = logging.getLogger(__name__)


class ReservableStackClient(Protocol):
    """Stack client with an explicit cooperative MCU reservation."""

    async def reserve(self) -> None:
        """Acquire the cooperative reservation."""
        ...

    async def release_reservation(self) -> None:
        """Release the cooperative reservation."""
        ...

    async def execute(
        self, stack: AngularStack, *, ready: bool, require_reservation: bool = False
    ) -> StackProgress:
        """Submit a single stack."""
        ...

    async def abort_stack(self, run_id: int) -> None:
        """Request a run-scoped abort."""
        ...


@dataclass(frozen=True, slots=True)
class UnassociatedStackResult:
    """Independent controller report and service count, without plane mapping."""

    controller: StackProgress
    capture: CaptureEvidence
    quality: str = "unassociated"


async def acquire_unassociated_stack(
    client: ReservableStackClient,
    camera: PvaCaptureEvidence,
    stack: AngularStack,
    uri: str,
    *,
    resources_exclusive: bool,
    external_trigger_configured: bool,
    illumination_ready: bool,
    timeout_s: float,
) -> UnassociatedStackResult:
    """Prepare camera, submit one stack, and retain separate evidence streams."""
    if resources_exclusive is not True or external_trigger_configured is not True:
        raise ValueError("exclusive resources and external trigger must be asserted")
    if illumination_ready is not True:
        raise ValueError("illumination readiness must be asserted")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be finite and positive")
    submitted = False
    reservation_acquired = False
    camera_attempted = False
    tasks: list[asyncio.Task[Any]] = []
    try:
        async with asyncio.timeout(timeout_s):
            await client.reserve()
            reservation_acquired = True
            camera_attempted = True
            await camera.prepare(uri, stack.count, external_trigger_configured=True)
        capture_task = asyncio.create_task(camera.collect())
        stack_task = asyncio.create_task(
            client.execute(stack, ready=True, require_reservation=True)
        )
        tasks = [capture_task, stack_task]
        submitted = True
        done, pending = await asyncio.wait(
            tasks, timeout=timeout_s, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            task.result()
        if pending:
            raise TimeoutError("stack capture timed out")
        progress, capture = stack_task.result(), capture_task.result()
        if not progress.complete or capture.stored_frames != stack.count:
            raise RuntimeError("stack or camera capture incomplete")
        return UnassociatedStackResult(progress, capture)
    except BaseException:
        stack_completed = (
            submitted
            and stack_task.done()
            and not stack_task.cancelled()
            and stack_task.exception() is None
            and stack_task.result().complete
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        if submitted and not stack_completed:
            abort = asyncio.create_task(client.abort_stack(stack.run_id))
            done, _ = await asyncio.wait({abort}, timeout=2)
            if not done:
                abort.cancel()
                abort.add_done_callback(
                    lambda task: None if task.cancelled() else task.exception()
                )
                logger.error("Abort request timed out; stop is unverified")
            else:
                try:
                    abort.result()
                except BaseException:
                    logger.exception("Abort request failed; stop is unverified")
        raise
    finally:
        error = sys.exc_info()[1]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=2)
            for task in done:
                if not task.cancelled():
                    task.exception()
            for task in pending:
                task.add_done_callback(
                    lambda completed: (
                        None if completed.cancelled() else completed.exception()
                    )
                )
            if pending:
                logger.error("Capture workers remain active; resources are uncertain")
            if pending and capture_task in pending:
                logger.error("Camera collector still active; release is unverified")
                if error is None:
                    raise RuntimeError("camera collector did not stop")
                return_cleanup = False
            else:
                return_cleanup = True
        else:
            return_cleanup = True
        cleanup_ok = False
        if return_cleanup and camera_attempted:
            try:
                await asyncio.wait_for(camera.close(), timeout=2)
                cleanup_ok = True
            except BaseException:
                if error is None:
                    raise
                logger.exception("Camera cleanup failed after acquisition failure")
        if error is None and reservation_acquired and cleanup_ok:
            await asyncio.wait_for(client.release_reservation(), timeout=2)
        elif reservation_acquired:
            logger.error("MCU reservation retained; reconcile before release")
