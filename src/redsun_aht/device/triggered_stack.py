"""Experimental orchestration for externally triggered finite stacks.

Camera and MCU adapters must have sole authority over their resources. This
workflow does not establish a lease, electrical compatibility or physical stop.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Any, Protocol

from redsun_aht.device.stack_reconciliation import Reconciliation, reconcile_stack

if TYPE_CHECKING:
    from redsun_aht.device.unified_mcu import AngularStack, StackProgress

logger = logging.getLogger(__name__)


class TriggeredCamera(Protocol):
    """Camera adapter contract; counters must be from the independent camera."""

    async def arm(self, frame_count: int) -> int:
        """Arm bounded external triggering and return the prior sequence baseline."""
        ...

    async def collect(self, frame_count: int) -> tuple[int, ...]:
        """Drain camera counters while capture proceeds, until collection ends."""
        ...

    async def close(self) -> None:
        """End capture and release this adapter's resources."""
        ...


class StackClient(Protocol):
    """One-start stack interface used by the experimental workflow."""

    async def execute(self, stack: AngularStack, *, ready: bool) -> StackProgress:
        """Submit once, with host-asserted readiness."""
        ...

    async def abort_stack(self, run_id: int) -> None:
        """Request an abort, without asserting physical stop."""
        ...


async def acquire_stack(
    client: StackClient,
    camera: TriggeredCamera,
    stack: AngularStack,
    *,
    capture_exclusive: bool,
    illumination_ready: bool,
    timeout_s: float,
) -> Reconciliation:
    """Run one bounded external-trigger capture and reconcile observed streams.

    Arm must finish before the stack is submitted. An exception after submission
    causes one best-effort abort request, which does not prove the motor stopped.
    A missing frame returns an unassociated result; it is never fabricated.
    The caller is responsible for external resource ownership and illumination.
    """
    if type(capture_exclusive) is not bool or not capture_exclusive:
        raise ValueError("exclusive camera capture must be asserted")
    if illumination_ready is not True:
        raise ValueError("illumination readiness must be asserted")
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    submitted = False
    tasks: list[asyncio.Task[Any]] = []
    try:
        baseline = await asyncio.wait_for(camera.arm(stack.count), timeout_s)
        if type(baseline) is not int or baseline < 0:
            raise ValueError("camera returned invalid baseline")
        submitted = True
        collect_task = asyncio.create_task(camera.collect(stack.count))
        stack_task = asyncio.create_task(client.execute(stack, ready=True))
        tasks = [collect_task, stack_task]
        done, pending = await asyncio.wait(
            tasks, timeout=timeout_s, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            task.result()
        if pending:
            raise TimeoutError("stack capture timed out")
        progress, frames = stack_task.result(), collect_task.result()
        return reconcile_stack(
            stack, progress, frames, camera_baseline=baseline, capture_exclusive=True
        )
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        if submitted:
            abort_task = asyncio.create_task(client.abort_stack(stack.run_id))
            try:
                done, _ = await asyncio.wait({abort_task}, timeout=2)
                if not done:
                    raise TimeoutError("abort request timed out")
                abort_task.result()
            except BaseException:
                logger.exception("Abort request was not confirmed; stop is unverified")
            finally:
                if not abort_task.done():
                    abort_task.cancel()
                    abort_task.add_done_callback(
                        lambda completed: (
                            None if completed.cancelled() else completed.exception()
                        )
                    )
        raise
    finally:
        pending_error = sys.exc_info()[1]
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
                logger.error("Capture tasks still running; resources remain uncertain")
            if collect_task in pending:
                logger.error("Camera collector still active; camera release unverified")
                continue_cleanup = False
            else:
                continue_cleanup = True
        else:
            continue_cleanup = True
        if not continue_cleanup:
            if pending_error is None:
                raise RuntimeError("camera collector did not stop; release unverified")
        else:
            try:
                await asyncio.shield(asyncio.wait_for(camera.close(), 2))
            except BaseException:
                if pending_error is None:
                    raise
                logger.exception("Camera cleanup failed after acquisition failure")
