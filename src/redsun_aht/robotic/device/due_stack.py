"""Host observation of the Due X-stack run over its existing PVA service."""

import asyncio
from dataclasses import dataclass
import json
import math
import sys
from typing import Any

from redsun_aht.device.unified_mcu import AngularStack

DUE_STACK_CLEANUP_CALL_TIMEOUT_S = 2
DUE_STACK_CLEANUP_BUDGET_S = 3 * DUE_STACK_CLEANUP_CALL_TIMEOUT_S


def validate_due_geometry(definition: dict[str, int]) -> None:
    stack = AngularStack(**definition)
    if abs(stack.step) > 7200:
        raise ValueError("Due stack step exceeds the angular window")
    if stack.count > 1 and stack.step == 0:
        raise ValueError("A multi-plane stack needs a nonzero step")


@dataclass(frozen=True)
class DueStackStatus:
    run_id: int
    state: str
    edges: int

    @classmethod
    def parse(cls, reply: str, run_id: int, count: int):
        value = json.loads(reply)
        if not isinstance(value, dict) or type(value.get("run_id")) is not int or value["run_id"] != run_id:
            raise ValueError("Due status does not identify the requested run")
        state = value.get("state")
        edges = value.get("edges")
        if state not in ("idle", "active", "waiting_capture", "complete", "stopping",
                         "abort_unverified", "fault_unverified", "cancelled"):
            raise ValueError("Invalid Due stack state")
        if type(edges) is not int or not 0 <= edges <= count:
            raise ValueError("Invalid Due stack edge count")
        return cls(run_id, state, edges)


async def wait_due_capture_plane(mcu: Any, run_id: int, count: int, index: int) -> None:
    """Wait for the expected settled plane or final completion."""
    async with asyncio.timeout(10):
        while True:
            await mcu.zdt_status_request.set(run_id)
            status = DueStackStatus.parse(await mcu.zdt_status.get_value(cached=False), run_id, count)
            if status.edges != index:
                raise RuntimeError(f"Due reported plane {status.edges}, expected {index}")
            expected = "complete" if index == count else "waiting_capture"
            if status.state == expected:
                return
            if status.state != "active":
                raise RuntimeError(f"Due stack reported {status.state} while awaiting {expected}")
            await asyncio.sleep(0.05)


async def wait_due_stack(
    mcu: Any, run_id: int, count: int, *, timeout_s: float
) -> DueStackStatus:
    """Wait for completion with the expected edge count, independent of frames."""
    if type(run_id) is not int or not 1 <= run_id <= 2147483647:
        raise ValueError("Invalid Due stack run ID")
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError("Invalid Due stack plane count")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Invalid Due stack timeout")
    previous_edges = 0
    async with asyncio.timeout(timeout_s):
        while True:
            await mcu.zdt_status_request.set(run_id, timeout=min(timeout_s, 8))
            status = DueStackStatus.parse(await mcu.zdt_status.get_value(cached=False), run_id, count)
            if status.edges < previous_edges:
                raise RuntimeError("Due stack edge count regressed")
            previous_edges = status.edges
            if status.state == "complete":
                if status.edges != count:
                    raise RuntimeError("Due completed with the wrong number of edges")
                return status
            if status.state != "active":
                raise RuntimeError(f"Due stack reported {status.state}")
            await asyncio.sleep(0.05)


async def capture_due_stack(
    mcu: Any, camera: Any, definition: dict[str, int], *,
    timeout_s: float, exclusive_camera: bool, illumination_ready: bool,
):
    """Capture ordered frames with an exclusively owned external-trigger adapter."""
    submitted = False
    try:
        if exclusive_camera is not True or illumination_ready is not True:
            raise ValueError("Confirm exclusive camera ownership and illumination readiness")
        run_id, count = definition["run_id"], definition["count"]
        if type(run_id) is not int or not 1 <= run_id <= 2147483647:
            raise ValueError("Invalid Due stack run ID")
        if type(count) is not int or not 1 <= count <= 32:
            raise ValueError("Invalid Due stack plane count")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Invalid Due stack timeout")
        async with asyncio.timeout(timeout_s):
            baseline = await camera.arm(count)
            if type(baseline) is not int or baseline < 0:
                raise ValueError("Invalid camera counter baseline")
            await camera.confirm_ready()
            submitted = True
            await mcu.zdt_start.set(json.dumps({**definition, "camera_ready": True}))

            async def collect_frames():
                frames = await camera.collect(count)
                if tuple(frame.sequence for frame in frames) != tuple(range(baseline + 1, baseline + count + 1)):
                    raise RuntimeError("Due edges and camera metadata counters could not be associated")
                return frames

            async with asyncio.TaskGroup() as tasks:
                camera_task = tasks.create_task(collect_frames())
                status_task = tasks.create_task(wait_due_stack(mcu, run_id, count, timeout_s=timeout_s))
            await camera.finish()
            return status_task.result(), camera_task.result()
    except BaseException as error:
        if submitted:
            try:
                await asyncio.wait_for(mcu.zdt_abort.set(run_id), timeout=DUE_STACK_CLEANUP_CALL_TIMEOUT_S)
            except Exception as abort_error:
                error.add_note(f"Due abort request was not acknowledged: {abort_error}")
            else:
                error.add_note(
                    f"Due acknowledged abort for run {run_id}. "
                    "Check due_z_stack_status for the run's outcome."
                )
            try:
                await asyncio.wait_for(mcu.blank.set(True), timeout=DUE_STACK_CLEANUP_CALL_TIMEOUT_S)
            except Exception as blank_error:
                error.add_note(f"Due panel blanking could not be confirmed: {blank_error}")
        raise
    finally:
        pending_error = sys.exception()
        try:
            await asyncio.wait_for(camera.close(), timeout=DUE_STACK_CLEANUP_CALL_TIMEOUT_S)
        except TimeoutError as close_error:
            message = "Camera release was not confirmed within two seconds. The adapter may still own the camera."
            if pending_error is None:
                raise RuntimeError(message) from close_error
            pending_error.add_note(message)
        except Exception as close_error:
            if pending_error is None:
                raise
            pending_error.add_note(f"Camera close could not be confirmed: {close_error}")
