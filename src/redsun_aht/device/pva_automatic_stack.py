"""Service-owned continuous camera capture for one autonomous MCU stack."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from redsun_aht.device.pva_capture_evidence import CaptureEvidence, PvaCaptureEvidence

if TYPE_CHECKING:
    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PvaAutomaticStack:
    """Stored frame count and ordered requested targets, inferred by host order."""

    evidence: CaptureEvidence
    attempted_pulses: int
    camera_count_delta: int
    requested_frames: tuple[tuple[int, int, int], ...]
    quality: str = "count_matched_host_order_inferred"


async def acquire_pva_automatic_stack(
    owner: PvaUnifiedMcu,
    camera: Any,
    definition: AngularStack,
    shots_per_plane: int,
    pattern_mask: int,
    interval_ms: int,
    store_uri: str,
    *,
    resources_exclusive: bool,
    external_trigger_configured: bool,
    continuous_capture_configured: bool,
    illumination_ready: bool,
    timeout_s: float = 60,
) -> PvaAutomaticStack:
    """Arm one continuous capture, start one MCU stack and store all frames.

    Trigger configuration and exclusive camera ownership are caller assertions.
    Counts and order do not prove electrical edges, exposure timing or position.
    """
    if any(
        value is not True
        for value in (
            resources_exclusive,
            external_trigger_configured,
            continuous_capture_configured,
            illumination_ready,
        )
    ):
        raise ValueError("exclusive continuous externally triggered capture required")
    if not owner.reservation_held:
        raise ValueError("MCU reservation required")
    if type(shots_per_plane) is not int or not 1 <= shots_per_plane <= 64:
        raise ValueError("invalid shots per plane")
    if type(pattern_mask) is not int or not 1 <= pattern_mask <= 31:
        raise ValueError("invalid pattern mask")
    if type(interval_ms) is not int or not 1 <= interval_ms < 30000:
        raise ValueError("invalid interval")
    if not 0 < timeout_s <= 300:
        raise ValueError("invalid timeout")
    count = definition.count * shots_per_plane
    capture = PvaCaptureEvidence(camera, timeout_s=timeout_s)
    tasks: list[asyncio.Task[Any]] = []
    failed = False
    closed = False
    try:
        baseline = await camera.frame_count.get_value(cached=False)
        if type(baseline) is not int or baseline < 0:
            raise ValueError("invalid camera frame count")
        await capture.prepare(store_uri, count, external_trigger_configured=True)

        async def confirm_ready() -> None:
            if not await camera.sequence_started.get_value(cached=False):
                raise RuntimeError("camera sequence not running")
            if await camera.frame_count.get_value(cached=False) != baseline:
                raise RuntimeError("unexpected camera frames before stack start")

        controller_task = asyncio.create_task(
            owner.automatic_shots(
                definition,
                shots_per_plane,
                pattern_mask,
                interval_ms,
                camera_armed=True,
                illumination_ready=True,
                before_start=confirm_ready,
            )
        )
        capture_task = asyncio.create_task(capture.collect())
        tasks = [controller_task, capture_task]
        done, pending = await asyncio.wait(
            tasks, timeout=timeout_s, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            task.result()
        if pending:
            raise TimeoutError("automatic stack capture timed out")
        report, evidence = controller_task.result(), capture_task.result()
        await capture.close()
        closed = True
        final_count = await camera.frame_count.get_value(cached=False)
        if type(final_count) is not int or final_count - baseline != count:
            raise RuntimeError("camera service frame count mismatch")
        if evidence.stored_frames != count or report.attempted_pulses != count:
            raise RuntimeError("automatic capture count mismatch")
        patterns = tuple(i + 1 for i in range(5) if pattern_mask & (1 << i))
        frames = tuple(
            (
                i // shots_per_plane,
                definition.first + definition.step * (i // shots_per_plane),
                patterns[i % len(patterns)],
            )
            for i in range(count)
        )
        return PvaAutomaticStack(
            evidence, report.attempted_pulses, final_count - baseline, frames
        )
    except BaseException:
        failed = True
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        try:
            if not closed:
                await capture.close()
        except Exception:
            if not failed:
                raise
            logger.exception("Camera cleanup failed after automatic stack error")
