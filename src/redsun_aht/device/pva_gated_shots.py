"""Host-paced MCU shots with one continuously armed camera capture."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ophyd_async.core import wait_for_value

from redsun_aht.device.pva_automatic_stack import PvaAutomaticStack
from redsun_aht.device.pva_capture_evidence import PvaCaptureEvidence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack


async def acquire_pva_gated_shots(
    owner: PvaUnifiedMcu,
    camera: Any,
    definition: AngularStack,
    patterns: Sequence[int],
    store_uri: str,
    *,
    resources_exclusive: bool,
    external_trigger_configured: bool,
    automatic_pulses_enabled: bool,
    illumination_ready: bool,
    bind_stage: bool = True,
    timeout_s: float = 20,
) -> PvaAutomaticStack:
    """Arm once, then wait for a camera frame after each c1-gated MCU shot.

    The caller configures Line0 triggering and c1 before reserving the MCU.
    Frame-to-pattern association is inferred from ordered, matched counts.
    """
    if (
        any(
            flag is not True
            for flag in (
                resources_exclusive,
                external_trigger_configured,
                automatic_pulses_enabled,
                illumination_ready,
            )
        )
        or not owner.reservation_held
    ):
        raise ValueError("reserved MCU and configured external trigger required")
    patterns = tuple(patterns)
    if not 1 <= len(patterns) <= 64 or any(
        type(pattern) is not int or not 1 <= pattern <= 5 for pattern in patterns
    ):
        raise ValueError("one to 64 valid patterns per plane required")
    if type(bind_stage) is not bool or not 0 < timeout_s <= 300:
        raise ValueError("invalid stage binding or timeout")
    count = definition.count * len(patterns)
    capture = PvaCaptureEvidence(camera, timeout_s=timeout_s)
    frames: list[tuple[int, int, int]] = []
    previous_immediate: bool | None = None
    try:
        if hasattr(camera, "immediate_frame_count"):
            previous_immediate = await camera.immediate_frame_count.get_value(
                cached=False
            )
            await camera.immediate_frame_count.set(True)
        baseline = await camera.frame_count.get_value(cached=False)
        if type(baseline) is not int or baseline < 0:
            raise ValueError("invalid camera frame count")
        await capture.prepare(store_uri, count, external_trigger_configured=True)
        if await camera.frame_count.get_value(cached=False) != baseline:
            raise RuntimeError("unexpected camera frame before first shot")

        async def frame(shot: int, plane: int, pattern: int, target: int) -> None:
            expected = baseline + shot + 1
            await wait_for_value(
                camera.frame_count, lambda value: value >= expected, timeout=timeout_s
            )
            actual = await camera.frame_count.get_value(cached=False)
            if actual != expected:
                raise RuntimeError("camera frame count mismatch")
            frames.append((plane, target, pattern))

        await asyncio.wait_for(
            owner.software_shots(
                definition,
                len(patterns),
                patterns * definition.count,
                frame,
                camera_exclusive=True,
                illumination_ready=True,
                bind_stage=bind_stage,
            ),
            timeout=timeout_s,
        )
        evidence = await capture.collect()
        if evidence.stored_frames != count or len(frames) != count:
            raise RuntimeError("camera frame count mismatch")
        return PvaAutomaticStack(evidence, count, count, tuple(frames))
    finally:
        try:
            await capture.close()
        finally:
            if previous_immediate is not None:
                await camera.immediate_frame_count.set(previous_immediate)
