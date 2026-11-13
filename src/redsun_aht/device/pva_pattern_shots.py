"""Changing-pattern software shots using the existing PVA camera service."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .pva_capture_evidence import CaptureEvidence, PvaCaptureEvidence

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from redsun_aht.device.unified_mcu import AngularStack


@dataclass(frozen=True, slots=True)
class PatternShot:
    """One host-ordered snap; the timestamp is a host service acknowledgement."""

    shot: int
    plane: int
    index: int
    pattern: int
    target_tenths: int
    camera_count: int
    host_receipt_ns: int
    quality: str = "host_order_inferred"


@dataclass(frozen=True, slots=True)
class PatternPlane:
    """One complete store and its host-associated pattern order."""

    plane: int
    target_tenths: int
    patterns: tuple[int, ...]
    evidence: CaptureEvidence
    quality: str = "host_order_inferred"


async def acquire_pva_pattern_shots(
    owner: Any,
    camera: Any,
    definition: AngularStack,
    patterns: Sequence[int],
    store_for_plane: Callable[[int], str],
    on_shot: Callable[[PatternShot], Awaitable[None]],
    on_plane: Callable[[PatternPlane], Awaitable[None]],
    *,
    resources_exclusive: bool,
    internal_trigger_configured: bool,
    illumination_ready: bool,
    software_trigger_configured: bool = False,
    bind_stage: bool = True,
    hold_illumination: bool = False,
    before_plane: Callable[[int, int], Awaitable[None]] | None = None,
    timeout_s: float = 10,
    open_shot_gate: Callable[[], Awaitable[None]] | None = None,
    close_shot_gate: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Run each MCU shot with a snap or an active software-trigger sequence."""
    if (
        resources_exclusive is not True
        or illumination_ready is not True
        or type(internal_trigger_configured) is not bool
        or type(software_trigger_configured) is not bool
        or internal_trigger_configured == software_trigger_configured
    ):
        raise ValueError(
            "exclusive resources, exactly one camera trigger mode and "
            "illumination required"
        )
    if type(bind_stage) is not bool or (before_plane is not None and bind_stage):
        raise ValueError("external plane hook requires unbound stage")
    if (open_shot_gate is None) != (close_shot_gate is None):
        raise ValueError("both per-shot gate callbacks are required")
    patterns = tuple(patterns)
    if not 1 <= len(patterns) <= 32 or any(
        type(p) is not int or not 1 <= p <= 5 for p in patterns
    ):
        raise ValueError("one to 32 valid patterns required")
    uris = tuple(store_for_plane(i) for i in range(definition.count))
    if any(not isinstance(uri, str) or not uri.strip() for uri in uris) or len(
        set(uris)
    ) != len(uris):
        raise ValueError("unique plane stores required")
    active: PvaCaptureEvidence | None = None

    async def capture(shot: int, plane: int, pattern: int, target: int) -> None:
        nonlocal active
        index = shot % len(patterns)
        if plane != shot // len(patterns) or pattern != patterns[index]:
            raise RuntimeError("MCU shot order mismatch")
        if index == 0:
            if active is not None:
                raise RuntimeError("previous plane remains open")
            if before_plane is not None:
                await before_plane(plane, target)
            active = PvaCaptureEvidence(camera, timeout_s=timeout_s)
            await active.prepare(
                uris[plane],
                len(patterns),
                internal_trigger_configured=internal_trigger_configured,
                software_trigger_configured=software_trigger_configured,
                software_snap=internal_trigger_configured,
            )
        if active is None:
            raise RuntimeError("plane capture is not open")
        try:
            if open_shot_gate is not None:
                await open_shot_gate()
            count = (
                await active.software_trigger()
                if software_trigger_configured
                else await active.snap()
            )
            await on_shot(
                PatternShot(
                    shot, plane, index, pattern, target, count, time.perf_counter_ns()
                )
            )
        except BaseException:
            if close_shot_gate is not None:
                try:
                    await close_shot_gate()
                except Exception:
                    logger.exception(
                        "Per-shot gate close failed after acquisition error"
                    )
            raise
        else:
            if close_shot_gate is not None:
                await close_shot_gate()
        if index == len(patterns) - 1:
            evidence = await active.collect()
            if evidence.stored_frames != len(patterns):
                raise RuntimeError("camera service returned incomplete plane")
            await active.close()
            active = None
            await on_plane(PatternPlane(plane, target, patterns, evidence))

    try:
        await owner.software_shots(
            definition,
            len(patterns),
            patterns * definition.count,
            capture,
            camera_exclusive=True,
            illumination_ready=True,
            bind_stage=bind_stage,
            hold_illumination=hold_illumination,
        )
    finally:
        if active is not None:
            await active.close()
