"""Service-owned camera captures coordinated by host-acknowledged MCU planes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .pva_capture_evidence import CaptureEvidence, PvaCaptureEvidence

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from redsun_aht.device.unified_mcu import AngularStack


@dataclass(frozen=True, slots=True)
class PlaneCapture:
    """One service store associated by exclusive host-controlled ordering."""

    plane: int
    target_tenths: int
    pattern: int
    evidence: CaptureEvidence
    quality: str = "host_order_inferred"


@dataclass(frozen=True, slots=True)
class PlaneTiming:
    """Host observation points, not camera exposure or electrical timestamps."""

    run_id: int
    plane: int
    ready_ns: int
    stage_ready_ns: int
    camera_started_ns: int
    capture_complete_ns: int
    camera_stopped_ns: int


async def acquire_pva_internal_planes(
    owner: Any,
    camera: Any,
    definition: AngularStack,
    frames_per_plane: int,
    patterns: Sequence[int],
    store_for_plane: Callable[[int], str],
    on_plane: Callable[[PlaneCapture], Awaitable[None]],
    *,
    resources_exclusive: bool,
    internal_trigger_configured: bool,
    illumination_ready: bool,
    bind_stage: bool = True,
    hold_illumination: bool = False,
    before_plane: Callable[[int, int], Awaitable[None]] | None = None,
    on_timing: Callable[[PlaneTiming], None] | None = None,
    timeout_s: float = 10.0,
) -> None:
    """Store N internal-triggered frames per ready plane, then advance once.

    Camera and MCU are separately owned by services. Caller owns their
    reservations and verifies the camera trigger property. Each plane gets a
    unique create-only store; associations are inferred by host ordering.
    """
    if (
        resources_exclusive is not True
        or internal_trigger_configured is not True
        or illumination_ready is not True
    ):
        raise ValueError(
            "exclusive resources, internal trigger and illumination required"
        )
    if type(frames_per_plane) is not int or not 1 <= frames_per_plane <= 32:
        raise ValueError("frames per plane must be 1..32")
    if before_plane is not None and bind_stage:
        raise ValueError("external plane hook requires unbound stage")
    patterns = tuple(patterns)
    if len(patterns) != definition.count or any(
        type(p) is not int or not 1 <= p <= 5 for p in patterns
    ):
        raise ValueError("one valid pattern per plane required")
    uris = tuple(store_for_plane(i) for i in range(definition.count))
    if any(not isinstance(uri, str) or not uri.strip() for uri in uris) or len(
        set(uris)
    ) != len(uris):
        raise ValueError("unique plane stores required")

    async def capture(shot: int, plane: int, pattern: int, target: int) -> None:
        if shot != plane:
            raise RuntimeError("MCU plane order mismatch")
        ready_ns = time.perf_counter_ns()
        if before_plane is not None:
            await before_plane(plane, target)
        stage_ready_ns = time.perf_counter_ns()
        acquisition = PvaCaptureEvidence(camera, timeout_s=timeout_s)
        try:
            await acquisition.prepare(
                uris[plane], frames_per_plane, internal_trigger_configured=True
            )
            camera_started_ns = time.perf_counter_ns()
            evidence = await acquisition.collect()
            capture_complete_ns = time.perf_counter_ns()
            if evidence.stored_frames != frames_per_plane:
                raise RuntimeError("camera service returned incomplete plane")
        finally:
            await acquisition.close()
        camera_stopped_ns = time.perf_counter_ns()
        if on_timing is not None:
            on_timing(
                PlaneTiming(
                    definition.run_id,
                    plane,
                    ready_ns,
                    stage_ready_ns,
                    camera_started_ns,
                    capture_complete_ns,
                    camera_stopped_ns,
                )
            )
        await on_plane(PlaneCapture(plane, target, pattern, evidence))

    await owner.software_shots(
        definition,
        1,
        patterns,
        capture,
        camera_exclusive=True,
        illumination_ready=True,
        bind_stage=bind_stage,
        hold_illumination=hold_illumination,
    )
