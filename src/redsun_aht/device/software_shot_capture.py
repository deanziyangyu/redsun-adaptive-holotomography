"""MCU-managed software shots with a separately owned software-snap camera."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from numpy.typing import NDArray

    from redsun_aht.device.unified_mcu import AngularStack, UnifiedMcu


class SnapCamera(Protocol):
    """Return one owned camera frame and its service count."""

    async def snap(self) -> tuple[int, NDArray[Any]]:
        """Receive one counted frame."""
        ...


@dataclass(frozen=True, slots=True)
class ShotFrame:
    """Host-ordered association; receipt time includes camera-service transfer."""

    shot: int
    plane: int
    pattern: int
    target_tenths: int
    camera_count: int
    host_receipt_perf_counter_ns: int
    quality: str = "host_order_inferred"


async def acquire_software_shots(
    owner: UnifiedMcu,
    camera: SnapCamera,
    definition: AngularStack,
    shots_per_plane: int,
    patterns: Sequence[int],
    frame_sink: Callable[[ShotFrame, NDArray[Any]], Awaitable[None]],
    *,
    camera_exclusive: bool,
    illumination_ready: bool,
    snap_timeout_s: float = 20.0,
) -> None:
    """Publish each owned frame before allowing the next MCU shot request.

    The sink owns persistence and may retain partial evidence after failure.
    A slow sink is included in the shot timeout and host-to-next-shot cadence.
    """
    previous_count: int | None = None

    async def capture(shot: int, plane: int, pattern: int, target: int) -> None:
        nonlocal previous_count
        count, image = await camera.snap()
        received = time.perf_counter_ns()
        if type(count) is not int or count < 1:
            raise RuntimeError("invalid camera-service count")
        if previous_count is not None and count != previous_count + 1:
            raise RuntimeError("camera-service count discontinuity")
        previous_count = count
        await frame_sink(
            ShotFrame(shot, plane, pattern, target, count, received), image
        )

    await owner.software_shots(
        definition,
        shots_per_plane,
        patterns,
        capture,
        camera_exclusive=camera_exclusive,
        illumination_ready=illumination_ready,
        snap_timeout_s=snap_timeout_s,
    )
