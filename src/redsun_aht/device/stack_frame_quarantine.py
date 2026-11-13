"""Bounded experimental quarantine for unreconciled camera frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from redsun_aht.device.stack_reconciliation import PlaneFrame, reconcile_stack

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from redsun_aht.device.unified_mcu import AngularStack, StackProgress


@dataclass(frozen=True, slots=True)
class UnassociatedFrame:
    """Owned pixel copy and camera counter, with no stage-plane assertion."""

    camera_sequence: int
    pixels: NDArray[Any]


@dataclass(frozen=True, slots=True)
class QuarantineResult:
    """Association status and raw evidence; association is order-inferred only."""

    status: str
    frames: tuple[UnassociatedFrame, ...]
    associations: tuple[PlaneFrame, ...]


class FrameQuarantine:
    """Bounded per-stack buffer, retaining no MDA-provided plane labels."""

    def __init__(self, *, max_frames: int, max_bytes: int) -> None:
        if type(max_frames) is not int or not 1 <= max_frames <= 64:
            raise ValueError("max_frames must be 1..64")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("positive byte limit required")
        self._max_frames = max_frames
        self._max_bytes = max_bytes
        self._frames: list[UnassociatedFrame] = []
        self._bytes = 0
        self._overflow = False
        self._finished = False

    def append(self, camera_sequence: int, pixels: NDArray[Any]) -> None:
        """Copy a frame or latch overflow; ignored MDA indices cannot leak in."""
        if self._finished:
            raise RuntimeError("quarantine already finalized")
        if type(camera_sequence) is not int or camera_sequence < 0:
            raise ValueError("invalid camera sequence")
        array = np.asarray(pixels)
        if array.dtype.hasobject or array.ndim < 2:
            raise ValueError("numeric image required")
        if (
            self._overflow
            or len(self._frames) >= self._max_frames
            or self._bytes + array.nbytes > self._max_bytes
        ):
            self._overflow = True
            return
        owned = array.copy()
        owned.setflags(write=False)
        self._frames.append(UnassociatedFrame(camera_sequence, owned))
        self._bytes += owned.nbytes

    def finish(
        self,
        stack: AngularStack,
        progress: StackProgress,
        *,
        camera_baseline: int,
        capture_exclusive: bool,
    ) -> QuarantineResult:
        """Finalize once, retaining unassociated pixels if reconciliation fails."""
        if self._finished:
            raise RuntimeError("quarantine already finalized")
        self._finished = True
        frames = tuple(self._frames)
        if self._overflow:
            return QuarantineResult("buffer_overflow", frames, ())
        result = reconcile_stack(
            stack,
            progress,
            tuple(frame.camera_sequence for frame in frames),
            camera_baseline=camera_baseline,
            capture_exclusive=capture_exclusive,
        )
        return QuarantineResult(result.status, frames, result.associations)
