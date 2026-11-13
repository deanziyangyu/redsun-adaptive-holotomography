"""Experimental finite UniMMCore camera adapter for external-trigger captures.

The caller owns the core, configures its external trigger and supplies a real,
validated camera counter extractor and baseline. This class does not establish
camera readiness, electrical compatibility or sensor exposure timestamps.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

from redsun_aht.device.automatic_shot_capture import CameraFrame

if TYPE_CHECKING:
    from collections.abc import Callable


class UniMMCoreTriggeredCamera:
    """One finite sequence with serialized core calls and independent counters."""

    def __init__(
        self,
        core: Any,
        *,
        device: str,
        trigger_property: str,
        trigger_value: str,
        counter_baseline: int,
        counter_from_metadata: Callable[[Any], int],
        poll_interval_s: float = 0.001,
        continuous: bool = False,
        required_properties: dict[str, str] | None = None,
    ) -> None:
        if not all(
            isinstance(x, str) and x for x in (device, trigger_property, trigger_value)
        ):
            raise ValueError("camera label and trigger property/value required")
        if type(counter_baseline) is not int or counter_baseline < 0:
            raise ValueError("independent counter baseline required")
        if not callable(counter_from_metadata) or not 0 < poll_interval_s <= 1:
            raise ValueError("counter extractor and positive polling interval required")
        if type(continuous) is not bool:
            raise ValueError("continuous must be boolean")
        if required_properties is not None and any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in required_properties.items()
        ):
            raise ValueError("required camera properties must be nonempty strings")
        self.continuous = continuous
        self.required_properties = dict(required_properties or {})
        self.core, self.device = core, device
        self.trigger_property, self.trigger_value = trigger_property, trigger_value
        self.baseline, self.counter_from_metadata = (
            counter_baseline,
            counter_from_metadata,
        )
        self.poll_interval_s = poll_interval_s
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="trigger-camera"
        )
        self._used = False
        self._closed = False
        self._started = False
        self._frames: list[CameraFrame] = []

    async def _call(self, func: Callable, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(
            self._worker, func, *args
        )

    def _arm_sync(self, count: int) -> None:
        if self.core.getCameraDevice() != self.device:
            raise RuntimeError("unexpected camera device")
        if (
            self.core.getProperty(self.device, self.trigger_property)
            != self.trigger_value
        ):
            raise RuntimeError("camera external trigger property mismatch")
        for prop, expected in self.required_properties.items():
            if self.core.getProperty(self.device, prop) != expected:
                raise RuntimeError(f"camera property mismatch: {prop}")
        if self.core.isSequenceRunning():
            raise RuntimeError("camera sequence already running")
        self.core.clearCircularBuffer()
        # Mark attempted first: an exception may occur after the camera started.
        self._started = True
        if self.continuous:
            self.core.startContinuousSequenceAcquisition(0)
        else:
            self.core.startSequenceAcquisition(count, 0, True)

    async def arm(self, frame_count: int) -> int:
        """Validate trigger configuration and start the finite sequence."""
        if self._used or self._closed:
            raise RuntimeError("camera adapter is one-use")
        if type(frame_count) is not int or not 1 <= frame_count <= 2048:
            raise ValueError("invalid finite frame count")
        self._used = True
        await self._call(self._arm_sync, frame_count)
        return self.baseline

    def _poll_sync(self) -> tuple[CameraFrame | None, bool]:
        if self.core.isBufferOverflowed():
            raise RuntimeError("camera circular buffer overflow")
        if self.core.getRemainingImageCount():
            image, metadata = self.core.popNextImageAndMD()
            received = time.perf_counter_ns()
            sequence = self.counter_from_metadata(metadata)
            if type(sequence) is not int or sequence < 0:
                raise ValueError("invalid camera metadata counter")
            pixels = np.asarray(image).copy()
            pixels.setflags(write=False)
            return CameraFrame(sequence, received, pixels), True
        return None, bool(self.core.isSequenceRunning())

    @property
    def partial_frames(self) -> tuple[CameraFrame, ...]:
        """Frames copied so far; an in-flight pop may not yet be present."""
        return tuple(self._frames)

    async def collect(self, frame_count: int) -> tuple[CameraFrame, ...]:
        """Drain copied frames and independently supplied metadata counters."""
        if not self._used or not self._started or self._closed:
            raise RuntimeError("camera sequence not armed")
        while len(self._frames) < frame_count:
            frame, running = await self._call(self._poll_sync)
            if frame is not None:
                self._frames.append(frame)
            elif not running:
                break
            else:
                await asyncio.sleep(self.poll_interval_s)
        if await self._call(self.core.isBufferOverflowed):
            raise RuntimeError("camera circular buffer overflow")
        return tuple(self._frames)

    def _ready_sync(self) -> None:
        if not self.core.isSequenceRunning():
            raise RuntimeError("camera sequence is not running")
        if self.core.isBufferOverflowed() or self.core.getRemainingImageCount():
            raise RuntimeError("unexpected camera frames before stack start")

    async def confirm_ready(self) -> None:
        """Check host-visible sequence state just before the MCU start request."""
        if self._frames:
            raise RuntimeError("unexpected camera frames before stack start")
        await self._call(self._ready_sync)
        if self._frames:
            raise RuntimeError("unexpected camera frames before stack start")

    def _finish_sync(self) -> None:
        if self._started:
            self.core.stopSequenceAcquisition()
            self._started = False
        if self.core.isBufferOverflowed():
            raise RuntimeError("camera circular buffer overflow")
        if self.core.getRemainingImageCount():
            raise RuntimeError("unexpected extra camera frames")

    async def finish(self) -> None:
        """Stop the completed sequence and reject unconsumed extra frames."""
        await self._call(self._finish_sync)

    def _close_sync(self) -> None:
        if self._started:
            self.core.stopSequenceAcquisition()
            self._started = False

    async def close(self) -> None:
        """Request camera sequence stop and close the serialized worker."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._call(self._close_sync)
        finally:
            self._worker.shutdown(wait=False)
