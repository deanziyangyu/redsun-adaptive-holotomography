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

DCAM_STACK_MAX_BYTES = 128 * 1024 * 1024

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
        owns_core: bool = False,
        max_frame_bytes: int | None = None,
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
        self.owns_core = owns_core
        if max_frame_bytes is not None and (type(max_frame_bytes) is not int or max_frame_bytes <= 0):
            raise ValueError("Frame byte limit must be a positive integer")
        self._max_frame_bytes = max_frame_bytes
        self._frame_bytes = 0
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
        self._cleanup = None
        self._started = False
        self._frames: list[CameraFrame] = []

    async def _call(self, func: Callable, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(
            self._worker, func, *args
        )

    def _check_trigger_configuration(self) -> None:
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

    def _arm_sync(self, count: int) -> None:
        self._check_trigger_configuration()
        if self.core.isSequenceRunning():
            raise RuntimeError("camera sequence already running")
        if self._max_frame_bytes is not None:
            width = int(self.core.getImageWidth())
            height = int(self.core.getImageHeight())
            bytes_per_pixel = int(self.core.getBytesPerPixel())
            if min(width, height, bytes_per_pixel) <= 0:
                raise RuntimeError("Invalid triggered camera frame layout")
            if width * height * bytes_per_pixel * count > self._max_frame_bytes:
                raise RuntimeError("Requested stack exceeds the triggered camera frame budget")
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
            view = np.asarray(image)
            if self._max_frame_bytes is not None and self._frame_bytes + view.nbytes > self._max_frame_bytes:
                raise RuntimeError("Triggered camera frame budget exceeded")
            pixels = view.copy()
            self._frame_bytes += pixels.nbytes
            pixels.setflags(write=False)
            return CameraFrame(sequence, received, pixels), True
        return None, bool(self.core.isSequenceRunning())

    @property
    def partial_frames(self) -> tuple[CameraFrame, ...]:
        """Frames copied so far; an in-flight pop may not yet be present."""
        return tuple(self._frames)

    def take_frames(self) -> list[CameraFrame]:
        """Transfer ownership of the collected frames to a one-shot consumer."""
        frames = self._frames
        self._frames = []
        return frames

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
        self._check_trigger_configuration()
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

    def _stop_sequence_sync(self) -> None:
        if self._started:
            if self.continuous or self.core.isSequenceRunning():
                self.core.stopSequenceAcquisition()
            self._started = False

    def _finish_sync(self) -> None:
        self._stop_sequence_sync()
        if self.core.isBufferOverflowed():
            raise RuntimeError("camera circular buffer overflow")
        if self.core.getRemainingImageCount():
            raise RuntimeError("unexpected extra camera frames")

    async def finish(self) -> None:
        """Stop the completed sequence and reject unconsumed extra frames."""
        await self._call(self._finish_sync)

    def _close_sync(self) -> None:
        try:
            self._stop_sequence_sync()
        finally:
            if self.owns_core:
                self.core.unloadAllDevices()

    async def close(self) -> None:
        """Request camera sequence stop and close the serialized worker."""
        if self._cleanup is None:
            self._closed = True
            self._cleanup = self._worker.submit(self._close_sync)
            self._worker.shutdown(wait=False)
        completion = asyncio.wrap_future(self._cleanup)
        # A timed-out caller must not cancel queued native cleanup.
        completion.add_done_callback(lambda future: future.exception() if not future.cancelled() else None)
        await asyncio.shield(completion)

    @property
    def released(self) -> bool:
        """Whether camera cleanup returned successfully."""
        cleanup = self._cleanup
        return cleanup is not None and cleanup.done() and not cleanup.cancelled() and cleanup.exception() is None


def dcam_image_number(metadata: Any) -> int:
    """Extract DCAM's adapter counter, without falling back to a core index."""
    try:
        raw = metadata.GetSingleTag("HamamatsuImageNr").GetValue()
    except (AttributeError, KeyError, RuntimeError) as error:
        raise ValueError("Missing HamamatsuImageNr metadata") from error
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"Invalid HamamatsuImageNr: {raw!r}")
    # The adapter reports zero for its first frame after arming.
    return int(raw) + 1


def dcam_external_stack_camera(
    core: Any, *, device: str, polarity: str, owns_core: bool = False,
) -> UniMMCoreTriggeredCamera:
    """Bind an exclusively owned DCAM core to the finite Due stack contract."""
    if polarity not in ("NEGATIVE", "POSITIVE"):
        raise ValueError("Supply the commissioned DCAM trigger polarity")
    return UniMMCoreTriggeredCamera(
        core,
        device=device,
        trigger_property="TRIGGER SOURCE",
        trigger_value="EXTERNAL",
        counter_baseline=0,
        counter_from_metadata=dcam_image_number,
        owns_core=owns_core,
        max_frame_bytes=DCAM_STACK_MAX_BYTES,
        required_properties={
            "TRIGGER ACTIVE": "EDGE",
            "TriggerPolarity": polarity,
            "Trigger": "NORMAL",
        },
    )
