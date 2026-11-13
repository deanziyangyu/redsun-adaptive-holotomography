"""Prepare one PVA camera capture without inventing a camera frame counter.

The caller configures the camera trigger and owns its resources. Sequence-start
acknowledgement is not a hardware-ready signal or proof of electrical triggering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ophyd_async.core import wait_for_value

if TYPE_CHECKING:
    from redsun_aht.device.mmcore.camera import MMCamera


@dataclass(frozen=True, slots=True)
class CaptureEvidence:
    """A bounded store count; frames have no inferred stage association."""

    store_uri: str
    expected_frames: int
    stored_frames: int
    quality: str = "unassociated_service_count"


class PvaCaptureEvidence:
    """One-use capture preparation for an exclusively owned camera service."""

    def __init__(self, camera: MMCamera, *, timeout_s: float = 10) -> None:
        if not 0 < timeout_s <= 300:
            raise ValueError("invalid timeout")
        self.camera = camera
        self.timeout_s = timeout_s
        self._used = False
        self._capture_attempted = False
        self._acquire_attempted = False
        self._snap_mode = False
        self._software_trigger_mode = False
        self._snaps = 0
        self._uri = ""
        self._count = 0

    async def prepare(
        self,
        uri: str,
        count: int,
        *,
        external_trigger_configured: bool = False,
        internal_trigger_configured: bool = False,
        software_trigger_configured: bool = False,
        software_snap: bool = False,
    ) -> None:
        """Open storage and await start, with exactly one trigger mode asserted.

        Internal mode is suitable for a bounded capture after host plane
        readiness; these assertions do not set or verify the camera property.
        Software-trigger mode starts a continuous sequence whose triggers the
        caller issues through the camera service.
        """
        if self._used:
            raise RuntimeError("capture adapter is one-use")
        trigger_modes = (
            external_trigger_configured,
            internal_trigger_configured,
            software_trigger_configured,
        )
        if (
            any(type(mode) is not bool for mode in trigger_modes)
            or sum(trigger_modes) != 1
        ):
            raise ValueError("exactly one trigger configuration must be asserted")
        if type(software_snap) is not bool or (
            software_snap and not internal_trigger_configured
        ):
            raise ValueError("software snaps require internal trigger")
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError("store URI is required")
        if type(count) is not int or not 1 <= count <= 2048:
            raise ValueError("frame count must be 1..2048")
        self._used = True
        camera = self.camera
        if (
            await camera.acquire.get_value(cached=False)
            or await camera.capture.get_value(cached=False)
            or await camera.sequence_started.get_value(cached=False)
        ):
            raise RuntimeError("camera service is not idle")
        self._uri, self._count = uri, count
        await camera.file_path.set(uri)
        await camera.num_capture.set(count)
        await camera.record_frame_timing.set(True)
        self._capture_attempted = True
        await camera.capture.set(True)
        if await camera.captured.get_value(cached=False) != 0:
            raise RuntimeError("capture did not start empty")
        self._snap_mode = software_snap
        self._software_trigger_mode = software_trigger_configured
        if software_snap:
            return
        self._acquire_attempted = True
        await camera.acquire.set(True)
        await wait_for_value(camera.sequence_started, True, timeout=self.timeout_s)

    async def collect(self) -> CaptureEvidence:
        """Wait for store completion; return counts without plane associations."""
        if not self._acquire_attempted and not self._snap_mode:
            raise RuntimeError("capture was not prepared")
        await wait_for_value(self.camera.capture, False, timeout=self.timeout_s)
        if await self.camera.state.get_value(cached=False) == "faulted":
            error = await self.camera.last_error.get_value(cached=False)
            raise RuntimeError(f"camera service fault: {error}")
        return await self.snapshot()

    async def snap(self) -> int:
        """Request one camera-service snap into the open bounded capture."""
        if not self._snap_mode or self._snaps >= self._count:
            raise RuntimeError("software capture not ready for another snap")
        before = await self.camera.frame_count.get_value(cached=False)
        await self.camera.snap_request.set(True)
        after = await self.camera.frame_count.get_value(cached=False)
        if type(before) is not int or type(after) is not int or after != before + 1:
            raise RuntimeError("snap frame count mismatch")
        self._snaps += 1
        return after

    async def software_trigger(self) -> int:
        """Request one frame from an active software-triggered sequence."""
        if not self._software_trigger_mode or not self._acquire_attempted:
            raise RuntimeError("software-triggered sequence is not ready")
        if self._snaps >= self._count:
            raise RuntimeError("software-triggered sequence is already complete")
        before = await self.camera.frame_count.get_value(cached=False)
        await self.camera.software_trigger.set(True)
        after = await self.camera.frame_count.get_value(cached=False)
        if type(before) is not int or type(after) is not int or after != before + 1:
            raise RuntimeError("software-trigger frame count mismatch")
        self._snaps += 1
        return after

    async def snapshot(self) -> CaptureEvidence:
        """Read the last service count, without asserting that acquisition stopped."""
        if not self._capture_attempted:
            raise RuntimeError("capture was not attempted")
        count = await self.camera.captured.get_value(cached=False)
        if type(count) is not int or not 0 <= count <= self._count:
            raise RuntimeError("invalid service capture count")
        return CaptureEvidence(self._uri, self._count, count)

    async def close(self) -> None:
        """Request acquisition and capture stop; a failed request is uncertain."""
        try:
            if self._acquire_attempted:
                await self.camera.acquire.set(False)
                await wait_for_value(
                    self.camera.sequence_started, False, timeout=self.timeout_s
                )
        finally:
            if self._capture_attempted:
                await self.camera.capture.set(False)
