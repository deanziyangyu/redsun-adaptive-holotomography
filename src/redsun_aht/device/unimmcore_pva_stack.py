"""Experimental UniMMCore sequence adapter for the unified MCU PVA service.

This supports only finite sequenced stacks. It has no manual move, physical
position readback, durable lease, camera-ready sensor, restart recovery or safety stop.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

from pymmcore_plus.experimental.unicore import StageDevice
from pymmcore_plus.mda import MDAEngine

from redsun_aht.device.unified_mcu import AngularStack

if TYPE_CHECKING:
    from concurrent.futures import Future

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import StackProgress


class DeferredPvaStage(StageDevice):
    """Sequence-only adapter; its normal UniMMCore start arms, not moves."""

    def __init__(
        self,
        client: PvaUnifiedMcu,
        loop: asyncio.AbstractEventLoop,
        *,
        run_id: int,
        um_per_rev: float,
        origin_um: float,
        lower_tenths: int,
        upper_tenths: int,
        ready: bool,
        trigger_mask: int = 1,
    ) -> None:
        super().__init__()
        if (
            not math.isfinite(um_per_rev)
            or um_per_rev <= 0
            or not math.isfinite(origin_um)
        ):
            raise ValueError("finite positive lead and finite origin required")
        if ready is not True:
            raise ValueError(
                "host camera and illumination readiness assertion required"
            )
        if type(trigger_mask) is not int or not 1 <= trigger_mask <= 7:
            raise ValueError("trigger mask must select outputs 1..3")
        self.client, self.loop = client, loop
        self.run_id, self.um_per_rev, self.origin_um = run_id, um_per_rev, origin_um
        self.lower, self.upper = lower_tenths, upper_tenths
        self.trigger_mask = trigger_mask
        self.definition: AngularStack | None = None
        self.future: Future[StackProgress] | None = None
        self.abort_future: Future[None] | None = None
        self._armed = False
        self._attempted = False

    def home(self) -> None:
        """Refuse unsupported homing."""
        raise NotImplementedError("homing is not supported")

    def set_origin(self) -> None:
        """Refuse unsupported re-referencing."""
        raise NotImplementedError("origin changes are not supported")

    def set_position_um(self, val: float) -> None:
        """Refuse unsupported manual motion."""
        raise NotImplementedError("manual movement is not supported")

    def get_position_um(self) -> float:
        """Refuse to invent a physical readback."""
        raise NotImplementedError("physical readback is not available")

    def get_sequence_max_length(self) -> int:
        """Return the firmware's maximum plane count."""
        return 32

    def send_sequence(self, sequence: tuple[float, ...]) -> None:
        """Validate the full sequence and convert micrometres to motor tenths."""
        if self._attempted or self._armed:
            raise RuntimeError("sequence unavailable or already attempted")
        self.definition = None
        if not sequence or len(sequence) > 32:
            raise ValueError("sequence must contain 1..32 positions")
        units = []
        for value in sequence:
            if not math.isfinite(value):
                raise ValueError("finite positions required")
            angle = (value - self.origin_um) * 3600 / self.um_per_rev
            rounded = round(angle)
            if not math.isclose(angle, rounded, rel_tol=0, abs_tol=1e-6):
                raise ValueError("position is not on the firmware command grid")
            units.append(rounded)
        step = units[1] - units[0] if len(units) > 1 else 0
        if any(value != units[0] + i * step for i, value in enumerate(units)):
            raise ValueError("firmware requires uniform steps")
        self.definition = AngularStack(
            self.run_id,
            units[0],
            step,
            len(units),
            self.lower,
            self.upper,
            self.trigger_mask,
        )

    def start_sequence(self) -> None:
        """Arm only; the camera-start hook performs the actual submission."""
        if self.definition is None or self._armed or self._attempted:
            raise RuntimeError("sequence is not available")
        self._armed = True

    def release_after_camera_start(self) -> None:
        """Schedule once; completion must be checked separately by the caller."""
        if not self._armed or self._attempted or self.definition is None:
            raise RuntimeError("sequence is not armed")
        if not self.loop.is_running():
            raise RuntimeError("PVA event loop is not running")
        if not self.client.reservation_held:
            raise RuntimeError("a held MCU reservation is required")
        self._armed = False
        self._attempted = True
        self.future = asyncio.run_coroutine_threadsafe(
            self.client.execute(self.definition, ready=True, require_reservation=True),
            self.loop,
        )

    def stop_sequence(self) -> None:
        """Disarm or request an asynchronous abort; stop is not verified."""
        self._armed = False
        if (
            self.future is not None
            and not self.future.done()
            and self.abort_future is None
        ):
            self.abort_future = asyncio.run_coroutine_threadsafe(
                self.client.abort_stack(self.run_id), self.loop
            )

    def stop(self) -> None:
        """Request the sequence abort without asserting motor stop."""
        self.stop_sequence()


class DeferredPvaEngine(MDAEngine):
    """Custom MDA engine for the paired deferred PVA stage."""

    def __init__(
        self, mmc: Any, stage: DeferredPvaStage, *, completion_timeout_s: float = 5.0
    ) -> None:
        super().__init__(mmc)
        if not math.isfinite(completion_timeout_s) or completion_timeout_s <= 0:
            raise ValueError("completion timeout must be positive and finite")
        self.stage = stage
        self.completion_timeout_s = completion_timeout_s

    def post_sequence_started(self, event: Any) -> None:
        """Submit after camera start returns; camera readiness is not sensed."""
        if event.z_sequence:
            self.stage.release_after_camera_start()

    def exec_sequenced_event(self, event: Any) -> Any:
        """Drain camera frames, then await the MCU result before normal teardown."""
        inner = super().exec_sequenced_event(event)
        signal = None
        cancelled = False
        try:
            while True:
                try:
                    payload = inner.send(signal)
                except StopIteration:
                    break
                signal = yield payload
                if signal == "cancel":
                    cancelled = True
                    self.stage.stop_sequence()
            if event.z_sequence and not cancelled:
                if self.stage.future is None:
                    raise RuntimeError("stage sequence was not submitted")
                try:
                    self.stage.future.result(timeout=self.completion_timeout_s)
                except BaseException:
                    self.stage.stop_sequence()
                    raise
        finally:
            inner.close()
