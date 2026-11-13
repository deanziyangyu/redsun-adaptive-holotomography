"""Experimental host-stepped axis through the unified PVA owner.

Each move is one autonomous one-plane MCU stack and attempts its selected GPIO
pulse. Camera exposures are separate software snaps. Controller readback is not
independent physical metrology, and abort is not a verified stop.
"""

from __future__ import annotations

import asyncio
import math
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from typing import TYPE_CHECKING, Any, TypeVar

from redsun_aht.device.unified_mcu import AngularStack

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.pva_x_state import PvaXState
    from redsun_aht.device.unified_mcu import XControllerState


class PvaSteppedAxis:
    """Non-sequenced axis; caller owns reservation and camera resources."""

    def __init__(
        self,
        client: PvaUnifiedMcu,
        state: PvaXState,
        loop: asyncio.AbstractEventLoop,
        *,
        first_run_id: int,
        um_per_rev: float,
        origin_um: float,
        lower_tenths: int,
        upper_tenths: int,
        trigger_mask: int,
        resources_exclusive: bool,
        ready: bool,
        timeout_s: float = 15,
    ) -> None:
        super().__init__()
        if resources_exclusive is not True or ready is not True:
            raise ValueError("exclusive resources and readiness must be asserted")
        if (
            not math.isfinite(um_per_rev)
            or um_per_rev <= 0
            or not math.isfinite(origin_um)
        ):
            raise ValueError("finite positive lead and finite origin required")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("finite positive timeout required")
        AngularStack(
            first_run_id, lower_tenths, 0, 1, lower_tenths, upper_tenths, trigger_mask
        )
        self.client, self.state, self.loop = client, state, loop
        self.run_id, self.lead, self.origin = first_run_id, um_per_rev, origin_um
        self.lower, self.upper, self.mask = lower_tenths, upper_tenths, trigger_mask
        self.timeout_s = timeout_s
        self._operation_guard = threading.Lock()
        self._uncertain = False
        self._active_id: int | None = None
        self.abort_future: Any = None

    def _call(self, coro: Coroutine[Any, Any, T]) -> T:
        if not self.loop.is_running():
            coro.close()
            raise RuntimeError("PVA event loop is not running")
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self.loop:
            coro.close()
            raise RuntimeError("stage call cannot block the PVA event loop")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=self.timeout_s)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError(
                "PVA stage request timed out; outcome uncertain"
            ) from None

    def _query(self) -> XControllerState:
        return self._call(self.state.read_state())

    def validate_position_um(self, val: float) -> int:
        """Validate a target and return its angular grid position without I/O."""
        if not math.isfinite(val):
            raise ValueError("finite target required")
        angle = (val - self.origin) * 3600 / self.lead
        target = round(angle)
        if not math.isclose(angle, target, rel_tol=0, abs_tol=1e-6):
            raise ValueError("target is not on the firmware command grid")
        AngularStack(
            self.run_id,
            target,
            0,
            1,
            self.lower,
            self.upper,
            self.mask,
        )
        return target

    def set_position_um(self, val: float) -> None:
        """Submit exactly one finite move and verify controller-reported state."""
        target = self.validate_position_um(val)
        with self._operation_guard:
            if self._uncertain:
                raise RuntimeError("stage outcome uncertain")
            if not self.client.reservation_held:
                raise RuntimeError("a held MCU reservation is required")
            stack = AngularStack(
                self.run_id, target, 0, 1, self.lower, self.upper, self.mask
            )
            self._active_id = stack.run_id
            try:
                report = self._call(
                    self.client.execute(stack, ready=True, require_reservation=True)
                )
                if self._uncertain:
                    raise RuntimeError("stop requested; stage outcome uncertain")
                if not report.complete or report.targets_tenths != (target,):
                    raise RuntimeError("controller report does not match move")
                state = self._query()
                if self._uncertain:
                    raise RuntimeError("stop requested; stage outcome uncertain")
                if (
                    abs(state.target_tenths - target) > 1
                    or abs(state.position_tenths - target) > 1
                    or state.speed_tenths_rpm != 0
                    or abs(state.error_hundredths_degree) > 10
                    or state.home_flags & 3 != 3
                    or state.home_flags & 0x3C
                    or state.motor_flags & 3 != 3
                    or state.motor_flags & 0x8C
                ):
                    raise RuntimeError("controller state does not confirm target")
            except BaseException as error:
                self._uncertain = True
                if isinstance(error, TimeoutError):
                    self.stop()
                raise
            finally:
                self._active_id = None

    def get_position_um(self) -> float:
        """Convert a fresh controller-reported angle using the caller's scale."""
        with self._operation_guard:
            if self._uncertain:
                raise RuntimeError("stage outcome uncertain")
            try:
                state = self._query()
                if not self.lower <= state.position_tenths <= self.upper:
                    raise RuntimeError("controller position outside configured bounds")
                return self.origin + state.position_tenths * self.lead / 3600
            except BaseException:
                self._uncertain = True
                raise

    def home(self) -> None:
        """Homing is intentionally not implemented."""
        raise NotImplementedError("homing is not supported")

    def stop(self) -> None:
        """Request an asynchronous run abort; delivery and stop are unverified."""
        run_id = self._active_id
        if run_id is not None and self.abort_future is None and self.loop.is_running():
            self._uncertain = True
            self.abort_future = asyncio.run_coroutine_threadsafe(
                self.client.abort_stack(run_id), self.loop
            )

    def set_origin(self) -> None:
        """Refuse unsupported reference changes."""
        raise NotImplementedError("origin changes are not supported")
