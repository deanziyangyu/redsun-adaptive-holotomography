"""Experimental ESP300 axis-3 transport for a caller-owned serial endpoint.

Admit from a fresh read-only snapshot; no port opening, motor enable or homing.
Controller feedback and stop requests are not independent mechanical evidence.
"""

from __future__ import annotations

import logging
import math
import time
from threading import Event, Lock
from typing import TYPE_CHECKING

from redsun_aht.device.esp300_readonly import Esp300ReadOnly

if TYPE_CHECKING:
    from redsun_aht.device.esp300_readonly import (
        AxisSnapshot,
        ControllerSnapshot,
        SerialEndpoint,
    )

logger = logging.getLogger(__name__)


class Esp300ReadbackUnavailable(RuntimeError):
    """A stopped or quarantined survey owner cannot publish valid positions."""


class _Esp300AxisSerial:
    """Bounded PA/MD/TP transport based on the historical MDA benchmark."""

    def __init__(
        self,
        endpoint: SerialEndpoint,
        snapshot: AxisSnapshot,
        *,
        expected_identity: str,
        lower_um: float,
        upper_um: float,
        timeout_s: float = 10.0,
        axis: int,
    ) -> None:
        values = (
            lower_um,
            upper_um,
            timeout_s,
            snapshot.actual_um,
            snapshot.target_um,
            snapshot.resolution_um,
            snapshot.lower_um,
            snapshot.upper_um,
        )
        if not all(math.isfinite(v) for v in values) or timeout_s <= 0:
            raise ValueError("finite bounds and positive timeout required")
        if (
            snapshot.axis != axis
            or snapshot.identity != expected_identity
            or not expected_identity
        ):
            raise ValueError("unexpected axis identity")
        if not snapshot.motion_done or snapshot.resolution_um <= 0:
            raise ValueError("stationary axis and positive resolution required")
        if abs(snapshot.actual_um - snapshot.target_um) > 2 * snapshot.resolution_um:
            raise ValueError("initial target and position differ")
        if not (
            snapshot.lower_um
            <= lower_um
            <= snapshot.actual_um
            <= upper_um
            <= snapshot.upper_um
        ):
            raise ValueError("bounds or position outside controller limits")
        if lower_um < snapshot.actual_um - 1000 or upper_um > snapshot.actual_um + 1000:
            raise ValueError("bounds exceed the authorized starting envelope")
        self._endpoint = endpoint
        self._axis = axis
        self._lower, self._upper = lower_um, upper_um
        self._tolerance = 2 * snapshot.resolution_um
        self._timeout = timeout_s
        self._io_guard, self._move_guard = Lock(), Lock()
        self._uncertain = Event()
        self._stop_guard = Lock()
        self._stop_sent = False

    def _exchange(self, command: str, *, query: bool) -> str:
        wire = (command + ("?" if query else "") + "\r").encode("ascii")
        with self._io_guard:
            if command != f"{self._axis}ST" and self._uncertain.is_set():
                raise RuntimeError("ESP300 state uncertain")
            if self._endpoint.write(wire) != len(wire):
                raise OSError("short ESP300 write")
            self._endpoint.flush()
            if not query:
                return ""
            raw = self._endpoint.readline(512)
        if not raw or len(raw) >= 512 or not raw.endswith(b"\n"):
            raise RuntimeError("invalid ESP300 response framing")
        text = raw.decode("ascii", "strict").strip()
        if not text or any(ord(ch) < 32 for ch in text):
            raise RuntimeError("invalid ESP300 response")
        return text

    def position_mm(self) -> float:
        """Read controller position; poison the transport on uncertain input."""
        if self._uncertain.is_set():
            raise RuntimeError("ESP300 state uncertain")
        try:
            value = float(self._exchange(f"{self._axis}TP", query=True))
            if (
                not math.isfinite(value)
                or not self._lower <= value * 1000 <= self._upper
            ):
                raise RuntimeError("controller position outside admitted bounds")
            return value
        except BaseException:
            self._uncertain.set()
            raise

    def move_and_settle_mm(
        self, target_mm: float, *, stop_on_failure: bool = False
    ) -> None:
        """Send one bounded absolute move and verify MD and TP, without replay."""
        if (
            not math.isfinite(target_mm)
            or not self._lower <= target_mm * 1000 <= self._upper
        ):
            raise ValueError("target outside admitted bounds")
        with self._move_guard:
            if self._uncertain.is_set():
                raise RuntimeError("ESP300 state uncertain")
            move_attempted = False
            try:
                if self._exchange(f"{self._axis}MD", query=True) != "1":
                    raise RuntimeError("ESP300 axis is not idle")
                position_um = self.position_mm() * 1000
                desired_um = float(self._exchange(f"{self._axis}DP", query=True)) * 1000
                if (
                    not math.isfinite(desired_um)
                    or abs(desired_um - position_um) > self._tolerance
                ):
                    raise RuntimeError("ESP300 desired and actual positions differ")
                move_attempted = True
                self._exchange(f"{self._axis}PA{target_mm:.12g}", query=False)
                deadline = time.monotonic() + self._timeout
                while True:
                    if self._uncertain.is_set():
                        raise RuntimeError("ESP300 stop requested")
                    done = self._exchange(f"{self._axis}MD", query=True)
                    if done == "1":
                        actual_um = self.position_mm() * 1000
                        if abs(actual_um - target_mm * 1000) <= self._tolerance:
                            return
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                f"ESP300 position does not match target: "
                                f"{actual_um:g} um vs {target_mm * 1000:g} um"
                            )
                    elif done != "0":
                        raise RuntimeError("invalid motion-done response")
                    elif time.monotonic() >= deadline:
                        raise TimeoutError("ESP300 move timed out")
                    time.sleep(0.005)
            except BaseException:
                self._uncertain.set()
                if move_attempted and stop_on_failure:
                    try:
                        self.stop()
                    except Exception:
                        logger.exception(
                            "ESP300 axis %s stop request failed", self._axis
                        )
                raise

    def stop(self) -> None:
        """Request stop and inhibit reuse; this does not verify physical stop."""
        with self._stop_guard:
            if self._stop_sent:
                return
            self._uncertain.set()
            self._exchange(f"{self._axis}ST", query=False)
            self._stop_sent = True


class Esp300ZSerial(_Esp300AxisSerial):
    """Preserve the axis-3 interface used by the experimental Z-stack path."""

    @classmethod
    def admit(
        cls,
        endpoint: SerialEndpoint,
        *,
        expected_identity: str,
        lower_um: float,
        upper_um: float,
        timeout_s: float = 10.0,
    ) -> Esp300ZSerial:
        """Read controller identity and all axes on the same exclusive endpoint."""
        return cls(
            endpoint,
            Esp300ReadOnly(endpoint).snapshot().axes[2],
            expected_identity=expected_identity,
            lower_um=lower_um,
            upper_um=upper_um,
            timeout_s=timeout_s,
        )

    def __init__(
        self,
        endpoint: SerialEndpoint,
        snapshot: AxisSnapshot,
        *,
        expected_identity: str,
        lower_um: float,
        upper_um: float,
        timeout_s: float = 10.0,
    ) -> None:
        super().__init__(
            endpoint,
            snapshot,
            expected_identity=expected_identity,
            lower_um=lower_um,
            upper_um=upper_um,
            timeout_s=timeout_s,
            axis=3,
        )


class Esp300SurveySerial:
    """Own XYZ movement and readback on one exclusive serial endpoint."""

    def __init__(
        self,
        endpoint: SerialEndpoint,
        axes: tuple[tuple[str, float, float], ...],
        *,
        timeout_s: float = 10.0,
    ) -> None:
        if len(axes) != 3:
            raise ValueError("expected X, Y and Z identities and bounds")
        self._lock = Lock()
        self._activity_lock = Lock()
        self._uncertain = Event()
        self._active: _Esp300AxisSerial | None = None
        self._failed_axis: _Esp300AxisSerial | None = None
        self._reader = Esp300ReadOnly(endpoint)
        initial = self._reader.snapshot()
        self._axes = tuple(
            _Esp300AxisSerial(
                endpoint,
                state,
                expected_identity=identity,
                lower_um=lower,
                upper_um=upper,
                timeout_s=timeout_s,
                axis=axis,
            )
            for axis, (state, (identity, lower, upper)) in enumerate(
                zip(initial.axes, axes, strict=True), start=1
            )
        )
        self._configuration = axes
        self._resolutions = tuple(state.resolution_um for state in initial.axes)

    def _snapshot(self) -> ControllerSnapshot:
        if self._uncertain.is_set():
            raise RuntimeError("ESP300 survey state uncertain; reconcile before reuse")
        try:
            result = self._reader.snapshot()
            for state, (identity, lower, upper), resolution in zip(
                result.axes, self._configuration, self._resolutions, strict=True
            ):
                if (
                    state.identity != identity
                    or state.resolution_um != resolution
                    or not state.motion_done
                    or not state.lower_um
                    <= lower
                    <= state.actual_um
                    <= upper
                    <= state.upper_um
                    or abs(state.target_um - state.actual_um) > 2 * state.resolution_um
                ):
                    raise RuntimeError(f"ESP300 survey axis {state.axis} state changed")
            return result
        except BaseException:
            self._uncertain.set()
            raise

    def snapshot(self) -> ControllerSnapshot:
        """Read all axes while excluding moves on this endpoint."""
        with self._lock:
            return self._snapshot()

    def axis_specs(self) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
        """Return the admitted bounds and encoder resolutions in XYZ order."""
        return tuple(
            (lower, upper) for _, lower, upper in self._configuration
        ), self._resolutions

    def move_um(self, axis: int, target_um: float) -> None:
        """Verify all axes before moving one within its configured envelope."""
        if type(axis) is not int or not 1 <= axis <= 3:
            raise ValueError("expected an ESP300 axis number in 1..3")
        _, lower, upper = self._configuration[axis - 1]
        if not math.isfinite(target_um) or not lower <= target_um <= upper:
            raise ValueError("target outside admitted bounds")
        with self._lock:
            self._snapshot()
            if not self._reader.motor_power(axis):
                raise RuntimeError(f"ESP300 axis {axis} motor power is off")
            transport = self._axes[axis - 1]
            with self._activity_lock:
                if self._uncertain.is_set():
                    raise RuntimeError("ESP300 survey stop requested")
                self._active = transport
            try:
                transport.move_and_settle_mm(target_um / 1000, stop_on_failure=True)
            except BaseException:
                self._uncertain.set()
                with self._activity_lock:
                    self._failed_axis = transport
                raise
            finally:
                with self._activity_lock:
                    self._active = None
            if self._uncertain.is_set():
                raise RuntimeError("ESP300 survey stop requested")

    def read_um(self, axis: int) -> float:
        """Read an axis after any in-flight survey operation completes."""
        if type(axis) is not int or not 1 <= axis <= 3:
            raise ValueError("expected an ESP300 axis number in 1..3")
        with self._lock:
            if self._uncertain.is_set():
                raise Esp300ReadbackUnavailable(
                    "ESP300 survey state uncertain; reconcile before reuse"
                )
            try:
                return self._axes[axis - 1].position_mm() * 1000
            except BaseException:
                self._uncertain.set()
                raise

    def stop(self) -> None:
        """Latch a stop request and interrupt any active axis on this owner."""
        with self._activity_lock:
            self._uncertain.set()
            axis = self._active or self._failed_axis
            if axis is not None:
                axis.stop()
