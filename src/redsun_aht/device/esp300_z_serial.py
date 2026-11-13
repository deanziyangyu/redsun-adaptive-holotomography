"""Experimental ESP300 axis-3 transport for a caller-owned serial endpoint.

Admit from a fresh read-only snapshot; no port opening, motor enable or homing.
Controller feedback and stop requests are not independent mechanical evidence.
"""

from __future__ import annotations

import math
import time
from threading import Event, Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redsun_aht.device.esp300_readonly import AxisSnapshot, SerialEndpoint


class Esp300ZSerial:
    """Bounded PA/MD/TP transport based on the historical MDA benchmark."""

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
        from redsun_aht.device.esp300_readonly import Esp300ReadOnly

        snapshot = Esp300ReadOnly(endpoint).snapshot()
        return cls(
            endpoint,
            snapshot.axes[2],
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
            snapshot.axis != 3
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
        self._lower, self._upper = lower_um, upper_um
        self._tolerance = 2 * snapshot.resolution_um
        self._timeout = timeout_s
        self._io_guard, self._move_guard = Lock(), Lock()
        self._uncertain = Event()

    def _exchange(self, command: str, *, query: bool) -> str:
        wire = (command + ("?" if query else "") + "\r").encode("ascii")
        with self._io_guard:
            if command != "3ST" and self._uncertain.is_set():
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
            value = float(self._exchange("3TP", query=True))
            if (
                not math.isfinite(value)
                or not self._lower <= value * 1000 <= self._upper
            ):
                raise RuntimeError("controller position outside admitted bounds")
            return value
        except BaseException:
            self._uncertain.set()
            raise

    def move_and_settle_mm(self, target_mm: float) -> None:
        """Send one bounded absolute move and verify MD and TP, without replay."""
        if (
            not math.isfinite(target_mm)
            or not self._lower <= target_mm * 1000 <= self._upper
        ):
            raise ValueError("target outside admitted bounds")
        with self._move_guard:
            if self._uncertain.is_set():
                raise RuntimeError("ESP300 state uncertain")
            try:
                if self._exchange("3MD", query=True) != "1":
                    raise RuntimeError("ESP300 axis is not idle")
                position_um = self.position_mm() * 1000
                desired_um = float(self._exchange("3DP", query=True)) * 1000
                if (
                    not math.isfinite(desired_um)
                    or abs(desired_um - position_um) > self._tolerance
                ):
                    raise RuntimeError("ESP300 desired and actual positions differ")
                self._exchange(f"3PA{target_mm:.12g}", query=False)
                deadline = time.monotonic() + self._timeout
                while True:
                    if self._uncertain.is_set():
                        raise RuntimeError("ESP300 stop requested")
                    done = self._exchange("3MD", query=True)
                    if done == "1":
                        break
                    if done != "0":
                        raise RuntimeError("invalid motion-done response")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("ESP300 move timed out")
                    time.sleep(0.005)
                if abs(self.position_mm() * 1000 - target_mm * 1000) > self._tolerance:
                    raise RuntimeError("ESP300 position does not match target")
            except BaseException:
                self._uncertain.set()
                raise

    def stop(self) -> None:
        """Request stop and inhibit reuse; this does not verify physical stop."""
        self._uncertain.set()
        self._exchange("3ST", query=False)
