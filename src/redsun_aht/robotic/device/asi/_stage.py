"""Bounded XY facade over the ASI legacy-ASCII client."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Final

from .._motion import MoveHalted
from ._client import TENTHS_PER_UM

#: The controller reports in 0.1 um units, so accept one half-micron landing
#: error after an idle readback. This is the same tolerance as the qualified
#: native Ludl path.
# The controller settles up to ~6 counts short of a target (0.2 um per count
# after the MR095 reticle calibration), i.e. ~1.2 um observed; 1.5 um covers the
# measured spread without hiding a whole 10 um step.
POSITION_TOLERANCE_UM: Final[float] = 1.5

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ._client import AsiXYClient


def quantize_tenths(position_um: float, counts_per_um: float = TENTHS_PER_UM) -> int:
    """Quantize a micrometre coordinate to the controller's tenth-micron units.

    The donor script documents a controller quirk: accepted targets must not
    end in a raw digit 3, so such targets are nudged by one tenth of a micron.
    This preserves that donor-verified command shape rather than inventing a
    new encoding.
    """
    if (
        isinstance(position_um, bool)
        or not isinstance(position_um, (int, float))
        or not math.isfinite(position_um)
    ):
        raise ValueError("target position must be finite")
    tenths = int(position_um * counts_per_um)
    if abs(tenths) % 10 == 3:
        tenths += 1
    return tenths


def tenths_to_um(tenths: float, counts_per_um: float = TENTHS_PER_UM) -> float:
    """Convert controller tenth-micron units to micrometres."""
    return float(tenths) / counts_per_um


class AsiXYStage:
    position_observer = None
    """Startup-bounded XY moves over one owned ASI serial session.

    Completion uses host-polled paired busy status. A timeout or a final
    readback outside tolerance is reported as unknown; this facade never
    retries a move automatically.
    """

    def __init__(
        self,
        client: AsiXYClient,
        *,
        max_xy_offset_um: float = 2000.0,
        move_timeout_s: float = 60.0,
        poll_period_s: float = 0.05,
        startup_speed_mm_s: float | None = None,
        startup_ramp_ms: float | None = None,
        startup_backlash_mm: float | None = None,
        absolute_takeup_um: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        counts_per_um: float = TENTHS_PER_UM,
    ) -> None:
        if (
            isinstance(max_xy_offset_um, bool)
            or not isinstance(max_xy_offset_um, (int, float))
            or not math.isfinite(max_xy_offset_um)
            or not 0 < max_xy_offset_um <= 2000.0
        ):
            raise ValueError("XY envelope must be positive and no greater than 2000 um")
        for value, name in (
            (move_timeout_s, "move timeout"),
            (poll_period_s, "poll period"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if startup_speed_mm_s is not None and (
            isinstance(startup_speed_mm_s, bool)
            or not isinstance(startup_speed_mm_s, (int, float))
            or not math.isfinite(startup_speed_mm_s)
            or startup_speed_mm_s <= 0
        ):
            raise ValueError("startup speed must be finite and positive")
        if startup_ramp_ms is not None and (
            isinstance(startup_ramp_ms, bool)
            or not isinstance(startup_ramp_ms, (int, float))
            or not math.isfinite(startup_ramp_ms)
            or startup_ramp_ms < 0
        ):
            raise ValueError("startup ramp time must be finite and non-negative")
        if startup_backlash_mm is not None and (
            isinstance(startup_backlash_mm, bool)
            or not isinstance(startup_backlash_mm, (int, float))
            or not math.isfinite(startup_backlash_mm)
            or not 0 <= startup_backlash_mm <= 1.0
        ):
            raise ValueError("startup backlash must be finite and between 0 and 1 mm")
        if (
            isinstance(absolute_takeup_um, bool)
            or not isinstance(absolute_takeup_um, (int, float))
            or not math.isfinite(absolute_takeup_um)
            or absolute_takeup_um < 0
        ):
            raise ValueError("absolute take-up must be finite and non-negative")
        if (
            isinstance(counts_per_um, bool)
            or not isinstance(counts_per_um, (int, float))
            or not math.isfinite(counts_per_um)
            or counts_per_um <= 0
        ):
            raise ValueError("counts per micrometre must be finite and positive")
        self._client = client
        self._counts_per_um = float(counts_per_um)
        self.travel_bounds_enabled = True
        self._max_xy_offset_um = float(max_xy_offset_um)
        self._move_timeout_s = float(move_timeout_s)
        self._poll_period_s = float(poll_period_s)
        self._startup_speed_mm_s = (
            None if startup_speed_mm_s is None else float(startup_speed_mm_s)
        )
        self._startup_ramp_ms = (
            None if startup_ramp_ms is None else float(startup_ramp_ms)
        )
        self._startup_backlash_mm = (
            None if startup_backlash_mm is None else float(startup_backlash_mm)
        )
        self._absolute_takeup_um = float(absolute_takeup_um)
        self._clock = clock
        self._sleep = sleep
        self._connect_lock = asyncio.Lock()
        self._move_lock = asyncio.Lock()
        # A stop request ends the wait, not the command: the controller comes to rest short of
        # the target, so the bounded move reports where it stopped instead of polling until its
        # own timeout. Measured 2026-10-31: a halted 400 um lane left the service busy for the
        # whole move timeout while the stage was already at rest 200 um in.
        self._halt_requested = asyncio.Event()
        # The last position the telemetry observer saw, kept so a settle timeout can report both
        # the direct read and the observed track (2026-10-30).
        self._last_observed_tenths: tuple[float, float] | None = None
        self._origin_xy_um: tuple[float, float] | None = None
        self._origin_tenths: tuple[float, float] | None = None
        self._last_commanded_xy_um: tuple[float, float] | None = None
        self._last_direction_xy: tuple[int, int] = (0, 0)
        self._original_speeds_mm_s: dict[str, float] | None = None
        self._original_ramps_ms: dict[str, float] | None = None
        self._closed = False

    async def connect(self) -> tuple[float, float]:
        """Handshake, capture the startup position and optional bounded profile.

        Run speed and ramp time are session-scoped and restored on close. The
        anti-backlash distance is an instrument setting and is kept after the
        session, so a later session or raw client finds the configured value.
        """
        async with self._connect_lock:
            if self._origin_xy_um is not None:
                raise RuntimeError("stage envelope has already been established")
            await self._client.connect()
            tenths = await self._client.read_xy_tenths()
            self._origin_tenths = tenths
            self._origin_xy_um = (
                tenths_to_um(tenths[0], self._counts_per_um),
                tenths_to_um(tenths[1], self._counts_per_um),
            )
            self._original_speeds_mm_s = {
                "x": await self._client.read_speed_mm_s("X"),
                "y": await self._client.read_speed_mm_s("Y"),
            }
            if self._startup_speed_mm_s is not None:
                try:
                    await self._client.set_speed_mm_s("X", self._startup_speed_mm_s)
                    await self._client.set_speed_mm_s("Y", self._startup_speed_mm_s)
                except BaseException:
                    await self._restore_profile()
                    self._client.close()
                    raise
            if self._startup_ramp_ms is not None:
                try:
                    self._original_ramps_ms = {
                        "x": await self._client.read_ramp_ms("X"),
                        "y": await self._client.read_ramp_ms("Y"),
                    }
                    await self._client.set_ramp_ms("X", self._startup_ramp_ms)
                    await self._client.set_ramp_ms("Y", self._startup_ramp_ms)
                except BaseException:
                    await self._restore_profile()
                    self._client.close()
                    raise
            if self._startup_backlash_mm is not None:
                try:
                    await self._client.set_backlash_xy_mm(
                        self._startup_backlash_mm, self._startup_backlash_mm
                    )
                except BaseException:
                    await self._restore_profile()
                    self._client.close()
                    raise
            return self._origin_xy_um

    async def read_xy(self) -> tuple[float, float]:
        """Read both controller coordinates in micrometres."""
        tenths = await self._client.read_xy_tenths()
        return (
            tenths_to_um(tenths[0], self._counts_per_um),
            tenths_to_um(tenths[1], self._counts_per_um),
        )

    async def read_xy_with_tenths(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return one raw tenth-micron sample and its micrometre conversion."""
        tenths = await self._client.read_xy_tenths()
        return tenths, (
            tenths_to_um(tenths[0], self._counts_per_um),
            tenths_to_um(tenths[1], self._counts_per_um),
        )

    @property
    def xy_limits_um(self) -> dict[str, tuple[float, float]]:
        """Return the startup-relative command interval for each XY axis."""
        if self._origin_xy_um is None:
            raise RuntimeError("call connect() before reading travel limits")
        limit = self._max_xy_offset_um
        return {
            axis: (origin - limit, origin + limit)
            for axis, origin in zip(("x", "y"), self._origin_xy_um, strict=True)
        }

    @property
    def calibration_metadata(self) -> dict[str, dict[str, int | float]]:
        """Return the fixed unit convention and the captured reference sample."""
        if self._origin_tenths is None or self._origin_xy_um is None:
            raise RuntimeError("call connect() before reading calibration metadata")
        result: dict[str, dict[str, int | float]] = {}
        for index, axis in enumerate(("x", "y")):
            result[axis] = {
                "reference_position_tenths": self._origin_tenths[index],
                "reference_position_um": self._origin_xy_um[index],
                "tenths_per_um": self._counts_per_um,
                "positive_direction": 1,
            }
        return result

    async def is_busy_xy(self) -> bool:
        """Return the controller's paired XY busy state."""
        return await self._client.is_busy_xy()

    async def move_xy(self, target_xy_um: tuple[float, float]) -> tuple[float, float]:
        """Move to a paired absolute target inside the startup envelope.

        A move is a single command unless the opt-in ``absolute_takeup_um`` profile
        setting is non-zero and an axis reverses its direction of travel. On a
        reversal that axis is first commanded to ``target - sign(step) * take-up``
        and then to ``target``, so the final approach arrives on the same flank of
        the play whatever the direction it came from. Same-direction moves, moves
        no longer than the take-up and a take-up of zero keep the single-command
        form.

        Measured basis: with the settled and frame-time readbacks identical (so the
        service was not lagging), the first probe after a reversal moved 5.15 px/um
        of reported travel against 5.29-5.31 px/um for the next three same-direction
        increments; after a return to the anchor the stage reported 0.00 um while the
        specimen was about 1.3 um away, and two independent runs put reversal moves
        at 5.60-5.64 px/um against 5.28-5.43 for the nine other directions. That is
        0.7-1.4 um of play between encoder and specimen which the readback cannot
        see, and every hand-eye pair after a reversal inherits it.

        Cost: one extra short move per direction reversal, so the default 0.0 keeps
        the single-command form for the long one-directional scanning moves. When the
        take-up point would leave the envelope it is clamped to the envelope edge and
        logged instead of being commanded outside the envelope.
        """
        _require_xy(target_xy_um, "XY target")
        async with self._move_lock:
            target = (float(target_xy_um[0]), float(target_xy_um[1]))
            self._bounded_target_counts(target)
            approach = self._absolute_takeup_approach(target)
            if approach is not None:
                logger.info(
                    f"move_xy reversal take-up approach={approach} target={target}"
                )
                await self._move_xy_locked(approach)
            return await self._move_xy_locked(target)

    async def move_axis(self, axis: str, value: float) -> tuple[float, float]:
        """Move only the selected lateral axis, without re-commanding its partner."""
        if axis not in ("x", "y"):
            raise ValueError(f"Unsupported lateral axis {axis!r}")
        async with self._move_lock:
            current = await self.read_xy()
            target = (value, current[1]) if axis == "x" else (current[0], value)
            self._bounded_target_counts(target)
            approach = self._absolute_takeup_approach(target)
            index = 0 if axis == "x" else 1
            if approach is not None and approach[index] != target[index]:
                point = (approach[0], target[1]) if axis == "x" else (target[0], approach[1])
                await self._move_xy_locked(point, axis=axis.upper())
            return await self._move_xy_locked(target, axis=axis.upper())

    async def move_relative_xy(
        self, delta_xy_um: tuple[float, float]
    ) -> tuple[float, float]:
        """Move by one paired relative displacement.

        A jog is a plain bounded relative move: one command, to the displacement the caller
        asked for. The take-up lives in the controller's anti-backlash setting, the
        ``--backlash-mm`` distance this stage writes as ``2HB`` on connect, and that setting is
        the single compensation for the play. An earlier version of this method added a second,
        client-side take-up: it stepped to one *backlash behind* the requested target first and
        then forward onto it. That 16 um was a configured policy, not a measurement - a revisit
        landed within 1.4 um after hundreds of micrometres of travel, which bounds the live
        residual play at roughly 1-2 um. A 16 um software take-up is therefore larger than the
        physical play: it lands the axis on the opposite flank and spends about 32 um of travel
        per reversal, so it is gone and the controller's setting is left to do the work alone.

        Absolute moves carry an extra approach move only through the opt-in
        ``absolute_takeup_um`` setting: a long one-directional scan does not reverse, and
        an extra move per frame would cost cadence. See ``move_xy``.
        """
        _require_xy(delta_xy_um, "XY displacement")
        async with self._move_lock:
            current = await self.read_xy()
            target = (
                current[0] + float(delta_xy_um[0]),
                current[1] + float(delta_xy_um[1]),
            )
            return await self._move_xy_locked(target)

    async def stop(self) -> None:
        """Issue the controller's paired XY halt without taking the move lock.

        The halt also releases the pending wait: :meth:`_wait_for_target` stops
        polling and raises :class:`MoveHalted` with the position the axis
        actually reached, so the owning service can publish a ``stopped``
        outcome rather than staying busy until the move timeout.
        """
        self._halt_requested.set()
        await self._client.halt_xy()

    async def close(self) -> None:
        """Restore the captured run profile and release the serial session."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._restore_profile()
        finally:
            self._client.close()

    async def _move_xy_locked(
        self, target_xy_um: tuple[float, float], *, axis: str | None = None,
    ) -> tuple[float, float]:
        if self._origin_xy_um is None:
            raise RuntimeError("call connect() before requesting stage motion")
        target_tenths = self._bounded_target_counts(target_xy_um)
        previous = self._last_commanded_xy_um
        self._halt_requested.clear()
        self._record_commanded_move(target_xy_um)
        logger.info(
            f"move_xy target counts={target_tenths} um={target_xy_um} "
            f"counts_per_um={self._counts_per_um}"
        )
        # Skip an axis the command does not move. Restating it is not a no-op on this controller: with its
        # anti-backlash distance configured - the written 2HB setting, which is the single compensation for
        # the play - a sub-play command (including zero) is treated as a reversal and the axis travels about
        # two backlashes and returns. Measured on the stage alone, a 0.16 um y command produced a 31.0 um
        # excursion, 0.00 um with compensation disabled, and 0.00 um when y was not commanded. A traverse
        # from one row to the next changes only x, so it must command only x.
        #
        # That configured distance is a policy value, not a measurement: a revisit landed within 1.4 um
        # after hundreds of micrometres of travel, which bounds the live residual play at about 1-2 um. So
        # the controller's setting is the one compensation, and neither the jog nor this axis skip adds a
        # second one of its own.
        unchanged = None
        if previous is not None:
            previous_tenths = tuple(
                quantize_tenths(value, self._counts_per_um) for value in previous
            )
            unchanged = tuple(
                target_tenths[index] == previous_tenths[index] for index in (0, 1)
            )
        if axis is None and unchanged is not None and unchanged[0] != unchanged[1]:
            axis = "X" if not unchanged[0] else "Y"
        if axis is not None:
            index = 0 if axis == "X" else 1
            logger.info(f"move_xy single-axis {axis} (the other axis is already there)")
            await self._client.move_axis_tenths(axis, target_tenths[index])
        else:
            await self._client.move_xy_tenths(*target_tenths)
        logger.info("move_xy command sent")
        return await self._wait_for_target(target_tenths, target_xy_um)

    def _record_commanded_move(self, target_xy_um: tuple[float, float]) -> None:
        """Remember the last commanded absolute target and its per-axis direction.

        Only called under the move lock, and only after the envelope check, so a
        rejected target never becomes the reference for the next reversal.
        """
        previous = self._last_commanded_xy_um
        reference = self._origin_xy_um if previous is None else previous
        if reference is None:
            return
        directions: list[int] = []
        for target, last in zip(target_xy_um, reference, strict=True):
            directions.append(0 if target == last else (1 if target > last else -1))
        self._last_commanded_xy_um = target_xy_um
        self._last_direction_xy = (directions[0], directions[1])

    def _absolute_takeup_approach(
        self, target_xy_um: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Return the reverse-direction approach point, or None for a single move.

        The approach steps one take-up behind the target on every axis whose
        direction reverses, so the final move arrives from the same side it would
        have arrived from without the play. The point is clamped to the
        startup-relative envelope, because the approach must never leave it.
        """
        takeup_um = self._absolute_takeup_um
        previous = self._last_commanded_xy_um
        origin = self._origin_xy_um
        if takeup_um <= 0.0 or previous is None or origin is None:
            return None
        approach = list(target_xy_um)
        offset_used = False
        for index, axis in enumerate(("x", "y")):
            step = target_xy_um[index] - previous[index]
            if step == 0.0:
                continue
            last_direction = self._last_direction_xy[index]
            direction = 1 if step > 0 else -1
            if last_direction == 0 or direction == last_direction:
                continue
            if abs(step) <= takeup_um:
                continue
            point = target_xy_um[index] - math.copysign(takeup_um, step)
            low = origin[index] - self._max_xy_offset_um
            high = origin[index] + self._max_xy_offset_um
            clamped = min(max(point, low), high) if self.travel_bounds_enabled else point
            if clamped != point:
                logger.warning(
                    f"absolute take-up for {axis} towards "
                    f"{target_xy_um[index]:.3f} um would approach to {point:.3f} um, "
                    f"outside the startup-relative envelope; clamped to "
                    f"{clamped:.3f} um"
                )
            if clamped == target_xy_um[index]:
                continue
            approach[index] = clamped
            offset_used = True
        return (approach[0], approach[1]) if offset_used else None

    async def _wait_for_target(
        self,
        target_tenths: tuple[int, int],
        target_xy_um: tuple[float, float],
    ) -> tuple[float, float]:
        # Compare the landing against the *quantised* target: the controller can
        # only reach whole counts, and the requested micrometre value may sit up
        # to half a count (0.1 um) away, which was enough to push the residual
        # just over the tolerance and hang until the move timeout.
        quantized_xy_um = (
            target_tenths[0] / self._counts_per_um,
            target_tenths[1] / self._counts_per_um,
        )
        deadline = self._clock() + self._move_timeout_s
        polls = 0
        # The position the move started from. Measured 2026-10-30 after a controller reboot: a
        # 133 um move went out about 58 um and came back to where it started while a 20 um move on
        # the same stage completed, so "returned to its start" is a signature worth naming.
        start_tenths: tuple[float, float] | None = None
        while True:
            busy = await self._client.is_busy_xy()
            polls += 1
            if polls == 1:
                start_tenths = await self._client.read_xy_tenths()
            if polls <= 3 or polls % 50 == 0:
                counts = await self._client.read_xy_tenths()
                logger.info(
                    f"wait_for_target poll {polls} busy={busy} counts={counts} "
                    f"target={target_tenths}"
                )
            if self._halt_requested.is_set() and not busy:
                halted_tenths = await self._client.read_xy_tenths()
                halted_xy_um = (
                    tenths_to_um(halted_tenths[0], self._counts_per_um),
                    tenths_to_um(halted_tenths[1], self._counts_per_um),
                )
                if all(
                    abs(actual - requested) <= POSITION_TOLERANCE_UM
                    for actual, requested in zip(
                        halted_xy_um, quantized_xy_um, strict=True
                    )
                ):
                    return halted_xy_um
                raise MoveHalted(
                    halted_xy_um,
                    "ASI XY move was halted before its target: "
                    f"counts={halted_tenths} target={target_tenths} um={halted_xy_um}",
                )
            if busy and self.position_observer is not None:
                started_ns = int(self._clock() * 1e9)
                tenths = await self._client.read_xy_tenths()
                self._last_observed_tenths = tenths
                self.position_observer(tenths, started_ns, int(self._clock() * 1e9))
            remaining = deadline - self._clock()
            if remaining <= 0:
                last_tenths = await self._client.read_xy_tenths()
                last_busy = await self._client.is_busy_xy()
                returned = (
                    start_tenths is not None
                    and abs(last_tenths[0] - start_tenths[0]) <= 20
                    and abs(last_tenths[1] - start_tenths[1]) <= 20
                )
                raise TimeoutError(
                    "ASI XY move did not reach a settled target before the "
                    f"deadline; outcome unknown (busy={last_busy}, "
                    f"counts={last_tenths}, target={target_tenths}, "
                    f"quantized={quantized_xy_um}, "
                    # The direct read and the last observed telemetry are both reported: on
                    # 2026-10-30 a move timed out with the direct read 78 um short while the
                    # telemetry had tracked the stage to within 9 um, and one number alone
                    # could not say which source was stale.
                    f"last_observed={self._last_observed_tenths}, "
                    f"start={start_tenths})"
                    + (
                        "; the stage is back where it started, so the controller aborted the "
                        "move - check its soft limits and travel range"
                        if returned
                        else ""
                    )
                )
            if not busy:
                actual_tenths = await self._client.read_xy_tenths()
                actual_xy_um = (
                    tenths_to_um(actual_tenths[0], self._counts_per_um),
                    tenths_to_um(actual_tenths[1], self._counts_per_um),
                )
                if all(
                    abs(actual - requested) <= POSITION_TOLERANCE_UM
                    for actual, requested in zip(
                        actual_xy_um, quantized_xy_um, strict=True
                    )
                ):
                    return actual_xy_um
            # Some controllers report idle before the position query settles.
            # Continue observing the same move; never reissue it.
            await self._sleep(min(self._poll_period_s, remaining))

    def _bounded_target_counts(self, target_xy_um: tuple[float, float]) -> tuple[int, int]:
        self._require_in_envelope(target_xy_um)
        x, y = target_xy_um
        counts = (
            quantize_tenths(x, self._counts_per_um),
            quantize_tenths(y, self._counts_per_um),
        )
        self._require_in_envelope((
            tenths_to_um(counts[0], self._counts_per_um),
            tenths_to_um(counts[1], self._counts_per_um),
        ))
        return counts

    def _require_in_envelope(self, target_xy_um: tuple[float, float]) -> None:
        if self._origin_xy_um is None:
            raise RuntimeError("call connect() before requesting stage motion")
        if self.travel_bounds_enabled and any(
            abs(target - origin) > self._max_xy_offset_um
            for target, origin in zip(target_xy_um, self._origin_xy_um, strict=True)
        ):
            raise ValueError(
                "XY target exceeds the startup-relative envelope of "
                f"+/-{self._max_xy_offset_um:g} um"
            )

    async def _restore_profile(self) -> None:
        speeds = self._original_speeds_mm_s
        if speeds is not None:
            for axis, key in (("X", "x"), ("Y", "y")):
                try:
                    await self._client.set_speed_mm_s(axis, speeds[key])
                except Exception:
                    logger.exception("Could not restore ASI %s run speed", axis)
        ramps = self._original_ramps_ms
        if ramps is not None:
            for axis, key in (("X", "x"), ("Y", "y")):
                try:
                    await self._client.set_ramp_ms(axis, ramps[key])
                except Exception:
                    logger.exception("Could not restore ASI %s ramp time", axis)


def _require_xy(value: tuple[float, float], label: str) -> None:
    if len(value) != 2 or any(
        isinstance(coordinate, bool)
        or not isinstance(coordinate, (int, float))
        or not math.isfinite(coordinate)
        for coordinate in value
    ):
        raise ValueError(f"{label} must contain two finite coordinates")


__all__ = [
    "POSITION_TOLERANCE_UM",
    "AsiXYStage",
    "quantize_tenths",
    "tenths_to_um",
]
