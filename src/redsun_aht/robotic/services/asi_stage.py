"""Serve the native serial ASI stage over PVAccess.

The service exposes the same public stage/motion PV shape as the native Ludl
services so the Redsun planner bindings and the shared client work unchanged.
It uses the donor-verified ``2H`` ASCII dialect at 115200 8N1 and reports
positions in micrometres. Optional Z uses the pyhololab donor's separate
``1H`` route over the same serial session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from fastcs.attributes import AttributeIO, AttrR, AttrRW, AttrW
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, String
from fastcs.logging import logger
from redsun.services import configure_logging

from redsun_aht.robotic.device.asi._client import TENTHS_PER_UM, AsiXYClient
from redsun_aht.robotic.device.asi._stage import AsiXYStage, tenths_to_um
from redsun_aht.robotic.device.ludl._transport import OxiSerialSession
from redsun_aht.robotic.services._process import (
    controller_id,
    identity_arguments,
    run,
    serve,
)
from redsun_aht.robotic.services._stage_telemetry import StageTelemetryIO
from redsun_aht.robotic.services.ludl_serial_stage import (
    DEFAULT_POLL_PERIOD_MS,
    LIVE_TELEMETRY_PERIOD,
    MAX_POLL_PERIOD_MS,
    MIN_POLL_PERIOD_MS,
    STAGE_GROUP,
    AxisRef,
    LudlAxisController,
    LudlMotionStateIO,
    LudlRelativeMoveIO,
    LudlSerialStageIO,
    LudlStopIO,
    LudlTraverseIO,
    MotionProfileRef,
    MotionStateRef,
    RelativeMoveRef,
    StopRef,
    TelemetryRef,
    TraverseRef,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

ASI_TELEMETRY_SCHEMA = "redsun.asi-stage-telemetry/v1"
ASI_PROFILE_SCHEMA = "redsun.asi-motion-profile/v1"
ASI_SET_SPEED_SCHEMA = "redsun.asi-set-speed/v1"
ASI_SET_BACKLASH_SCHEMA = "redsun.asi-set-backlash/v1"


class AsiClient(AsiXYClient):
    """Add the LX4000 Z route to the existing serial client."""

    async def read_z_counts(self) -> float:
        reply = await self._query("1HW Z")
        if reply.startswith(":A"):
            value = float(reply[2:].strip())
            if math.isfinite(value):
                return value
        raise ValueError(f"unexpected ASI Z position reply: {reply!r}")

    async def move_z_counts(self, target: int) -> None:
        reply = await self._query(f"1HM Z= {target:d}")
        if reply != ":A":
            raise RuntimeError(f"unexpected ASI Z move reply: {reply!r}")

    async def is_busy_z(self) -> bool:
        reply = await self._query("1H/")
        if reply == "B":
            return True
        if reply == "N":
            return False
        raise ValueError(f"unexpected ASI Z status reply: {reply!r}")

    async def halt_z(self) -> None:
        reply = await self._query("1HHALT")
        if reply != ":A":
            raise RuntimeError(f"unexpected ASI Z halt reply: {reply!r}")


class AsiAxisIO(LudlSerialStageIO):
    """Expose an opt-in bounded Z axis alongside the inherited XY axis I/O."""

    def __init__(
        self,
        stage: AsiXYStage,
        client: AsiClient,
        *,
        z_counts_per_um: float,
        max_z_offset_um: float | None,
        poll_period_s: float,
    ) -> None:
        super().__init__(stage)
        self._client = client
        self._z_counts_per_um = z_counts_per_um
        self._max_z_offset_um = max_z_offset_um
        self._poll_period_s = poll_period_s
        self._z_origin_um: float | None = None
        self._z_command_lock = asyncio.Lock()
        self._z_stop_generation = 0
        self._closing = False
        self._z_faulted = False

    async def capture_z_origin(self) -> None:
        if self._max_z_offset_um is not None:
            if await self._client.is_busy_z():
                raise RuntimeError("ASI Z must be idle before capturing the service origin")
            self._z_origin_um = await self._read_z_um()

    async def _read_z_um(self) -> float:
        value = await self._client.read_z_counts() / self._z_counts_per_um
        if not math.isfinite(value):
            raise ValueError("ASI Z readback cannot be represented in micrometres")
        return value

    def interrupt_z(self) -> None:
        self._z_stop_generation += 1

    def begin_shutdown(self) -> None:
        self._closing = True
        self.interrupt_z()

    async def wait_for_axis_io(self) -> None:
        async with self._axis_lock:
            pass

    async def halt_z(self) -> None:
        if self._z_origin_um is not None:
            async with self._z_command_lock:
                await self._client.halt_z()

    async def send(self, attr: AttrW[float, AxisRef], value: float) -> None:
        if self._closing:
            raise RuntimeError("ASI stage service is shutting down")
        if attr.io_ref.axis != "z":
            await super().send(attr, value)
            return
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("Z target must be finite")
        origin, limit = self._z_origin_um, self._max_z_offset_um
        if origin is None or limit is None:
            raise RuntimeError("Z requires a configured travel envelope and startup readback")
        if abs(value - origin) > limit:
            raise ValueError("Z target exceeds the startup-relative travel envelope")
        raw_target = value * self._z_counts_per_um
        if not math.isfinite(raw_target):
            raise ValueError("Z target cannot be represented in controller counts")
        target = int(raw_target)
        # The legacy ASIXYStage skips absolute counts whose last digit is 3.
        if abs(target) % 10 == 3:
            target += 1
        if abs(target / self._z_counts_per_um - origin) > limit:
            raise ValueError("Z target exceeds the startup-relative travel envelope")

        generation = self._z_stop_generation
        async with self._axis_lock:
            async with self._z_command_lock:
                if self._closing:
                    raise RuntimeError("ASI stage service is shutting down")
                if generation != self._z_stop_generation:
                    raise RuntimeError("Z move cancelled by stop before submission")
                if self.busy:
                    raise RuntimeError("stage motion is already active")
                self.busy = True
                self.outcome = "moving"
                try:
                    if await self._client.is_busy_z():
                        raise RuntimeError("ASI Z is already moving")
                    await self._client.move_z_counts(target)
                except BaseException:
                    await self._latch_z_fault()
                    raise
            try:
                deadline = time.monotonic() + 60.0
                while True:
                    if generation != self._z_stop_generation:
                        raise RuntimeError("ASI Z move interrupted by stop")
                    if not await self._client.is_busy_z():
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("ASI Z did not become idle")
                    await asyncio.sleep(self._poll_period_s)
                actual = await self._read_z_um()
                if generation != self._z_stop_generation:
                    raise RuntimeError("ASI Z move interrupted by stop")
                if abs(actual - target / self._z_counts_per_um) > 1 / self._z_counts_per_um:
                    raise RuntimeError("ASI Z stopped outside one count of its target")
                if isinstance(attr, AttrR):
                    await attr.update(actual)
                self.outcome = "completed"
            except BaseException:
                await self._latch_z_fault()
                raise
            finally:
                self.busy = self._z_faulted

    async def _latch_z_fault(self) -> None:
        self._z_faulted = True
        self.busy = True
        self.outcome = "unknown"
        # The caller holds axis I/O and may still hold the Z submission lock.
        try:
            await self._client.halt_z()
        except Exception:
            logger.exception("ASI Z halt failed after a move fault")

    async def update(self, attr: AttrR[float, AxisRef]) -> None:
        if self._closing:
            return
        if attr.io_ref.axis != "z":
            await super().update(attr)
            return
        async with self._axis_lock:
            await attr.update(await self._read_z_um())


class AsiStopIO(LudlStopIO):
    """Route the service stop to both enabled controller routes."""

    def __init__(self, stage: AsiXYStage, axis_io: AsiAxisIO) -> None:
        super().__init__(stage, axis_io)
        self._asi_axis_io = axis_io

    async def send(self, attr: AttrW[bool, StopRef], value: bool) -> None:
        if not value:
            return
        self._asi_axis_io.interrupt_z()
        try:
            await super().send(attr, value)
        finally:
            await self._asi_axis_io.halt_z()
            if self._asi_axis_io._z_faulted:
                self._asi_axis_io.outcome = "unknown"


class AsiProfileIO(AttributeIO[str, MotionProfileRef]):
    """Read and explicitly bound the ASI run-speed profile."""

    def __init__(
        self,
        client: AsiXYClient,
        axis_io: LudlSerialStageIO,
        *,
        maximum_speed_mm_s: float,
    ) -> None:
        super().__init__()
        if (
            isinstance(maximum_speed_mm_s, bool)
            or not isinstance(maximum_speed_mm_s, (int, float))
            or not math.isfinite(maximum_speed_mm_s)
            or maximum_speed_mm_s <= 0
        ):
            raise ValueError("maximum speed must be finite and positive")
        self._client = client
        self._axis_io = axis_io
        self._maximum_speed_mm_s = float(maximum_speed_mm_s)

    async def update(self, attr: AttrR[str, MotionProfileRef]) -> None:
        """Read both axis speeds, ramp times and backlash values."""
        async with self._axis_io._axis_lock:
            profile = await self._read_profile()
        await attr.update(json.dumps(profile, separators=(",", ":")))

    async def send(self, attr: AttrW[str, MotionProfileRef], value: str) -> None:
        """Apply an explicit bounded run-speed request without touching ramps."""
        try:
            request = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid ASI motion-profile request") from exc
        if not isinstance(request, dict) or request.get("schema") not in {
            ASI_SET_SPEED_SCHEMA,
            ASI_SET_BACKLASH_SCHEMA,
        }:
            raise ValueError("unsupported ASI motion-profile schema")
        axis = request.get("axis", "xy")
        if axis not in ("x", "y", "xy"):
            raise ValueError("axis must be 'x', 'y' or 'xy'")
        speed = request.get("speed_mm_s")
        backlash = request.get("backlash_mm")
        if request.get("schema") == ASI_SET_BACKLASH_SCHEMA:
            # The anti-backlash distance (2HB) is the controller's only play
            # compensation and it is worth changing between a survey and a manual
            # navigation: a survey step tolerates a large approach, a small jog does
            # not. It is written under the same axis lock and verified by readback, so
            # an unconfirmed write raises instead of leaving an unknown setting.
            if backlash is None:
                raise ValueError("a backlash request must carry backlash_mm")
            if speed is not None:
                raise ValueError("a backlash request must not carry speed_mm_s")
            if axis != "xy":
                # The controller's paired form writes both axes in one command and the
                # client has no single-axis setter, so accepting "x" or "y" here would
                # silently change the other axis as well.
                raise ValueError(
                    "a backlash request writes both axes; omit axis or set it to 'xy'"
                )
            if (
                isinstance(backlash, bool)
                or not isinstance(backlash, (int, float))
                or not math.isfinite(backlash)
                or backlash < 0
                or backlash > 1.0
            ):
                raise ValueError(
                    "backlash_mm must be finite, non-negative and at most 1 mm"
                )
        else:
            if (
                isinstance(speed, bool)
                or not isinstance(speed, (int, float))
                or not math.isfinite(speed)
                or speed <= 0
            ):
                raise ValueError("speed_mm_s must be finite and positive")
            if float(speed) > self._maximum_speed_mm_s:
                raise ValueError(
                    "requested speed exceeds the configured ASI service ceiling of "
                    f"{self._maximum_speed_mm_s:g} mm/s"
                )
            if backlash is not None:
                raise ValueError("a speed request must not carry backlash_mm")
        async with self._axis_io._axis_lock:
            if self._axis_io.busy:
                raise RuntimeError("cannot change the ASI profile while moving")
            if speed is not None:
                if axis in ("x", "xy"):
                    await self._client.set_speed_mm_s("X", float(speed))
                if axis in ("y", "xy"):
                    await self._client.set_speed_mm_s("Y", float(speed))
            if backlash is not None:
                await self._client.set_backlash_xy_mm(float(backlash), float(backlash))
            actual = await self._read_profile()
        if backlash is not None:
            for readback_axis in ("x", "y"):
                reported = float(actual[readback_axis]["backlash_mm"])
                if abs(reported - float(backlash)) > 1e-6:
                    raise ValueError(
                        f"anti-backlash write was not confirmed on {readback_axis}: "
                        f"controller reports {reported:.4f} mm, requested "
                        f"{float(backlash):.4f} mm"
                    )
        await attr.update(json.dumps(actual, separators=(",", ":")))

    async def _read_profile(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": ASI_PROFILE_SCHEMA,
            "units": {
                "speed_mm_s": "mm/s",
                "ramp_ms": "ms",
                "backlash_mm": "mm",
            },
            "maximum_speed_mm_s": self._maximum_speed_mm_s,
        }
        for axis in ("x", "y"):
            result[axis] = {
                "speed_mm_s": await self._client.read_speed_mm_s(axis),
                "ramp_ms": await self._client.read_ramp_ms(axis),
                "backlash_mm": await self._client.read_backlash_mm(axis),
            }
        return result


class AsiStageController(Controller):
    """Managed bounded XYZ service backed by one ASI serial client."""

    def __init__(
        self,
        stage: AsiXYStage,
        client: AsiClient,
        axes: Iterable[str] = ("x", "y"),
        *,
        poll_period_ms: int = DEFAULT_POLL_PERIOD_MS,
        maximum_speed_mm_s: float = 1.0,
        telemetry_period_s: float = LIVE_TELEMETRY_PERIOD,
        z_counts_per_um: float = TENTHS_PER_UM,
        max_z_offset_um: float | None = None,
    ) -> None:
        axes = tuple(axes)
        if not axes or len(set(axes)) != len(axes) or set(axes) - {"x", "y", "z"}:
            raise ValueError("select unique ASI axes from x, y and z")
        if ("z" in axes) != (max_z_offset_um is not None):
            raise ValueError("select Z and specify its maximum startup-relative travel together")
        if max_z_offset_um is not None and (
            isinstance(max_z_offset_um, bool)
            or not math.isfinite(max_z_offset_um)
            or max_z_offset_um <= 0
        ):
            raise ValueError("Z travel envelope must be finite and positive")
        if (
            isinstance(z_counts_per_um, bool)
            or not math.isfinite(z_counts_per_um)
            or z_counts_per_um <= 0
        ):
            raise ValueError("Z counts per micrometre must be finite and positive")
        if not isinstance(telemetry_period_s, (int, float)) or telemetry_period_s <= 0:
            raise ValueError("telemetry_period_s must be positive")
        if (
            isinstance(poll_period_ms, bool)
            or not isinstance(poll_period_ms, int)
            or not MIN_POLL_PERIOD_MS <= poll_period_ms <= MAX_POLL_PERIOD_MS
        ):
            raise ValueError(
                f"position polling period must be {MIN_POLL_PERIOD_MS}.."
                f"{MAX_POLL_PERIOD_MS} ms"
            )
        poll_period_s = poll_period_ms / 1000
        stage_any: Any = stage
        axis_io = AsiAxisIO(
            stage_any,
            client,
            z_counts_per_um=z_counts_per_um,
            max_z_offset_um=max_z_offset_um,
            poll_period_s=poll_period_s,
        )
        stop_io = AsiStopIO(stage_any, axis_io)
        state_io = LudlMotionStateIO(axis_io)
        traverse_io = LudlTraverseIO(stage_any, axis_io)
        relative_move_io = LudlRelativeMoveIO(stage.move_relative_xy, axis_io)
        profile_io = AsiProfileIO(
            client,
            axis_io,
            maximum_speed_mm_s=maximum_speed_mm_s,
        )
        telemetry_io = StageTelemetryIO()
        super().__init__(
            ios=[
                axis_io,
                stop_io,
                state_io,
                traverse_io,
                relative_move_io,
                profile_io,
            ]
        )
        self._stage = stage
        self._client = client
        self._axis_io = axis_io
        self._traverse_io = traverse_io
        self._relative_move_io = relative_move_io
        self._telemetry_io = telemetry_io
        stage_any.position_observer = self._publish_position
        self._poll_period_s = poll_period_s
        self._telemetry_task: asyncio.Task[None] | None = None

        group = Controller(ios=[axis_io])
        for axis in axes:
            group.add_sub_controller(
                axis, LudlAxisController(axis, axis_io, poll_period_s)
            )
        stage_controller = Controller(ios=[axis_io])
        stage_controller.add_sub_controller("axis", group)
        self.add_sub_controller(STAGE_GROUP, stage_controller)

        motion = Controller(
            ios=[
                stop_io,
                state_io,
                traverse_io,
                relative_move_io,
                profile_io,
                telemetry_io,
            ]
        )
        motion.stop = AttrW(
            Bool(), io_ref=StopRef(), description="Halt ASI XY and enabled Z"
        )
        motion.busy = AttrR(
            Bool(), io_ref=MotionStateRef(key="busy", update_period=poll_period_s)
        )
        motion.outcome = AttrR(
            String(), io_ref=MotionStateRef(key="outcome", update_period=poll_period_s)
        )
        motion.traverse = AttrW(
            String(),
            io_ref=TraverseRef(),
            description="Start a bounded absolute XY move in micrometres",
        )
        motion.move_relative = AttrW(
            String(),
            io_ref=RelativeMoveRef(),
            description=(
                "Start a bounded paired relative ASI XY move from a versioned "
                "JSON displacement in micrometres"
            ),
        )
        motion.profile = AttrRW(
            String(),
            io_ref=MotionProfileRef(update_period=5.0),
            description=(
                "ASI run speed (mm/s), ramp time (ms) and backlash (mm); speed "
                "writes require a service ceiling"
            ),
        )
        motion.telemetry_json = AttrR(
            String(),
            io_ref=TelemetryRef(update_period=float(telemetry_period_s)),
            description=(
                "Host-bracketed sampled ASI XY readbacks; not exposure-synchronized"
            ),
        )
        self.add_sub_controller("motion", motion)

    async def connect(self) -> None:
        """Handshake, capture the origin, start sampled telemetry."""
        try:
            origin = await self._stage.connect()
            await self._axis_io.capture_z_origin()
            if self._axis_io.busy:
                controller_busy = await self._stage.is_busy_xy()
                traverse_live = (
                    self._traverse_io._task is not None
                    and not self._traverse_io._task.done()
                )
                if not controller_busy and not traverse_live and not self._axis_io._z_faulted:
                    self._axis_io.busy = False
                    self._axis_io.outcome = "idle"
                    logger.warning(
                        "cleared a stale ASI axis-busy flag at connect "
                        "(controller idle, no live traverse)"
                    )
            self._telemetry_io.start_run(
                str(uuid4()),
                source="ASI XY stage telemetry",
                schema=ASI_TELEMETRY_SCHEMA,
                calibration_metadata=self._stage.calibration_metadata,
            )
            self._telemetry_io.publish(
                {
                    "type": "startup_position",
                    "host_monotonic_ns": time.monotonic_ns(),
                    "x_um": origin[0],
                    "y_um": origin[1],
                }
            )
            self._telemetry_task = asyncio.create_task(
                self._sample_positions(), name="asi-position-sampler"
            )
            await super().connect()
        except BaseException:
            if self._telemetry_task is not None:
                self._telemetry_task.cancel()
                await asyncio.gather(self._telemetry_task, return_exceptions=True)
            await self._stage.close()
            raise

    async def disconnect(self) -> None:
        """Stop motion, restore run speeds and release serial ownership."""
        try:
            self._axis_io.begin_shutdown()
            try:
                await self._stage.stop()
            finally:
                await self._axis_io.halt_z()
        finally:
            try:
                try:
                    await self._traverse_io.cancel_and_wait()
                finally:
                    await self._relative_move_io.cancel_and_wait()
            finally:
                if self._telemetry_task is not None:
                    self._telemetry_task.cancel()
                    await asyncio.gather(self._telemetry_task, return_exceptions=True)
                try:
                    await self._axis_io.wait_for_axis_io()
                    await self._stage.close()
                finally:
                    await super().disconnect()

    def _publish_position(
        self, tenths: tuple[int, int], started_ns: int, finished_ns: int
    ) -> None:
        """Publish the move loop's own position readback as telemetry."""
        for axis, raw in (("x", tenths[0]), ("y", tenths[1])):
            self._telemetry_io.publish(
                {
                    "type": "ascii_position_poll",
                    "axis": axis,
                    "receive_call_started_ns": started_ns,
                    "receive_call_finished_ns": finished_ns,
                    "position_um": tenths_to_um(
                        raw, getattr(self._stage, "_counts_per_um", TENTHS_PER_UM)
                    ),
                    "raw_position_tenths_um": raw,
                    "raw_position_semantics": (
                        "controller 2HW reply in tenths of a micrometre"
                    ),
                    "move_requested": True,
                }
            )

    async def _sample_positions(self) -> None:
        while True:
            started_ns = time.monotonic_ns()
            try:
                tenths, (x_um, y_um) = await self._stage.read_xy_with_tenths()
                finished_ns = time.monotonic_ns()
                took_ms = (finished_ns - started_ns) / 1e6
                if took_ms > 3 * self._poll_period_s * 1000.0:
                    logger.info(f"slow telemetry poll took {took_ms:.1f} ms")
                for axis, position_um, raw in (
                    ("x", x_um, tenths[0]),
                    ("y", y_um, tenths[1]),
                ):
                    self._telemetry_io.publish(
                        {
                            "type": "ascii_position_poll",
                            "axis": axis,
                            "receive_call_started_ns": started_ns,
                            "receive_call_finished_ns": finished_ns,
                            "position_um": position_um,
                            "raw_position_tenths_um": raw,
                            "raw_position_semantics": (
                                "controller 2HW reply in tenths of a micrometre"
                            ),
                            "move_requested": self._axis_io.busy,
                        }
                    )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep transient telemetry errors local.
                logger.exception("ASI position poll failed")
            await asyncio.sleep(self._poll_period_s)


def build_controller(
    *,
    port: str,
    baudrate: int = 115200,
    axes: Iterable[str] = ("x", "y"),
    max_xy_offset_um: float = 2000.0,
    speed_mm_s: float | None = None,
    ramp_ms: float | None = None,
    backlash_mm: float | None = None,
    absolute_takeup_um: float = 0.0,
    poll_period_ms: int = DEFAULT_POLL_PERIOD_MS,
    telemetry_period_s: float = LIVE_TELEMETRY_PERIOD,
    counts_per_um: float = TENTHS_PER_UM,
    z_counts_per_um: float = TENTHS_PER_UM,
    max_z_offset_um: float | None = None,
) -> AsiStageController:
    """Open a sole-owned ASI session with explicit bounds and optional profile."""
    if not port:
        raise ValueError("serial port is required")
    if (
        isinstance(max_xy_offset_um, bool)
        or not math.isfinite(max_xy_offset_um)
        or not 0 < max_xy_offset_um <= 2000.0
    ):
        raise ValueError("XY envelope must be positive and no greater than 2000 um")
    if (
        isinstance(absolute_takeup_um, bool)
        or not isinstance(absolute_takeup_um, (int, float))
        or not math.isfinite(absolute_takeup_um)
        or absolute_takeup_um < 0
    ):
        raise ValueError("absolute take-up must be finite and non-negative")
    session = OxiSerialSession.open(
        port,
        baudrate=baudrate,
        # A moving controller can take a long time to answer a status or position query, and a
        # timed-out exchange is indistinguishable from a stage that did not move. Measured
        # 2026-10-30: the same absolute move that timed out through this service completed on the
        # same port with a 2 s session timeout (a +500 unit move landed within 3 units), so the
        # 0.5 s default was the failure, not the motion. 10 s leaves room for a long jog, which is
        # the common case on this stage.
        timeout=10.0,
        stopbits=1,
        parity="N",
    )
    client = AsiClient(session)
    try:
        stage = AsiXYStage(
            client,
            max_xy_offset_um=max_xy_offset_um,
            move_timeout_s=60.0,
            poll_period_s=poll_period_ms / 1000,
            startup_speed_mm_s=speed_mm_s,
            startup_ramp_ms=ramp_ms,
            startup_backlash_mm=backlash_mm,
            absolute_takeup_um=absolute_takeup_um,
            counts_per_um=counts_per_um,
        )
        # The configured run speed is the ceiling unless the caller explicitly
        # starts the service faster; a remote profile write may not raise it.
        maximum_speed = 1.0 if speed_mm_s is None else min(1.0, speed_mm_s)
        return AsiStageController(
            stage,
            client,
            axes,
            poll_period_ms=poll_period_ms,
            maximum_speed_mm_s=maximum_speed,
            telemetry_period_s=telemetry_period_s,
            z_counts_per_um=z_counts_per_um,
            max_z_offset_um=max_z_offset_um,
        )
    except BaseException:
        client.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="sole-owned ASI serial port")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--poll-period-ms", type=int, default=DEFAULT_POLL_PERIOD_MS)
    parser.add_argument(
        "--telemetry-period-s", type=float, default=LIVE_TELEMETRY_PERIOD
    )
    parser.add_argument("--counts-per-um", type=float, default=TENTHS_PER_UM)
    parser.add_argument("--axes", default="x,y")
    parser.add_argument("--z-counts-per-um", type=float, default=TENTHS_PER_UM)
    parser.add_argument("--max-z-offset-um", type=float)
    parser.add_argument("--max-xy-offset-um", type=float, default=2000.0)
    parser.add_argument(
        "--speed-mm-s",
        type=float,
        help="run speed applied on connect and restored on disconnect",
    )
    parser.add_argument(
        "--ramp-ms",
        type=float,
        help="acceleration/ramp time applied on connect and restored on disconnect",
    )
    parser.add_argument(
        "--backlash-mm",
        type=float,
        help=(
            "anti-backlash approach distance applied to both axes on connect "
            "and restored on disconnect (0-1 mm)"
        ),
    )
    parser.add_argument(
        "--absolute-takeup-um",
        type=float,
        default=0.0,
        help=(
            "opt-in reverse-direction take-up for absolute XY moves in um; a "
            "reversal approaches from this far behind the target first "
            "(0 disables the extra move)"
        ),
    )
    identity_arguments(parser, "asi-serial-stage")
    options = parser.parse_args(argv)
    configure_logging()
    axes = [axis.strip() for axis in options.axes.split(",") if axis.strip()]
    controller = build_controller(
        port=options.port,
        baudrate=options.baudrate,
        axes=axes,
        max_xy_offset_um=options.max_xy_offset_um,
        speed_mm_s=options.speed_mm_s,
        ramp_ms=options.ramp_ms,
        backlash_mm=options.backlash_mm,
        absolute_takeup_um=options.absolute_takeup_um,
        poll_period_ms=options.poll_period_ms,
        telemetry_period_s=options.telemetry_period_s,
        counts_per_um=options.counts_per_um,
        z_counts_per_um=options.z_counts_per_um,
        max_z_offset_um=options.max_z_offset_um,
    )
    return run(serve(controller, controller_id(options)))


if __name__ == "__main__":
    sys.exit(main())
