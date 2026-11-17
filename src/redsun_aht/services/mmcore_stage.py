"""One Micro-Manager stage, served over PVAccess.

Ported, with adaptation, from
``redsun_searching_is_imaging/services/mmcore_stage.py`` at donor revision
``3a3bf06`` (``redsun-searching-is-imaging``). The process owns its own
``CMMCorePlus``, so no Micro-Manager call runs on the session's event loop.

Run as ``python -m redsun_aht.services.mmcore_stage --prefix AHT:XY: --axes x,y
--mm-config ix83.cfg --device XY``, or, without a configuration,
``--adapter DemoCamera --device DXYStage --axes x,y``.

Changed for this bundle:

- The service is served with ``redsun.services.serve_controller``, which
  Redsun 0.16 provides, so the donor's private ``_process`` module is not
  ported. Readiness and the stop request are the session's, as before.
- A Micro-Manager system configuration can be loaded (``--mm-config``) and the
  installation chosen (``--mm-path``), so a configured instrument is served
  without naming an adapter and device.
- Every axis publishes the travel limits it enforces, from ``--limits``, and
  refuses a move outside them. The donor had no limits in the service; the
  envelope was a single startup-relative offset for both lateral axes.
- The Ludl-specific ``LudlStageIO`` and its composed adapter stack are
  dropped. The abstraction they sat beside is kept and generalised: the motion
  profile is read and written as raw MMCore properties of whichever adapter
  the stage uses, named by ``--motion-properties``.
- A stage-wide and a per-axis ``StopRequest`` reach ``CMMCorePlus.stop``, so a
  client can stop motion through the service instead of only through the
  session.
- Position telemetry is the same bounded JSON batch, with the Ludl schema and
  calibration identifiers replaced by this bundle's own.
- Every axis publishes the finest step it can be trusted to make, taken from
  the adapter's own step-size property when it names one and from
  :data:`DEFAULT_PRECISION_UM` when it does not. The attribute's description
  says which applied, and the device turns it into a
  ``redsun_aht.contracts.AxisSpec``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import uuid4

import numpy as np
from fastcs.attributes import AttributeIO, AttributeIORef, AttrR, AttrRW, AttrW
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, Float, String, Waveform
from fastcs.logging import logger
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.services import configure_logging, identity, serve_controller

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterable, Mapping

    from pymmcore_plus import CMMCorePlus

#: Seconds between two reads of a position.
POLL_PERIOD: Final = 0.2

#: What a position is served in.
UNITS: Final = "um"

#: The axes a lateral stage moves. The rest go through ``setPosition``.
LATERAL: Final = ("x", "y")

#: Where the axes sit under the service's prefix.
STAGE_GROUP: Final = "stage"

#: Where the motion profile and the position telemetry sit under the prefix.
MOTION_GROUP: Final = "motion"

#: Raw MMCore adapter properties read as the stage's motion profile. The names
#: are the ones the Micro-Manager Ludl and ASI adapters use; an adapter that
#: exposes none of them reports them as unsupported.
MOTION_PROPERTIES: Final = ("Speed", "StartSpeed", "Acceleration")

#: The motion properties a raw-setting ceiling bounds.
RATE_PROPERTIES: Final = ("Speed", "StartSpeed")

#: Raw MMCore adapter properties a stage may publish its finest reliable step
#: in, in micrometres. The first one the device exposes wins.
PRECISION_PROPERTIES: Final = ("StepSizeUm", "Precision", "Resolution")

#: The precision an axis is published with when its adapter names no step size.
DEFAULT_PRECISION_UM: Final = 0.001

MIN_TELEMETRY_PERIOD_MS: Final = 20
MAX_TELEMETRY_PERIOD_MS: Final = 5000

#: Seconds of samples a telemetry batch may hold before it drops the oldest.
TELEMETRY_CAPACITY: Final = 512

#: Schema of the motion-profile readback.
MOTION_PROFILE_SCHEMA: Final = "redsun.mmcore-stage-motion-profile/v1"

#: Schema of a batch of sampled stage positions.
TELEMETRY_SCHEMA: Final = "redsun.mmcore-stage-telemetry/v1"

#: Clock the telemetry samples are stamped with.
TELEMETRY_CLOCK: Final = "python.time.monotonic_ns"

MMCORE_TELEMETRY_TIMESTAMP_SEMANTICS: Final = (
    "host monotonic bracket around MMCore getXYPosition; not encoder or camera time"
)


@dataclass
class AxisRef(AttributeIORef):  # type: ignore[misc]
    """Names the axis an attribute reads and moves."""

    axis: str = ""


@dataclass
class TelemetryRef(AttributeIORef):  # type: ignore[misc]
    """Selects the bounded stage-position batch signal."""


class StageTelemetryIO(AttributeIO[str, TelemetryRef]):  # type: ignore[misc]
    """Publish sampled stage observations over a bounded JSON PV.

    Samples are queued by the sampler and published by the attribute's own
    poll, so a slow client never blocks motion: the queue is bounded and
    reports the frames it dropped instead.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pending: deque[dict[str, Any]] = deque()
        self._sequence = 0
        self._overflow = False
        self._lock = asyncio.Lock()
        self._run_id: str | None = None
        self._source = "Micro-Manager XYStage"
        self._timestamp_semantics = MMCORE_TELEMETRY_TIMESTAMP_SEMANTICS

    def start_run(self, run_id: str, *, source: str, timestamp_semantics: str) -> None:
        """Set the provenance every sample of this connection carries."""
        self._pending.clear()
        self._sequence = 0
        self._overflow = False
        self._run_id = run_id
        self._source = source
        self._timestamp_semantics = timestamp_semantics

    def publish(self, record: dict[str, Any]) -> None:
        """Queue one observation without blocking the stage communication path."""
        if len(self._pending) >= TELEMETRY_CAPACITY:
            self._overflow = True
            return
        self._sequence += 1
        self._pending.append({"sample_id": self._sequence, **record})

    async def update(self, attr: AttrR[str, TelemetryRef]) -> None:
        """Publish queued observations, and retain them if the update fails."""
        async with self._lock:
            if not self._pending:
                return
            count = len(self._pending)
            payload = json.dumps(
                {
                    "schema": TELEMETRY_SCHEMA,
                    "run_id": self._run_id,
                    "source": self._source,
                    "clock": TELEMETRY_CLOCK,
                    "timestamp_semantics": self._timestamp_semantics,
                    "overflow": self._overflow,
                    "published_records": self._sequence,
                    "records": list(self._pending),
                },
                separators=(",", ":"),
            )
            await attr.update(payload)
            for _ in range(count):
                self._pending.popleft()


class MotionProfileIO(AttributeIO[str, AttributeIORef]):  # type: ignore[misc]
    """Access a stage adapter's raw motion profile properties.

    The profile is the adapter's own settings, in its own units, not calibrated
    physical quantities. A write is refused unless the service was started with
    an explicit ceiling, and never while the stage is moving.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        stage_io: StageIO,
        properties: Iterable[str],
        max_rate_setting: int | None,
    ) -> None:
        super().__init__()
        if max_rate_setting is not None and max_rate_setting <= 0:
            raise ValueError("maximum raw motion-rate setting must be positive")
        self._core = core
        self._label = label
        self._stage_io = stage_io
        self._properties = tuple(properties)
        self._max_rate_setting = max_rate_setting

    async def update(self, attr: AttrR[str, AttributeIORef]) -> None:
        """Publish the stage's current motion state."""
        async with self._stage_io.move_lock:
            value = await asyncio.to_thread(self._read_profile)
        await attr.update(value)

    async def send(self, attr: AttrW[str, AttributeIORef], value: str) -> None:
        """Apply a requested motion-profile write.

        Raises
        ------
        ValueError
            For a request that is not ``{"property": ..., "value": ...}`` with
            a numeric value in range for the property.
        RuntimeError
            When writes are disabled or the stage is moving.
        """
        try:
            request = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("motion profile request must be JSON") from exc
        if not isinstance(request, dict) or set(request) != {"property", "value"}:
            raise ValueError("request must contain exactly property and value")
        prop = request["property"]
        if prop not in self._properties:
            raise ValueError(f"unsupported motion property: {prop!r}")
        raw_value = request["value"]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError("motion property value must be numeric")
        number = float(raw_value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("motion property value must be finite and positive")
        if not number.is_integer():
            raise ValueError("motion property values must be integers")
        if prop == "Acceleration" and not 1 <= number <= 255:
            raise ValueError("Acceleration must be an integer in 1..255")
        if self._max_rate_setting is None:
            raise RuntimeError(
                "motion-profile writes are disabled; configure an explicit "
                "maximum raw motion-rate setting"
            )
        if prop in RATE_PROPERTIES and number > self._max_rate_setting:
            raise ValueError("motion rate exceeds the configured raw-setting ceiling")

        async with self._stage_io.move_lock:
            await asyncio.to_thread(self._write_property, prop, number)
        await self.update(cast("AttrR[str, AttributeIORef]", attr))

    def _read_profile(self) -> str:
        names = set(self._core.getDevicePropertyNames(self._label))
        properties: dict[str, str] = {}
        for prop in self._properties:
            if prop in names:
                properties[prop] = str(self._core.getProperty(self._label, prop))
        return json.dumps(
            {
                "schema": MOTION_PROFILE_SCHEMA,
                "device": self._label,
                "units": "raw MMCore adapter settings; not calibrated physical units",
                "properties": properties,
                "unsupported": [prop for prop in self._properties if prop not in names],
                "busy": bool(self._core.deviceBusy(self._label)),
                "writes_enabled": self._max_rate_setting is not None,
                "maximum_raw_rate_setting": self._max_rate_setting,
            },
            separators=(",", ":"),
        )

    def _write_property(self, prop: str, value: float) -> None:
        if self._core.deviceBusy(self._label):
            raise RuntimeError("cannot change the motion profile while stage is moving")
        names = set(self._core.getDevicePropertyNames(self._label))
        if prop not in names:
            raise ValueError(f"stage adapter does not expose {prop!r}")
        if self._core.isPropertyReadOnly(self._label, prop):
            raise ValueError(f"stage adapter property {prop!r} is read-only")
        previous = str(self._core.getProperty(self._label, prop))
        requested = str(int(value) if value.is_integer() else value)
        self._core.setProperty(self._label, prop, requested)
        actual = str(self._core.getProperty(self._label, prop))
        try:
            actual_number = float(actual)
        except ValueError as exc:
            raise RuntimeError(
                f"MMCore returned a nonnumeric {prop} readback: {actual!r}"
            ) from exc
        if not math.isfinite(actual_number) or actual_number <= 0:
            raise RuntimeError(
                f"MMCore returned an invalid {prop} readback: {actual!r}"
            )
        if (
            prop in RATE_PROPERTIES
            and self._max_rate_setting is not None
            and actual_number > self._max_rate_setting
        ):
            self._core.setProperty(self._label, prop, previous)
            restored = str(self._core.getProperty(self._label, prop))
            if not math.isclose(
                float(restored), float(previous), rel_tol=1e-6, abs_tol=1e-6
            ):
                raise RuntimeError(
                    f"MMCore normalized {prop} above the configured ceiling and "
                    f"failed to restore prior value {previous}; current readback "
                    f"is {restored}"
                )
            raise RuntimeError(
                f"MMCore normalized {prop} above the configured raw-setting "
                f"ceiling: requested {requested}, got {actual}; prior value "
                f"{previous} was restored (ceiling {self._max_rate_setting})"
            )


class AxisPositionIO(AttributeIO[float, AxisRef]):  # type: ignore[misc]
    """Backend interface for the shared PVA positioner groups."""

    def limits_for(self, axis: str) -> tuple[float, float]:
        """Return the enforced travel bounds for an axis."""
        raise NotImplementedError

    def precision_for(self, axis: str) -> tuple[float, str]:
        """Return the step size and its source for an axis."""
        raise NotImplementedError

    def stop(self) -> None:
        """Request a stage stop."""
        raise NotImplementedError


class StageIO(AxisPositionIO):
    """Reads and moves one stage through Micro-Manager.

    *limits* are the travel bounds the service enforces, in the units the
    stage reports; an axis without an entry is unbounded. They are refused
    before the device is asked to move, so a client that ignores the published
    limits still cannot drive the stage outside them.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        max_xy_offset_um: float | None = None,
        limits: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        if max_xy_offset_um is not None and (
            not math.isfinite(max_xy_offset_um) or max_xy_offset_um <= 0
        ):
            raise ValueError("max XY offset must be a finite positive number")
        for axis, (low, high) in (limits or {}).items():
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(
                    f"{axis} limits must be finite and increasing, not {low}..{high}"
                )
        self._core = core
        self._label = label
        self._move_lock = asyncio.Lock()
        self._move_active = False
        self._max_xy_offset_um = max_xy_offset_um
        self._limits = dict(limits or {})
        self._origin_xy = (
            tuple(map(float, core.getXYPosition(label)))
            if max_xy_offset_um is not None
            else None
        )

    @property
    def move_lock(self) -> asyncio.Lock:
        """The lock one move at a time is held under."""
        return self._move_lock

    def limits_for(self, axis: str) -> tuple[float, float]:
        """Return the travel bounds of *axis*, unbounded when none are declared."""
        return self._limits.get(axis, (-math.inf, math.inf))

    def precision_for(self, axis: str) -> tuple[float, str]:
        """Return the axis's finest reliable step in micrometres, and its source.

        The adapter's own step-size property wins when it exposes one, because
        that is the only figure a driver can be asked for. An adapter that
        names none falls back on :data:`DEFAULT_PRECISION_UM`, and the returned
        source says so, so the published attribute never claims a precision the
        hardware did not declare.
        """
        del axis  # one MMCore device serves every axis of the stage
        names = set(self._core.getDevicePropertyNames(self._label))
        for prop in PRECISION_PROPERTIES:
            if prop not in names:
                continue
            try:
                value = float(self._core.getProperty(self._label, prop))
            except ValueError:
                continue
            if math.isfinite(value) and value > 0:
                return (value, f"the adapter's {prop!r} property")
        return (
            DEFAULT_PRECISION_UM,
            "the bundled default; the adapter exposes no step size",
        )

    async def send(self, attr: AttrW[float, AxisRef], value: float) -> None:
        """Move the axis, and read back where it stopped.

        One move at a time: a lateral move reads and writes the pair, so a
        concurrent move on the other axis would put this one back where it
        started.
        """
        logger.debug(f"Moving {attr.io_ref.axis} to {value}")
        async with self._move_lock:
            self._move_active = True
            try:
                await asyncio.to_thread(self._move, attr.io_ref.axis, float(value))
            finally:
                self._move_active = False
        if isinstance(attr, AttrR):
            await self.update(attr)

    async def update(self, attr: AttrR[float, AxisRef]) -> None:
        """Read where the axis is."""
        await attr.update(await asyncio.to_thread(self._position, attr.io_ref.axis))

    def stop(self) -> None:
        """Stop whichever device this stage moves."""
        self._core.stop(self._label)

    def _position(self, axis: str) -> float:
        if axis not in LATERAL:
            return float(self._core.getPosition(self._label))
        x, y = self._core.getXYPosition(self._label)
        return float(x if axis == "x" else y)

    def _move(self, axis: str, value: float) -> None:
        """Move the axis and block until the device reports it arrived."""
        self._check_limits(axis, value)
        if axis not in LATERAL:
            self._core.setPosition(self._label, value)
        else:
            target = self._xy_target(axis, value)
            self._core.setXYPosition(
                self._label,
                *target,
            )
        self._core.waitForDevice(self._label)

    def _check_limits(self, axis: str, value: float) -> None:
        """Refuse a target outside the axis's declared travel bounds."""
        low, high = self.limits_for(axis)
        if not low <= value <= high:
            raise ValueError(
                f"{axis} target {value} lies outside the axis limits {low}..{high}"
            )

    def _xy_target(self, axis: str, value: float) -> tuple[float, float]:
        """Preserve the sibling axis and enforce the configured travel box."""
        x, y = self._core.getXYPosition(self._label)
        target = (value if axis == "x" else x, value if axis == "y" else y)
        if (
            self._max_xy_offset_um is not None
            and self._origin_xy is not None
            and any(
                abs(requested - origin) > self._max_xy_offset_um
                for requested, origin in zip(target, self._origin_xy, strict=True)
            )
        ):
            raise ValueError(
                "XY target exceeds the configured startup-position envelope "
                f"of +/-{self._max_xy_offset_um:g} um"
            )
        return float(target[0]), float(target[1])


class MMAxisController(Controller):  # type: ignore[misc]
    """One axis of the stage: its position, its limits and its stop request."""

    def __init__(self, axis: str, io: AxisPositionIO) -> None:
        super().__init__(ios=[io])
        self._io = io
        self._axis = axis
        low, high = io.limits_for(axis)
        self.position = AttrRW(
            Float(units=UNITS),
            io_ref=AxisRef(axis=axis, update_period=POLL_PERIOD),
        )
        self.limits = AttrR(
            Waveform(np.float64, shape=(2,)),
            initial_value=np.array([low, high], dtype=np.float64),
            description=(
                f"{axis} travel bounds in {UNITS}, as the service enforces them"
            ),
        )
        precision, source = io.precision_for(axis)
        if not math.isfinite(precision) or precision <= 0:
            precision = DEFAULT_PRECISION_UM
            source = "the bundled default; the adapter's step size is unusable"
        self.precision = AttrR(
            Float(units=UNITS),
            initial_value=precision,
            description=f"{axis} finest reliable step in {UNITS}, from {source}",
        )
        self.stop_request = AttrRW(
            Bool(),
            description="write True to stop the stage at the device",
        )
        self.stop_request.add_on_update_callback(self._on_stop_request, always=True)

    async def _on_stop_request(self, requested: bool) -> None:
        """Stop the stage once, then clear the request."""
        if not requested:
            return
        await asyncio.to_thread(self._io.stop)
        await self.stop_request.update(False)


class MMStageGroup(Controller):  # type: ignore[misc]
    """The stage itself: its named axes and its stage-wide stop request."""

    def __init__(
        self, io: AxisPositionIO, axes: Iterable[str], provenance: str = "unknown",
        *,
        axis_controller: Callable[[str, AxisPositionIO], MMAxisController]
        = MMAxisController,
    ) -> None:
        super().__init__(ios=[io])
        self._io = io
        if provenance not in ("simulated", "measured", "unknown"):
            raise ValueError("invalid stage readback provenance")
        self.provenance = AttrR(
            String(),
            initial_value=provenance,
            description="Declared readback source; unknown is not measured",
        )
        group = Controller(ios=[io])
        for axis in axes:
            group.add_sub_controller(axis, axis_controller(axis, io))
        self.add_sub_controller("axis", group)
        self.stop_request = AttrRW(
            Bool(),
            description="write True to stop every axis of the stage at the device",
        )
        self.stop_request.add_on_update_callback(self._on_stop_request, always=True)

    async def _on_stop_request(self, requested: bool) -> None:
        """Stop the stage once, then clear the request."""
        if not requested:
            return
        await asyncio.to_thread(self._io.stop)
        await self.stop_request.update(False)


class MMMotionController(Controller):  # type: ignore[misc]
    """Serve motion-profile controls and sampled stage positions."""

    def __init__(
        self,
        profile_io: MotionProfileIO,
        telemetry_io: StageTelemetryIO,
        telemetry_period_s: float | None,
    ) -> None:
        super().__init__(ios=[profile_io, telemetry_io])
        self.profile = AttrRW(
            String(),
            io_ref=AttributeIORef(update_period=1.0),
            description=(
                "Raw adapter profile JSON; write one property with "
                '{"property":"Speed","value":100}'
            ),
        )
        self.telemetry_json = AttrR(
            String(),
            io_ref=TelemetryRef(update_period=telemetry_period_s),
            description=(
                "Run-scoped host-bracketed XY readbacks; sampling is opt-in and "
                "not exposure-synchronized"
            ),
        )


class MMStageController(Controller):  # type: ignore[misc]
    """A stage, its axes served as the named entries of ``stage``'s ``axis``.

    *axis_limits* applies when this controller builds its own ``StageIO``; an
    *io* passed in carries the bounds it enforces itself.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        axes: Iterable[str],
        max_xy_offset_um: float | None = None,
        io: StageIO | None = None,
        max_motion_rate_setting: int | None = None,
        telemetry_period_ms: int | None = None,
        *,
        axis_limits: Mapping[str, tuple[float, float]] | None = None,
        motion_properties: Iterable[str] = MOTION_PROPERTIES,
        provenance: str = "unknown",
    ) -> None:
        if telemetry_period_ms is not None and not (
            MIN_TELEMETRY_PERIOD_MS <= telemetry_period_ms <= MAX_TELEMETRY_PERIOD_MS
        ):
            raise ValueError(
                "telemetry period must be in "
                f"{MIN_TELEMETRY_PERIOD_MS}..{MAX_TELEMETRY_PERIOD_MS} ms"
            )
        io = io or StageIO(core, label, max_xy_offset_um, axis_limits)
        super().__init__(ios=[io])
        self._core = core
        self._label = label
        self._stage_io = io
        self._telemetry_period_s = (
            telemetry_period_ms / 1000 if telemetry_period_ms is not None else None
        )
        self._telemetry_task: asyncio.Task[None] | None = None
        self._telemetry_io: StageTelemetryIO | None = None

        axes = tuple(axes)
        self.add_sub_controller(STAGE_GROUP, MMStageGroup(io, axes, provenance))
        properties = tuple(name for name in motion_properties if name)
        if properties or self._telemetry_period_s is not None:
            profile_io = MotionProfileIO(
                core, label, io, properties, max_rate_setting=max_motion_rate_setting
            )
            telemetry_io = StageTelemetryIO()
            self._telemetry_io = telemetry_io
            self.add_sub_controller(
                MOTION_GROUP,
                MMMotionController(profile_io, telemetry_io, self._telemetry_period_s),
            )

    async def connect(self) -> None:
        """Start opt-in position sampling after FastCS has connected its I/O."""
        await super().connect()
        if self._telemetry_period_s is None or self._telemetry_io is None:
            return
        self._telemetry_io.start_run(
            str(uuid4()),
            source="Micro-Manager XYStage",
            timestamp_semantics=MMCORE_TELEMETRY_TIMESTAMP_SEMANTICS,
        )
        self._telemetry_task = asyncio.create_task(
            self._sample_positions(), name="mmcore-position-sampler"
        )

    async def _sample_positions(self) -> None:
        """Publish paired XY readbacks with the host call's monotonic bracket."""
        assert self._telemetry_io is not None
        assert self._telemetry_period_s is not None
        while True:
            started_ns = time.monotonic_ns()
            try:
                move_active_before = self._stage_io._move_active
                read_task = asyncio.create_task(
                    asyncio.to_thread(self._core.getXYPosition, self._label)
                )
                try:
                    x_um, y_um = await asyncio.shield(read_task)
                except asyncio.CancelledError:
                    await asyncio.gather(read_task, return_exceptions=True)
                    raise
                finished_ns = time.monotonic_ns()
                move_active = move_active_before or self._stage_io._move_active
                for axis, device_number, position_um in (
                    ("x", 1, x_um),
                    ("y", 2, y_um),
                ):
                    self._telemetry_io.publish(
                        {
                            "type": "mmcore_xy_poll",
                            "axis": axis,
                            "device_number": device_number,
                            "read_call_started_ns": started_ns,
                            "read_call_finished_ns": finished_ns,
                            "position_um": float(position_um),
                            "move_requested": move_active,
                        }
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("MMCore XY telemetry poll failed")
            await asyncio.sleep(self._telemetry_period_s)

    async def disconnect(self) -> None:
        """Stop the stage and let go of the device."""
        if self._telemetry_task is not None:
            self._telemetry_task.cancel()
            await asyncio.gather(self._telemetry_task, return_exceptions=True)
        try:
            await asyncio.to_thread(self._stage_io.stop)
        finally:
            try:
                await asyncio.to_thread(self._core.unloadAllDevices)
            finally:
                await super().disconnect()


def parse_axis_limits(text: str, axes: Iterable[str]) -> dict[str, tuple[float, float]]:
    """Read ``x=-1000:1000,y=-500:500`` into per-axis bounds.

    Raises
    ------
    ValueError
        For an entry that is not ``axis=low:high`` with two numbers, an axis
        the stage does not move, or a pair that is not increasing.
    """
    limits: dict[str, tuple[float, float]] = {}
    known = set(axes)
    for entry in (part.strip() for part in text.split(",")):
        if not entry:
            continue
        axis, separator, bounds = entry.partition("=")
        low_text, colon, high_text = bounds.partition(":")
        if not separator or not colon or not axis.strip():
            raise ValueError(
                f"invalid --limits entry {entry!r}; expected axis=low:high"
            )
        axis = axis.strip()
        if axis not in known:
            raise ValueError(
                f"--limits names axis {axis!r}, which the stage does not move"
            )
        try:
            low, high = float(low_text), float(high_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid --limits entry {entry!r}: not two numbers"
            ) from exc
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(
                f"--limits for {axis!r} must be finite and increasing, "
                f"not {low}..{high}"
            )
        limits[axis] = (low, high)
    return limits


def _readback_provenance(library: str, requested: str) -> str:
    """Resolve the configured evidence label without upgrading DemoCamera."""
    if requested not in ("auto", "unknown", "measured", "simulated"):
        raise ValueError("invalid requested readback provenance")
    if library == "DemoCamera":
        if requested == "measured":
            raise ValueError("DemoCamera cannot provide measured readback")
        return "simulated"
    return "unknown" if requested == "auto" else requested


def build_controller(
    adapter: str | None,
    device: str | None,
    label: str,
    axes: Iterable[str],
    *,
    mm_config: str | None = None,
    mm_path: str | None = None,
    axis_limits: Mapping[str, tuple[float, float]] | None = None,
    max_xy_offset_um: float | None = None,
    motion_properties: Iterable[str] = MOTION_PROPERTIES,
    max_motion_rate_setting: int | None = None,
    telemetry_period_ms: int | None = None,
    readback_provenance: str = "auto",
) -> MMStageController:
    """Compose a selected MMCore stage and wrap it in a FastCS controller.

    Raises
    ------
    ValueError
        If neither a configuration nor an adapter device is given.
    """
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus(mm_path=mm_path) if mm_path else CMMCorePlus()
    if mm_config is not None:
        core.loadSystemConfiguration(mm_config)
        if device:
            label = device
        elif not label:
            raise ValueError(
                f"no stage is selected in {mm_config!r}; give --device to select one"
            )
    else:
        if adapter is None or device is None:
            raise ValueError(
                "an adapter and device are required unless a Micro-Manager "
                "configuration is loaded"
            )
        core.loadDevice(label, adapter, device)
        core.initializeDevice(label)
    # A known demo adapter supplies simulated readbacks. Other adapters remain
    # unknown until an instrument profile provides verified provenance.
    try:
        library: str = str(core.getDeviceLibrary(label))
    except Exception:  # adapter unable to identify itself; fail closed
        library = ""
    provenance = _readback_provenance(library, readback_provenance)
    io = StageIO(core, label, max_xy_offset_um, axis_limits)
    return MMStageController(
        core,
        label,
        axes,
        max_xy_offset_um,
        io=io,
        max_motion_rate_setting=max_motion_rate_setting,
        telemetry_period_ms=telemetry_period_ms,
        motion_properties=motion_properties,
        provenance=provenance,
    )


def run(serving: Coroutine[Any, Any, None]) -> int:
    """Run *serving* until the session or Ctrl+C stops it; return the exit code.

    Ctrl+C, the way a service run on its own is stopped, counts as a clean
    stop: `asyncio.run` has cancelled and awaited *serving* before it raises.
    """
    with suppress(KeyboardInterrupt):
        asyncio.run(serving)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the service, taking its identity from the session that launched it."""
    service = identity()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default=service.prefix if service else "",
        help="PV prefix, the session's unless given",
    )
    parser.add_argument(
        "--name",
        default=service.name if service else "stage",
        help="name of this service, the session's unless given",
    )
    parser.add_argument(
        "--mm-config",
        help="Micro-Manager system configuration to load, instead of an adapter device",
    )
    parser.add_argument(
        "--mm-path",
        help="Micro-Manager installation to load device adapters from",
    )
    parser.add_argument("--adapter", help="Micro-Manager adapter")
    parser.add_argument(
        "--device",
        help=(
            "stage to serve: the adapter's device name without a configuration, "
            "or the device label within one"
        ),
    )
    parser.add_argument(
        "--axes", default="x,y", help="axes the stage moves, comma separated"
    )
    parser.add_argument(
        "--limits",
        default="",
        help="per-axis travel bounds, x=-1000:1000,y=-500:500",
    )
    parser.add_argument(
        "--max-xy-offset-um",
        type=float,
        help="limit XY targets to this offset from the service startup readback",
    )
    parser.add_argument(
        "--motion-properties",
        default=",".join(MOTION_PROPERTIES),
        help="comma-separated raw MMCore properties served as the motion profile",
    )
    parser.add_argument(
        "--max-motion-rate-setting",
        type=int,
        help="enable Speed/StartSpeed writes up to this raw adapter setting",
    )
    parser.add_argument(
        "--readback-provenance",
        choices=("auto", "unknown", "simulated", "measured"),
        default="auto",
        help=(
            "auto identifies DemoCamera as simulated; other adapters remain "
            "unknown unless the instrument profile declares their provenance"
        ),
    )
    parser.add_argument(
        "--telemetry-period-ms",
        type=int,
        help=(
            "enable host-bracketed MMCore XY sampling at this period "
            f"({MIN_TELEMETRY_PERIOD_MS}..{MAX_TELEMETRY_PERIOD_MS} ms)"
        ),
    )
    options = parser.parse_args(argv)

    configure_logging()
    axes = [axis.strip() for axis in str(options.axes).split(",") if axis.strip()]
    try:
        axis_limits = parse_axis_limits(str(options.limits), axes)
    except ValueError as error:
        parser.error(str(error))
    motion_properties = tuple(
        name.strip()
        for name in str(options.motion_properties).split(",")
        if name.strip()
    )
    controller = build_controller(
        options.adapter,
        options.device,
        str(options.name),
        axes,
        mm_config=options.mm_config,
        mm_path=options.mm_path,
        axis_limits=axis_limits,
        max_xy_offset_um=options.max_xy_offset_um,
        motion_properties=motion_properties,
        max_motion_rate_setting=options.max_motion_rate_setting,
        telemetry_period_ms=options.telemetry_period_ms,
        readback_provenance=options.readback_provenance,
    )
    return run(
        serve_controller(controller, EpicsPVATransport(), options.prefix or None)
    )


if __name__ == "__main__":
    sys.exit(main())
