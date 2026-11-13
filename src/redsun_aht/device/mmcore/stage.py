"""A Micro-Manager stage, reached through the PVA service that owns it.

Ported from ``redsun_searching_is_imaging/device/mmcore/_stage.py`` at donor
revision ``3a3bf06`` (``redsun-searching-is-imaging``).

Changed:

- The module is ``device/mmcore/stage.py``, not ``_stage.py``, so a session
  imports the stage from a public module.
- Every axis publishes the travel limits the service enforces, so a move
  outside them fails before it is put, and the limit readback a plan wants is
  part of the axis's configuration rather than of the service's command line.
- ``MMAxis`` and ``MMStage`` carry a ``stop()`` that reaches the service: the
  service stops the MMCore device and holds no motion of its own.
- The Ludl adapter path is not ported; the donor's client was adapter-neutral
  and this bundle keeps it that way.
- The device also implements the plan-facing surface of
  ``redsun_aht.contracts.devices.Stage``: ``axes``, ``move``,
  ``read_position`` and ``stop``. They are an adapter over the same signals,
  not a second device, and the axes are read from the service rather than
  declared here, because the service is what enforces the range.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from typing import Annotated as A

import numpy as np
from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    MovableLogic,
    SignalR,
    SignalRW,
    StandardMovable,
    StandardReadable,
    StandardReadableFormat,
    set_and_wait_for_other_value,
)
from ophyd_async.fastcs.core import fastcs_connector
from redsun.log import Loggable

from ...contracts import (
    AxisReading,
    AxisSpec,
    QualityLabel,
    StageReading,
)
from ._containers import ReadableDeviceMap  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Final

    from ophyd_async.core import DeviceMock, TimeoutCalculator

#: Where the stage's axes sit under the service's prefix.
#: ``fastcs`` names a PV path after its controller, in title case.
STAGE_GROUP = "Stage"

#: The Ludl XY readback is quantized to 0.1 um and a bounded physical +50 um
#: move landed 0.2 um from its target. The earlier 0.02 um tolerance therefore
#: timed out after the stage had arrived. Half a micron still resolves these
#: small moves while allowing the measured adapter/controller landing error.
# The ASI service settles up to ~1 um (5 counts at 0.2 um/count); the client
# completion predicate must not be stricter than the service's acceptance.
LOGGER = logging.getLogger("redsun.aht.device.mmcore.stage")
"""Module logger for figures a service publishes but cannot mean."""

POSITION_TOLERANCE: Final[float] = 1.5

#: Seconds a move may take before it is given up as failed. The service waits
#: up to 60 s for MMCore; a 3 mm Ludl traverse was measured at 29.594 s.
MOVE_TIMEOUT: Final[float] = 60.0

#: The units every position and limit is served in, and therefore the units of
#: a declared axis. The service serves ``Float(units="um")`` whatever the
#: adapter's own encoder counts are, so a plan always reads micrometres.
AXIS_UNITS: Final = "um"


@dataclass
class MMAxisLogic(MovableLogic[float]):
    """Move logic for one axis of a Micro-Manager stage."""

    limits: SignalR[np.ndarray]
    stop_request: SignalRW[bool]
    tolerance: float = POSITION_TOLERANCE

    async def check_move(self, new_position: float) -> None:
        """Refuse a target the service would refuse, before it is put.

        The service is the authority and checks the same bounds; refusing here
        keeps an out-of-range move off the wire and out of a scan's timing.
        """
        low, high = (float(bound) for bound in await self.limits.get_value())
        if not low <= new_position <= high:
            raise ValueError(
                f"{self.readback.name} target {new_position} lies outside "
                f"the axis limits {low}..{high}"
            )

    async def stop(self) -> None:
        """Ask the service to stop this stage."""
        await self.stop_request.set(True)

    async def calculate_timeout(
        self, old_position: float, new_position: float
    ) -> float:
        """Give every move `MOVE_TIMEOUT`, so a readback that never lands raises."""
        return MOVE_TIMEOUT

    async def move(self, new_position: float, timeout: TimeoutCalculator) -> None:
        """Write the setpoint and wait for the readback to land within tolerance."""
        await set_and_wait_for_other_value(
            self.setpoint,
            new_position,
            self.readback,
            # rtol defaults to 1e-5, which would widen the tolerance with the
            # distance travelled
            lambda value: bool(
                np.isclose(value, new_position, atol=self.tolerance, rtol=0)
            ),
            timeout=timeout(),
        )


class MMAxis(StandardReadable, StandardMovable[float]):
    """One axis of a Micro-Manager stage, moved through its service."""

    position: A[SignalRW[float], StandardReadableFormat.HINTED_SIGNAL]
    limits: A[SignalR[np.ndarray], StandardReadableFormat.CONFIG_SIGNAL]
    precision: A[SignalR[float], StandardReadableFormat.CONFIG_SIGNAL]
    stop_request: SignalRW[bool]

    @cached_property
    def movable_logic(self) -> MovableLogic[float]:
        """Setpoint and readback of this axis, which are one signal."""
        return MMAxisLogic(
            setpoint=self.position,
            readback=self.position,
            limits=self.limits,
            stop_request=self.stop_request,
        )

    async def declared_spec(self, name: str) -> AxisSpec:
        """Return this axis as the contract's declared axis, named *name*.

        Raises
        ------
        ValueError
            When the service publishes no finite travel bounds for the axis,
            which it does when it was started without ``--limits`` for it: a
            plan cannot be given an axis whose range is unknown, and a
            substituted range would be a bound the hardware never declared.
        """
        limits = await self.limits.get_value()
        lower = float(limits[0])
        upper = float(limits[1])
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(
                f"axis {name!r} publishes no travel limits; start the stage "
                "service with --limits for it"
            )
        return AxisSpec(
            name=name,
            units=AXIS_UNITS,
            lower=lower,
            upper=upper,
            precision=_published_precision(await self.precision.get_value(), name),
        )


#: Precision used when the service publishes none, or publishes an unusable
#: figure. It matches the service's own default, so a client that cannot read
#: the attribute still describes the axis the service meant to declare.
FALLBACK_PRECISION_UM: Final = 0.001


def _published_precision(value: float, name: str) -> float:
    """Return *value* when it can describe a step size, else the default.

    Precision is a hint about the finest reliable step, not a bound: an
    unusable figure (zero, negative or non-finite, which a service can publish
    before its axis is updated) must not make the whole axis undescribable,
    which would hide a movable axis from every plan.
    """
    number = float(value)
    if math.isfinite(number) and number > 0:
        return number
    LOGGER.warning(
        f"axis {name!r} published precision {number!r}, which cannot describe a "
        f"step; using {FALLBACK_PRECISION_UM} um"
    )
    return FALLBACK_PRECISION_UM


class MMStage(StandardReadable, Loggable):
    """A Micro-Manager stage, reached through the service that owns it.

    The stage has the axes the service serves: the adapter, the device and
    the axis names belong to the service's declaration.

    The plan-facing surface of ``redsun_aht.contracts.devices.Stage`` is served
    from the same signals: :attr:`axes` and :meth:`read_position` read what the
    service publishes, and :meth:`move` is a sequence of axis moves, because a
    Micro-Manager XY stage takes one move at a time. The axes are read once at
    connect and refreshed by :meth:`refresh_axes`, since the contract's
    ``axes`` is a synchronous mapping.

    Parameters
    ----------
    prefix :
        PV prefix of the service, ending in ``:``; a device declared with
        ``service=`` receives it from that service.
    """

    axis: A[ReadableDeviceMap[MMAxis], StandardReadableFormat.CHILD]
    stop_request: SignalRW[bool]
    provenance: SignalR[str]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(
            name=name,
            connector=fastcs_connector(f"{prefix}{STAGE_GROUP}:", self),
        )
        self._axes: dict[str, AxisSpec] = {}

    @property
    def axes(self) -> Mapping[str, AxisSpec]:
        """Return the declared axes, as last read from the service.

        An axis whose service publishes no finite travel limits is absent: it
        cannot be described without inventing a bound. It is still movable by
        name through :meth:`move`, which the service bounds on its own.
        """
        return MappingProxyType(dict(self._axes))

    async def refresh_axes(self) -> Mapping[str, AxisSpec]:
        """Read every axis's declared range and precision from the service.

        An axis the service publishes no finite limits for is skipped with a
        warning rather than failing the read: a session that only views frames
        must still connect to a stage service started without ``--limits``.
        """
        specs: dict[str, AxisSpec] = {}
        for name, axis in self.axis.items():
            try:
                specs[name] = await axis.declared_spec(name)
            except ValueError as error:
                self.logger.warning(f"axis {name!r} is not describable: {error}")
        self._axes = specs
        return MappingProxyType(dict(self._axes))

    async def connect(
        self,
        mock: bool | DeviceMock[Any] = False,
        timeout: float = DEFAULT_TIMEOUT,
        force_reconnect: bool = False,
    ) -> None:
        """Connect, then read the axes the service declares."""
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        await self.refresh_axes()

    async def move(self, position: Mapping[str, float]) -> StageReading:
        """Move every named axis and return the reading after they settle.

        Every name is checked before any axis moves, so a plan cannot arrive
        half-way to a position that names an axis this stage does not have.

        Raises
        ------
        ValueError
            For an axis the stage does not serve, or a target the axis's
            published limits refuse.
        """
        unknown = tuple(name for name in position if name not in self.axis)
        if unknown:
            raise ValueError(
                f"the stage has no axis named {unknown[0]!r}; it serves "
                f"{sorted(self.axis)}"
            )
        targets = {name: float(target) for name, target in position.items()}
        for name, target in targets.items():
            if not math.isfinite(target):
                raise ValueError(f"axis {name!r} target must be finite")
            await self.axis[name].movable_logic.check_move(target)
        for name, target in targets.items():
            await self.axis[name].set(target)
        return await self.read_position()

    async def read_position(self) -> StageReading:
        """Return every axis position, labelled by where it came from.

        The service must declare whether readbacks are simulated or measured.
        Unknown provenance yields no usable positions rather than a false claim.
        """
        provenance = await self.provenance.get_value()
        if provenance not in ("simulated", "measured"):
            return StageReading(
                axes=(),
                monotonic_ns=time.monotonic_ns(),
                quality=QualityLabel.UNAVAILABLE,
            )
        measured = provenance == "measured"
        readings = tuple(
            [
                AxisReading(
                    name=name,
                    value=float(location["readback"]),
                    setpoint=float(location["setpoint"]),
                    measured=measured,
                )
                for name, axis in self.axis.items()
                for location in [await axis.locate()]
            ]
        )
        if not readings:
            return StageReading(
                axes=(),
                monotonic_ns=time.monotonic_ns(),
                quality=QualityLabel.UNAVAILABLE,
            )
        return StageReading(
            axes=readings,
            monotonic_ns=time.monotonic_ns(),
            quality=QualityLabel.MEASURED if measured else QualityLabel.SIMULATED,
        )

    async def stop(self) -> None:
        """Ask the service to stop every axis of the stage, and nothing else.

        A Micro-Manager XY stage is one device, so stopping is not per axis;
        the service calls ``CMMCorePlus.stop`` on the device the axes share.
        """
        await self.stop_request.set(True)


__all__ = [
    "AXIS_UNITS",
    "MOVE_TIMEOUT",
    "POSITION_TOLERANCE",
    "MMAxis",
    "MMStage",
]
