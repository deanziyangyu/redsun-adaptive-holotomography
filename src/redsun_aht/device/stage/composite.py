"""Limit-checked composite stage with independently replaceable axis backends."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic, monotonic_ns
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from redsun_aht.domain import PoseSample

if TYPE_CHECKING:
    from collections.abc import Mapping

_IDENTITY_4X4 = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


class StageSettleError(RuntimeError):
    """Raised when a stage does not reach its commanded pose in time."""


class AxisBackend(Protocol):
    """Minimal asynchronous boundary implemented by one physical or mock axis."""

    async def move_to(self, position: float) -> None:
        """Start an absolute move in the axis' declared engineering unit."""

    async def read_position(self) -> float:
        """Read the current axis position."""

    async def motion_complete(self) -> bool:
        """Return whether the most recent move has completed."""

    async def stop(self) -> None:
        """Stop ordinary motion as promptly as the backend permits."""


@dataclass(frozen=True, slots=True)
class AxisSpec:
    """Named axis metadata and software admission bounds."""

    name: str
    unit: str
    lower_limit: float
    upper_limit: float
    settle_tolerance: float
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        """Validate a complete finite axis declaration."""
        if not self.name or not self.unit:
            raise ValueError("axis name and unit cannot be empty")
        values = (
            self.lower_limit,
            self.upper_limit,
            self.settle_tolerance,
            self.uncertainty,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("axis bounds and tolerances must be finite")
        if self.lower_limit >= self.upper_limit:
            raise ValueError("axis lower_limit must be below upper_limit")
        if self.settle_tolerance <= 0:
            raise ValueError("axis settle_tolerance must be positive")
        if self.uncertainty < 0:
            raise ValueError("axis uncertainty cannot be negative")


@dataclass(slots=True)
class SimulatedAxisBackend:
    """Deterministic axis model that can delay or stall settling."""

    position: float = 0.0
    settle_polls: int = 0
    stalled: bool = False
    commands: list[float] = field(default_factory=list)
    stopped: bool = False
    _target: float | None = field(default=None, init=False)
    _polls_remaining: int = field(default=0, init=False)

    async def move_to(self, position: float) -> None:
        """Record a target and initialize its deterministic poll delay."""
        self.commands.append(position)
        self.stopped = False
        self._target = position
        self._polls_remaining = self.settle_polls

    async def read_position(self) -> float:
        """Return the current simulated position."""
        return self.position

    async def motion_complete(self) -> bool:
        """Advance one simulated settling poll."""
        if self.stalled:
            return False
        if self._polls_remaining:
            self._polls_remaining -= 1
            return False
        if self._target is not None:
            self.position = self._target
            self._target = None
        return True

    async def stop(self) -> None:
        """Cancel the pending simulated target."""
        self._target = None
        self.stopped = True


@dataclass(slots=True)
class CompositeStageDevice:
    """Compose named axes into one validated stage and pose source."""

    specs: tuple[AxisSpec, ...]
    backends: Mapping[str, AxisBackend]
    coordinate_frame: str = "aht-sample-stage"
    source: str = "composite-stage"
    settle_timeout: float = 5.0
    settle_poll_interval: float = 0.01
    _commanded: dict[str, float] = field(default_factory=dict, init=False)
    _measured: dict[str, float] = field(default_factory=dict, init=False)
    _pending_axes: set[str] = field(default_factory=set, init=False)
    _spec_by_name: dict[str, AxisSpec] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Freeze and validate the axis-to-backend composition."""
        if not self.specs:
            raise ValueError("composite stage must contain at least one axis")
        self._spec_by_name = {spec.name: spec for spec in self.specs}
        if len(self._spec_by_name) != len(self.specs):
            raise ValueError("composite stage axis names must be unique")
        if set(self.backends) != set(self._spec_by_name):
            raise ValueError("every declared axis must have exactly one backend")
        if self.settle_timeout <= 0 or self.settle_poll_interval < 0:
            raise ValueError("settle timing must be positive and non-negative")
        self.backends = MappingProxyType(dict(self.backends))

    @property
    def axes(self) -> Mapping[str, str]:
        """Return the stable public axis-to-unit mapping."""
        return MappingProxyType({spec.name: spec.unit for spec in self.specs})

    async def move(self, positions: Mapping[str, float]) -> PoseSample:
        """Validate the complete request before starting any axis move."""
        if not positions:
            raise ValueError("move request cannot be empty")
        requested = {name: float(value) for name, value in positions.items()}
        for name, value in requested.items():
            spec = self._spec_by_name.get(name)
            if spec is None:
                raise ValueError(f"unknown stage axis: {name}")
            if not isfinite(value):
                raise ValueError(f"axis {name} target must be finite")
            if not spec.lower_limit <= value <= spec.upper_limit:
                raise ValueError(
                    f"axis {name} target {value} is outside "
                    f"[{spec.lower_limit}, {spec.upper_limit}] {spec.unit}"
                )
        changed = {
            name: value
            for name, value in requested.items()
            if name not in self._commanded or value != self._commanded[name]
        }
        try:
            for name, value in changed.items():
                await self.backends[name].move_to(value)
        except BaseException:
            await self.stop()
            raise
        self._commanded.update(requested)
        self._pending_axes = set(changed)
        return await self._sample("commanded", overrides=requested)

    async def settle(self) -> PoseSample:
        """Wait for motion-complete plus tolerance agreement, then read pose."""
        deadline = monotonic() + self.settle_timeout
        pending_axes = tuple(
            spec.name for spec in self.specs if spec.name in self._pending_axes
        )
        if not pending_axes:
            return await self._sample("measured", read_axes=frozenset())
        while monotonic() < deadline:
            complete = [
                await self.backends[name].motion_complete() for name in pending_axes
            ]
            if all(complete):
                sample = await self._sample(
                    "measured", read_axes=frozenset(pending_axes)
                )
                if self._within_tolerance(sample):
                    self._pending_axes.clear()
                    return sample
            await asyncio.sleep(self.settle_poll_interval)
        await self.stop()
        raise StageSettleError("stage did not settle before the configured timeout")

    async def read(self) -> PoseSample:
        """Read all axes into one explicitly framed pose sample."""
        return await self._sample("measured", read_axes=None)

    async def stop(self) -> None:
        """Issue ordinary stop to every composed axis backend."""
        for spec in self.specs:
            await self.backends[spec.name].stop()

    async def _sample(
        self,
        sample_kind: str,
        overrides: Mapping[str, float] | None = None,
        read_axes: frozenset[str] | None = None,
    ) -> PoseSample:
        values: list[float] = []
        overrides = overrides or {}
        for spec in self.specs:
            if spec.name in overrides:
                values.append(overrides[spec.name])
            elif read_axes is not None and spec.name not in read_axes:
                try:
                    values.append(self._measured[spec.name])
                except KeyError:
                    observed = await self.backends[spec.name].read_position()
                    self._measured[spec.name] = observed
                    values.append(observed)
            else:
                observed = await self.backends[spec.name].read_position()
                self._measured[spec.name] = observed
                values.append(observed)
        return PoseSample(
            monotonic_ns=monotonic_ns(),
            sample_kind=sample_kind,
            joints=tuple(values),
            joint_units=tuple(spec.unit for spec in self.specs),
            coordinate_frame=self.coordinate_frame,
            transform_to_parent=_IDENTITY_4X4,
            transform_units="mm",
            uncertainty=tuple(spec.uncertainty for spec in self.specs),
            source=self.source,
        )

    def _within_tolerance(self, sample: PoseSample) -> bool:
        for spec, observed in zip(self.specs, sample.joints, strict=True):
            commanded = self._commanded.get(spec.name)
            if (
                commanded is not None
                and abs(observed - commanded) > spec.settle_tolerance
            ):
                return False
        return True
