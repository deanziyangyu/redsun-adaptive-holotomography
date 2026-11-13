"""Experimental in-process coordinator for already-owned stage devices.

All callers must use the same instance. This is not a service-side or
cross-process lease, physical watchdog, synchronized move, or verified stop.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4

from redsun_aht.contracts import QualityLabel

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from redsun_aht.contracts import Stage, StageReading


@dataclass(frozen=True, slots=True)
class AxisBinding:
    """Map one public axis name to an existing stage and its axis name."""

    stage: Stage
    axis: str
    max_jog: float

    def __post_init__(self) -> None:
        """Reject a missing axis or invalid per-request jog bound."""
        if self.axis not in self.stage.axes:
            raise ValueError("binding axis is not declared by stage")
        if (
            isinstance(self.max_jog, bool)
            or not math.isfinite(self.max_jog)
            or self.max_jog <= 0
        ):
            raise ValueError("max jog must be finite and positive")


class CompositeMotionCoordinator:
    """Serialize moves across injected stages with a cooperative local token."""

    def __init__(self, axes: Mapping[str, AxisBinding]) -> None:
        if not axes or any(not isinstance(k, str) or not k.strip() for k in axes):
            raise ValueError("named axes are required")
        if len({(id(v.stage), v.axis) for v in axes.values()}) != len(axes):
            raise ValueError("physical axis cannot have multiple names")
        self._axes = dict(axes)
        self._token: str | None = None
        self._busy = False
        self._uncertain = False

    @property
    def uncertain(self) -> bool:
        """Whether a submitted move needs external reconciliation."""
        return self._uncertain

    def reserve(self) -> str:
        """Reserve all configured axes for one acquisition client."""
        if self._busy or self._uncertain or self._token is not None:
            raise RuntimeError("coordinator busy, reserved or uncertain")
        self._token = uuid4().hex
        return self._token

    def release(self, token: str) -> None:
        """Release a certain idle acquisition reservation."""
        if self._busy or self._uncertain or self._token is None or token != self._token:
            raise RuntimeError("coordinator busy, uncertain or reservation mismatch")
        self._token = None

    def _admit(self, token: str | None) -> None:
        if self._busy or self._uncertain:
            raise RuntimeError("coordinator busy or uncertain")
        if (self._token is None and token is not None) or (
            self._token is not None and token != self._token
        ):
            raise RuntimeError("coordinator reservation mismatch")
        self._busy = True

    def _validate(self, targets: Mapping[str, float]) -> dict[str, float]:
        if not targets or not set(targets) <= set(self._axes):
            raise ValueError("nonempty known axis targets required")
        return {
            name: self._axes[name].stage.axes[self._axes[name].axis].check(value)
            for name, value in targets.items()
        }

    async def _execute(self, targets: Mapping[str, float]) -> dict[str, StageReading]:
        groups: dict[int, tuple[Stage, dict[str, float]]] = {}
        for name, value in targets.items():
            binding = self._axes[name]
            groups.setdefault(id(binding.stage), (binding.stage, {}))[1][
                binding.axis
            ] = value
        readings: dict[str, StageReading] = {}
        try:
            for key, (stage, values) in groups.items():
                if self._uncertain:
                    raise RuntimeError("stop requested; outcome uncertain")
                reading = await stage.move(values)
                if self._uncertain:
                    raise RuntimeError("stop requested; outcome uncertain")
                for name in targets:
                    if id(self._axes[name].stage) == key:
                        readings[name] = reading
        except BaseException:
            self._uncertain = True
            raise
        return readings

    async def move(
        self, targets: Mapping[str, float], *, token: str | None = None
    ) -> dict[str, StageReading]:
        """Prevalidate all targets, then move the owned stages sequentially."""
        self._admit(token)
        try:
            return await self._execute(self._validate(targets))
        finally:
            self._busy = False

    async def jog_by(self, increments: dict[str, float]) -> None:
        """Read and apply one bounded manual increment while no run is reserved."""
        self._admit(None)
        try:
            if not increments or not set(increments) <= set(self._axes):
                raise ValueError("nonempty known jog axes required")
            for name, value in increments.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or abs(value) > self._axes[name].max_jog
                ):
                    raise ValueError("invalid or oversized jog")
            positions: dict[int, StageReading] = {}
            for name in increments:
                stage = self._axes[name].stage
                key = id(stage)
                if key not in positions:
                    positions[key] = await stage.read_position()
                    if positions[key].quality is QualityLabel.UNAVAILABLE:
                        raise RuntimeError("stage position unavailable")
            targets = {
                name: positions[id(self._axes[name].stage)].positions[
                    self._axes[name].axis
                ]
                + value
                for name, value in increments.items()
            }
            await self._execute(self._validate(targets))
        finally:
            self._busy = False

    async def run_reserved(
        self, token: str, operation: Callable[[], Awaitable[T]]
    ) -> T:
        """Run one acquisition operation under an existing local reservation.

        The operation must use the same owned backends and be bounded by the
        caller. Failure leaves its device outcome uncertain.
        """
        if self._token is None or token != self._token:
            raise RuntimeError("coordinator reservation mismatch")
        self._admit(token)
        try:
            try:
                result = await operation()
                if self._uncertain:
                    raise RuntimeError("stop requested; outcome uncertain")
                return result
            except BaseException:
                self._uncertain = True
                raise
        finally:
            self._busy = False

    async def stop(self) -> None:
        """Request stop from each stage and quarantine the coordinator.

        A returned request does not establish physical stop. This operation is
        permitted during a move or reservation and never clears uncertainty.
        """
        self._uncertain = True
        stages = {id(binding.stage): binding.stage for binding in self._axes.values()}
        results = await asyncio.gather(
            *(stage.stop() for stage in stages.values()), return_exceptions=True
        )
        if any(isinstance(result, BaseException) for result in results):
            raise RuntimeError("one or more stage stop requests failed")
