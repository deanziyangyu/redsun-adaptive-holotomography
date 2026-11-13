"""Stage-contract adapter for the synthetic PVA service only."""

from __future__ import annotations

import time
from types import MappingProxyType
from typing import TYPE_CHECKING

from redsun_aht.contracts import AxisReading, AxisSpec, QualityLabel, StageReading

if TYPE_CHECKING:
    from collections.abc import Mapping

    from redsun_aht.simulation.mcu_stack import Stack
    from redsun_aht.simulation.pva_stack import PvaSimulatedStack, StackEvidence


class PvaSyntheticSurveyStage:
    """Adapt the fake service without presenting its readback as physical."""

    def __init__(self, client: PvaSimulatedStack) -> None:
        self.client = client
        self._active: str | None = None
        self._axes = {
            name: AxisSpec(name, "um", -100, 100, 0.25)
            for name in ("survey_x", "survey_y", "coarse_z")
        }

    @property
    def axes(self) -> Mapping[str, AxisSpec]:
        """Bounds declared by the fixed synthetic service configuration."""
        return MappingProxyType(self._axes)

    async def read_position(self) -> StageReading:
        """Read the last service-published synthetic positions."""
        values = await self.client.jog_positions()
        return StageReading(
            tuple(
                AxisReading(name, values[name], values[name], False)
                for name in self._axes
            ),
            time.monotonic_ns(),
            QualityLabel.SIMULATED,
        )

    async def move(self, position: Mapping[str, float]) -> StageReading:
        """Apply one absolute fake survey request through the service."""
        if not position or not set(position) <= set(self._axes):
            raise ValueError("invalid simulated survey targets")
        targets = {
            name: self._axes[name].check(value) for name, value in position.items()
        }
        await self.client.manual.set(
            self.client._payload_with_token("targets", targets)
        )
        return await self.read_position()

    async def execute_stack(self, stack: Stack) -> StackEvidence:
        """Run a fake stack; a stop request is scoped to this active identity."""
        if self._active is not None:
            raise RuntimeError("simulated stack already active")
        self._active = stack.run_id
        try:
            return await self.client.execute(stack)
        finally:
            self._active = None

    async def stop(self) -> None:
        """Request simulated stack abort when active; no physical stop exists."""
        if self._active is not None:
            await self.client.abort_stack(self._active)
