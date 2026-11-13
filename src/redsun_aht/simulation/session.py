"""Headless Redsun composition for the experimental simulated stack."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4

from redsun import AsPresenter, Session

from .mcu_stack import Axis, SimulatedMcu, Stack
from .stack_plan import simulated_stack_plan

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from typing import Any

    from redsun.registry import PlanEntry


class SimulatedStackPresenter:
    """Offer finite stacks through one shared in-process fake MCU."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.mcu = SimulatedMcu(Axis("fast_z", -100.0, 100.0, 0.25))

    def plan_map(self) -> Mapping[str, PlanEntry]:
        """Expose the rehearsal plan using Redsun's plan registry shape."""
        return {"simulated_zstack": {"plan": self.run_stack}}

    def run_stack(self) -> Generator[Any, Any, None]:
        """Run synthetic positions in micrometres; no hardware is opened."""
        stack = Stack(uuid4().hex, (0.0, 0.25, 0.5), 2)
        yield from simulated_stack_plan(self.mcu, stack)


class SimulatedStackSession(Session):
    """Headless in-process rehearsal; not an MCU or camera PVA service."""

    config: ClassVar[dict[str, object]] = {
        "session": "simulated-zstack",
        "strict": True,
    }
    stack: AsPresenter[SimulatedStackPresenter]
