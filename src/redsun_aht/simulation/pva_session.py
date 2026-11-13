"""Headless Redsun session for the experimental PVA stack rehearsal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Protocol, cast, runtime_checkable
from uuid import uuid4

from redsun import AsDevice, AsPresenter, Declare, DevicesOf, Session

from .mcu_stack import Stack
from .pva_jog import SimulatedPvaJog
from .pva_stack import (  # noqa: TC001 - Redsun resolves declaration
    PvaSimulatedStack,
    StackEvidence,
)
from .pva_stack_plan import pva_stack_plan

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Generator, Mapping
    from typing import Any

    from redsun.registry import PlanEntry

    from redsun_aht.operator_input.xinput import XInputPoller


@runtime_checkable
class StackService(Protocol):
    """Narrow injected capability; it carries simulated edges only."""

    async def execute(self, stack: Stack) -> StackEvidence:
        """Submit a stack once and return its simulated records."""
        ...


@runtime_checkable
class JogService(Protocol):
    """Synthetic-only interface shared by manual controls and controller input."""

    async def jog_by(self, increments: dict[str, float]) -> None:
        """Request one bounded increment."""
        ...

    async def jog_positions(self) -> dict[str, float]:
        """Return the latest published synthetic position snapshot."""
        ...


class SimulatedJogPresenter:
    """Headless view-facing adapter to one simulated PVA service."""

    def __init__(self, name: str, *, joggers: DevicesOf[JogService]) -> None:
        self.name = name
        self._joggers = dict(joggers)
        self._controller_running = False

    def _one(self) -> JogService:
        if len(self._joggers) != 1:
            raise RuntimeError("expected one simulated jog service")
        return next(iter(self._joggers.values()))

    async def jog_by(self, increments: dict[str, float]) -> None:
        """Request a synthetic jog through the same service as XInput."""
        await self._one().jog_by(increments)

    async def positions(self) -> dict[str, float]:
        """Read the latest published synthetic positions."""
        return await self._one().jog_positions()

    async def run_controller(
        self, poller: XInputPoller, stop: asyncio.Event, *, period_s: float = 0.05
    ) -> None:
        """Run one caller-managed controller loop against the shared fake service."""
        if self._controller_running:
            raise RuntimeError("synthetic controller already running")
        bridge = SimulatedPvaJog(self._one())
        self._controller_running = True
        try:
            await bridge.run(poller, stop, period_s=period_s)
        finally:
            self._controller_running = False


class PvaStackPresenter:
    """Expose the PVA-backed fake stack as a Redsun plan."""

    def __init__(self, name: str, *, stacks: DevicesOf[StackService]) -> None:
        self.name = name
        self._stacks = dict(stacks)

    def plan_map(self) -> Mapping[str, PlanEntry]:
        """Offer the plan only when one simulated stack service exists."""
        return (
            {"pva_simulated_zstack": {"plan": self.run_stack}}
            if len(self._stacks) == 1
            else {}
        )

    def run_stack(self, store_uri: str | None = None) -> Generator[Any, Any, None]:
        """Submit one fixed synthetic stack through the injected service."""
        if len(self._stacks) != 1:
            raise RuntimeError("expected one simulated stack service")
        stack = Stack(uuid4().hex, (0.0, 0.25, 0.5), 2)
        yield from pva_stack_plan(
            cast("PvaSimulatedStack", next(iter(self._stacks.values()))),
            stack,
            store_uri=store_uri,
        )


class PvaStackSession(Session):
    """Headless simulated stack session; service is started separately."""

    config: ClassVar[dict[str, object]] = {
        "session": "pva-simulated-zstack",
        "strict": True,
    }
    stack: Annotated[AsDevice[PvaSimulatedStack], Declare(prefix="SIMSTACK:")]
    presenter: AsPresenter[PvaStackPresenter]
    jog_presenter: AsPresenter[SimulatedJogPresenter]
