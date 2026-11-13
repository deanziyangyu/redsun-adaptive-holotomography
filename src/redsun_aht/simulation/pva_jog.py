"""XInput-to-PVA bridge for the simulated survey service only."""

from __future__ import annotations

import asyncio
import math
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol

from redsun_aht.operator_input.xinput import SurveyController, survey_intent

if TYPE_CHECKING:
    from redsun_aht.operator_input.xinput import (
        ControllerState,
        SurveyIntent,
        XInputPoller,
    )


class JogTarget(Protocol):
    """Synthetic increment sink shared by the presenter and controller."""

    async def jog_by(self, increments: dict[str, float]) -> None:
        """Submit one bounded synthetic increment."""
        ...


class SimulatedPvaJog:
    """Send one bounded quantized fake increment per enabled input sample.

    This is not a physical coordinator, real stop or input polling watchdog.
    The service itself validates increments and owns the simulated reservation.
    """

    def __init__(self, device: JogTarget) -> None:
        self.device = device
        self.input = SurveyController()
        self._pending = False
        self._running = False

    async def run(
        self, poller: XInputPoller, stop: asyncio.Event, *, period_s: float = 0.05
    ) -> None:
        """Poll while one write is pending; never queue another write."""
        if not math.isfinite(period_s) or period_s < 0.01:
            raise ValueError("poll period must be at least 0.01 seconds")
        if self._running:
            raise RuntimeError("synthetic poll loop already running")
        self._running = True
        pending: asyncio.Task[SurveyIntent] | None = None
        try:
            while not stop.is_set():
                state = poller.read()
                if pending is not None and pending.done():
                    pending.result()
                    pending = None
                if pending is None:
                    pending = asyncio.create_task(self.update(state))
                elif survey_intent(state).stop_requested:
                    self.input.disarm()
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), period_s)
        finally:
            self.input.disarm()
            if pending is not None:
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            self._running = False

    async def update(self, state: ControllerState | None) -> SurveyIntent:
        """Map one sample; reject overlap instead of queueing stale motion."""
        if self._pending:
            self.input.disarm()
            raise RuntimeError("synthetic jog already pending")
        intent = self.input.update(state)
        if intent.stop_requested:
            return intent
        increments = {
            name: math.trunc(amount * 4) / 4
            for name, amount in zip(
                ("survey_x", "survey_y", "coarse_z"),
                (intent.x, intent.y, intent.z),
                strict=True,
            )
            if math.trunc(amount * 4)
        }
        if increments:
            self._pending = True
            try:
                await self.device.jog_by(increments)
            except BaseException:
                self.input.disarm()
                raise
            finally:
                self._pending = False
        return intent
