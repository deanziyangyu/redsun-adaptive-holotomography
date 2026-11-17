"""Serve the ESP300 survey axes through one bounded serial owner."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, cast

from fastcs.attributes import AttrR, AttrW
from fastcs.controllers import Controller
from fastcs.datatypes import Bool
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from oxiserial import serial_for_url
from redsun.services import configure_logging, serve_controller

from redsun_aht.device.esp300_z_serial import (
    Esp300ReadbackUnavailable,
    Esp300SurveySerial,
)
from redsun_aht.services.mmcore_stage import (
    AxisPositionIO,
    MMAxisController,
    MMStageGroup,
    parse_axis_limits,
)

if TYPE_CHECKING:
    from redsun_aht.services.mmcore_stage import AxisRef

AXES = ("x", "y", "z")


class Esp300StageIO(AxisPositionIO):
    """Adapt the serial owner's micrometre operations to positioner PVs."""

    def __init__(self, owner: Esp300SurveySerial) -> None:
        super().__init__()
        self._owner = owner
        self._closing = False
        self._tasks: set[asyncio.Task] = set()
        self._moves: dict[asyncio.Task, str] = {}
        self._limits, self._precision = owner.axis_specs()

    def limits_for(self, axis: str) -> tuple[float, float]:
        """Return the enforced bounds in micrometres."""
        return self._limits[AXES.index(axis)]

    def precision_for(self, axis: str) -> tuple[float, str]:
        """Return the encoder resolution and its source."""
        return self._precision[AXES.index(axis)], "ESP300 controller resolution"

    async def update(self, attr: AttrR[float, AxisRef]) -> None:
        """Publish a fresh controller position."""
        if self._closing:
            return
        task = asyncio.create_task(
            asyncio.to_thread(self._owner.read_um, AXES.index(attr.io_ref.axis) + 1)
        )
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        try:
            position = await asyncio.shield(task)
        except Esp300ReadbackUnavailable:
            position = float("nan")
        if not self._closing:
            await attr.update(position)

    async def send(self, attr: AttrW[float, AxisRef], value: float) -> None:
        """Settle the requested axis before publishing its readback."""
        if self._closing:
            raise RuntimeError("ESP300 service is closing")
        task = asyncio.create_task(
            asyncio.to_thread(
                self._owner.move_um, AXES.index(attr.io_ref.axis) + 1, value
            )
        )
        self._tasks.add(task)
        self._moves[task] = attr.io_ref.axis
        task.add_done_callback(self._finished)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.to_thread(self.stop)
            raise
        if isinstance(attr, AttrR):
            await self.update(attr)

    def _finished(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        self._moves.pop(task, None)
        if not task.cancelled():
            task.exception()

    def stop(self) -> None:
        """Interrupt active movement and latch the owner."""
        self._owner.stop()

    async def stop_active_axis(self, axis: str) -> None:
        """Stop only if that axis still has an unresolved movement request."""
        if any(name == axis and not task.done() for task, name in self._moves.items()):
            await asyncio.to_thread(self.stop)

    async def drain(self) -> None:
        """Stop before closing the endpoint and let serial workers finish."""
        self._closing = True
        try:
            await asyncio.to_thread(self.stop)
        finally:
            await asyncio.gather(*self._tasks, return_exceptions=True)


class Esp300AxisController(MMAxisController):
    """Expose the routine cleanup request alongside the explicit stop."""

    def __init__(self, axis: str, io: AxisPositionIO) -> None:
        super().__init__(axis, io)
        self.stop_active_request = AttrW(
            Bool(), description="Stop if the ESP300 axis has an active move",
        )
        self.stop_active_request.set_on_put_callback(self._stop_active)

    async def _stop_active(self, attr: AttrW, requested: bool) -> None:
        del attr
        if requested:
            await cast("Esp300StageIO", self._io).stop_active_axis(self._axis)


def build_controller(io: Esp300StageIO, *, provenance: str = "measured") -> Controller:
    """Build the shared positioner PV tree with encoder readback."""
    controller = Controller()
    controller.add_sub_controller("stage", MMStageGroup(
        io, AXES, provenance, axis_controller=Esp300AxisController,
    ))
    return controller


async def _serve(
    options: argparse.Namespace, configuration: tuple[tuple[str, float, float], ...]
) -> None:
    port = serial_for_url(
        options.port,
        baudrate=19200,
        timeout=1,
        write_timeout=1,
        rtscts=True,
    )
    try:
        owner = Esp300SurveySerial(port, configuration)
        io = Esp300StageIO(owner)
        try:
            await serve_controller(
                build_controller(io), EpicsPVATransport(), options.prefix
            )
        finally:
            await io.drain()
    finally:
        port.close()


def main() -> int:
    """Validate arguments and serve a single serial endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--axis-identities", nargs=3, required=True)
    parser.add_argument("--limits", required=True)
    options = parser.parse_args()
    if not options.port.strip() or not options.prefix.strip():
        parser.error("explicit serial port and PVA prefix required")
    try:
        limits = parse_axis_limits(options.limits, AXES)
    except ValueError as error:
        parser.error(str(error))
    if set(limits) != set(AXES):
        parser.error("limits must be supplied for x, y and z")
    configuration = tuple(
        (identity, *limits[axis])
        for axis, identity in zip(AXES, options.axis_identities, strict=True)
    )
    configure_logging()
    with suppress(KeyboardInterrupt):
        asyncio.run(_serve(options, configuration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
