"""FastCS PVA service for the MMStudio-compatible 91-LED v1 Arduino controller.

This process is the only owner of the Arduino serial port.  It exposes the
MMStudio-compatible command set (``00``, ``01`` and the patterns
``11``..``15``) as typed attributes, plus the last command and its transport
outcome.  The 91-LED v1 layout has four semicircle patterns (``11``..``14``) and
one centre pattern (``15``); LED indices live in firmware.  It does not invent
optical readback: ``LastCommand`` and ``LastOutcome`` describe what was sent,
not what the sample received.

Writing ``Pattern`` sends a pattern command. The AHT MM ASCII 91 firmware does
not emit a camera TTL pulse; the software DPCT plan reads a fresh frame after
each accepted pattern. ``10`` (array off) is omitted from the service surface.

Transport failures are recorded in ``LastOutcome``/``LastDetail`` instead of
taking the PVA server down.  ``Port`` and ``Baudrate`` reconfigure the owned
endpoint: a replacement port is blanked before the old one is released.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from fastcs.attributes import AttributeIO, AttributeIORef, AttrR, AttrRW, AttrW
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, Int, String, Table
from fastcs.util import ONCE

from redsun_aht.device.arduino.commands import (
    SUPPORTED_ORDER,
    CommandOutcome,
    CommandStatus,
    pattern_command,
)
from redsun_aht.device.arduino.service import (
    BAUDRATE,
    ArduinoLedProfile,
    ArduinoLedService,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

AttributeField = Literal[
    "port",
    "baudrate",
    "pattern",
    "all_off",
    "open_shutter",
    "last_command",
    "last_outcome",
    "last_detail",
    "supported_commands",
    "panel_profile",
    "led_count",
]

SUPPORTED_COMMANDS_TABLE: Final = Table([("command", "U2")])
"""One string column, which is how PVA carries the read-only command array."""


@dataclass
class ArduinoLedRef(AttributeIORef):  # type: ignore[misc]
    """Identifies which service-backed value an attribute reads or writes."""

    field: AttributeField = "last_command"


def supported_commands_array() -> NDArray[Any]:
    """Return the advertised commands as a one-row-per-command string table."""
    return np.array(
        [(command,) for command in SUPPORTED_ORDER],
        dtype=[("command", "U2")],
    )


class ArduinoLedIO(AttributeIO[Any, ArduinoLedRef]):  # type: ignore[misc]
    """Translate FastCS attribute operations into serial-service calls."""

    def __init__(self, service: ArduinoLedService) -> None:
        super().__init__()
        self._service = service
        self._controller: ArduinoLedController | None = None

    def bind(self, controller: ArduinoLedController) -> None:
        """Attach the controller whose readbacks follow every command."""
        self._controller = controller

    async def update(self, attr: AttrR[Any, ArduinoLedRef]) -> None:
        """Publish the service's current configuration or last known outcome."""
        field = attr.io_ref.field
        if field == "port":
            await attr.update(self._service.port_name)
        elif field == "baudrate":
            await attr.update(self._service.baudrate)
        elif field == "pattern":
            pattern = self._service.last_pattern
            await attr.update(1 if pattern is None else pattern)
        elif field == "last_command":
            await attr.update(self._service.last_command)
        elif field == "last_outcome":
            outcome = self._service.last_outcome
            await attr.update("" if outcome is None else outcome.status.value)
        elif field == "last_detail":
            await attr.update(self._service.last_detail)
        elif field == "supported_commands":
            await attr.update(supported_commands_array())
        elif field == "panel_profile":
            await attr.update(self._service.panel_profile.name)
        elif field == "led_count":
            await attr.update(self._service.led_count)
        else:
            raise ValueError(f"unknown Arduino LED attribute field {field!r}")

    async def send(self, attr: AttrW[Any, ArduinoLedRef], value: Any) -> None:
        """Apply one writable attribute and publish the resulting readbacks."""
        field = attr.io_ref.field
        if field == "port":
            text = str(value).strip()
            await self._service.reconfigure(port=text)
            if isinstance(attr, AttrR):
                await attr.update(text)
            await self._publish(self._service.last_outcome)
        elif field == "baudrate":
            baudrate = int(value)
            await self._service.reconfigure(baudrate=baudrate)
            if isinstance(attr, AttrR):
                await attr.update(baudrate)
            await self._publish(self._service.last_outcome)
        elif field == "pattern":
            number = int(value)
            outcome = await self._service.send(pattern_command(number))
            if outcome.delivered and isinstance(attr, AttrR):
                await attr.update(number)
            await self._publish(outcome)
        elif field == "all_off":
            if not bool(value):
                return
            outcome = await self._service.all_off()
            await self._publish(outcome)
        elif field == "open_shutter":
            if not bool(value):
                return
            outcome = await self._service.open_shutter()
            await self._publish(outcome)
        else:
            raise ValueError(f"attribute {field!r} is read-only")

    async def _publish(self, outcome: CommandOutcome | None) -> None:
        """Mirror one command outcome into the read-only diagnostic attributes."""
        controller = self._controller
        if controller is None:
            return
        await controller.last_command.update("" if outcome is None else outcome.command)
        await controller.last_outcome.update(
            "" if outcome is None else outcome.status.value
        )
        await controller.last_detail.update("" if outcome is None else outcome.detail)


class ArduinoLedController(Controller):  # type: ignore[misc]
    """PVA surface for one serialized MMStudio-compatible Arduino."""

    def __init__(self, service: ArduinoLedService) -> None:
        self.service = service
        self.io = ArduinoLedIO(service)
        super().__init__(ios=[self.io])

        self.port = AttrRW(
            String(),
            io_ref=ArduinoLedRef(field="port", update_period=ONCE),
            description="Serial port owned by this service",
        )
        self.baudrate = AttrRW(
            Int(min=1),
            io_ref=ArduinoLedRef(field="baudrate", update_period=ONCE),
            description="Serial line speed; 9600 in the validated MMStudio sketch",
        )
        self.pattern = AttrRW(
            Int(min=1, max=5),
            io_ref=ArduinoLedRef(field="pattern", update_period=ONCE),
            description=(
                "Pattern 1..5 (four semicircles plus centre); writing it sends "
                "11..15 and is the camera trigger"
            ),
        )
        self.all_off = AttrW(
            Bool(),
            io_ref=ArduinoLedRef(field="all_off"),
            description="Send 00 to blank every output without a camera trigger",
        )
        self.open_shutter = AttrW(
            Bool(),
            io_ref=ArduinoLedRef(field="open_shutter"),
            description="Send 01 to clear the shutter-closed flag without a trigger",
        )
        self.last_command = AttrR(
            String(),
            io_ref=ArduinoLedRef(field="last_command", update_period=ONCE),
            description="Two-digit command most recently attempted",
        )
        self.last_outcome = AttrR(
            String(),
            io_ref=ArduinoLedRef(field="last_outcome", update_period=ONCE),
            description="sent, echoed, uncertain or failed; never optical evidence",
        )
        self.last_detail = AttrR(
            String(),
            io_ref=ArduinoLedRef(field="last_detail", update_period=ONCE),
            description="Diagnostics for the most recent command attempt",
        )
        self.supported_commands = AttrR(
            SUPPORTED_COMMANDS_TABLE,
            io_ref=ArduinoLedRef(field="supported_commands", update_period=ONCE),
            description="Commands this panel dialect supports: 00, 01, 11..15",
        )
        self.panel_profile = AttrR(
            String(),
            io_ref=ArduinoLedRef(field="panel_profile", update_period=ONCE),
            description="Panel the host believes it is driving: panel91 v1",
        )
        self.led_count = AttrR(
            Int(),
            io_ref=ArduinoLedRef(field="led_count", update_period=ONCE),
            description="LEDs in the installed panel (91 for panel91 v1)",
        )
        self.io.bind(self)

    async def initialise(self) -> None:
        """Put the outputs in their dark state before admitting PVA commands.

        ``00`` blanks without emitting a camera trigger, so a service started
        after an interrupted session cannot leave a pattern illuminated.
        Refuse startup unless the blank is echoed and the sketch identifies
        itself as the 91-LED MM ASCII implementation.
        """
        outcome = await self.service.all_off()
        if outcome.status is not CommandStatus.ECHOED:
            raise RuntimeError(
                "Arduino LED startup blank was not echoed: "
                f"{outcome.status.value}: {outcome.detail}"
            )
        await self.service.verify_firmware()


def build_controller(service: ArduinoLedService) -> ArduinoLedController:
    """Build the controller without starting a transport."""
    return ArduinoLedController(service)


async def _serve(
    controller: ArduinoLedController,
    service: ArduinoLedService,
    prefix: str,
) -> int:
    """Serve *controller* over PVA and blank the outputs when the loop ends."""
    from fastcs.transports.epics.pva.transport import EpicsPVATransport
    from redsun.services import serve_controller

    try:
        await serve_controller(controller, EpicsPVATransport(), prefix=prefix)
    finally:
        await service.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the Arduino LED controller until the session or Ctrl+C stops it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="PVA prefix for this service")
    parser.add_argument(
        "--port", required=True, help="serial port the sketch is attached to"
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=BAUDRATE,
        help="serial line speed (9600 in the validated setup)",
    )
    options = parser.parse_args(argv)

    try:
        profile = ArduinoLedProfile(
            port=options.port,
            baudrate=options.baudrate,
        )
        service = ArduinoLedService.open(profile)
    except (OSError, ValueError) as error:
        print(f"arduino-led: {error}", file=sys.stderr)
        return 1

    controller = build_controller(service)
    try:
        return asyncio.run(_serve(controller, service, options.prefix))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
