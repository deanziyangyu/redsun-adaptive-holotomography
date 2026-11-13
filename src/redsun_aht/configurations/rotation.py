"""Redsun motor widget for the unified MCU rotation axis."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Annotated, ClassVar, cast

from redsun import (
    AsDevice,
    AsPresenter,
    AsService,
    AsView,
    Declare,
    Launch,
    links_between,
)
from redsun.presenter import PositionerPresenter  # noqa: TC002 - resolved by Redsun
from redsun.qt import QtSession
from redsun.view.qt.builtins import PositionerView  # noqa: TC002 - Redsun declaration

from redsun_aht.device.pva_rotation import PvaRotationMotor  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import NoReturn

    from redsun import Link


class RotationSession(QtSession):
    """Positioner widget connected to one separately served unified MCU."""

    config: ClassVar[dict[str, object]] = {
        "session": "rotation",
        "strict": True,
        "services": {"transport": "pv-access"},
        "presenters": {"motor_ctrl": {}},
        "views": {"motor_widget": {"steps": [0.2, 1.0, 5.0, 10.0, 45.0]}},
    }
    rotation: Annotated[AsDevice[PvaRotationMotor], Declare(prefix="TLMCU:")]
    motor_ctrl: AsPresenter[PositionerPresenter]
    motor_widget: AsView[PositionerView]

    def wire(self) -> Iterator[Link]:
        """Connect the built-in jog and stop controls to the presenter."""
        yield from links_between(self.motor_widget, self.motor_ctrl)


def build_rotation_session(port: str | None = None) -> RotationSession:
    """Use an existing TLMCU service, or launch its sole owner on *port*."""
    if port is None:
        return RotationSession()
    if not port.strip() or "<" in port or ">" in port:
        raise ValueError("an explicit serial endpoint is required")
    annotations = {
        "mcu_service": Annotated[
            AsService,
            Launch(
                "redsun_aht.services.unified_mcu",
                prefix="TLMCU:",
                args=("--prefix", "TLMCU:", "--port", port),
                stop_timeout=30,
            ),
        ]
    }
    session_type = type(
        "ManagedRotationSession", (RotationSession,), {"__annotations__": annotations}
    )
    return cast("RotationSession", session_type())


def run_rotation(argv: Sequence[str]) -> NoReturn:
    """Run the motor widget against a shared or managed unified MCU service."""
    parser = argparse.ArgumentParser(prog="aht session rotation")
    parser.add_argument(
        "--port", help="launch the unified MCU owner on this serial port"
    )
    options = parser.parse_args(list(argv))
    build_rotation_session(options.port).run()
