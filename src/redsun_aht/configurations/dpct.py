"""The DPCT session: one camera, one stage and the 91-LED v1 panel.

This session adds what a four-pattern DPCT run needs to the editable camera
session: an ``mmcore-stage`` device for the measured position each frame
records, an ``arduino-led`` device for illumination,
and :class:`~redsun_aht.presenter.dpct.DpctPresenter`, which offers the
``dpct`` plan. Everything else — the camera, its settings view, the live stream
and the acquisition page — is :class:`~redsun_aht.configurations.camera.
CameraSession`, so the operator can watch frames arrive while the plan runs.

The plan opens its own run, so it must be launched on its own: the acquisition
page's Run button does that, and the page lists ``dpct`` only when the profile
declares both a camera and an illumination device.
:class:`~redsun_aht.presenter.dpct.DpctPresenter` names whichever of the two the
session lacks rather than failing to build.

The packaged profile is simulated except for the serial port: Micro-Manager's
``DemoCamera`` serves the camera and the stage, while the Arduino service opens
a real endpoint. ``profiles/dpct.yaml`` therefore carries a placeholder port
that must name the port the sketch is attached to before this session is
started on the instrument; see the comment beside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from redsun import AsDevice, AsPresenter, Declare

from ..device.arduino.device import ArduinoLedDevice  # noqa: TC001 - resolved by Redsun
from ..device.mmcore import MMStage  # noqa: TC001 - resolved by Redsun
from ..presenter.dpct import DpctPresenter  # noqa: TC001 - resolved by Redsun
from .camera import CameraSession

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import NoReturn

__all__ = [
    "DPCT_PROFILE",
    "DpctSession",
    "build_dpct_session",
    "run_dpct",
]

_PROFILES = Path(__file__).resolve().parents[1] / "profiles"

DPCT_PROFILE = _PROFILES / "dpct.yaml"
"""Profile for the DPCT session: camera, stage and Arduino LED panel."""


class DpctSession(CameraSession):
    """One camera, one stage and the panel, with the ``dpct`` plan on the page.

    The three services are named after the devices they serve — ``camera1``,
    ``stage1`` and ``led1`` — so the profile, the device, the detector and the
    run record agree about one identity.

    The stage is declared even though the plan never moves it: every frame
    records the measured position, and a run without a stage can only record
    that it had none. The default plan waits for a fresh free-running camera
    frame after each illumination pattern command; the panel emits no TTL.

    Notes
    -----
    ``wire`` is the camera session's. The DPCT presenter has no view of its own:
    its plan reaches the operator through the acquisition page's plan list, and
    its outcome through the log and ``sig_outcome``.
    """

    config = DPCT_PROFILE

    stage1: Annotated[AsDevice[MMStage], Declare(service="stage1")]
    led1: Annotated[AsDevice[ArduinoLedDevice], Declare(service="led1")]
    dpct_ctrl: AsPresenter[DpctPresenter]


def build_dpct_session(profile: Path | None = None) -> DpctSession:
    """Return an unbuilt DPCT session.

    Parameters
    ----------
    profile : Path | None
        A session profile to use instead of the packaged one. It must declare
        the ``camera1``, ``stage1`` and ``led1`` services and the presenter and
        view entries this session annotates.

    Returns
    -------
    DpctSession
        The session, not built, so a caller can inspect it before ``run``.
    """
    if profile is None:
        return DpctSession()

    resolved = profile.expanduser()

    class ConfiguredDpct(DpctSession):
        """The DPCT session in a caller-supplied profile."""

        config = resolved

    return ConfiguredDpct()


def run_dpct(argv: Sequence[str]) -> NoReturn:
    """Run the DPCT session.

    Parameters
    ----------
    argv : Sequence[str]
        Arguments after ``aht session dpct --``: an optional ``--profile PATH``.
    """
    parser = argparse.ArgumentParser(
        prog="aht session dpct",
        description=__doc__,
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="session profile to use instead of the packaged one",
    )
    options = parser.parse_args(list(argv))
    build_dpct_session(options.profile).run()
