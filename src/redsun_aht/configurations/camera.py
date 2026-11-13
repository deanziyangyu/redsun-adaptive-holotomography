"""The editable PVA camera session.

A session is a container: the configuration file declares which services run,
and the class below names the devices, presenters and views it wants beside
them. This one is the first milestone of the reset: one or two Micro-Manager
cameras served over PVAccess, their settings editable, their live frames
forwarded as Event documents, and an acquisition page whose ``live_stream``
action records a finite run or keeps one open until it is released.

``DetectorPresenter`` still offers ``plan_map()["live_stream"]``;
:class:`~redsun_aht.presenter.acquisition.AcquisitionPresenter` gathers it,
routes it to the detector presenter, and keeps a readable control for a camera
role the profile does not declare. The page is
:class:`~redsun_aht.view.acquisition.AcquisitionView`, the built-in page with
the bundle's missing-device control installed. Live frames are drawn by
:class:`~redsun_aht.view.image.ImageView` in the napari viewer that
:class:`~redsun_aht.hooks.NapariApplication` makes, one layer and one preview
driver per detector. The session also wires ``sig_locks_changed`` to the
settings view, so the camera settings are disabled while a plan holds the
detector.

``--detectors 2`` selects :class:`DualCameraSession`, which adds a second
camera device and reads the two-camera profile. The packaged profiles are
simulated (Micro-Manager's ``DemoCamera``) so the session starts without
hardware; an instrument profile replaces the adapter and device of each camera
service and keeps the service names, which are also the detector ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from redsun import AsDevice, AsHook, AsPresenter, AsView, Declare, Serves
from redsun.qt import QtHook, QtSession

from ..device.mmcore import MMCamera  # noqa: TC001 - resolved by Redsun
from ..hooks import NapariApplication  # noqa: TC001 - resolved by Redsun
from ..presenter.acquisition import AcquisitionPresenter  # noqa: TC001 - Redsun
from ..presenter.detector import DetectorPresenter  # noqa: TC001 - resolved by Redsun
from ..view.acquisition import AcquisitionView  # noqa: TC001 - resolved by Redsun
from ..view.detector import DetectorView  # noqa: TC001 - resolved by Redsun
from ..view.image import ImageView
from .wiring import (
    DetectorPreview,
    DetectorPreviewSource,
    wire_acquisition,
    wire_detector,
    wire_locks,
    wire_preview,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import NoReturn

    from redsun import Link

__all__ = [
    "CAMERA_PROFILE",
    "DUAL_CAMERA_PROFILE",
    "CameraSession",
    "DualCameraSession",
    "build_camera_session",
    "run_camera",
]

_PROFILES = Path(__file__).resolve().parents[1] / "profiles"

CAMERA_PROFILE = _PROFILES / "camera.yaml"
"""Profile for the one-camera session."""

DUAL_CAMERA_PROFILE = _PROFILES / "camera-dual.yaml"
"""Profile for the two-camera session."""


class CameraSession(QtSession):
    """One editable Micro-Manager camera, its settings view and live stream.

    The camera service is named ``camera1`` and the device takes its prefix
    from it, so the detector id, the service name and the device name agree.
    The presenter's ``live_stream`` plan accepts that device, one of the
    session's detectors, or both.

    The detector presenter is declared before the acquisition presenter, so the
    plan sources are gathered with the detector first and the acquisition
    presenter's routed action last; the page then runs the routed action.
    """

    config = CAMERA_PROFILE

    napari: Annotated[AsHook[NapariApplication], Serves(QtHook.CREATE_APPLICATION)]
    camera1: Annotated[AsDevice[MMCamera], Declare(service="camera1")]
    det_ctrl: AsPresenter[DetectorPresenter]
    acq_ctrl: AsPresenter[AcquisitionPresenter]
    det_widget: AsView[DetectorView]
    acq_widget: AsView[AcquisitionView]
    img_widget: AsView[ImageView]
    preview: DetectorPreview

    def wire(self) -> Iterator[Link]:
        """Link the presenters, their views, the preview and the plan locks.

        Yields
        ------
        Link
            The detector links (including the image view), the presenter's
            frame signal into the preview buffers, the acquisition page paired
            with its presenter, and the lock signal that disables the settings
            of a held detector. The session holds the buffers as ``preview``
            and gives the image view one preview driver per detector, so the
            newest frame of each camera is drawn on its own layer.
        """
        self.preview = DetectorPreview()
        yield from wire_detector(self.det_ctrl, self.det_widget, self.img_widget)
        yield from wire_preview(self.det_ctrl, self.preview)
        yield from wire_acquisition(self.acq_ctrl, self.acq_widget)
        yield from wire_locks(self.acq_ctrl, self.det_widget)
        if isinstance(self.img_widget, ImageView):
            for detector in self.det_ctrl.detectors:
                self.img_widget.attach_preview(
                    detector, DetectorPreviewSource(self.preview, detector)
                )


class DualCameraSession(CameraSession):
    """Two editable Micro-Manager cameras, for the two-detector live stream.

    Both cameras satisfy the presenter's device protocol, so the settings view
    shows one tab per detector and the live plan can select either or both.
    """

    config = DUAL_CAMERA_PROFILE

    camera2: Annotated[AsDevice[MMCamera], Declare(service="camera2")]


def build_camera_session(
    profile: Path | None = None,
    *,
    detectors: int = 1,
) -> CameraSession:
    """Return an unbuilt camera session.

    Parameters
    ----------
    profile : Path | None
        A session profile to use instead of the packaged one. It must declare
        the services the chosen session declares devices for.
    detectors : int
        ``1`` for :class:`CameraSession`, ``2`` for
        :class:`DualCameraSession`.

    Returns
    -------
    CameraSession
        The session, not built, so a caller can inspect it before ``run``.

    Raises
    ------
    ValueError
        If *detectors* is neither ``1`` nor ``2``.
    """
    if detectors not in (1, 2):
        raise ValueError("detectors must be 1 or 2")
    if profile is None:
        session_type: type[CameraSession] = (
            DualCameraSession if detectors == 2 else CameraSession
        )
        return session_type()

    resolved = profile.expanduser()
    if detectors == 2:

        class ConfiguredDualCamera(DualCameraSession):
            """The two-camera session in a caller-supplied profile."""

            config = resolved

        return ConfiguredDualCamera()

    class ConfiguredCamera(CameraSession):
        """The one-camera session in a caller-supplied profile."""

        config = resolved

    return ConfiguredCamera()


def run_camera(argv: Sequence[str]) -> NoReturn:
    """Run the editable camera session selected on the command line.

    Parameters
    ----------
    argv : Sequence[str]
        Arguments after ``aht session camera --``: ``--detectors {1,2}`` and
        an optional ``--profile PATH``.
    """
    parser = argparse.ArgumentParser(
        prog="aht session camera",
        description=__doc__,
    )
    parser.add_argument(
        "--detectors",
        type=int,
        choices=(1, 2),
        default=1,
        help="detectors to declare: one camera service, or two",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="session profile to use instead of the packaged one",
    )
    options = parser.parse_args(list(argv))
    build_camera_session(options.profile, detectors=options.detectors).run()
