"""HoloSPIM operator desktop over the devices declared in its profile."""

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, NoReturn

from redsun import AsHook, AsPresenter, AsView, Link, Serves, links_between
from redsun.presenter import PositionerPresenter
from redsun.qt import QtHook, QtSession
from redsun.view.qt.builtins import PositionerView

from ..hooks import NapariApplication
from ..presenter.acquisition import AcquisitionPresenter
from ..presenter.detector import DetectorPresenter
from ..presenter.led_patterns import LedPatternPresenter
from ..view.acquisition import AcquisitionView
from ..view.detector import DetectorView
from ..view.image import ImageView
from ..view.led_patterns import LedPatternView
from .profile import read_instrument_profile
from .wiring import (
    DetectorPreview, DetectorPreviewSource, wire_acquisition, wire_detector,
    wire_led_patterns, wire_locks, wire_preview,
)


class HolospimSession(QtSession):
    """Camera, acquisition, positioner and 91-LED preview for HoloSPIM."""

    config: ClassVar[dict[str, object]] = {
        "presenters": {
            "det_ctrl": {"timeout": 2.0},
            "acq_ctrl": {},
            "positioner_ctrl": {},
            "led_pattern_ctrl": {"panel": "holospim"},
        },
        "views": {
            "det_widget": {}, "acq_widget": {}, "img_widget": {},
            "positioner_widget": {}, "led_pattern_widget": {},
        },
    }

    napari: Annotated[AsHook[NapariApplication], Serves(QtHook.CREATE_APPLICATION)]
    det_ctrl: AsPresenter[DetectorPresenter]
    acq_ctrl: AsPresenter[AcquisitionPresenter]
    positioner_ctrl: AsPresenter[PositionerPresenter]
    led_pattern_ctrl: AsPresenter[LedPatternPresenter]
    det_widget: AsView[DetectorView]
    acq_widget: AsView[AcquisitionView]
    img_widget: AsView[ImageView]
    positioner_widget: AsView[PositionerView]
    led_pattern_widget: AsView[LedPatternView]
    preview: DetectorPreview

    def wire(self) -> Iterator[Link]:
        self.preview = DetectorPreview()
        yield from wire_detector(self.det_ctrl, self.det_widget, self.img_widget)
        yield from wire_preview(self.det_ctrl, self.preview)
        yield from wire_acquisition(self.acq_ctrl, self.acq_widget)
        yield from wire_locks(self.acq_ctrl, self.det_widget)
        yield from links_between(self.positioner_widget, self.positioner_ctrl)
        yield from wire_led_patterns(self.led_pattern_ctrl, self.led_pattern_widget)
        if isinstance(self.img_widget, ImageView):
            for name in self.det_ctrl.detectors:
                self.img_widget.attach_preview(name, DetectorPreviewSource(self.preview, name))


def run_holospim(argv: Sequence[str]) -> NoReturn:
    parser = argparse.ArgumentParser(prog="aht session holospim")
    parser.add_argument("--profile", type=Path, help="Complete HoloSPIM session profile")
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--mock", action="store_true", help="Override the profile to use mock devices")
    args = parser.parse_args(list(argv))
    packaged = Path(__file__).resolve().parents[1] / "profiles" / "holospim.yaml"
    try:
        config = [read_instrument_profile(path) for path in (args.profile or packaged, *args.overlay)]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.mock:
        config.append({"mock": True})
    HolospimSession(config).run()
