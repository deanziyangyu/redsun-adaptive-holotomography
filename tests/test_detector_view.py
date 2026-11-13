"""Tests for the detector view's import, annotation and pure-logic surfaces.

The suite is headless: constructing a Qt widget needs a ``QApplication``, and
no test here may create one. What is checked is what a headless environment can
honestly check - that importing the module creates no application, that the
widget classes and signals the session connects against exist, that ``setup``
is annotated with the presenter protocol it reads, and that the ROI and
descriptor arithmetic behaves before any widget sees it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_type_hints

import pytest
from psygnal import Signal
from qtpy import QtWidgets
from redsun.qt import Dock

from redsun_aht.contracts.devices import Roi
from redsun_aht.presenter.detector import DescribesDetectors
from redsun_aht.view.detector import (
    DetectorView,
    RoiPanel,
    SettingsControlWidget,
    clamp_roi,
    detector_key,
    sensor_and_roi,
)

if TYPE_CHECKING:
    from bluesky.protocols import Reading


def test_importing_the_view_creates_no_application() -> None:
    """Importing the module must not construct a ``QApplication``."""
    assert QtWidgets.QApplication.instance() is None


def test_the_view_widgets_and_signals_exist() -> None:
    """Define the widget classes, placement and signals the session connects to."""
    assert issubclass(DetectorView, QtWidgets.QWidget)
    assert issubclass(SettingsControlWidget, QtWidgets.QWidget)
    assert issubclass(RoiPanel, QtWidgets.QWidget)
    assert DetectorView.placement == Dock("right")
    assert isinstance(DetectorView.sig_property_changed, Signal)
    assert str(DetectorView.sig_property_changed.signature) == (
        "(p0: str, p1: str, p2: object, /)"
    )
    assert isinstance(DetectorView.sig_roi_selection, Signal)
    assert isinstance(DetectorView.sig_roi_edited, Signal)
    assert isinstance(RoiPanel.sig_roi_requested, Signal)
    assert isinstance(RoiPanel.sig_roi_edited, Signal)
    assert isinstance(RoiPanel.sig_selection_toggled, Signal)


def test_setup_is_annotated_with_the_presenter_protocol() -> None:
    """Annotate ``setup`` with the protocol a session injects the presenter as."""
    hints = get_type_hints(DetectorView.setup)
    assert hints["detectors"] is DescribesDetectors


def test_a_presenter_shaped_object_satisfies_the_protocol() -> None:
    """Accept any object that reports descriptors and readings."""

    class Describes:
        def detector_descriptors(self) -> dict[str, Any]:
            return {}

        def detector_readings(self) -> dict[str, Any]:
            return {}

    assert isinstance(Describes(), DescribesDetectors)
    assert not isinstance(object(), DescribesDetectors)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("cam-exposure", ("cam", "exposure")),
        ("cam-a-buffer", ("cam", "a-buffer")),
        ("camera-Gain", ("camera", "Gain")),
        ("cam", None),
        ("", None),
    ],
    ids=["exposure", "nested", "property", "no-separator", "empty"],
)
def test_a_descriptor_key_splits_into_detector_and_setting(
    key: str, expected: tuple[str, str] | None
) -> None:
    """Split a flat ``name-property`` key, or refuse one without a separator."""
    assert detector_key(key) == expected


@pytest.mark.parametrize(
    ("sensor", "roi", "expected"),
    [
        ([6, 4], "0,0,6,4", ((6, 4), Roi(0, 0, 6, 4))),
        ([6, 4], "1,1,3,2", ((6, 4), Roi(1, 1, 3, 2))),
        (None, "0,0,6,4", None),
        ([6, 4], None, None),
        ([6, 4], "not-a-roi", None),
        ("wide", "0,0,6,4", None),
    ],
    ids=["whole", "cropped", "no-sensor", "no-roi", "bad-roi", "bad-sensor"],
)
def test_a_sensor_and_roi_are_extracted_from_flat_readings(
    sensor: object, roi: object, expected: tuple[tuple[int, int], Roi] | None
) -> None:
    """Read the detector's sensor and ROI out of a merged configuration read."""
    readings: dict[str, Reading[Any]] = {}
    if sensor is not None:
        readings["cam-sensor_size"] = {"value": sensor, "timestamp": 0.0}
    if roi is not None:
        readings["cam-roi"] = {"value": roi, "timestamp": 0.0}

    assert sensor_and_roi(readings) == expected


@pytest.mark.parametrize(
    ("roi", "expected"),
    [
        (Roi(0, 0, 6, 4), Roi(0, 0, 6, 4)),
        (Roi(1, 1, 3, 2), Roi(1, 1, 3, 2)),
        (Roi(4, 2, 9, 9), Roi(4, 2, 2, 2)),
        (Roi(5, 3, 6, 4), Roi(5, 3, 1, 1)),
        (Roi(9, 9, 1, 1), Roi(5, 3, 1, 1)),
    ],
    ids=["whole", "inside", "past-edge", "corner", "origin-outside"],
)
def test_a_dragged_roi_is_kept_on_the_sensor(roi: Roi, expected: Roi) -> None:
    """Move and shrink a region until it lies inside a 6 by 4 sensor."""
    assert clamp_roi(roi, (6, 4)) == expected
