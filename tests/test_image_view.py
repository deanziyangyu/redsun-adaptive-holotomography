"""Tests for the live image view and its napari application hook.

The suite is headless: no test constructs a ``QApplication``, a napari viewer
or a widget. The layer manager is exercised through the two protocols the
donor view defines, with an in-memory viewer double and a counting layer
controller, and the session declaration is read the way Redsun reads it.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from bluesky.protocols import Reading
from redsun import Layer
from redsun.qt import Qt, QtHook
from redsun.session._declarations import check, read_hooks

from redsun_aht.configurations.camera import CameraSession
from redsun_aht.contracts.devices import Roi
from redsun_aht.hooks import NapariApplication
from redsun_aht.view.image import (
    DetectorLayerController,
    DetectorLayers,
    DetectorLayerSpec,
    ImageView,
    bounds_from_roi,
    layer_key,
    layer_setting,
    layer_specs,
    roi_from_bounds,
)
from redsun_aht.view.preview import (
    PreviewFrame,
    PreviewImage,
    PreviewLayerController,
    PreviewViewer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _Layer:
    """Record what a napari image layer would have shown."""

    def __init__(self, data: NDArray[Any]) -> None:
        self.data = data
        self.name = ""
        self.scale: Any = (1.0, 1.0)
        self.colormap: Any = "gray"
        self.refreshes = 0

    def refresh(self) -> None:
        self.refreshes += 1


class _Layers:
    """A layer list keyed by each layer's current name."""

    def __init__(self) -> None:
        self.items: list[_Layer] = []

    def add(self, layer: _Layer) -> None:
        self.items.append(layer)

    def __contains__(self, name: str) -> bool:
        return any(layer.name == name for layer in self.items)

    def __getitem__(self, name: str) -> _Layer:
        return next(layer for layer in self.items if layer.name == name)

    def remove(self, name: str) -> None:
        layer = self[name]
        self.items.remove(layer)


class _Viewer:
    """A viewer double that records the layers added to it."""

    def __init__(self) -> None:
        self._layers = _Layers()

    @property
    def layers(self) -> _Layers:
        return self._layers

    def add_image(self, data: NDArray[Any], *, name: str) -> _Layer:
        layer = _Layer(data)
        layer.name = name
        self._layers.add(layer)
        return layer


class _CountingController(DetectorLayerController):
    """A controller that counts the frames the manager lets through."""

    def __init__(self, viewer: PreviewViewer, *, name: str) -> None:
        super().__init__(viewer, name=name)
        self.calls = 0

    def refresh(self, image: PreviewImage | None = None) -> bool:
        self.calls += 1
        return super().refresh(image)


def _controller_factory(viewer: PreviewViewer, name: str) -> PreviewLayerController:
    return _CountingController(viewer, name=name)


def _manager(viewer: _Viewer) -> DetectorLayers:
    return DetectorLayers(viewer, controller_factory=_controller_factory)


def test_layer_specs_are_pure_and_keyed_by_detector() -> None:
    """Map the presenter's specs without importing or constructing a viewer."""
    mapped = layer_specs(
        {
            "camera1": {"shape": (4, 6), "dtype": "uint16"},
            "camera2": {"shape": (2, 3), "dtype": np.dtype("uint8")},
        }
    )

    assert set(mapped) == {"camera1", "camera2"}
    assert mapped["camera1"] == DetectorLayerSpec(
        key="camera1", name="camera1", shape=(4, 6), dtype="uint16"
    )
    assert mapped["camera2"].shape == (2, 3)
    assert mapped["camera2"].dtype == "uint8"
    assert layer_key("camera1") != layer_key("camera2")


@pytest.mark.parametrize(
    "spec",
    [
        {"shape": (0, 2), "dtype": "uint8"},
        {"shape": (2,), "dtype": "uint8"},
        {"shape": (2, 2)},
        {"shape": (2, 2), "dtype": "not-a-dtype"},
    ],
    ids=["non-positive", "one-dimensional", "no-dtype", "bad-dtype"],
)
def test_an_unusable_layer_spec_is_refused(spec: dict[str, Any]) -> None:
    """Refuse a detector spec that cannot size or type a layer."""
    with pytest.raises(ValueError):
        layer_specs({"camera1": spec})


def test_two_detectors_get_two_distinct_layer_keys() -> None:
    """Create one independently named layer for each detector."""
    viewer = _Viewer()
    manager = _manager(viewer)

    manager.setup(
        {
            "camera1": {"shape": (4, 6), "dtype": "uint8"},
            "camera2": {"shape": (2, 3), "dtype": "uint16"},
        }
    )

    assert manager.layer_keys == ("camera1", "camera2")
    assert set(manager.layer_keys) == {"camera1", "camera2"}
    assert viewer.layers["camera1"].data.shape == (4, 6)
    assert viewer.layers["camera2"].data.dtype == np.uint16


def test_a_repeated_sequence_does_not_refresh_a_layer() -> None:
    """Offer one frame twice and let only the first reach the controller."""
    viewer = _Viewer()
    manager = _manager(viewer)
    manager.setup({"camera1": {"shape": (2, 2), "dtype": "uint8"}})
    frame = PreviewFrame(image=np.ones((2, 2), dtype=np.uint8), sequence=7)
    readings: dict[str, Reading[Any]] = {
        "camera1-buffer": {"value": frame, "timestamp": 1.0}
    }

    manager.update_layers(readings)
    manager.update_layers(readings)

    controller = manager.controller("camera1")
    assert isinstance(controller, _CountingController)
    assert controller.calls == 1
    assert viewer.layers["camera1"].refreshes == 1


def test_a_raw_frame_takes_the_event_timestamp_as_its_sequence() -> None:
    """Skip the same Event document offered twice, without a preview frame."""
    viewer = _Viewer()
    manager = _manager(viewer)
    manager.setup({"camera1": {"shape": (2, 2), "dtype": "uint8"}})
    readings: dict[str, Reading[Any]] = {
        "camera1-buffer": {
            "value": np.ones((2, 2), dtype=np.uint8),
            "timestamp": 12.5,
        }
    }

    manager.update_layers(readings)
    manager.update_layers(readings)

    controller = manager.controller("camera1")
    assert isinstance(controller, _CountingController)
    assert controller.calls == 1


def test_configuration_edits_rename_and_restyle_a_layer() -> None:
    """Apply the layer name, scale and colormap a detector reports."""
    viewer = _Viewer()
    manager = _manager(viewer)
    manager.setup({"camera1": {"shape": (4, 6), "dtype": "uint8"}})

    manager.on_new_configuration("camera1", "camera1-name", "Left camera")
    assert "camera1" not in viewer.layers
    layer = viewer.layers["Left camera"]

    manager.on_new_configuration("camera1", "camera1-scale", 0.5)
    assert layer.scale == (0.5, 0.5)

    manager.on_new_configuration("camera1", "camera1-colormap", "magma")
    assert layer.colormap == "magma"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("camera1-name", "name"),
        ("camera1-scale", "scale"),
        ("camera1-colormap", "colormap"),
        ("camera1-roi", "roi"),
        ("scale", "scale"),
        ("camera1-exposure", None),
        ("camera2-name", None),
    ],
)
def test_a_layer_setting_is_recognised_by_its_canonical_key(
    name: str, expected: str | None
) -> None:
    """Recognise the four settings this view applies, and nothing else."""
    assert layer_setting("camera1", name) == expected


def test_a_dragged_rectangle_round_trips_through_sensor_bounds() -> None:
    """Turn a shape's corners into a ROI and back without leaving the sensor."""
    roi = roi_from_bounds(((1.2, 2.4), (4.8, 9.1)), (6, 10))

    assert roi == Roi(2, 1, 7, 4)
    assert bounds_from_roi(roi) == ((1.0, 2.0), (5.0, 9.0))


def test_the_image_view_and_napari_hook_are_declarable() -> None:
    """Resolve the view and the hook through the checks a session runs."""
    assert check(ImageView, Layer.VIEW, "img_widget", Qt) is ImageView

    hooks = read_hooks(CameraSession, CameraSession.hook_points)
    providers = [declaration.cls for declaration in hooks[QtHook.CREATE_APPLICATION]]

    assert providers == [NapariApplication]


def test_importing_hooks_does_not_import_napari() -> None:
    """Keep napari out of a headless import; only the hook may load it."""
    code = (
        "import sys\n"
        "import redsun_aht.hooks\n"
        "import redsun_aht.view.image\n"
        "print(','.join(name for name in ('napari',) if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
