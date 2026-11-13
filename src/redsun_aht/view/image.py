"""Live image view over the embedded napari viewer.

Adapted from ``redsun_searching_is_imaging/view/image.py`` at donor revision
``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed; copyright
the Redsun contributors. The arrangement of a ``ViewerModel`` with an embedded
``QtViewer``, one image layer per detector, the ROI signal and the layer-data
rules are the donor's.

Changed for :mod:`redsun_aht`:

- The viewer and its Qt widget come from
  :class:`redsun_aht.hooks.NapariApplication`, which makes one for the session.
  The view hosts them instead of building a second viewer, and it creates its
  own only when no hook ran.
- Layers are managed by :class:`DetectorLayers`, a Qt-free and napari-free
  object that talks to :class:`redsun_aht.view.preview.PreviewLayerController`
  through the viewer protocol. That keeps the layer mapping, sequencing and
  configuration edits testable without constructing a viewer.
- ``update_layers`` accepts the presenter's ``<detector>-buffer`` readings.
  A reading whose value carries an image and a sequence is used as it is;
  otherwise the Event timestamp becomes the sequence, so the controller skips
  the same document twice. ``DetectorLayers`` also skips a repeated sequence
  before it reaches the controller.
- ``on_new_configuration`` applies the ``<detector>-name``,
  ``<detector>-scale``, ``<detector>-colormap`` and ``<detector>-roi`` settings
  to the detector's layer. The layer key stays the detector name, so a renamed
  layer is still updated by the same controller entry.
- The selection box is a napari shapes layer. Its last rectangle is read back
  into a :class:`redsun_aht.contracts.devices.Roi` and announced on
  ``sig_roi_drawn``, which the detector view's ROI panel follows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from psygnal import Signal
from qtpy import QtCore, QtWidgets
from redsun import Placement, slot
from redsun.log import Loggable
from redsun.qt import Central

from redsun_aht.contracts.devices import Roi

from ..hooks import current_application
from .preview import (
    PreviewFrame,
    PreviewImage,
    PreviewLayerController,
    PreviewViewer,
)
from .preview_driver import PreviewDriver, PreviewSource

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bluesky.protocols import Reading
    from numpy.typing import NDArray

__all__ = [
    "DetectorLayerController",
    "DetectorLayerSpec",
    "DetectorLayers",
    "ImageView",
    "bounds_from_roi",
    "layer_key",
    "layer_setting",
    "layer_specs",
    "place",
    "roi_from_bounds",
]

ControllerFactory = Callable[[PreviewViewer, str], PreviewLayerController]
"""Build one layer controller for a viewer and a layer name."""

_LAYER_SETTINGS = ("name", "scale", "colormap", "roi")


@runtime_checkable
class EditableLayer(Protocol):
    """The parts of a napari image layer the configuration edits change."""

    data: NDArray[Any]
    name: str
    scale: Any
    colormap: Any

    def refresh(self) -> None:
        """Redraw the layer."""
        ...


@runtime_checkable
class DescribesLayers(Protocol):
    """The part of a presenter image layers are created from."""

    def detector_layer_specs(self) -> dict[str, dict[str, Any]]:
        """Return the layer spec of every detector, keyed by detector name."""
        ...


@dataclass(frozen=True, slots=True)
class DetectorLayerSpec:
    """The layer one detector draws its frames on.

    Attributes
    ----------
    key
        The stable key ``DetectorLayers`` stores the layer under. It is the
        detector name, so two detectors never share a key even when their
        display names are edited.
    name
        The display name the layer starts with.
    shape
        The sensor as ``(height, width)``.
    dtype
        The readout dtype, as a numpy dtype name.
    """

    key: str
    name: str
    shape: tuple[int, int]
    dtype: str


def layer_key(detector: str) -> str:
    """Return the stable layer key of *detector*.

    Raises
    ------
    ValueError
        If *detector* is empty, since no layer can be addressed by it.
    """
    if type(detector) is not str or not detector.strip():
        raise ValueError("a detector name must be a nonempty string")
    return detector


def layer_specs(
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, DetectorLayerSpec]:
    """Map ``detector_layer_specs`` output to validated layer specs.

    Parameters
    ----------
    specs : Mapping[str, Mapping[str, Any]]
        The presenter's spec per detector, each naming ``shape`` and
        ``dtype``.

    Returns
    -------
    dict[str, DetectorLayerSpec]
        One spec per detector, keyed by the detector name.

    Raises
    ------
    ValueError
        If a detector has no two-dimensional positive shape or no readable
        dtype.
    """
    result: dict[str, DetectorLayerSpec] = {}
    for detector, spec in specs.items():
        key = layer_key(detector)
        try:
            height, width = (int(value) for value in spec["shape"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"detector {detector!r} has no two-dimensional shape"
            ) from error
        if height <= 0 or width <= 0:
            raise ValueError(
                f"detector {detector!r} has a non-positive sensor shape "
                f"{(height, width)}"
            )
        dtype = spec.get("dtype")
        if dtype is None:
            raise ValueError(f"detector {detector!r} declares no dtype")
        try:
            dtype_name = str(np.dtype(dtype))
        except TypeError as error:
            raise ValueError(
                f"detector {detector!r} declares an unreadable dtype {dtype!r}"
            ) from error
        result[key] = DetectorLayerSpec(key, detector, (height, width), dtype_name)
    return result


def layer_setting(detector: str, name: str) -> str | None:
    """Return the layer setting *name* addresses, or ``None``.

    A canonical ``"<detector>-<setting>"`` key and a bare setting name are
    both accepted, because a one-detector presenter may emit either.
    """
    if not detector or not name:
        return None
    prefix = f"{detector}-"
    setting = name.removeprefix(prefix) if name.startswith(prefix) else name
    return setting if setting in _LAYER_SETTINGS else None


def roi_from_bounds(
    bounds: tuple[tuple[float, float], tuple[float, float]],
    sensor: tuple[int, int],
) -> Roi:
    """Turn a rectangle's two corners into a ROI on the sensor.

    The corners are ``(y, x)`` as napari keeps them, and *sensor* is
    ``(height, width)``. The rectangle is rounded to whole pixels, clamped to
    the sensor, and at least one pixel wide and high.
    """
    height, width = sensor
    (y0, x0), (y1, x1) = bounds
    left = min(max(round(min(x0, x1)), 0), width - 1)
    top = min(max(round(min(y0, y1)), 0), height - 1)
    right = min(max(round(max(x0, x1)), left + 1), width)
    bottom = min(max(round(max(y0, y1)), top + 1), height)
    return Roi(left, top, right - left, bottom - top)


def bounds_from_roi(roi: Roi) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the ``(y, x)`` corners a selection box draws for *roi*."""
    return (
        (float(roi.y), float(roi.x)),
        (float(roi.y + roi.height), float(roi.x + roi.width)),
    )


def place(canvas: NDArray[Any], frame: NDArray[Any], origin: tuple[int, int]) -> bool:
    """Write *frame* into *canvas* with its top-left corner at *origin*.

    *origin* is ``(x, y)``. A frame the size of the canvas replaces it whole.
    Returns whether the frame fit; nothing is written when it does not.
    """
    x, y = origin
    height, width = frame.shape[:2]
    if (height, width) == canvas.shape[:2]:
        canvas[...] = frame
        return True
    if x < 0 or y < 0 or y + height > canvas.shape[0] or x + width > canvas.shape[1]:
        return False
    canvas[y : y + height, x : x + width] = frame
    return True


class DetectorLayerController(PreviewLayerController):
    """A preview controller whose detector layer can be renamed in place.

    A driver samples its controller for the whole session, so a layer renamed
    by a detector configuration change has to keep the same controller. The
    donor controller names its layer for the layer list to find; this subclass
    renames that layer and moves its own name with it.
    """

    def rename(self, name: str) -> None:
        """Rename the layer this controller owns, keeping its data.

        Raises
        ------
        ValueError
            If *name* is empty, since no layer can be addressed by it.
        """
        name = name.strip()
        if not name:
            raise ValueError("a layer name must be nonempty")
        if name == self._name:
            return
        if self._name in self._viewer.layers:
            layer = cast("Any", self._viewer.layers[self._name])
            layer.name = name
        self._name = name


def _make_controller(viewer: PreviewViewer, name: str) -> PreviewLayerController:
    """Return the default controller for one detector layer."""
    return DetectorLayerController(viewer, name=name)


def _frame_from_reading(reading: Reading[Any]) -> PreviewImage | None:
    """Return the preview frame one ``<detector>-buffer`` reading carries.

    A reading whose value is already a preview image is used unchanged. A raw
    array is given the Event timestamp in nanoseconds as its sequence, so the
    same document offered twice is skipped.
    """
    value: Any = reading["value"]
    if value is None:
        return None
    if isinstance(value, PreviewImage):
        return value
    sequence = max(round(float(reading["timestamp"]) * 1_000_000_000), 1)
    return PreviewFrame(image=np.asarray(value), sequence=sequence)


class DetectorLayers:
    """One image layer per detector, drawn through its layer controller.

    The manager is Qt-free and napari-free: it works against the
    :class:`~redsun_aht.view.preview.PreviewViewer` protocol, so a test can
    pass a small viewer double and a controller that counts refreshes.

    Parameters
    ----------
    viewer
        The viewer the layers are made in.
    controller_factory
        Builds a controller for a viewer and a layer name. The default makes
        :class:`~redsun_aht.view.preview.PreviewLayerController`.
    """

    def __init__(
        self,
        viewer: PreviewViewer,
        *,
        controller_factory: ControllerFactory = _make_controller,
    ) -> None:
        self._viewer = viewer
        self._controller_factory = controller_factory
        self._specs: dict[str, DetectorLayerSpec] = {}
        self._controllers: dict[str, PreviewLayerController] = {}
        self._names: dict[str, str] = {}
        self._sequences: dict[str, int] = {}
        self._rois: dict[str, Roi] = {}

    @property
    def layer_keys(self) -> tuple[str, ...]:
        """Return the key of every detector layer, in creation order."""
        return tuple(self._specs)

    def spec(self, detector: str) -> DetectorLayerSpec | None:
        """Return *detector*'s layer spec, or ``None`` before it is set up."""
        return self._specs.get(detector)

    def controller(self, detector: str) -> PreviewLayerController | None:
        """Return *detector*'s layer controller, or ``None`` before setup."""
        return self._controllers.get(detector)

    def setup(self, specs: Mapping[str, Mapping[str, Any]]) -> None:
        """Create an empty sensor-sized layer and a controller per detector.

        Calling it again for a detector already set up keeps its controller
        and layer, so a rebuilt view does not orphan either.
        """
        for key, spec in layer_specs(specs).items():
            self._specs[key] = spec
            self._names.setdefault(key, spec.name)
            self._rois.setdefault(key, Roi(0, 0, spec.shape[1], spec.shape[0]))
            self._controllers.setdefault(
                key, self._controller_factory(self._viewer, self._names[key])
            )
            if self._names[key] not in self._viewer.layers:
                self._viewer.add_image(
                    np.zeros(spec.shape, dtype=np.dtype(spec.dtype)),
                    name=self._names[key],
                )

    def refresh(self, detector: str, frame: PreviewImage) -> bool:
        """Draw *frame* on *detector*'s layer and return whether it changed.

        A frame carrying the same sequence as the last one drawn is ignored.
        A frame smaller than the sensor is placed at the detector's ROI, so a
        cropped readout occupies the region it was taken from and the rest of
        the sensor stays blank.
        """
        controller = self._controllers.get(detector)
        if controller is None:
            return False
        if self._sequences.get(detector) == frame.sequence:
            return False
        spec = self._specs.get(detector)
        image = frame.image
        if spec is not None and tuple(image.shape[:2]) != spec.shape:
            canvas = np.zeros(spec.shape, dtype=image.dtype)
            roi = self._rois.get(detector)
            if roi is None or not place(canvas, image, (roi.x, roi.y)):
                return False
            image = canvas
        changed = controller.refresh(PreviewFrame(image=image, sequence=frame.sequence))
        if changed:
            self._sequences[detector] = frame.sequence
        return changed

    def update_layers(self, readings: Mapping[str, Reading[Any]]) -> None:
        """Draw every ``<detector>-buffer`` reading and record its ROI.

        Readings that are not a detector buffer are ignored. A buffer whose
        value is ``None`` is skipped, as a monitor's first placeholder is.
        """
        for key, reading in readings.items():
            if key.endswith("-roi"):
                value = reading["value"]
                if isinstance(value, Roi):
                    self._rois[key.removesuffix("-roi")] = value
                continue
            if not key.endswith("-buffer"):
                continue
            frame = _frame_from_reading(reading)
            if frame is not None:
                self.refresh(key.removesuffix("-buffer"), frame)

    def on_new_configuration(self, detector: str, name: str, value: Any) -> None:
        """Apply a detector setting to its layer, when the layer reads it.

        ``name``, ``scale``, ``colormap`` and ``roi`` are this view's to
        apply; every other setting is ignored. A ROI is kept as the region a
        later cropped frame is placed in.
        """
        setting = layer_setting(detector, name)
        if setting is None:
            return
        if setting == "roi":
            if isinstance(value, Roi):
                self._rois[detector] = value
            elif isinstance(value, str):
                try:
                    self._rois[detector] = Roi.parse(value)
                except ValueError:
                    return
            return
        if setting == "name":
            self.rename(detector, str(value))
            return
        layer = self._layer(detector)
        if layer is None:
            return
        if setting == "scale":
            self._apply_scale(layer, value)
        elif setting == "colormap":
            layer.colormap = str(value)

    def rename(self, detector: str, name: str) -> None:
        """Rename *detector*'s layer, keeping its controller and frame."""
        new_name = name.strip()
        current = self._names.get(detector)
        if not new_name or current is None or new_name == current:
            return
        controller = self._controllers.get(detector)
        rename = getattr(controller, "rename", None)
        if callable(rename):
            # a driver keeps sampling this controller, so it renames in place
            try:
                rename(new_name)
            except ValueError:
                return
            self._names[detector] = new_name
            return
        existing = self._layer(detector)
        data = None if existing is None else existing.data
        if controller is not None:
            controller.close()
        self._names[detector] = new_name
        self._controllers[detector] = self._controller_factory(self._viewer, new_name)
        if data is not None:
            self._viewer.add_image(data, name=new_name)

    def close(self) -> None:
        """Remove every detector layer this manager made."""
        for controller in self._controllers.values():
            controller.close()
        self._controllers.clear()
        self._specs.clear()
        self._names.clear()
        self._sequences.clear()
        self._rois.clear()

    def _layer(self, detector: str) -> EditableLayer | None:
        """Return *detector*'s layer once its controller has made it."""
        controller = self._controllers.get(detector)
        if controller is None or controller.name not in self._viewer.layers:
            return None
        return cast("EditableLayer", self._viewer.layers[controller.name])

    @staticmethod
    def _apply_scale(layer: EditableLayer, value: Any) -> None:
        """Put a scalar or per-axis *value* on *layer*'s scale."""
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            try:
                values = (float(value),)
            except (TypeError, ValueError):
                return
        if len(values) == 1:
            values = values * 2
        if len(values) == 2:
            layer.scale = values


class ImageView(QtWidgets.QWidget, Loggable):
    """View for live image display in the session's napari viewer.

    The viewer model and its Qt widget come from
    :class:`~redsun_aht.hooks.NapariApplication`; the view reparents the Qt
    widget into its own layout with a layer-control panel beside it. When no
    hook ran, the view makes a standalone viewer instead.

    One image layer is created per detector from the presenter's
    ``detector_layer_specs``. Frames reach the layers through a
    :class:`~redsun_aht.view.preview.PreviewLayerController` and a
    :class:`~redsun_aht.view.preview_driver.PreviewDriver` attached for each
    detector, or immediately through :meth:`update_layers`.

    Each detector layer carries a selection box on a shapes layer. Dragging
    the box changes nothing on the camera: the box is announced on
    ``sig_roi_drawn`` and applied by whoever confirms it, after which it
    follows the region the camera reads.
    """

    placement: Placement = Central()

    sig_roi_drawn = Signal(str, object)
    """Emitted when a detector's selection box is dragged, with the detector
    name and the box as a :class:`~redsun_aht.contracts.devices.Roi` in sensor
    pixels. The box is hidden, and cannot be dragged, until
    :meth:`set_roi_selection` shows it."""

    def __init__(self, name: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.name = name
        application = current_application()
        self._viewer: Any
        if application is not None:
            try:
                self._viewer = application.viewer
                self._qt_viewer = application.qt_viewer
            except RuntimeError:
                self._viewer, self._qt_viewer = self._make_viewer()
                self._owns_viewer = True
            else:
                self._owns_viewer = False
        else:
            self._viewer, self._qt_viewer = self._make_viewer()
            self._owns_viewer = True
        self._layers = DetectorLayers(self._viewer)
        self._drivers: dict[str, PreviewDriver] = {}
        self._roi_layers: dict[str, Any] = {}
        self._build_layout()
        self.logger.info("Initialized")

    def setup(self, detectors: DescribesLayers) -> None:
        """Create one image layer, and its selection box, per detector."""
        specs = detectors.detector_layer_specs()
        self._layers.setup(specs)
        for detector in self._layers.layer_keys:
            spec = self._layers.spec(detector)
            if spec is not None:
                self._add_roi_layer(detector, spec.shape)

    def attach_preview(
        self,
        detector: str,
        source: PreviewSource,
        *,
        interval_ms: int = 200,
    ) -> None:
        """Refresh *detector*'s layer from *source* on a timer.

        Parameters
        ----------
        detector : str
            The detector whose layer is drawn.
        source : PreviewSource
            A source with a ``snapshot`` method returning the newest preview
            frame, or ``None``; a
            :class:`~redsun_aht.view.preview.LatestPreview` qualifies.
        interval_ms : int
            Refresh interval in milliseconds.
        """
        controller = self._layers.controller(detector)
        if controller is None:
            self.logger.warning(
                f"No image layer for detector {detector!r}; not starting a preview"
            )
            return
        previous = self._drivers.pop(detector, None)
        if previous is not None:
            previous.stop()
        driver = PreviewDriver(source, controller, interval_ms=interval_ms, parent=self)
        driver.start()
        self._drivers[detector] = driver

    @slot
    def update_layers(self, readings: Mapping[str, Reading[Any]]) -> None:
        """Draw the frames in one live Event's readings."""
        self._layers.update_layers(readings)

    @slot
    def on_new_configuration(self, detector: str, name: str, value: Any) -> None:
        """Apply a detector setting to its layer and its selection box."""
        self._layers.on_new_configuration(detector, name, value)
        setting = layer_setting(detector, name)
        if setting == "name":
            self._rename_roi_layer(detector, str(value))
        elif setting == "roi":
            try:
                self.set_roi_box(detector, Roi.parse(str(value)))
            except ValueError:
                return

    @slot
    def set_roi_selection(self, detector: str, enabled: bool) -> None:
        """Show the box on *detector*'s layer and let it be edited, or hide it."""
        layer = self._roi_layers.get(detector)
        if layer is None:
            return
        layer.visible = enabled
        if enabled:
            self._viewer.layers.selection = {layer}

    @slot
    def set_roi_box(self, detector: str, roi: Roi) -> None:
        """Put the box on *detector*'s layer over *roi*, without announcing it."""
        layer = self._roi_layers.get(detector)
        if layer is None:
            return
        layer.data = []
        self._add_rectangle(layer, roi)

    def shutdown(self) -> None:
        """Stop every preview, remove every layer, and close an owned viewer."""
        for driver in self._drivers.values():
            driver.stop()
        self._drivers.clear()
        self._layers.close()
        for layer in self._roi_layers.values():
            name = getattr(layer, "name", None)
            if isinstance(name, str) and name in self._viewer.layers:
                self._viewer.layers.remove(name)
        self._roi_layers.clear()
        if self._owns_viewer:
            self._qt_viewer.close()

    @staticmethod
    def _make_viewer() -> tuple[Any, QtWidgets.QWidget]:
        """Make a standalone viewer for a view built without the hook."""
        from napari._qt.qt_event_loop import get_qapp
        from napari._qt.qt_viewer import QtViewer
        from napari.components import ViewerModel

        get_qapp()
        viewer = ViewerModel(title="aht-viewer", ndisplay=2, order=(), axis_labels=())
        return viewer, QtViewer(viewer, show_welcome_screen=False)

    def _build_layout(self) -> None:
        """Put the layer controls and the viewer into this widget."""
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        for attribute in ("controls", "layerButtons", "layers", "viewerButtons"):
            panel = getattr(self._qt_viewer, attribute, None)
            if panel is not None:
                left_layout.addWidget(panel)
        left_panel.setLayout(left_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self._qt_viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setLayout(layout)

    def _add_roi_layer(self, detector: str, shape: tuple[int, int]) -> None:
        """Add the hidden shapes layer a detector's selection box lives on."""
        add_shapes = getattr(self._viewer, "add_shapes", None)
        if add_shapes is None:
            return
        try:
            layer = add_shapes(
                name=f"{detector} ROI",
                edge_color="yellow",
                face_color="transparent",
                edge_width=1,
                visible=False,
            )
            self._add_rectangle(layer, Roi(0, 0, shape[1], shape[0]))
            layer.events.data.connect(partial(self._on_roi_drawn, detector))
        except Exception as error:  # a viewer without shapes is still usable
            self.logger.warning(
                f"Could not add a selection box for {detector!r}: {error}"
            )
            return
        self._roi_layers[detector] = layer

    @staticmethod
    def _add_rectangle(layer: Any, roi: Roi) -> None:
        """Draw *roi* as the one rectangle of a shapes layer."""
        (top, left), (bottom, right) = bounds_from_roi(roi)
        layer.add_rectangles(
            np.array(
                [
                    [top, left],
                    [top, right],
                    [bottom, right],
                    [bottom, left],
                ]
            )
        )

    def _on_roi_drawn(self, detector: str, event: object | None = None) -> None:
        """Announce where the box on *detector*'s layer now stands."""
        layer = self._roi_layers.get(detector)
        spec = self._layers.spec(detector)
        if layer is None or spec is None:
            return
        data = list(layer.data)
        if not data:
            return
        rectangle = np.asarray(data[-1], dtype=float)
        if rectangle.ndim != 2 or rectangle.shape[0] < 2:
            return
        top_left = rectangle.min(axis=0)
        bottom_right = rectangle.max(axis=0)
        self.sig_roi_drawn.emit(
            detector,
            roi_from_bounds(
                (
                    (float(top_left[0]), float(top_left[1])),
                    (float(bottom_right[0]), float(bottom_right[1])),
                ),
                spec.shape,
            ),
        )

    def _rename_roi_layer(self, detector: str, name: str) -> None:
        """Rename a detector's selection box when its layer is renamed."""
        layer = self._roi_layers.get(detector)
        if layer is not None:
            layer.name = f"{name} ROI"
