"""Qt controls for detector settings and the ROI a detector reads.

Ported from ``redsun_searching_is_imaging/view/detector.py`` at donor revision
``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed; copyright
the Redsun contributors. The widget structure, the ROI panel's interaction and
the signals are the donor's.

Changed for :mod:`redsun_aht`:

- The ROI value type is :class:`redsun_aht.contracts.devices.Roi`, the frozen
  contract shared with the plan and storage packages, not the searching
  bundle's local ``common.Roi``. It travels as ``roi.as_text()``, because the
  contract type carries no ``__str__``.
- ``DescribesDetectors`` is this package's presenter protocol. It names the two
  methods the view calls, and nothing about the plan side.
- The descriptor split is a pure helper, :func:`detector_key`, and the sensor
  and ROI extraction is :func:`sensor_and_roi`, so the import, annotation and
  pure-logic surfaces can be checked without constructing a widget or a
  ``QApplication``. :func:`clamp_roi` keeps a dragged region on the sensor.
- A reading the tree cannot place is skipped with a warning, as in the donor; a
  sensor or ROI reading of the wrong shape leaves the panel out instead of
  failing the view.
- The module imports ``QtWidgets`` at import time and builds no widget until a
  caller constructs one. Importing it never creates an application.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from psygnal import Signal
from qtpy import QtWidgets
from redsun import Placement, slot
from redsun.log import Loggable
from redsun.qt import Dock
from redsun.utils.descriptors import parse_key
from redsun.view.qt.treeview import DescriptorTreeView

from redsun_aht.contracts.devices import Roi

# Imported at run time, not under ``TYPE_CHECKING``: a session evaluates
# ``setup``'s annotation when it injects the presenter, and a headless test
# resolves the same hint to pin the wiring.
from redsun_aht.presenter.detector import DescribesDetectors  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bluesky.protocols import Descriptor, Reading

__all__ = [
    "DetectorView",
    "RoiPanel",
    "SettingsControlWidget",
    "clamp_roi",
    "detector_key",
    "sensor_and_roi",
]


def detector_key(key: str) -> tuple[str, str] | None:
    """Split a ``"<detector>-<setting>"`` descriptor key, or return ``None``.

    The searching bundle's ``DescriptorTreeView`` keys carry a flat
    ``name-property`` form. A key without the separator belongs to no detector
    and is skipped rather than raising through a view build.
    """
    try:
        return parse_key(key)
    except ValueError:
        return None


def sensor_and_roi(
    readings: Mapping[str, Reading[Any]],
) -> tuple[tuple[int, int], Roi] | None:
    """Return a detector's ``(width, height)`` sensor and applied ROI.

    The readings are the flat output of a detector's ``read_configuration``.
    ``None`` is returned when either reading is absent or malformed, so a
    detector that does not report a sensor or a ROI simply has no ROI panel.
    """
    sensor = next(
        (reading for key, reading in readings.items() if key.endswith("-sensor_size")),
        None,
    )
    roi = next(
        (reading for key, reading in readings.items() if key.endswith("-roi")), None
    )
    if sensor is None or roi is None:
        return None
    try:
        width, height = (int(item) for item in sensor["value"])
        return (width, height), Roi.parse(str(roi["value"]))
    except (TypeError, ValueError):
        return None


def clamp_roi(roi: Roi, sensor: tuple[int, int]) -> Roi:
    """Return *roi* moved and shrunk until it lies inside *sensor*.

    The ROI panel's spin boxes are limited to the sensor, but a region dragged
    on an image can start outside it; this keeps the two editors showing one
    addressable region without constructing a widget to do it.
    """
    width, height = sensor
    x = min(max(roi.x, 0), width - 1)
    y = min(max(roi.y, 0), height - 1)
    return Roi(
        x,
        y,
        min(max(roi.width, 1), width - x),
        min(max(roi.height, 1), height - y),
    )


class RoiPanel(QtWidgets.QWidget):
    """An editor for the region a detector reads.

    Select ROI opens the editor and shows the box on the image, announced on
    ``sig_selection_toggled``. The editor's spin boxes and the box show one
    region: dragging the box fills the spin boxes through :meth:`draw`, editing
    a spin box announces the region on ``sig_roi_edited``. Full fills in the
    whole sensor. OK emits ``sig_roi_requested`` with the region as
    ``"x,y,width,height"`` and closes the editor; nothing reaches the camera
    before that.

    Parameters
    ----------
    sensor
        The whole sensor, ``(width, height)``.
    applied
        The region the camera reads now.
    """

    sig_roi_requested = Signal(str)
    sig_roi_edited = Signal(object)
    sig_selection_toggled = Signal(bool)

    def __init__(
        self,
        sensor: tuple[int, int],
        applied: Roi,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.sensor = sensor
        self.applied = applied
        width, height = sensor

        self.select_button = QtWidgets.QPushButton("Select ROI", self)
        self.select_button.setToolTip("Show a box on the image to drag over the region")
        self.select_button.setCheckable(True)
        self.select_button.toggled.connect(self._on_select)

        self.editor = QtWidgets.QWidget(self)
        self.editor.setVisible(False)
        self.x_box = QtWidgets.QSpinBox(self.editor)
        self.x_box.setRange(0, width - 1)
        self.y_box = QtWidgets.QSpinBox(self.editor)
        self.y_box.setRange(0, height - 1)
        self.width_box = QtWidgets.QSpinBox(self.editor)
        self.width_box.setRange(1, width)
        self.height_box = QtWidgets.QSpinBox(self.editor)
        self.height_box.setRange(1, height)
        for box in self._boxes:
            box.valueChanged.connect(self._on_edit)
        self.full_button = QtWidgets.QPushButton("Full", self.editor)
        self.full_button.setToolTip("Fill in the whole sensor")
        self.full_button.clicked.connect(self._on_full)
        self.ok_button = QtWidgets.QPushButton("OK", self.editor)
        self.ok_button.setToolTip("Read out this region")
        self.ok_button.clicked.connect(self._on_ok)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        for row, pairs in enumerate(
            (
                (("x", self.x_box), ("y", self.y_box)),
                (("width", self.width_box), ("height", self.height_box)),
            )
        ):
            for column, (name, box) in enumerate(pairs):
                grid.addWidget(QtWidgets.QLabel(name, self.editor), row, 2 * column)
                grid.addWidget(box, row, 2 * column + 1)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.full_button, 1)
        buttons.addWidget(self.ok_button, 1)
        grid.addLayout(buttons, 2, 0, 1, 4)
        self.editor.setLayout(grid)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QtWidgets.QLabel("ROI", self), 1)
        header.addWidget(self.select_button)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.editor)
        self.setLayout(layout)
        self._load(applied)
        self._update_ok_button()

    @property
    def pending(self) -> Roi:
        """The region the editor shows."""
        return Roi(
            self.x_box.value(),
            self.y_box.value(),
            self.width_box.value(),
            self.height_box.value(),
        )

    def draw(self, roi: Roi) -> None:
        """Fill the editor with *roi*, the region dragged on the image."""
        self._load(clamp_roi(roi, self.sensor))
        self._update_ok_button()

    def apply(self, roi: Roi) -> None:
        """Show *roi* as the region the camera now reads."""
        self.applied = roi
        if not self.editor.isVisible():
            self._load(roi)
        self._update_ok_button()

    @property
    def _boxes(self) -> tuple[QtWidgets.QSpinBox, ...]:
        return (self.x_box, self.y_box, self.width_box, self.height_box)

    def _load(self, roi: Roi) -> None:
        """Put *roi* in the spin boxes without announcing an edit."""
        for box in self._boxes:
            box.blockSignals(True)
        try:
            self.x_box.setValue(roi.x)
            self.y_box.setValue(roi.y)
            self._limit_size_to_sensor()
            self.width_box.setValue(roi.width)
            self.height_box.setValue(roi.height)
        finally:
            for box in self._boxes:
                box.blockSignals(False)

    def _limit_size_to_sensor(self) -> None:
        """Keep width and height inside the sensor from the corner chosen."""
        sensor_width, sensor_height = self.sensor
        self.width_box.setMaximum(sensor_width - self.x_box.value())
        self.height_box.setMaximum(sensor_height - self.y_box.value())

    def _update_ok_button(self) -> None:
        self.ok_button.setEnabled(self.pending != self.applied)

    def _on_select(self, checked: bool) -> None:
        if checked:
            self._load(self.applied)
        self.editor.setVisible(checked)
        self._update_ok_button()
        self.sig_selection_toggled.emit(checked)

    def _on_edit(self) -> None:
        for box in self._boxes:
            box.blockSignals(True)
        try:
            self._limit_size_to_sensor()
        finally:
            for box in self._boxes:
                box.blockSignals(False)
        self._update_ok_button()
        self.sig_roi_edited.emit(self.pending)

    def _on_full(self) -> None:
        width, height = self.sensor
        self._load(Roi(0, 0, width, height))
        self._update_ok_button()
        self.sig_roi_edited.emit(self.pending)

    def _on_ok(self) -> None:
        self.sig_roi_requested.emit(self.pending.as_text())
        self.select_button.setChecked(False)


class SettingsControlWidget(QtWidgets.QWidget):
    """Widget for a detector's settings, backed by a descriptor tree view.

    *descriptors* and *readings* are the detector's ``describe_configuration``
    and ``read_configuration`` output.
    """

    def __init__(
        self,
        descriptors: dict[str, Descriptor],
        readings: dict[str, Reading[Any]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self.tree_view = DescriptorTreeView(descriptors, readings, parent=self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree_view)
        self.roi_panel = self._roi_panel(readings)
        if self.roi_panel is not None:
            layout.addWidget(self.roi_panel)
        self.setLayout(layout)

    def set_locked(self, locked: bool) -> None:
        """Disable the property editors and the ROI panel, or enable them again.

        The editors keep showing each new reading; a read-only property's label
        is left as it is.
        """
        items = QtWidgets.QTreeWidgetItemIterator(self.tree_view)
        while item := items.value():
            editor = self.tree_view.itemWidget(item, 1)
            if editor is not None and not isinstance(editor, QtWidgets.QLabel):
                editor.setEnabled(not locked)
            items += 1
        if self.roi_panel is not None:
            self.roi_panel.setEnabled(not locked)

    def _roi_panel(self, readings: dict[str, Reading[Any]]) -> RoiPanel | None:
        """Build the ROI panel for a detector reporting a sensor size and a ROI."""
        extracted = sensor_and_roi(readings)
        if extracted is None:
            return None
        sensor, applied = extracted
        return RoiPanel(sensor, applied, self)


class DetectorView(QtWidgets.QWidget, Loggable):
    """View for interactive detector settings control.

    One property panel per detector in a tabbed widget; edits go out on
    ``sig_property_changed``. Images are shown by a separate view, which shares
    only signals with this view: the region drawn there reaches the ROI panel
    over one, an edit in the panel reaches the box the same way, and only OK
    sends a region to the camera, as a property change like any other.
    """

    placement: Placement = Dock("right")

    sig_property_changed = Signal(str, str, object)
    """Emitted when the user changes a detector property, with the detector
    name, the property name and the new value."""

    sig_roi_selection = Signal(str, bool)
    """Emitted when the user asks to select a region on a detector's image,
    or stops: the detector name, and whether the box is wanted."""

    sig_roi_edited = Signal(str, object)
    """Emitted with the detector name and a `Roi` when the user edits the
    region in a detector's panel, for the box on its image to follow."""

    def __init__(self, name: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.name = name

        self.settings_tab_widget = QtWidgets.QTabWidget()
        self.settings_tab_widget.setMinimumWidth(300)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.settings_tab_widget)
        self.setLayout(layout)

        self.settings_controls: dict[str, SettingsControlWidget] = {}

        self.logger.info("Initialized")

    def setup(self, detectors: DescribesDetectors) -> None:
        """Build the settings panels of every detector *detectors* describes."""
        self.setup_ui(detectors.detector_descriptors(), detectors.detector_readings())

    def setup_ui(
        self,
        descriptors: dict[str, Descriptor],
        readings: dict[str, Reading[Any]],
    ) -> None:
        r"""Build one settings panel per detector.

        *descriptors* and *readings* are the ``describe_configuration`` and
        ``read_configuration`` output of every detector, merged flat.
        """
        devices: dict[str, dict[str, Descriptor]] = {}
        for key, descriptor in descriptors.items():
            parsed = detector_key(key)
            if parsed is None:
                self.logger.warning(f"Skipping malformed descriptor key: {key!r}")
                continue
            name, _ = parsed
            devices.setdefault(name, {})[key] = descriptor

        for device_label, dev_descriptors in devices.items():
            dev_readings = {k: v for k, v in readings.items() if k in dev_descriptors}

            widget = SettingsControlWidget(dev_descriptors, dev_readings, self)
            widget.tree_view.sig_property_changed.connect(self.sig_property_changed)
            if widget.roi_panel is not None:
                widget.roi_panel.sig_roi_requested.connect(
                    partial(self.sig_property_changed.emit, device_label, "roi")
                )
                widget.roi_panel.sig_selection_toggled.connect(
                    partial(self.sig_roi_selection.emit, device_label)
                )
                widget.roi_panel.sig_roi_edited.connect(
                    partial(self.sig_roi_edited.emit, device_label)
                )
            self.settings_controls[device_label] = widget
            self.settings_tab_widget.addTab(widget, device_label)

    @slot
    def set_locked(self, names: frozenset[str]) -> None:
        """Disable the settings of the detectors in *names*, and enable the rest.

        Their readings keep updating: only what edits them is disabled.
        """
        for detector, widget in self.settings_controls.items():
            widget.set_locked(detector in names)

    @slot
    def on_roi_drawn(self, detector: str, roi: Roi) -> None:
        """Fill *detector*'s editor with the region dragged on its image."""
        widget = self.settings_controls.get(detector)
        if widget is not None and widget.roi_panel is not None:
            widget.roi_panel.draw(roi)

    @slot
    def on_new_configuration(self, detector: str, key: str, value: Any) -> None:
        """Show *value*, what the device read back, for the setting *key*.

        Settles an edit pending on *key*, the ``name-property`` key of the
        setting. Reached only after a successful ``set``: a failure is logged
        by the presenter and leaves the pending value in place.
        """
        widget = self.settings_controls.get(detector)
        if widget is None:
            self.logger.warning(f"No settings panel for detector {detector!r}")
            return
        widget.tree_view.set_value(key, value)
        if key == f"{detector}-roi" and widget.roi_panel is not None:
            widget.roi_panel.apply(Roi.parse(str(value)))
