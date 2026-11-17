"""Matplotlib dock for visualizing LED maps already available to the app."""

from pathlib import Path

from psygnal import Signal
from qtpy import QtWidgets
from redsun import slot
from redsun.qt import Dock

from redsun_aht.presenter.led_patterns import draw_pattern


class LedPatternView(QtWidgets.QWidget):
    placement = Dock("right", group="led-pattern-preview")
    sig_load = Signal(str)
    sig_select = Signal(str)

    def __init__(self, name: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.name = name
        self.setObjectName(name)
        self.setWindowTitle("LED Pattern Preview")
        self.setMinimumWidth(360)
        self._figure = Figure(figsize=(4.8, 4.2), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._selector = QtWidgets.QComboBox(self)
        self._selector.currentIndexChanged.connect(self._select)
        load = QtWidgets.QPushButton("Open map…", self)
        load.clicked.connect(self._load)
        self._message = QtWidgets.QLabel(self)
        self._message.hide()
        self._message.setWordWrap(True)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._selector, 1)
        row.addWidget(load)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self._canvas)
        layout.addWidget(self._message)

    def _select(self, index: int) -> None:
        if index >= 0:
            self.sig_select.emit(self._selector.itemData(index))

    def _load(self) -> None:
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Preview LED map", "", "JSON maps (*.json)")
        if file:
            self.sig_load.emit(file)

    @slot(thread="main")
    def set_patterns(self, labels: tuple, selected: str) -> None:
        previous = self._selector.blockSignals(True)
        try:
            self._selector.clear()
            for label in labels:
                self._selector.addItem(Path(label).name if Path(label).is_absolute() else label, label)
            self._selector.setCurrentIndex(self._selector.findData(selected))
        finally:
            self._selector.blockSignals(previous)

    @slot(thread="main")
    def set_pattern(self, panel, label: str, weights: tuple) -> None:
        draw_pattern(self._figure, panel, Path(label).name if Path(label).is_absolute() else label, weights)
        self._message.hide()
        self._canvas.draw_idle()

    @slot(thread="main")
    def show_error(self, message: str) -> None:
        self._message.setText(message)
        self._message.show()
