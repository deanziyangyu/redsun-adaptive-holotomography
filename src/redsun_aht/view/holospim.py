"""HoloSPIM hardware actions routed through the acquisition RunEngine."""

import json
from typing import get_args

from psygnal import Signal
from qtpy import QtWidgets as QtW
from redsun import slot
from redsun.qt import Dock

from redsun_aht.presenter.holospim import LaserPreset


class HolospimControls(QtW.QWidget):
    """AA AOTF presets, RF settings and service readback."""

    placement = Dock("right", group="holospim-controls")
    sig_launch = Signal(str, dict)

    def __init__(self, name: str, parent: QtW.QWidget) -> None:
        super().__init__(parent)
        self.name = name
        self.setObjectName(name)
        self.setWindowTitle("Hardware")
        self._available = False
        self._active_plan = None
        self._requested_plan = None
        self._healthy = False
        self._locked = False
        self._device = ""
        layout = QtW.QVBoxLayout(self)
        group = QtW.QGroupBox("AA AOTF laser shutter", self)
        layout.addWidget(group)
        grid = QtW.QFormLayout(group)
        self._profile = QtW.QComboBox(group)
        self._profile.addItems(get_args(LaserPreset))
        grid.addRow("Preset", self._profile)
        self._apply_profile = QtW.QPushButton("Apply preset", group)
        self._apply_profile.clicked.connect(
            lambda: self.launch(
                "holospim_laser_profile", profile=self._profile.currentText()
            )
        )
        grid.addRow(self._apply_profile)
        self._off = QtW.QPushButton("All Off", group)
        self._off.clicked.connect(
            lambda: self.launch("holospim_laser_profile", profile="All Off")
        )
        grid.addRow(self._off)
        self._channel = QtW.QSpinBox(group)
        self._channel.setRange(1, 8)
        grid.addRow("Channel", self._channel)
        self._frequency = QtW.QDoubleSpinBox(group)
        self._frequency.setRange(50, 200)
        self._frequency.setDecimals(3)
        self._frequency.setSuffix(" MHz")
        grid.addRow("Frequency", self._frequency)
        self._maximum = QtW.QSpinBox(group)
        self._maximum.setRange(0, 2200)
        grid.addRow("Maximum intensity (adapter units)", self._maximum)
        self._power = QtW.QDoubleSpinBox(group)
        self._power.setRange(0, 100)
        self._power.setSuffix("%")
        grid.addRow("Power", self._power)
        self._apply_settings = QtW.QPushButton(
            "Apply RF settings with slots off", group
        )
        self._apply_settings.clicked.connect(self._settings)
        grid.addRow(self._apply_settings)
        self._readback = QtW.QLabel("Service readback unavailable", group)
        self._readback.setWordWrap(True)
        grid.addRow(self._readback)
        self._status = QtW.QLabel(self)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()
        self._update_enabled()

    def set_device(self, name: str | None) -> None:
        """Admit controls only for a uniquely resolved session laser."""
        self._device = name or ""
        self._available = name is not None
        self._update_enabled()

    def _update_enabled(self) -> None:
        enabled = self._available and self._active_plan is None and not self._locked
        self._off.setEnabled(enabled)
        for control in (self._apply_profile, self._apply_settings):
            control.setEnabled(enabled and self._healthy)

    def _settings(self) -> None:
        self.launch(
            "holospim_laser_settings",
            channel=self._channel.value(),
            frequency_mhz=self._frequency.value(),
            maximum_intensity=self._maximum.value(),
            power_percent=self._power.value(),
        )

    def launch(self, plan: str, **values: object) -> None:
        """Submit a hardware action to the session's acquisition presenter."""
        self._requested_plan = plan
        self._status.setText("Request submitted")
        self.sig_launch.emit(plan, values)

    @slot(thread="main")
    def on_status(self, reading: dict) -> None:
        """Display adapter readback and disable RF writes on reported faults."""
        try:
            (sample,) = reading.values()
            state = json.loads(sample["value"])
            if state.get("schema") != "redsun.aaaotf-status/v1":
                raise ValueError("unrecognized service status")
            self._healthy = state.get("connected") is True and not state.get("fault")
            properties = state.get("properties", {})
            active = [
                slot for slot, value in state.get("slot_states", {}).items() if value
            ]
            self._readback.setText(
                f"Profile: {state.get('active_profile', 'unknown')}\n"
                f"Active slots: {', '.join(active) or 'none'}\n"
                f"Channel: {properties.get('Channel', 'unknown')}\n"
                f"Frequency: {properties.get('Frequency (MHz)', 'unknown')} MHz\n"
                "Maximum intensity: "
                f"{properties.get('Maximum intensity (dB)', 'unknown')}\n"
                f"Power: {properties.get('Power (% of max)', 'unknown')}%\n"
                f"Fault: {state.get('fault') or 'none'}"
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            self._healthy = False
            self._readback.setText("Service readback unavailable")
        self._update_enabled()

    @slot(thread="main")
    def set_locked(self, names: frozenset[str]) -> None:
        """Respect acquisition ownership of the bound laser."""
        self._locked = self._device in names
        self._update_enabled()

    @slot(thread="main")
    def plan_started(self, plan: str, values: dict) -> None:
        """Disable manual requests while the RunEngine is occupied."""
        self._active_plan = plan
        self._update_enabled()

    @slot(thread="main")
    def plan_finished(self, plan: str) -> None:
        """Release the busy state and report a completed hardware request."""
        if self._active_plan == plan:
            self._active_plan = None
        if self._requested_plan == plan:
            self._requested_plan = None
            self._status.setText("Request finished")
        self._update_enabled()

    @slot(thread="main")
    def plan_stopped(self, plan: str) -> None:
        """Report an interrupted hardware request."""
        requested = self._requested_plan == plan
        self.plan_finished(plan)
        if requested:
            self._status.setText("Request stopped")

    @slot(thread="main")
    def plan_failed(self, plan: str, reason: str) -> None:
        """Show the command failure without asserting a shutter state."""
        requested = self._requested_plan == plan
        self.plan_finished(plan)
        if requested:
            self._status.setText(reason)
