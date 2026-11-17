"""Manual ARH controls routed through the acquisition RunEngine."""

from __future__ import annotations

import json
from pathlib import Path

from psygnal import Signal
from qtpy import QtWidgets as QtW
from redsun import Placement, slot
from redsun.qt import Dock

from redsun_aht.presenter.arh import CHANNELS, FILTER_SLOTS


class ArhControls(QtW.QWidget):
    """Instrument-specific controls for light, wheel and commanded rotary axes."""

    sig_launch = Signal(str, dict)
    sig_led_map_uploaded = Signal(str, tuple)
    placement: Placement = Dock("right", group="arh-controls")

    def __init__(self, name: str, parent: QtW.QWidget) -> None:
        super().__init__(parent)
        self.name = name
        self.setObjectName(name)
        layout = QtW.QVBoxLayout(self)
        self._stage_limits = QtW.QLabel(self)
        self._stage_limits.setWordWrap(True)
        layout.addWidget(self._stage_limits)
        self._stage_limits.hide()

        illumination = QtW.QGroupBox("Spectra illumination", self)
        self._illumination = illumination
        light_grid = QtW.QGridLayout(illumination)
        channel = QtW.QComboBox(illumination)
        self._channel = channel
        channel.addItems(CHANNELS)
        apply = QtW.QPushButton("Set channel", illumination)
        apply.clicked.connect(lambda: self.launch(
            "spectra_channel", light="illumination", channel=channel.currentText()
        ))
        light_grid.addWidget(channel, 0, 0)
        light_grid.addWidget(apply, 0, 1)
        for column, (label, opened) in enumerate((("Open shutter", True), ("Close shutter", False))):
            button = QtW.QPushButton(label, illumination)
            if opened:
                self._open_shutter = button
                button.setEnabled(False)
            button.clicked.connect(lambda checked=False, opened=opened: self.launch(
                "spectra_shutter", light="illumination", opened=opened
            ))
            light_grid.addWidget(button, 1, column)
        intensity = QtW.QSpinBox(illumination)
        self._intensity = intensity
        self._reported_intensity: int | None = None
        intensity.setRange(0, 100)
        intensity.setSuffix("%")
        set_intensity = QtW.QPushButton("Set intensity", illumination)
        self._set_intensity = set_intensity
        set_intensity.setEnabled(False)
        set_intensity.clicked.connect(lambda: self.launch(
            "spectra_intensity", light="illumination", intensity=intensity.value()
        ))
        light_grid.addWidget(intensity, 2, 0)
        light_grid.addWidget(set_intensity, 2, 1)
        layout.addWidget(illumination)
        self._illumination_state = QtW.QLabel(self)
        self._reported_channel: str | None = None
        self._reported_shutter_open: bool | None = None
        layout.addWidget(self._illumination_state)
        self._illumination_state.hide()

        wheel = QtW.QGroupBox("FW1000 filter wheel", self)
        self._wheel = wheel
        wheel_grid = QtW.QGridLayout(wheel)
        slots = QtW.QComboBox(wheel)
        for index, label in enumerate(FILTER_SLOTS):
            slots.addItem(f"{index}  {label}", index)
        slots.setCurrentIndex(4)
        set_slot = QtW.QPushButton("Set slot", wheel)
        self._set_slot = set_slot
        set_slot.setEnabled(False)
        set_slot.clicked.connect(lambda: self.launch(
            "filter_wheel_slot", wheel="wheel", slot=slots.currentData()
        ))
        wheel_grid.addWidget(slots, 0, 0)
        wheel_grid.addWidget(set_slot, 0, 1)
        self._wheel_state = QtW.QLabel("Status unknown", wheel)
        wheel_grid.addWidget(self._wheel_state, 1, 0)
        self._wheel_slot = QtW.QLabel("Last completed slot: unknown", wheel)
        wheel_grid.addWidget(self._wheel_slot, 1, 1)
        layout.addWidget(wheel)

        self._led_array = QtW.QGroupBox("LED Array", self)
        led_grid = QtW.QGridLayout(self._led_array)
        self._led_pixels = 0
        self._led_pixel = QtW.QSpinBox(self._led_array)
        self._led_pixel.setToolTip("Zero-based LED index")
        pattern = QtW.QComboBox(self._led_array)
        pattern.addItems(["Off", "Group 1", "Group 2", "Group 3", "Group 4", "Center"])
        pwm = QtW.QSpinBox(self._led_array)
        pwm.setRange(0, 4)
        pwm.setToolTip("0 turns PWM off")
        l4 = QtW.QSpinBox(self._led_array)
        l4.setRange(0, 12)
        l4.setToolTip("0 blanks the strip")
        color = QtW.QComboBox(self._led_array)
        color.addItems(["Blue", "Green", "Red"])
        intensity = QtW.QSpinBox(self._led_array)
        intensity.setRange(0, 255)
        intensity.setValue(255)
        for row, (label, editor, read, action) in enumerate((
            ("Pattern", pattern, pattern.currentIndex, "pattern"),
            ("Individual LED", self._led_pixel, self._led_pixel.value, "pixel"),
            ("L4 position", l4, l4.value, "l4"),
            ("PWM channel", pwm, pwm.value, "pwm"),
            ("Color", color, lambda: color.currentText().lower(), "color"),
            ("Intensity", intensity, intensity.value, "intensity"),
        )):
            led_grid.addWidget(QtW.QLabel(label, self._led_array), row, 0)
            led_grid.addWidget(editor, row, 1)
            apply = QtW.QPushButton("Set", self._led_array)
            apply.clicked.connect(
                lambda checked=False, action=action, read=read: self._apply_led(action, read())
            )
            led_grid.addWidget(apply, row, 2)
        self._led_map_button = QtW.QPushButton("Load static map…", self._led_array)
        self._led_map_button.clicked.connect(self._load_led_map)
        led_grid.addWidget(self._led_map_button, 6, 0, 1, 2)
        blank_led = QtW.QPushButton("Blank", self._led_array)
        blank_led.clicked.connect(lambda: self._apply_led("blank", None))
        led_grid.addWidget(blank_led, 6, 2)
        self._dpct_camera = QtW.QComboBox(self._led_array)
        self._dpct_camera.setAccessibleName("Dynamic DPCT camera")
        led_grid.addWidget(self._dpct_camera, 7, 0, 1, 2)
        self._dpct_demo = QtW.QPushButton("Run dynamic DPCT demo", self._led_array)
        self._dpct_demo.clicked.connect(self._confirm_dynamic_dpct)
        self._dpct_demo.setEnabled(False)
        led_grid.addWidget(self._dpct_demo, 7, 2)
        self.set_led_array_capability(0, False)
        layout.addWidget(self._led_array)

        motors = QtW.QGroupBox("Due outputs", self)
        self._motors = motors
        motor_rows = QtW.QFormLayout(motors)
        for label, plan in (("Enable rotary outputs", "emmsh_enable"), ("Stop Due motion", "emmsh_stop"), ("Blank panel", "emmsh_blank")):
            action = QtW.QPushButton(label, motors)
            action.clicked.connect(lambda checked=False, plan=plan: self.launch(plan, mcu="mcu"))
            motor_rows.addRow(action)
        zero = QtW.QPushButton("Set current roll, pitch and yaw to 0°", motors)
        zero.setToolTip("Reset the Due rotary coordinates without moving the motors")
        zero.clicked.connect(self.confirm_rotary_zero)
        motor_rows.addRow(zero)
        self._translation_bounds = QtW.QCheckBox("Enforce X, Y, F and ZDT travel bounds", motors)
        self._translation_bounds.setChecked(True)
        self._translation_bounds.clicked.connect(lambda checked: self.request_translation_bounds(checked))
        motor_rows.addRow(self._translation_bounds)
        layout.addWidget(motors)
        self._status = QtW.QLabel(self)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._pending_led_map: tuple[int, ...] | None = None
        self._plan: str | None = None
        self._active_plan: str | None = None

    def set_led_array_capability(self, pixels: int, supports_map: bool) -> None:
        self._led_pixels = max(0, pixels)
        self._led_pixel.setRange(0, max(0, pixels - 1))
        self._led_array.setEnabled(pixels > 0)
        self._led_array.setToolTip("" if pixels > 0 else "Compact LED Array unavailable")
        self._led_map_button.setEnabled(pixels > 0 and supports_map)
        self._led_map_button.setToolTip(
            f"JSON array of {pixels} integers from 0 to 255" if supports_map
            else "Static maps are unavailable on this firmware"
        )

    def set_dynamic_dpct_cameras(self, names: list[str]) -> None:
        self._dpct_camera.clear()
        self._dpct_camera.addItems(names)
        if "dcam" in names:
            self._dpct_camera.setCurrentText("dcam")
        self._dpct_demo.setEnabled(self._led_pixels == 61 and bool(names))

    def _confirm_dynamic_dpct(self) -> None:
        camera = self._dpct_camera.currentText()
        answer = QtW.QMessageBox.question(
            self, "Run dynamic DPCT demo",
            f"Reserve {camera} for software snaps and display the two semicircles, "
            "then graded and solid groups 1–4?",
            QtW.QMessageBox.StandardButton.Yes | QtW.QMessageBox.StandardButton.No,
            QtW.QMessageBox.StandardButton.No,
        )
        if answer == QtW.QMessageBox.StandardButton.Yes:
            self.launch(
                "arh_dynamic_dpct", camera=camera, mcu="mcu", confirm_exclusive_camera=True,
            )

    def _apply_led(self, action: str, value: object) -> None:
        request = json.dumps({"action": action, "value": value}, separators=(",", ":"))
        self.launch("emmsh_led_array", mcu="mcu", request=request)

    def _load_led_map(self) -> None:
        filename, _ = QtW.QFileDialog.getOpenFileName(self, "Load static LED map", "", "JSON (*.json)")
        if not filename:
            return
        try:
            with Path(filename).open("rb") as stream:
                raw = stream.read(8193)
            if len(raw) > 8192:
                raise ValueError("LED map file is too large")
            values = json.loads(raw)
            if (not isinstance(values, list) or len(values) != self._led_pixels
                    or any(type(v) is not int or not 0 <= v <= 255 for v in values)):
                raise ValueError(f"Expected {self._led_pixels} integers from 0 to 255")
        except (OSError, ValueError) as error:
            self._status.setText(f"LED map: {error}")
            return
        self._apply_led("map", values)

    def request_translation_bounds(self, enabled: bool) -> None:
        self._translation_bounds.setChecked(not enabled)
        self.launch("translation_bounds", stage="stage", mcu="mcu", enabled=enabled)

    def confirm_rotary_zero(self) -> None:
        answer = QtW.QMessageBox.question(
            self, "Set rotary zero",
            "Set the current roll, pitch and yaw positions to 0°? "
            "The motors will stay in place and future targets and limits will use this zero.",
            QtW.QMessageBox.StandardButton.Yes | QtW.QMessageBox.StandardButton.No,
            QtW.QMessageBox.StandardButton.No,
        )
        if answer == QtW.QMessageBox.StandardButton.Yes:
            self.launch("emmsh_zero_rotary", mcu="mcu")

    def launch(self, plan: str, **values: object) -> None:
        """Submit the same named plan as the acquisition page."""
        if self._plan is not None or self._active_plan is not None:
            return
        self._requested_bounds = values.get("enabled") if plan == "translation_bounds" else None
        if plan == "emmsh_led_array":
            request = json.loads(values["request"])
            if request["action"] == "map":
                self._pending_led_map = tuple(request["value"])
        self._plan = plan
        self._status.setText("Request submitted")
        self.sig_launch.emit(plan, values)

    def set_stage_limits(self, bounds: dict[str, list[float]]) -> None:
        intervals = ", ".join(f"{'F (focus)' if axis == 'z' else axis.upper()} [{low:g}, {high:g}]"
                              for axis, (low, high) in bounds.items())
        self._stage_limits.setText(f"LX4000 travel limits (µm)\n{intervals}")
        self._stage_limits.show()

    def set_available_devices(self, names: set[str]) -> None:
        self._translation_bounds.setEnabled({"stage", "mcu"} <= names)
        for name, group in (
            ("illumination", self._illumination),
            ("wheel", self._wheel),
            ("mcu", self._motors),
        ):
            available = name in names
            group.setEnabled(available)
            group.setToolTip("" if available else "Device is absent from this profile")

    @slot(thread="main")
    def on_wheel_busy(self, reading: dict) -> None:
        sample, = reading.values()
        self.set_wheel_busy(bool(sample["value"]))

    def set_wheel_busy(self, busy: bool) -> None:
        self._wheel_state.setText("Busy" if busy else "Idle")
        self._set_slot.setEnabled(not busy)

    @slot(thread="main")
    def on_wheel_slot(self, reading: dict) -> None:
        sample, = reading.values()
        self.set_wheel_slot(sample["value"])

    def set_wheel_slot(self, slot: int) -> None:
        value = str(slot) if 0 <= slot < len(FILTER_SLOTS) else "unknown"
        self._wheel_slot.setText(f"Last completed slot: {value}")

    def set_initial_illumination(self, channel: str, shutter_open: bool) -> None:
        if channel in CHANNELS:
            self._channel.setCurrentText(channel)
        self.set_illumination_readback(channel, shutter_open)

    def set_illumination_readback(self, channel: str | None, shutter_open: bool | None) -> None:
        self._reported_channel = channel
        self._reported_shutter_open = shutter_open
        active = channel in CHANNELS and channel not in ("off", "brightfield")
        self._open_shutter.setEnabled(active)
        self._set_intensity.setEnabled(active)
        state = "unknown" if shutter_open is None else "open" if shutter_open else "closed"
        self._illumination_state.setText(f"Spectra readback: {channel or 'unknown'}, shutter {state}")
        self._illumination_state.show()

    @slot(thread="main")
    def on_spectra_channel(self, reading: dict) -> None:
        sample, = reading.values()
        self.set_illumination_readback(sample["value"], self._reported_shutter_open)

    @slot(thread="main")
    def on_spectra_intensity(self, reading: dict) -> None:
        sample, = reading.values()
        value = sample["value"]
        if value != self._reported_intensity:
            self._reported_intensity = value
            self._intensity.setValue(value)

    @slot(thread="main")
    def on_spectra_shutter(self, reading: dict) -> None:
        sample, = reading.values()
        self.set_illumination_readback(self._reported_channel, sample["value"])

    @slot(thread="main")
    def plan_started(self, plan: str, values: dict) -> None:
        self._active_plan = plan
        if self._plan is not None and self._plan != plan:
            self._pending_led_map = None
            self._status.setText("Request was not started")
            self._plan = None
        self.setEnabled(False)

    @slot(thread="main")
    def plan_finished(self, plan: str) -> None:
        if self._active_plan == plan:
            self._active_plan = None
            self.setEnabled(True)
        if self._plan == plan:
            if plan == "translation_bounds" and self._requested_bounds is not None:
                self._translation_bounds.setChecked(bool(self._requested_bounds))
            if plan == "emmsh_led_array" and self._pending_led_map is not None:
                self.sig_led_map_uploaded.emit("Due · last uploaded static map", self._pending_led_map)
                self._pending_led_map = None
            self._status.setText("Request finished")
            self._plan = None

    @slot(thread="main")
    def plan_stopped(self, plan: str) -> None:
        requested = self._plan == plan
        if requested and plan == "emmsh_led_array":
            self._pending_led_map = None
        if self._plan == plan and plan == "translation_bounds":
            self._requested_bounds = None
        self.plan_finished(plan)
        if requested:
            self._status.setText("Request stopped")

    @slot(thread="main")
    def plan_failed(self, plan: str, reason: str) -> None:
        if self._plan == plan and plan == "emmsh_led_array":
            self._pending_led_map = None
        if self._plan == plan and plan == "translation_bounds":
            self._requested_bounds = None
        self.plan_finished(plan)
        self._status.setText(reason)
