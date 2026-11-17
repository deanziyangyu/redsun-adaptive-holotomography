"""Translate live-preview controller samples into finite Positioner jogs."""

from dataclasses import replace

from .xinput import BUTTON_A, DPAD_LEFT, DPAD_RIGHT, DPAD_UP, LEFT_SHOULDER, RIGHT_SHOULDER, SurveyController, _stick

JOG_STEPS = (1.0, 5.0, 20.0, 100.0)
ROTARY_JOG_STEPS = (0.2, 0.5, 1.0, 5.0)


class LiveJog:
    """Use the existing Positioner to submit one small move at a time."""

    def __init__(self, read, move, halt, step_changed=None, halt_mcu=None, *,
                 zdt_is_focus=False, supported_axes=None):
        self.read = read
        self.move = move
        self.halt = halt
        self.halt_mcu = halt_mcu
        self.zdt_is_focus = zdt_is_focus
        self.supported_axes = None if supported_axes is None else frozenset(supported_axes)
        self.rotary_jogging = False
        self.controller = SurveyController()
        self._step_index = 0
        self._shoulders = LEFT_SHOULDER | RIGHT_SHOULDER
        self._step_changed = step_changed
        self.jogging = False

    def poll(self) -> None:
        state = self.read()
        shoulders = state.buttons & (LEFT_SHOULDER | RIGHT_SHOULDER) if state is not None else 0
        pressed = shoulders & ~self._shoulders
        self._shoulders = shoulders if state is not None else LEFT_SHOULDER | RIGHT_SHOULDER
        if pressed in (LEFT_SHOULDER, RIGHT_SHOULDER):
            direction = 1 if pressed == RIGHT_SHOULDER else -1
            index = min(max(self._step_index + direction, 0), len(JOG_STEPS) - 1)
            if index != self._step_index:
                self._step_index = index
                if self._step_changed is not None:
                    self._step_changed(JOG_STEPS[index])
        focus = 0.0
        if state is not None:
            focus = float(bool(state.buttons & DPAD_RIGHT)) - float(bool(state.buttons & DPAD_LEFT))
            # Include horizontal D-pad in neutral arming without changing the
            # shared simulator mapping.
            if (state.buttons & (DPAD_LEFT | DPAD_RIGHT)
                or _stick(state.right_x, self.controller.deadzone)
                or _stick(state.right_y, self.controller.deadzone)
                or state.left_trigger > 30 or state.right_trigger > 30):
                arming_state = replace(state, buttons=state.buttons | DPAD_UP)
            else:
                arming_state = state
        else:
            arming_state = None
        intent = self.controller.update(state if state is not None and state.buttons & BUTTON_A else arming_state)
        if intent.stop_requested:
            self._halt_jog()
            return
        rotary = (("roll", _stick(state.right_x, self.controller.deadzone)),
                  ("pitch", _stick(state.right_y, self.controller.deadzone)),
                  ("yaw", float(state.right_trigger > 30) - float(state.left_trigger > 30)))
        z_axes = (("z", intent.z), ("zdt", focus)) if self.zdt_is_focus else (
            ("zdt", intent.z), ("z", focus),
        )
        candidates = (*z_axes, ("x", intent.x), ("y", intent.y), *rotary)
        if self.supported_axes is not None:
            candidates = tuple(item for item in candidates if item[0] in self.supported_axes)
        axis, delta = max(candidates, key=lambda item: abs(item[1]), default=("", 0.0))
        if delta:
            is_rotary = axis in ("roll", "pitch", "yaw")
            self.jogging = self.jogging or axis in ("x", "y", "z") or (
                self.zdt_is_focus and axis == "zdt"
            )
            self.rotary_jogging = self.rotary_jogging or is_rotary
            step = (ROTARY_JOG_STEPS if is_rotary else JOG_STEPS)[self._step_index]
            self.move(axis, step if delta > 0 else -step)

    def _halt_jog(self) -> None:
        if self.jogging:
            self.jogging = False
            self.halt()
        if self.rotary_jogging:
            self.rotary_jogging = False
            if self.halt_mcu is not None:
                self.halt_mcu()

    def stop(self) -> None:
        self.controller.disarm()
        self._shoulders = LEFT_SHOULDER | RIGHT_SHOULDER
        self._halt_jog()
