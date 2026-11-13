"""Windows XInput polling and pure survey-intent mapping.

This component only reports intent. It is not a motion controller, watchdog,
or physical stop; a coordinator must enforce leases, limits and stop behavior.
"""

from __future__ import annotations

import ctypes
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class _Gamepad(ctypes.Structure):
    _fields_ = [
        ("buttons", ctypes.c_uint16),
        ("left_trigger", ctypes.c_uint8),
        ("right_trigger", ctypes.c_uint8),
        ("left_x", ctypes.c_int16),
        ("left_y", ctypes.c_int16),
        ("right_x", ctypes.c_int16),
        ("right_y", ctypes.c_int16),
    ]


class _State(ctypes.Structure):
    _fields_ = [("packet", ctypes.c_uint32), ("gamepad", _Gamepad)]


BUTTON_A = 0x1000
BUTTON_B = 0x2000
DPAD_UP = 0x0001
DPAD_DOWN = 0x0002


@dataclass(frozen=True, slots=True)
class ControllerState:
    """One XInput sample; stick values are native signed 16-bit integers."""

    buttons: int
    left_x: int
    left_y: int

    def __post_init__(self) -> None:
        """Reject malformed states before they can become motion intent."""
        if type(self.buttons) is not int or not 0 <= self.buttons <= 0xFFFF:
            raise ValueError("buttons must be a 16-bit mask")
        if any(
            type(v) is not int or not -32768 <= v <= 32767
            for v in (self.left_x, self.left_y)
        ):
            raise ValueError("sticks must be signed 16-bit integers")


@dataclass(frozen=True, slots=True)
class SurveyIntent:
    """Normalized axis requests and an explicit stop request."""

    connected: bool
    enabled: bool
    x: float
    y: float
    z: float
    stop_requested: bool


def _stick(value: int, deadzone: float) -> float:
    normalized = value / (32768 if value < 0 else 32767)
    magnitude = abs(normalized)
    if magnitude <= deadzone:
        return 0.0
    return math.copysign((magnitude - deadzone) / (1.0 - deadzone), normalized)


def survey_intent(
    state: ControllerState | None,
    *,
    deadzone: float = 0.15,
    invert_x: bool = False,
    invert_y: bool = False,
    invert_z: bool = False,
) -> SurveyIntent:
    """Map left stick to XY and D-pad to Z while A is held; B requests stop."""
    if not math.isfinite(deadzone) or not 0 <= deadzone < 1:
        raise ValueError("deadzone must be finite and in [0, 1)")
    if state is None:
        return SurveyIntent(False, False, 0.0, 0.0, 0.0, True)
    enabled = bool(state.buttons & BUTTON_A) and not bool(state.buttons & BUTTON_B)
    if not enabled:
        return SurveyIntent(True, False, 0.0, 0.0, 0.0, True)
    z = float(bool(state.buttons & DPAD_UP)) - float(bool(state.buttons & DPAD_DOWN))
    return SurveyIntent(
        True,
        True,
        _stick(state.left_x, deadzone) * (-1 if invert_x else 1),
        _stick(state.left_y, deadzone) * (-1 if invert_y else 1),
        z * (-1 if invert_z else 1),
        False,
    )


class XInputPoller:
    """Poll XInput 1.4 on Windows, or an injected function in tests."""

    def __init__(
        self, index: int = 0, *, get_state: Callable[..., int] | None = None
    ) -> None:
        if type(index) is not int or not 0 <= index <= 3:
            raise ValueError("XInput index must be 0..3")
        if get_state is None:
            if sys.platform != "win32":
                raise OSError("XInput is available only on Windows")
            dll = ctypes.WinDLL("xinput1_4.dll")
            get_state = dll.XInputGetState
            get_state.argtypes = [ctypes.c_uint32, ctypes.POINTER(_State)]
            get_state.restype = ctypes.c_uint32
        self._get_state = get_state
        self.index = index

    def read(self) -> ControllerState | None:
        """Return None on any nonzero XInput status, including disconnect."""
        state = _State()
        if self._get_state(self.index, ctypes.byref(state)) != 0:
            return None
        pad = state.gamepad
        return ControllerState(int(pad.buttons), int(pad.left_x), int(pad.left_y))


class SurveyController:
    """Require released buttons and neutral axes before accepting a new jog."""

    def __init__(self, *, deadzone: float = 0.15) -> None:
        # Validate once, before any input can become intent.
        survey_intent(None, deadzone=deadzone)
        self.deadzone = deadzone
        self._ready = False

    def disarm(self) -> None:
        """Require a released enable button before accepting motion again."""
        self._ready = False

    def update(self, state: ControllerState | None) -> SurveyIntent:
        """Return a stopped intent until buttons and motion inputs are neutral."""
        if state is None:
            self._ready = False
            return survey_intent(None, deadzone=self.deadzone)
        if state.buttons & BUTTON_B:
            self._ready = False
            return SurveyIntent(True, False, 0.0, 0.0, 0.0, True)
        if not state.buttons & BUTTON_A:
            mapped = survey_intent(
                ControllerState(
                    BUTTON_A | (state.buttons & (DPAD_UP | DPAD_DOWN)),
                    state.left_x,
                    state.left_y,
                ),
                deadzone=self.deadzone,
            )
            self._ready = mapped.x == mapped.y == 0.0 and not state.buttons & (
                DPAD_UP | DPAD_DOWN
            )
            return SurveyIntent(True, False, 0.0, 0.0, 0.0, True)
        if not self._ready:
            return SurveyIntent(True, False, 0.0, 0.0, 0.0, True)
        return survey_intent(state, deadzone=self.deadzone)
