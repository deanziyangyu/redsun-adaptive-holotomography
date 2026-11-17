"""PVAccess clients for the ARH illumination and legacy MCU services."""

import hashlib
import json
import math
import zlib
from functools import partial
from secrets import randbelow
from uuid import uuid4

from ophyd_async.core import (
    SignalR, SignalRW, SignalW, StandardReadable, StandardReadableFormat,
    callback_on_mock_put, set_mock_value, soft_signal_rw,
)
from ophyd_async.fastcs.core import fastcs_connector

from ...device.dynamic_patterns import DynamicPattern
from .due_stack import wait_due_stack
from .mmcore._stage import MMAxis

_DEGREES_PER_DUE_COUNT = 360.0 / 2048


class LumencorSpectra(StandardReadable):
    intensity: SignalRW[int]
    shutter_open: SignalRW[bool]
    channel: SignalRW[str]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))

    async def connect(self, *args, **kwargs) -> None:
        await super().connect(*args, **kwargs)
        if not kwargs.get("mock", args[0] if args else False):
            return

        def select_channel(_channel: str) -> None:
            set_mock_value(self.shutter_open, False)

        async def set_shutter(opened: bool) -> None:
            if opened and await self.channel.get_value() in ("off", "brightfield"):
                raise RuntimeError("Select an illumination preset before opening the shutter")

        set_mock_value(self.intensity, 0)
        set_mock_value(self.channel, "off")
        set_mock_value(self.shutter_open, False)
        callback_on_mock_put(self.channel, select_channel)
        callback_on_mock_put(self.shutter_open, set_shutter)


class UnifiedEmmsh(StandardReadable):
    commanded_steps: SignalR[str]
    rotary_enabled: SignalR[bool]
    roll_target: SignalW[int]
    pitch_target: SignalW[int]
    yaw_target: SignalW[int]
    software_dpct_enabled: SignalR[bool]
    dpct_pattern: SignalW[int]
    led_pixels: SignalR[int]
    led_map_supported: SignalR[bool]
    led_array_request: SignalW[str]
    dynamic_pattern_result: SignalR[str]
    zdt_bounds: SignalW[bool]
    zdt_start: SignalW[str]
    zdt_captured: SignalW[str]
    zdt_software_enabled: SignalR[bool]
    zdt_stack_enabled: SignalR[bool]
    zdt_abort: SignalW[int]
    zdt_status_request: SignalW[int]
    zdt_status: SignalR[str]
    stop_: SignalW[bool]
    enable_rotary: SignalW[bool]
    rotary_zero: SignalW[bool]
    rotary_range: SignalW[str]
    rotary_jog_request: SignalW[str]
    blank: SignalW[bool]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))

    async def upload_dynamic_pattern(self, slot: int, pixels: bytes) -> DynamicPattern:
        """Store drive codes in one controller slot and return the publication receipt."""
        if type(slot) is not int or not 1 <= slot <= 5 or not isinstance(pixels, bytes):
            raise ValueError("a slot in 1..5 and immutable pixel bytes are required")
        if len(pixels) != await self.led_pixels.get_value():
            raise ValueError("expected one intensity byte per panel pixel")
        request_id = uuid4().hex
        await self.led_array_request.set(json.dumps({
            "action": "dynamic_map",
            "value": {"id": request_id, "slot": slot, "pixels": list(pixels)},
        }))
        result = json.loads(await self.dynamic_pattern_result.get_value())
        if (
            not isinstance(result, dict)
            or set(result) != {"id", "slot", "generation", "crc32", "sha256"}
            or result["id"] != request_id
            or type(result["slot"]) is not int or result["slot"] != slot
            or type(result["generation"]) is not int
            or not 1 <= result["generation"] <= 0xFFFFFFFF
            or type(result["crc32"]) is not int or result["crc32"] != zlib.crc32(pixels)
            or result["sha256"] != hashlib.sha256(pixels).hexdigest()
        ):
            raise ValueError("dynamic upload response mismatch")
        return DynamicPattern(slot, result["generation"], result["crc32"], result["sha256"])

    async def connect(self, *args, **kwargs) -> None:
        await super().connect(*args, **kwargs)
        mock = kwargs.get("mock", args[0] if args else False)
        if not hasattr(self, "roll"):
            for name in ("roll", "pitch", "yaw", "zdt"):
                axis = MMAxis()
                if name == "zdt":
                    origin = 0.0
                    if not mock and await self.zdt_software_enabled.get_value(cached=False):
                        try:
                            origin = await self.read_zdt_position()
                        except (RuntimeError, TimeoutError, ValueError, KeyError):
                            origin = float("nan")
                    self._zdt_position_um = origin
                    axis.position = soft_signal_rw(
                        float, origin, units="µm", precision=2, setter=self._set_zdt,
                        getter=lambda: self._zdt_position_um, poll_period=1,
                    )
                else:
                    axis.position = soft_signal_rw(
                        float, 0, units="deg", precision=2, getter=partial(self._rotary_position, name),
                        setter=partial(self._set_rotary, name), poll_period=1,
                    )
                    axis.travel_limits = soft_signal_rw(
                        str, "", getter=partial(self._read_rotary_range, name),
                        setter=partial(self._write_rotary_range, name),
                    )
                    axis.add_readables([axis.travel_limits], StandardReadableFormat.CONFIG_SIGNAL)
                axis.add_readables([axis.position])
                setattr(self, name, axis)
        if not mock:
            return

        axes = ("roll", "pitch", "yaw")
        limits = dict.fromkeys(axes, (0, 2047))
        targets = dict.fromkeys(axes, 0)
        stopped = False

        def publish_targets() -> None:
            set_mock_value(self.commanded_steps, json.dumps({
                "rotary_limits": limits, "steps": targets,
            }))

        def set_target(steps: int, *, axis: str) -> None:
            if stopped:
                raise RuntimeError("Due rotary outputs are disabled after a stop")
            low, high = limits[axis]
            if not low <= steps <= high:
                raise ValueError(f"Mock Due {axis} target outside [{low}, {high}]")
            targets[axis] = steps
            publish_targets()

        def stop(requested: bool) -> None:
            nonlocal stopped
            if requested:
                stopped = True
                set_mock_value(self.rotary_enabled, False)

        for axis in axes:
            callback_on_mock_put(
                getattr(self, f"{axis}_target"),
                lambda steps, axis=axis: set_target(steps, axis=axis),
            )
        publish_targets()
        set_mock_value(self.rotary_enabled, True)
        callback_on_mock_put(self.stop_, stop)
        def enable(requested: bool) -> None:
            nonlocal stopped
            if requested:
                stopped = False
                set_mock_value(self.rotary_enabled, True)

        callback_on_mock_put(self.enable_rotary, enable)
        def configure_range(value: str) -> None:
            if not stopped:
                raise RuntimeError("Stop Due motion before changing rotary limits")
            name, bounds = value.split("=")
            low, high = map(int, bounds.split(":"))
            if not -2047 <= low < high < 2048:
                raise ValueError("Enter rotary limits as min:max within -2047 to 2047 counts")
            limits[name] = (low, high)
            publish_targets()

        callback_on_mock_put(self.rotary_range, configure_range)

        def zero_rotary(requested: bool) -> None:
            if requested:
                targets.update(dict.fromkeys(axes, 0))
                publish_targets()

        callback_on_mock_put(self.rotary_zero, zero_rotary)
        def jog(value: str) -> None:
            name, target = value.split("=")
            enable(True)
            try:
                set_target(int(target), axis=name)
            finally:
                stop(True)

        callback_on_mock_put(self.rotary_jog_request, jog)
        self._mock_zdt = True
        self._mock_zdt_bounds = True

        def set_zdt_bounds(enabled: bool) -> None:
            self._mock_zdt_bounds = enabled

        callback_on_mock_put(self.zdt_bounds, set_zdt_bounds)
        set_mock_value(self.software_dpct_enabled, True)
        set_mock_value(self.led_pixels, 61)
        set_mock_value(self.led_map_supported, True)

    async def _rotary_position(self, name):
        status = json.loads(await self.commanded_steps.get_value(cached=False))
        return status["steps"][name] * _DEGREES_PER_DUE_COUNT

    async def _read_rotary_range(self, name):
        status = json.loads(await self.commanded_steps.get_value(cached=False))
        bounds = status["rotary_limits"].get(name)
        return "" if bounds is None else ":".join(
            f"{count * _DEGREES_PER_DUE_COUNT:.3f}" for count in bounds
        )

    async def _write_rotary_range(self, name, value):
        try:
            lower, upper = map(float, value.split(":"))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError
            low, high = (round(angle / _DEGREES_PER_DUE_COUNT) for angle in (lower, upper))
        except (ValueError, OverflowError) as error:
            raise ValueError("Enter rotary limits in degrees as min:max") from error
        await self.rotary_range.set(f"{name}={low}:{high}", timeout=5)
        return await self._read_rotary_range(name)

    async def _set_rotary(self, name, target):
        if not math.isfinite(target):
            raise ValueError("Rotary target must be a finite angle in degrees")
        counts = round(target / _DEGREES_PER_DUE_COUNT)
        status = json.loads(await self.commanded_steps.get_value(cached=False))
        bounds = status["rotary_limits"].get(name)
        if bounds is None or not bounds[0] <= counts <= bounds[1]:
            raise ValueError(f"Due {name} target is outside its configured limits")
        await self.rotary_jog_request.set(f"{name}={counts}", timeout=12)
        return counts * _DEGREES_PER_DUE_COUNT

    async def read_zdt_position(self):
        if getattr(self, "_mock_zdt", False):
            return self._zdt_position_um
        await self.zdt_status_request.set(0, timeout=5)
        status = json.loads(await self.zdt_status.get_value(cached=False))
        if status.get("run_id") != 0:
            raise RuntimeError("Due did not return a fresh ZDT position")
        return float(status["position_um"])

    async def _set_zdt(self, target):
        return await self.move_zdt(target)

    async def move_zdt(self, target: float, *, window: tuple[int, int] | None = None) -> float:
        if not math.isfinite(target) or abs(target) > 10000:
            raise ValueError("ZDT target exceeds the protocol range of ±10000 µm")
        if getattr(self, "_mock_zdt", False):
            if window is not None:
                if not window[0] <= round(target * 3.6) <= window[1]:
                    raise ValueError("ZDT target exceeds the requested jog window")

            elif self._mock_zdt_bounds and abs(target) > 1000:
                raise ValueError("ZDT target exceeds the configured travel envelope")
            self._zdt_position_um = round(target * 3.6) / 3.6
            return self._zdt_position_um
        run_id = 1 + randbelow(2147483647)
        request = dict(run_id=run_id, jog_target=round(target * 3.6), camera_ready=True, software=True)
        if window is not None:
            request["jog_window"] = list(window)
        await self.zdt_start.set(json.dumps(request), timeout=10)
        try:
            await wait_due_stack(self, run_id, 1, timeout_s=10)
        except BaseException as error:
            try:
                await self.zdt_abort.set(run_id, timeout=3)
            except Exception as cleanup:
                error.add_note(f"ZDT stop could not be confirmed: {cleanup}")
            raise
        status = json.loads(await self.zdt_status.get_value(cached=False))
        self._zdt_position_um = float(status["target_um"])
        return self._zdt_position_um

    async def stop(self, success=False):
        await self.stop_.set(True, timeout=5)
