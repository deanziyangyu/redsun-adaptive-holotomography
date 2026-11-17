"""Serve status, stop and illumination blanking for the legacy ARH MCU."""

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import re

from fastcs.attributes import AttrR, AttrW, AttributeIO, AttributeIORef
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, Int, String
from redsun.services import configure_logging

from ...device.dynamic_patterns import DynamicPatternLink
from ..device.due_stack import validate_due_geometry
from ..device.ludl._transport import OxiSerialSession
from ._process import controller_id, identity_arguments, run, serve


class ZdtStateNotSettled(RuntimeError):
    """The Due received a state frame that did not pass admission."""


@dataclass
class EmmshRef(AttributeIORef):
    action: str = ""


class EmmshIO(AttributeIO[bool | int | str, EmmshRef]):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    async def update(self, attr: AttrR):
        match attr.io_ref.action:
            case "enabled":
                value = bool(self.controller.rotary_limits) and not self.controller.rotary_stopped
            case "zdt_status":
                value = self.controller.zdt_status_json
            case "software_dpct_enabled":
                value = self.controller.software_dpct
            case _:
                await self.controller.publish_snapshot(attr)
                return
        await attr.update(value)

    async def send(self, attr: AttrW, value: bool | int):
        if attr.io_ref.action in ("roll", "pitch", "yaw"):
            await self.controller.move_rotary(attr.io_ref.action, value)
        elif attr.io_ref.action == "zdt_start":
            await self.controller.start_zdt_stack(value)
        elif attr.io_ref.action == "zdt_status_request":
            await self.controller.query_zdt_stack(value)
        elif attr.io_ref.action == "zdt_abort":
            await self.controller.abort_zdt_stack(value)
        elif attr.io_ref.action == "zdt_captured":
            await self.controller.acknowledge_zdt_capture(value)
        elif attr.io_ref.action == "led_array":
            await self.controller.set_led_array(value)
        elif attr.io_ref.action == "dpct":
            await self.controller.set_dpct_pattern(value)
        elif attr.io_ref.action == "rotary_range":
            await self.controller.set_rotary_range(value)
        elif attr.io_ref.action == "zdt_bounds":
            await self.controller.set_zdt_bounds(value)
        elif attr.io_ref.action == "rotary_zero":
            if value:
                await self.controller.zero_rotary()
        elif attr.io_ref.action == "rotary_jog":
            await self.controller.jog_rotary(value)
        elif value:
            await self.controller.safe_action(attr.io_ref.action)


class EmmshController(Controller):
    """One serial owner. Firmware counts report targets, with modulo wrap."""

    def __init__(self, session, firmware_revision, *, rotary_limits=None, zdt_stack=False,
                 software_dpct=False, zdt_software=False, zdt_software_only=False, led_array=False):
        self.session = session
        self.firmware_revision = firmware_revision
        self._reported_firmware_identity: str | None = None
        self.rotary_limits = rotary_limits or {}
        self.rotary_stopped = False
        self.lock = asyncio.Lock()
        self.closed = False
        self.software_dpct = software_dpct
        self.led_array_requested = led_array
        self._led_pixels = 0
        self._led_map = False
        self._dynamic_uncertain = False
        self._dynamic_replay: set[str] = set()
        self._dynamic = DynamicPatternLink(
            self._write_dynamic, self._read_dynamic, self._cleanup_dynamic, 1.0
        )
        self.zdt_stack = zdt_stack
        self.zdt_software = zdt_software
        self.zdt_software_only = zdt_software_only
        self.zdt_bounds_enabled = True
        self.zdt_status_json = "{}"
        self._zdt_jog = None
        io = EmmshIO(self)
        super().__init__(ios=[io])
        self.commanded_steps = AttrR(String(), io_ref=EmmshRef(update_period=1.0))
        self.stop = AttrW(Bool(), io_ref=EmmshRef(action="stop"))
        self.enable_rotary = AttrW(Bool(), io_ref=EmmshRef(action="enable_rotary"))
        self.rotary_zero = AttrW(Bool(), io_ref=EmmshRef(action="rotary_zero"))
        self.rotary_range = AttrW(String(), io_ref=EmmshRef(action="rotary_range"))
        self.rotary_jog_request = AttrW(String(), io_ref=EmmshRef(action="rotary_jog"))
        self.blank = AttrW(Bool(), io_ref=EmmshRef(action="blank"))

        self.rotary_enabled = AttrR(Bool(), io_ref=EmmshRef(action="enabled", update_period=1.0))
        self.roll_target = self._rotary_attribute("roll")
        self.pitch_target = self._rotary_attribute("pitch")
        self.yaw_target = self._rotary_attribute("yaw")
        self.software_dpct_enabled = AttrR(
            Bool(), io_ref=EmmshRef(action="software_dpct_enabled", update_period=1.0)
        )
        self.dpct_pattern = AttrW(Int(), io_ref=EmmshRef(action="dpct"))
        self.led_pixels = AttrR(Int(), description="Number of supported compact LED array pixels")
        self.led_map_supported = AttrR(Bool(), description="Static 8-bit LED map supported")
        self.led_array_request = AttrW(String(), io_ref=EmmshRef(action="led_array"))
        self.dynamic_pattern_result = AttrR(
            String(), initial_value="", description="Correlated dynamic LED slot receipt"
        )
        self.zdt_bounds = AttrW(Bool(), io_ref=EmmshRef(action="zdt_bounds"))
        self.zdt_start = AttrW(String(), io_ref=EmmshRef(action="zdt_start"))
        self.zdt_captured = AttrW(String(), io_ref=EmmshRef(action="zdt_captured"))
        self.zdt_software_enabled = AttrR(Bool(), description="Verified Due software capture capability")
        self.zdt_stack_enabled = AttrR(Bool(), description="Verified Due X-stack capability")
        self.zdt_abort = AttrW(Int(), io_ref=EmmshRef(action="zdt_abort"))
        self.zdt_status_request = AttrW(Int(), io_ref=EmmshRef(action="zdt_status_request"))
        self.zdt_status = AttrR(String(), io_ref=EmmshRef(action="zdt_status", update_period=1.0))

    def _rotary_attribute(self, axis: str) -> AttrW:
        return AttrW(Int(units="steps"), io_ref=EmmshRef(action=axis))

    async def set_rotary_range(self, value: str) -> None:
        match = re.fullmatch(r"(roll|pitch|yaw)=(-?[0-9]{1,4}):(-?[0-9]{1,4})", value)
        if match is None or not -2047 <= int(match[2]) < int(match[3]) < 2048:
            raise ValueError("Enter rotary limits as min:max within -2047 to 2047 counts")
        async with self.lock:
            if self.closed or await self._exchange(b"?i") != "ROTARY idle":
                raise RuntimeError("Due rotary axes must be idle to change limits")
            snapshot = json.loads(await self._snapshot())
            self.rotary_limits[match[1]] = (int(match[2]), int(match[3]))
            snapshot["rotary_limits"][match[1]] = self.rotary_limits[match[1]]
            await self.commanded_steps.update(json.dumps(snapshot))

    async def set_zdt_bounds(self, enabled: bool) -> None:
        async with self.lock:
            if self.closed or not self.zdt_stack:
                raise RuntimeError("ZDT bounds control is unavailable")
            self.zdt_bounds_enabled = enabled

    async def start_zdt_stack(self, request: str) -> None:
        """Start an opt-in X stack after fresh firmware admission readbacks."""
        if not self.zdt_stack:
            raise ValueError("ZDT stack start is disabled")
        payload = json.loads(request)
        keys = ("run_id", "first", "step", "count", "lower", "upper")
        required = {*keys, "camera_ready"}
        jog_keys = {"run_id", "camera_ready", "software"}
        is_jog = isinstance(payload, dict) and set(payload) in (
            jog_keys | {"jog"}, jog_keys | {"jog_target"},
            jog_keys | {"jog_target", "jog_window"},
        )
        if not isinstance(payload, dict) or (not is_jog and set(payload) not in (required, required | {"software"})):
            raise ValueError("Expected a stack definition and camera_ready")
        software = payload.get("software", False)
        if self.zdt_software_only and software is not True:
            raise ValueError("This Due profile requires software capture")
        if type(software) is not bool or (software and not self.zdt_software_enabled.get()):
            raise ValueError("Due software capture mode is unavailable")
        if payload["camera_ready"] is not True:
            raise ValueError("Require camera readiness")
        async with self.lock:
            if self.closed:
                raise RuntimeError("MCU service is closed")
            window = payload.get("jog_window")
            wide_jog = is_jog and ("jog_window" in payload or not self.zdt_bounds_enabled)
            if is_jog:
                absolute = "jog_target" in payload
                request_value = payload["jog_target" if absolute else "jog"]
                if software is not True or type(request_value) is not int:
                    raise ValueError("ZDT jog requires software mode and integer tenths of a degree")
                origin = await self._verify_zdt(b"zastate?", "state", wait_for_settle=True)
                if origin is None:
                    raise RuntimeError("Due firmware does not report ZDT position for jogging")
                target = request_value if absolute else origin + request_value
                if not absolute and request_value == 0:
                    raise ValueError("ZDT relative jog must be nonzero")
                # Use the travel envelope, including the firmware's 1 mm limit from origin.
                low, high = max(-3600, origin - 3600), min(3600, origin + 3600)
                if wide_jog:
                    low, high = -36000, 36000
                if "jog_window" in payload:
                    if (
                        not isinstance(window, list) or len(window) != 2
                        or any(type(bound) is not int for bound in window)
                        or not -36000 <= window[0] < window[1] <= 36000
                        or window[1] - window[0] > 3600
                    ):
                        raise ValueError("ZDT jog window must be at most 1 mm within the protocol range")
                    low, high = window
                    if not low <= origin <= high:
                        raise ValueError("ZDT position is outside the requested jog window")
                if not low <= target <= high:
                    raise ValueError("ZDT jog exceeds the configured firmware travel window")
                payload.update(first=target, step=0, count=1, lower=low, upper=high)
            if wide_jog:
                if type(payload["run_id"]) is not int or not 1 <= payload["run_id"] <= 2147483647:
                    raise ValueError("Invalid Due jog run ID")
            else:
                validate_due_geometry({key: payload[key] for key in keys})
            run_id, first, step, count, lower, upper = (payload[k] for k in keys)
            # No retry after any command with an uncertain result.
            verb = "JOGW" if wide_jog else "JOG" if is_jog else "COMMIT"
            commit = f"{verb} {run_id} {first} {step} {count} {lower} {upper} 1"
            rejected = False
            try:
                reply = await self._exchange(commit.encode(), stack_reply=True)
                if reply != "ZSTACK committed":
                    rejected = reply.startswith("ZSTACK error=")
                    raise RuntimeError(f"Due did not commit the ZDT stack: {reply!r}")
                if software and not is_jog:
                    reply = await self._exchange(f"SOFTWARE {run_id}".encode(), stack_reply=True)
                    if reply != "ZSTACK software":
                        raise RuntimeError(f"Due rejected software capture mode: {reply!r}")
                await self._verify_zdt(b"zacfg?", "config")
                observed = await self._verify_zdt(b"zastate?", "state", wait_for_settle=is_jog)
                if is_jog and not absolute and observed != origin:
                    raise RuntimeError("ZDT position changed during jog admission")
                for verb, expected in (("ARM", "ZSTACK armed_host_asserted"), ("START", "ZSTACK started")):
                    reply = await self._exchange(f"{verb} {run_id}".encode(), stack_reply=True)
                    if reply != expected:
                        raise RuntimeError(f"Due rejected ZDT {verb}: {reply!r}")
                self._zdt_jog = (run_id, first) if is_jog else None
            except BaseException as error:
                if rejected:
                    raise
                try:
                    reply = await self._exchange(f"ABORT {run_id}".encode(), stack_reply=True)
                    if reply not in ("ZSTACK stopping", "ZSTACK cancelled"):
                        raise RuntimeError(f"Unexpected abort reply: {reply!r}")
                except Exception as cleanup:
                    error.add_note(f"ZDT abort could not be confirmed: {cleanup}")
                raise

    async def _verify_zdt(self, command: bytes, field: str, *, wait_for_settle: bool = False) -> int | None:
        if not wait_for_settle:
            return await self._read_zdt_verification(command, field)
        # Immediately after motion the X controller can still report residual
        # speed or following error. Retry only a rejected read, never a move.
        deadline = asyncio.get_running_loop().time() + 1.5
        while True:
            try:
                return await self._read_zdt_verification(command, field)
            except ZdtStateNotSettled:
                if asyncio.get_running_loop().time() >= deadline:
                    raise
                await asyncio.sleep(0.05)

    async def _read_zdt_verification(self, command: bytes, field: str) -> int | None:
        pending = await self._exchange(command, verification_reply=True)
        if pending != "ZA pending":
            raise RuntimeError(f"Due did not begin ZDT {field} verification: {pending!r}")
        reply = await self._read_reply(await self.session.read_line(), verification_reply=True)
        if field == "state" and (match := re.fullmatch(r"ZA state_verified=1 position_tenths_deg=(-?[0-9]{1,5})", reply)):
            return int(match[1])
        if field == "state" and reply == "ZA state_verified=0":
            raise ZdtStateNotSettled(f"Due did not verify ZDT state: {reply!r}")
        if reply != f"ZA {field}_verified=1":
            raise RuntimeError(f"Due did not verify ZDT {field}: {reply!r}")

    async def query_zdt_stack(self, run_id: int) -> None:
        if not self.zdt_stack or type(run_id) is not int or not 0 <= run_id <= 2147483647:
            raise ValueError("ZDT stack status is disabled or the run ID is invalid")
        states = ("idle", "active", "complete", "stopping", "abort_unverified", "fault_unverified", "cancelled")
        async with self.lock:
            if self.closed:
                raise RuntimeError("MCU service is closed")
            if run_id == 0:
                position = await self._verify_zdt(b"zastate?", "state", wait_for_settle=True)
                if position is None:
                    raise RuntimeError("Due firmware does not report ZDT position")
                self.zdt_status_json = json.dumps({"run_id": 0, "state": "idle", "position_um": position / 3.6})
                await self.zdt_status.update(self.zdt_status_json)
                return
            reply = await self._exchange(f"STATUS {run_id}".encode("ascii"), stack_reply=True)
            waiting = re.fullmatch(r"ZSTACK waiting_capture index=([0-9]|[12][0-9]|3[01])", reply)
            match = re.fullmatch(r"ZSTACK state=([0-6]) edges=([0-9]{1,10})", reply)
            if waiting is not None:
                status = {"run_id": run_id, "state": "waiting_capture", "edges": int(waiting[1])}
            elif match is not None:
                status = {"run_id": run_id, "state": states[int(match[1])], "edges": int(match[2])}
            else:
                raise RuntimeError(f"Unexpected Due ZDT status: {reply!r}")
            if status["state"] == "complete" and self._zdt_jog is not None and run_id == self._zdt_jog[0]:
                _, target = self._zdt_jog
                # Firmware completion already checks controller position against
                # the target. A second admission query can reject residual speed.
                status["target_um"] = target / 3.6
                self._zdt_jog = None
            self.zdt_status_json = json.dumps(status)
            await self.zdt_status.update(self.zdt_status_json)

    async def acknowledge_zdt_capture(self, value: str) -> None:
        match = re.fullmatch(r"([1-9][0-9]{0,9}) ([0-9]|[12][0-9]|3[01])", value)
        if not self.zdt_software_enabled.get() or match is None or int(match[1]) > 2147483647:
            raise ValueError("Expected a software capture run ID and plane index")
        async with self.lock:
            if self.closed:
                raise RuntimeError("MCU service is closed")
            reply = await self._exchange(f"CAPTURED {value}".encode("ascii"), stack_reply=True)
            if reply != "ZSTACK captured":
                raise RuntimeError(f"Due rejected capture acknowledgement: {reply!r}")

    async def abort_zdt_stack(self, run_id):
        """Abort an identified firmware stack without issuing another motion request."""
        if not self.zdt_stack or type(run_id) is not int or not 1 <= run_id <= 2147483647:
            raise ValueError("ZDT stack abort is disabled or the run ID is invalid")
        async with self.lock:
            if self.closed:
                raise RuntimeError("MCU service is closed")
            command = f"ABORT {run_id}".encode("ascii")
            reply = await self._exchange(command, stack_reply=True)
            if reply == "ZSTACK error=reset_required":
                raise RuntimeError("Due reports an unverified stop. Confirm the Z axis is stationary before resetting.")
            if reply not in ("ZSTACK stopping", "ZSTACK cancelled"):
                raise RuntimeError(f"Due did not acknowledge ZDT stack abort: {reply!r}")

    async def set_led_array(self, request: str) -> None:
        if len(request) > 8192:
            raise ValueError("LED Array request is too long")
        payload = json.loads(request)
        if not isinstance(payload, dict) or set(payload) != {"action", "value"}:
            raise ValueError("Expected LED Array action and value")
        action, value = payload["action"], payload["value"]
        limits = {"pattern": ("lp", 5), "pwm": ("lw", 4), "l4": ("lf", 12),
                  "pixel": ("li", self._led_pixels - 1), "intensity": ("lb", 255)}
        if isinstance(action, str) and action.startswith("dynamic_"):
            if action == "dynamic_map":
                await self._upload_dynamic(value)
            else:
                await self._dynamic_recipe(action, value)
            return
        if action == "map":
            if not self._led_map:
                raise ValueError("This firmware does not support static LED maps")
            if (not isinstance(value, list) or len(value) != self._led_pixels
                    or any(type(v) is not int or not 0 <= v <= 255 for v in value)):
                raise ValueError(f"Expected {self._led_pixels} integer intensities from 0 to 255")
            commands = ["lmb", *(f"lmw{i} {bytes(value[i:i + 20]).hex()}"
                                 for i in range(0, len(value), 20)), "lmc"]
        elif action == "blank" and value is None:
            commands = ["ls0"]
        elif action == "color" and value in ("red", "green", "blue"):
            commands = ["lc" + value[0]]
        elif (isinstance(action, str) and action in limits and type(value) is int
              and 0 <= value <= limits[action][1]):
            commands = [limits[action][0] + str(value)]
        else:
            raise ValueError("Invalid LED Array action or value")

        async with self.lock:
            if self.closed or not self._led_pixels:
                raise RuntimeError("Compact LED Array unavailable")
            begun = False
            try:
                for command in commands:
                    if command == "lmb":
                        begun = True
                    await self._send_led_command(command)
            except BaseException as error:
                if begun:
                    try:
                        await self._send_led_command("lmx")
                    except Exception as cleanup:
                        error.add_note(f"Could not discard staged LED map: {cleanup}")
                raise

    async def _upload_dynamic(self, value) -> None:
        if (
            not isinstance(value, dict)
            or set(value) != {"id", "slot", "pixels"}
            or not isinstance(value["id"], str)
            or re.fullmatch(r"[0-9a-f]{32}", value["id"]) is None
            or type(value["slot"]) is not int or not 1 <= value["slot"] <= 5
            or not isinstance(value["pixels"], list)
            or len(value["pixels"]) != self._led_pixels
            or any(type(pixel) is not int or not 0 <= pixel <= 255 for pixel in value["pixels"])
        ):
            raise ValueError("Expected a dynamic slot in 1..5 and one byte per panel pixel")
        async with self.lock:
            if self.closed or not self._led_pixels or self._dynamic_uncertain:
                raise RuntimeError("Dynamic LED upload unavailable")
            result = await self._dynamic.upload_dynamic_pattern(
                value["slot"], bytes(value["pixels"])
            )
            await self.dynamic_pattern_result.update(json.dumps({"id": value["id"], **asdict(result)}))

    async def _dynamic_recipe(self, action: str, value) -> None:
        if not isinstance(value, list) or not value or type(value[0]) is not int:
            raise ValueError("Expected a dynamic recipe request")
        run, *args = value
        if not 1 <= run <= 0xFFFFFFFF:
            raise ValueError("Invalid dynamic run ID")

        def numbers(values, bounds):
            return len(values) == len(bounds) and all(
                type(number) is int and low <= number <= high
                for number, (low, high) in zip(values, bounds, strict=True)
            )

        commands = []
        if action == "dynamic_arm":
            if not 1 <= len(args) <= 5 or any(
                not isinstance(step, list)
                or not numbers(step, [(1, 5), (1, 0xFFFFFFFF), (0, 0xFFFFFFFF)])
                for step in args
            ):
                raise ValueError("Expected 1..5 dynamic slot receipts")
            for slot, generation, crc in args:
                commands.append((f"DPSLOT? {slot}", f"DPSLOT {slot} {generation} {crc}"))
            commands.append((f"DPCFG {run} {len(args)} 1", f"DPCFG {run}"))
            for index, (slot, generation, _) in enumerate(args):
                commands.append((f"DPSTEP {run} {index} {slot} {generation}", f"DPSTEP {run} {index}"))
            commands.append((f"DPARM {run}", f"DPARM {run} {len(args)}"))
        elif action == "dynamic_next" and numbers(args, [(0, 4), (1, 5), (1, 0xFFFFFFFF)]):
            ordinal, slot, generation = args
            commands.append((f"DPNEXT {run} {ordinal}", f"DPREADY {run} {ordinal} {slot} {generation} {ordinal + 1}"))
        elif action == "dynamic_ack" and numbers(args, [(0, 4)]):
            ordinal = args[0]
            line = f"DPACK {run} {ordinal} {ordinal + 1}"
            commands.append((line, line))
        elif action == "dynamic_finish" and numbers(args, [(1, 5)]):
            commands.append((f"DPSTATUS? {run}", f"DPSTATUS {run} 4 {args[0]} 0"))
            line = f"DPSTOP {run}"
            commands.append((line, line))
        else:
            raise ValueError("Invalid dynamic recipe action or arguments")

        async with self.lock:
            if self.closed or self._led_pixels != 61 or self._dynamic_uncertain:
                raise RuntimeError("Dynamic Due capture unavailable")
            if action == "dynamic_arm":
                await self._dynamic.capabilities()
            for command, reply in commands:
                await self._dynamic.exchange(
                    command, reply, self._dynamic_replay,
                    retries=0 if command.startswith(("DPSLOT?", "DPSTOP")) else 1,
                )
            if action == "dynamic_finish":
                self._dynamic_replay.clear()

    async def _write_dynamic(self, line: str) -> None:
        await asyncio.wait_for(self.session.send_unframed(line.encode("ascii") + b"\n"), 1.0)

    async def _read_dynamic(self, timeout: float) -> str:
        async with asyncio.timeout(timeout):
            data = await self.session.read_line()
            if not data:
                raise TimeoutError("No dynamic pattern response")
            return await self._read_reply(data, dynamic_reply=True)

    async def _cleanup_dynamic(self, command: str) -> None:
        try:
            await self._write_dynamic(command)
            async with asyncio.timeout(1.0):
                while await self._read_dynamic(1.0) != command:
                    pass
        except Exception:
            self._dynamic_uncertain = True

    async def _send_led_command(self, command: str) -> None:
        reply = await self._exchange(command.encode("ascii"))
        if reply != command:
            raise RuntimeError(f"Due rejected LED Array command {command.split()[0]}: {reply}")

    async def set_dpct_pattern(self, value):
        if not self.software_dpct or type(value) is not int or value not in (11, 12, 13, 14):
            raise ValueError("Software DPCT is disabled or pattern is outside 11..14")
        async with self.lock:
            if self.closed:
                raise RuntimeError("MCU service is closed")
            command = f"DPCT {value}"
            if await self._exchange(command.encode("ascii")) != command:
                raise RuntimeError("Due did not acknowledge the DPCT pattern")

    async def zero_rotary(self) -> None:
        async with self.lock:
            if self.closed:
                raise RuntimeError("MCU service is closed")
            reply = await self._exchange(b"ROTARY ZERO")
            if reply != "ROTARY zeroed":
                raise RuntimeError(f"Due did not set the rotary zero: {reply!r}")
            await self.commanded_steps.update(await self._snapshot())

    async def jog_rotary(self, value: str) -> None:
        match = re.fullmatch(r"(roll|pitch|yaw)=(-?[0-9]{1,4})", value)
        if match is None:
            raise ValueError("Expected axis=signed target counts")
        axis, target_text = match.groups()
        target = int(target_text)
        async with self.lock:
            bounds = self.rotary_limits.get(axis)
            if self.closed or bounds is None or not bounds[0] <= target <= bounds[1]:
                raise ValueError(f"{axis} target outside the configured commissioning range")
            if await self._exchange(b"?i") != "ROTARY idle":
                raise RuntimeError("Due rotary axes are busy")
            try:
                await self._safe_action("enable_rotary")
                await self._send_rotary_target(axis, target)
                async with asyncio.timeout(8):
                    while await self._exchange(b"?i") != "ROTARY idle":
                        await asyncio.sleep(0.05)
                await self._release_rotary_outputs()
            except BaseException as error:
                try:
                    await self._safe_action("stop")
                except Exception as cleanup:
                    error.add_note(f"Due stop could not be confirmed: {cleanup}")
                raise

    async def _send_rotary_target(self, axis, target):
        command, label = {"roll": ("r", "Roll"), "pitch": ("t", "Pitch"), "yaw": ("y", "Yaw")}[axis]
        reply = await self._exchange(f"{command}{target}".encode("ascii"))
        if reply == "ERR rotary stopped":
            self.rotary_stopped = True
            await self.rotary_enabled.update(False)
            raise RuntimeError("Due rotary outputs are disabled after a stop")
        if reply != f"{label}: {target}":
            raise RuntimeError(f"Unexpected {axis} target acknowledgement: {reply!r}")

    async def move_rotary(self, axis, target):
        async with self.lock:
            bounds = self.rotary_limits.get(axis)
            if bounds is None or type(target) is not int or not bounds[0] <= target <= bounds[1]:
                raise ValueError(f"{axis} target outside the configured commissioning range")
            if self.closed:
                raise RuntimeError("MCU service is closed")
            if self.rotary_stopped:
                raise RuntimeError("Due rotary outputs are disabled after a stop")
            await self._send_rotary_target(axis, target)

    async def _exchange(self, command, *, stack_reply=False, verification_reply=False):
        reply = await self.session.exchange(command + b"\n", terminator=b"\n")
        return await self._read_reply(
            reply, stack_reply=stack_reply, verification_reply=verification_reply,
        )

    async def _read_reply(self, reply, *, stack_reply=False, verification_reply=False, dynamic_reply=False):
        # The downstream Z controller can emit notices before or after the
        # MCU's local reply. Read past them without resending the command.
        for _ in range(8):
            if not reply.endswith(b"\n"):
                raise RuntimeError("Incomplete MCU response")
            line = reply.decode("ascii").strip()
            if line == "ZSTACK busy" and not stack_reply:
                raise RuntimeError("Due manual command refused while a ZDT stack is reserved")
            unsolicited = line.startswith(("Z ", "ZSTACK EDGE ", "ZSTACK complete ", "ZSTACK stop_unverified "))
            if line.startswith("ZA state_frame=") or (line.startswith("ZA ") and not verification_reply):
                unsolicited = True
            if not dynamic_reply and re.match(
                r"DP(?:CAPS|SLOT|BEGIN|WRITE|COMMIT|ABORT|CFG|STEP|ARM|READY|ACK|STATUS|STOP|ERR) ", line
            ):
                unsolicited = True
            if not unsolicited and (stack_reply or not line.startswith("ZSTACK ")):
                return line
            reply = await self.session.read_line()
        raise RuntimeError("Too many downstream Z notices before MCU response")

    async def snapshot(self):
        async with self.lock:
            return await self._snapshot()

    async def publish_snapshot(self, attr):
        async with self.lock:
            await attr.update(await self._snapshot())

    async def _snapshot(self):
        if self.closed:
            raise RuntimeError("MCU service is closed")
        targets = {}
        for axis, command in (("roll", b"?r"), ("pitch", b"?t"), ("yaw", b"?y")):
            reply = await self._exchange(command)
            if not re.fullmatch(r"-?[0-9]{1,4}", reply) or not -2047 <= int(reply) <= 2047:
                raise RuntimeError(f"Unexpected MCU {axis} target reply: {reply!r}")
            targets[axis] = int(reply)
        return json.dumps({
            "provenance": "commanded_target",
            "configured_firmware_revision": self.firmware_revision,
            "reported_firmware_identity": self._reported_firmware_identity,
            "rotary_limits": self.rotary_limits,
            "steps": targets,
        })

    async def _safe_action(self, action):
        command, expected = {
            "stop": (b"k", "Stopped local axes"),
            "blank": (b"q", "Exited Fixed Mode (Ready)."),
            "enable_rotary": (b"j", "Enabled"),
        }[action]
        if action == "enable_rotary" and not self.rotary_limits:
            raise RuntimeError("Configure rotary limits before enabling outputs")
        if action == "stop":
            self.rotary_stopped = True
            await self.rotary_enabled.update(False)
        reply = await self._exchange(command)
        if reply != expected and not (action == "blank" and reply == "Panel blanked"):
            raise RuntimeError(f"MCU did not acknowledge {action}")
        if action == "enable_rotary":
            self.rotary_stopped = False
            await self.rotary_enabled.update(True)

    async def safe_action(self, action):
        async with self.lock:
            if self.closed:
                raise RuntimeError("MCU service is closed")
            await self._safe_action(action)

    async def connect(self):
        if self.zdt_stack:
            identity = await self._exchange(b"ID?")
            if identity != "ARH-DUE ZDT-X protocol=1":
                raise RuntimeError(f"Expected Due ZDT-X protocol 1, received {identity!r}")
            reply = await self._exchange(b"ZSTACK?", stack_reply=True)
            if reply != "ZSTACK due-x 1":
                raise RuntimeError("Installed Due firmware lacks the commissioned X-stack interface")
            self._reported_firmware_identity = identity
        if self.zdt_software:
            if not self.zdt_stack:
                raise ValueError("Due software capture requires ZDT-X stacks")
            if await self._exchange(b"ZSOFTWARE?", stack_reply=True) != "ZSTACK software 1":
                raise RuntimeError("Installed Due firmware lacks software stack capture")
            await self.zdt_software_enabled.update(True)
        if self.software_dpct and await self._exchange(b"DPCT?") != "DPCT software-snap 1":
            raise RuntimeError("Installed Due firmware lacks software DPCT")
        if self.rotary_limits:
            reply = await self._exchange(b"?s")
            if reply not in ("ROTARY ready", "ROTARY stopped"):
                raise RuntimeError(f"Unexpected Due rotary output state: {reply!r}")
            self.rotary_stopped = reply == "ROTARY stopped"
        if self.led_array_requested:
            try:
                capability = await self._exchange(b"LED?")
            except (RuntimeError, TimeoutError):
                capability = ""
            match = re.fullmatch(r"LED compact=1 pixels=([1-9][0-9]{0,2}) map=(0|8) ttl=software", capability)
            if match:
                self._led_pixels = int(match[1])
                self._led_map = match[2] == "8"
        await self.led_pixels.update(self._led_pixels)
        await self.led_map_supported.update(self._led_map)
        snapshot = await self.snapshot()
        await self.safe_action("blank")
        await self.commanded_steps.update(snapshot)
        await self.rotary_enabled.update(bool(self.rotary_limits) and not self.rotary_stopped)
        await self.software_dpct_enabled.update(self.software_dpct)
        await self.zdt_stack_enabled.update(self.zdt_stack)
        await super().connect()

    async def disconnect(self):
        try:
            await super().disconnect()
        finally:
            await self.close()

    async def _release_rotary_outputs(self):
        if await self._exchange(b"w") != "Rotary outputs off":
            raise RuntimeError("Due did not confirm rotary outputs off")
        self.rotary_stopped = True
        await self.rotary_enabled.update(False)

    async def close(self):
        async with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                try:
                    try:
                        if await self._exchange(b"?i") != "ROTARY idle":
                            raise RuntimeError("Due rotary axes are not idle")
                        await self._release_rotary_outputs()
                    except Exception:
                        # Reserved ZDT work refuses ?i. Stop it before releasing the port.
                        await self._safe_action("stop")
                except BaseException as error:
                    try:
                        await self._safe_action("blank")
                    except Exception as blank_error:
                        error.add_note(f"Due panel blanking could not be confirmed: {blank_error}")
                    raise
                else:
                    await self._safe_action("blank")
            finally:
                self.session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baudrate", type=int, required=True)
    parser.add_argument("--firmware-revision", required=True)
    parser.add_argument("--rotary-limits", help="Commissioned absolute step ranges: roll=0:100,pitch=0:100,yaw=0:100")
    parser.add_argument("--zdt-stack", action="store_true", help="Enable status and abort for commissioned Due X-stack firmware")
    parser.add_argument("--software-dpct", action="store_true")
    parser.add_argument("--led-array", action="store_true", help="Detect and expose compact LED Array commands")
    parser.add_argument("--zdt-software", action="store_true")
    parser.add_argument("--zdt-software-only", action="store_true",
                        help="Reject stack requests that would emit external triggers")
    parser.add_argument("--programming-host", action="store_true",
                        help="Synchronize the Due Programming Port before capability queries")
    identity_arguments(parser, "unified-emmsh")
    options = parser.parse_args(argv)
    if options.zdt_software_only and not options.zdt_software:
        parser.error("--zdt-software-only requires --zdt-software")
    if options.zdt_software and not options.zdt_stack:
        parser.error("--zdt-software requires --zdt-stack")
    if any(not value or "REPLACE_" in value for value in (options.port, options.firmware_revision)):
        parser.error("Supply the MCU serial port and installed firmware revision")
    limits = {}
    for entry in options.rotary_limits.split(",") if options.rotary_limits else ():
        match = re.fullmatch(r"(roll|pitch|yaw)=(-?[0-9]{1,4}):(-?[0-9]{1,4})", entry)
        if match is None:
            parser.error("Invalid rotary range; expected axis=min:max")
        axis, lower, upper = match.groups()
        low, high = int(lower), int(upper)
        if axis in limits or not -2047 <= low < high < 2048:
            parser.error("Rotary limits must be unique and within steps -2047 through 2047")
        limits[axis] = (low, high)
    configure_logging()

    async def run_service():
        session = OxiSerialSession.open(
            options.port, baudrate=options.baudrate, stopbits=1, timeout=0.5,
        )
        controller = EmmshController(
            session, options.firmware_revision, rotary_limits=limits, software_dpct=options.software_dpct,
            zdt_stack=options.zdt_stack, zdt_software=options.zdt_software,
            zdt_software_only=options.zdt_software_only, led_array=options.led_array,
        )
        if options.programming_host:
            try:
                await synchronize_programming_host(session)
            except BaseException:
                session.close()
                raise
        try:
            await serve(controller, controller_id(options))
        except BaseException as error:
            try:
                await controller.close()
            except Exception as cleanup_error:
                error.add_note(f"Due service cleanup also failed: {cleanup_error}")
            raise
        else:
            await controller.close()

    return run(run_service())


async def synchronize_programming_host(session):
    """Wait for reset, terminate a partial line and discard bounded startup output."""
    await asyncio.sleep(1.0)
    await session.exchange(b"\n", terminator=b"\n")
    async with asyncio.timeout(3):
        for _ in range(32):
            if not await session.read_line():
                return
    raise RuntimeError("Due Programming Port startup output did not settle")


if __name__ == "__main__":
    raise SystemExit(main())
