"""One AA AOTF controller held open by a standalone Redsun PVA service.

The service owns one Micro-Manager Core and its SerialManager/AAAOTF devices
for the service lifetime. Profile requests do not close or reopen COM ports.
Laser-source controls are not loaded or commanded by this service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

from fastcs.attributes import AttributeIO, AttributeIORef, AttrR, AttrRW, AttrW
from fastcs.controllers import Controller
from fastcs.datatypes import String
from fastcs.logging import logger
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from pymmcore import CMMCore
from redsun.services import configure_logging, serve_controller

if TYPE_CHECKING:
    from collections.abc import Mapping

AOTF_CHANNELS: Final = tuple(range(1, 9))
PROFILE_SCHEMA: Final = "redsun.aaaotf-profiles/v1"
FREQUENCY_PROPERTY: Final = "Frequency (MHz)"
INTENSITY_PROPERTY: Final = "Maximum intensity (dB)"
POWER_PROPERTY: Final = "Power (% of max)"
STATE_PROPERTY: Final = "State"
CHANNEL_PROPERTY: Final = "Channel"
PROPERTY_LIMITS: Final = {
    CHANNEL_PROPERTY: (1, 8, int),
    STATE_PROPERTY: (0, 1, int),
    FREQUENCY_PROPERTY: (50, 200, float),
    INTENSITY_PROPERTY: (0, 2200, int),
    POWER_PROPERTY: (0, 100, float),
}
PVA_PREFIX: Final = "HOLOSPIM-AAAOTF:"
LEASE_SECONDS: Final = 5.0


@dataclass(frozen=True)
class SlotSetting:
    """Static adapter settings for one wavelength slot."""

    frequency_mhz: float
    maximum_intensity: int
    power_percent: float


@dataclass(frozen=True)
class AotfProfile:
    """Instrument-bundle-owned slot settings and named shutter states."""

    device_label: str
    slots: Mapping[int, SlotSetting]
    profiles: Mapping[str, frozenset[int]]

    @classmethod
    def from_json(cls, path: Path) -> AotfProfile:
        """Read and validate a local profile without opening hardware."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"profile schema must be {PROFILE_SCHEMA!r}")
        label = raw.get("device_label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("device_label must be a nonempty string")

        slot_settings: dict[int, SlotSetting] = {}
        for slot_text, row in raw.get("slots", {}).items():
            slot = int(slot_text)
            if slot not in AOTF_CHANNELS:
                raise ValueError(f"AOTF slot must be in 1..8: {slot}")
            frequency = float(row["frequency_mhz"])
            intensity = row["maximum_intensity"]
            power = float(row["power_percent"])
            if not math.isfinite(frequency) or not 50 <= frequency <= 200:
                raise ValueError(f"slot {slot} frequency must be in 50..200 MHz")
            if type(intensity) is not int or not 0 <= intensity <= 2200:
                raise ValueError(f"slot {slot} maximum_intensity must be in 0..2200")
            if not math.isfinite(power) or not 0 <= power <= 100:
                raise ValueError(f"slot {slot} power_percent must be in 0..100")
            slot_settings[slot] = SlotSetting(frequency, intensity, power)
        preserved = raw.get("preserve_rf_slots", [])
        if not isinstance(preserved, list) or any(
            type(slot) is not int or slot not in AOTF_CHANNELS for slot in preserved
        ):
            raise ValueError("preserve_rf_slots must list integer slots in 1..8")
        if preserved and slot_settings:
            raise ValueError("choose either slots or preserve_rf_slots")
        enabled_slots = set(preserved) | slot_settings.keys()
        if not enabled_slots:
            raise ValueError("declare at least one AOTF slot")

        profiles: dict[str, frozenset[int]] = {}
        for name, values in raw.get("profiles", {}).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("profile names must be nonempty strings")
            slots = frozenset(values)
            if any(
                type(slot) is not int or slot not in enabled_slots for slot in slots
            ):
                raise ValueError(f"profile {name!r} references an unconfigured slot")
            profiles[name] = slots
        if profiles.get("all_off") != frozenset():
            raise ValueError("profiles must define all_off as an empty slot set")
        return cls(label.strip(), slot_settings, profiles)


class AAAOTFCore:
    """Synchronous MMCore owner; one instance retains the COM port handle."""

    def __init__(self, core: Any, profile: AotfProfile) -> None:
        self.core = core
        self.profile = profile
        self.states = {slot: 0 for slot in AOTF_CHANNELS}
        self.active_profile = "unknown"
        self.fault = ""
        self.connected = False
        self.last_transition: dict[str, Any] = {}
        self.properties: dict[str, str] = {}
        self.instance_id = uuid4().hex

    def initialize(self) -> None:
        """Verify the adapter, turn every slot off, and configure while dark."""
        label = self.profile.device_label
        loaded = {str(item) for item in self.core.getLoadedDevices()}
        if label not in loaded:
            raise RuntimeError(f"MMCore config did not load AOTF device {label!r}")
        if self.core.getDeviceLibrary(label) != "AAAOTF":
            raise RuntimeError(f"Unexpected adapter for {label!r}")
        try:
            self.confirm_all_off()
            for slot, setting in sorted(self.profile.slots.items()):
                self._write_settings(slot, setting)
            self._capture_properties()
            self.active_profile = "all_off"
            self.connected = True
        except Exception:
            self.safe_shutdown()
            raise

    def apply_profile(self, name: str) -> dict[str, Any]:
        """Switch a named profile with off-before-on ordering and readback."""
        if name not in self.profile.profiles:
            raise ValueError(f"unknown AOTF profile {name!r}")
        if self.fault:
            raise RuntimeError(f"AOTF service faulted; restart service: {self.fault}")
        target = self.profile.profiles[name]
        started_ns = time.perf_counter_ns()
        try:
            # Break-before-make for slots that are being replaced.
            for slot in AOTF_CHANNELS:
                if self.states[slot] and slot not in target:
                    self._set_state(slot, 0)
            # Slot settings are written once during startup. The service is
            # the sole command owner, so profile transitions only touch State.
            for slot in sorted(target):
                if not self.states[slot]:
                    self._set_state(slot, 1)
            self._capture_properties()
            # Each State write above already has an MMCore readback. Unchanged
            # slots remain under this service's exclusive command ownership.
            self.active_profile = name
            self.last_transition = {
                "profile": name,
                "states": self.states.copy(),
                "elapsed_ms": (time.perf_counter_ns() - started_ns) / 1e6,
                "result": "completed",
            }
            return self.snapshot()
        except BaseException as exc:
            self._record_failure(name, started_ns, exc)
            raise

    def apply_properties(self, values: Mapping[str, int | float | str]) -> None:
        """Apply a leased batch to the selected channel, with State written last."""
        updates: dict[str, int | float] = {}
        if not values:
            raise ValueError("No AOTF properties supplied")
        for name, raw in values.items():
            if name not in self.properties or name not in PROPERTY_LIMITS:
                raise ValueError(f"AOTF property {name!r} is read-only or unknown")
            lower, upper, value_type = PROPERTY_LIMITS[name]
            value = float(raw)
            if not lower <= value <= upper or (
                value_type is int and not value.is_integer()
            ):
                raise ValueError(f"Invalid AOTF {name} value: {raw!r}")
            updates[name] = value_type(value)

        selected = int(self.properties[CHANNEL_PROPERTY])
        slot = int(updates.pop(CHANNEL_PROPERTY, selected))
        state = updates.pop(STATE_PROPERTY, None)
        if any(self.states.values()) and (slot != selected or updates):
            raise ValueError("Set all_off before changing AOTF channel or RF settings")
        if self.fault:
            raise RuntimeError(f"AOTF service faulted; restart service: {self.fault}")

        started_ns = time.perf_counter_ns()
        try:
            self._select(slot)
            if INTENSITY_PROPERTY in updates and POWER_PROPERTY not in updates:
                updates[POWER_PROPERTY] = float(
                    self.core.getProperty(self.profile.device_label, POWER_PROPERTY)
                )
            for name in (FREQUENCY_PROPERTY, INTENSITY_PROPERTY, POWER_PROPERTY):
                if name in updates:
                    self._set_numeric(name, updates[name], slot)
            if state is not None:
                self._set_state(slot, int(state))
            self._capture_properties()
            self.active_profile = "manual" if any(self.states.values()) else "all_off"
            self.last_transition = {
                "profile": "properties",
                "states": self.states.copy(),
                "elapsed_ms": (time.perf_counter_ns() - started_ns) / 1e6,
                "result": "completed",
            }
        except BaseException as exc:
            self._record_failure("properties", started_ns, exc)
            raise

    def _record_failure(self, name: str, started_ns: int, exc: BaseException) -> None:
        self.fault = f"{type(exc).__name__}: {exc}"
        self.active_profile = "unknown"
        self.last_transition = {
            "profile": name,
            "elapsed_ms": (time.perf_counter_ns() - started_ns) / 1e6,
            "result": "outcome_unknown",
            "error": self.fault,
        }
        logger.exception("AA AOTF command outcome is unknown")

    def _capture_properties(self) -> None:
        label = self.profile.device_label
        self.properties = {
            name: self.core.getProperty(label, name)
            for name in self.core.getDevicePropertyNames(label)
        }

    def snapshot(self) -> dict[str, Any]:
        """Return service-side readback and last-command timing."""
        return {
            "schema": "redsun.aaaotf-status/v1",
            "instance_id": self.instance_id,
            "connected": self.connected,
            "active_profile": self.active_profile,
            "slot_states": {str(key): value for key, value in self.states.items()},
            "properties": self.properties.copy(),
            "writable_properties": [
                name for name in self.properties if name in PROPERTY_LIMITS
            ],
            "fault": self.fault,
            "last_transition": self.last_transition,
        }

    def confirm_all_off(self) -> None:
        """Request and check State 0 on every slot without releasing the port."""
        errors: list[str] = []
        selected = int(
            self.core.getProperty(self.profile.device_label, CHANNEL_PROPERTY)
        )
        # The adapter can open the next slot when its selector moves while ON.
        for slot in (selected, *(item for item in AOTF_CHANNELS if item != selected)):
            try:
                self._set_state(slot, 0)
            except Exception as exc:
                errors.append(f"slot {slot}: {exc!r}")
        if errors:
            self.fault = "; ".join(filter(None, (self.fault, *errors)))
            self.active_profile = "unknown"
            raise RuntimeError(f"Could not confirm all slots OFF: {errors}")
        else:
            self._capture_properties()
            self.active_profile = "all_off"

    def safe_shutdown(self) -> None:
        """Attempt all-off cleanup before unloading MMCore devices."""
        try:
            self.confirm_all_off()
        except Exception:
            logger.exception("AA AOTF shutdown could not confirm all slots OFF")
        self.connected = False

    def close(self) -> None:
        """Release the Core and its serial adapter exactly once at service stop."""
        self.safe_shutdown()
        self.core.unloadAllDevices()

    def _select(self, slot: int) -> None:
        self.core.setProperty(self.profile.device_label, CHANNEL_PROPERTY, str(slot))
        actual = int(
            float(self.core.getProperty(self.profile.device_label, CHANNEL_PROPERTY))
        )
        if actual != slot:
            raise RuntimeError(f"Channel readback {actual} != requested {slot}")

    def _read_state(self, slot: int) -> int:
        self._select(slot)
        actual = int(
            float(self.core.getProperty(self.profile.device_label, STATE_PROPERTY))
        )
        if actual not in (0, 1):
            raise RuntimeError(f"slot {slot} has invalid State readback {actual}")
        return actual

    def _set_state(self, slot: int, state: int) -> None:
        self._select(slot)
        self.core.setProperty(self.profile.device_label, STATE_PROPERTY, str(state))
        actual = int(
            float(self.core.getProperty(self.profile.device_label, STATE_PROPERTY))
        )
        if actual != state:
            raise RuntimeError(
                f"slot {slot} State readback {actual} != requested {state}"
            )
        self.states[slot] = state

    def _write_settings(self, slot: int, setting: SlotSetting) -> None:
        if self._read_state(slot) != 0:
            raise RuntimeError(f"refusing to change slot {slot} settings while ON")
        self._select(slot)
        for prop, requested in (
            (FREQUENCY_PROPERTY, setting.frequency_mhz),
            (INTENSITY_PROPERTY, setting.maximum_intensity),
            (POWER_PROPERTY, setting.power_percent),
        ):
            self.core.setProperty(self.profile.device_label, prop, str(requested))
            actual = float(self.core.getProperty(self.profile.device_label, prop))
            if not math.isfinite(actual) or not math.isclose(
                actual, float(requested), rel_tol=0.0, abs_tol=0.001
            ):
                raise RuntimeError(
                    f"slot {slot} {prop} readback {actual} != requested {requested}"
                )

    def _set_numeric(self, prop: str, requested: int | float, slot: int) -> None:
        self.core.setProperty(self.profile.device_label, prop, str(requested))
        actual = float(self.core.getProperty(self.profile.device_label, prop))
        if not math.isfinite(actual) or not math.isclose(
            actual, float(requested), rel_tol=0.0, abs_tol=0.001
        ):
            raise RuntimeError(f"slot {slot} {prop} readback {actual} != {requested}")


@dataclass
class AotfRef(AttributeIORef):
    """Selects one state signal published by the PVA service."""

    field: str = "profile"


class AotfProfileIO(AttributeIO[str, AotfRef]):
    """Serialize Core calls and serve profile, state and timing readbacks."""

    def __init__(self, aotf: AAAOTFCore, *, persistent: bool) -> None:
        super().__init__()
        self.aotf = aotf
        self.persistent = persistent
        self._lock = asyncio.Lock()
        self.owner: str | None = None
        self.deadline = 0.0

    @staticmethod
    async def _core_call(operation: Any, *args: Any) -> None:
        # A cancelled PV request must finish MMCore I/O before releasing the lock.
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _expire_if_due(self) -> None:
        if self.owner is not None and time.monotonic() >= self.deadline:
            self.owner = None
            await self._core_call(self.aotf.confirm_all_off)

    async def watch_owner(self) -> None:
        """Close the shutter if the attached session stops renewing its claim."""
        while True:
            await asyncio.sleep(0.25)
            async with self._lock:
                try:
                    await self._expire_if_due()
                except RuntimeError:
                    logger.exception("AA AOTF writer expired with OFF unconfirmed")

    async def close(self) -> None:
        """Finish outstanding commands before the Core releases the port."""
        async with self._lock:
            await self._core_call(self.aotf.close)

    async def writer_command(self, value: str) -> None:
        """Process a session claim, renewal, named profile or release."""
        command = json.loads(value)
        action, token = command["action"], command["token"]
        if not isinstance(token, str) or not token:
            raise ValueError("AOTF writer token is required")
        async with self._lock:
            await self._expire_if_due()
            if action == "claim":
                if self.owner is not None:
                    raise RuntimeError("AOTF already has a writer")
                if self.aotf.fault or not self.aotf.connected:
                    raise RuntimeError("AOTF is not ready")
                await self._core_call(self.aotf.confirm_all_off)
                self.owner = token
            elif token != self.owner:
                raise RuntimeError("AOTF writer claim is missing or expired")
            elif action == "release":
                self.owner = None
                await self._core_call(self.aotf.confirm_all_off)
                logger.info("AAAOTF session released with all eight slots OFF")
                return
            elif action in ("profile", "properties"):
                operation = (
                    self.aotf.apply_profile
                    if action == "profile"
                    else self.aotf.apply_properties
                )
                try:
                    await self._core_call(operation, command[action])
                except BaseException as error:
                    if self.aotf.fault or isinstance(error, asyncio.CancelledError):
                        self.owner = None
                        try:
                            await self._core_call(self.aotf.confirm_all_off)
                        except RuntimeError:
                            logger.exception(
                                "AA AOTF command fault with OFF unconfirmed"
                            )
                    raise
            elif action != "renew":
                raise ValueError(f"Unknown AOTF writer action: {action!r}")
            self.deadline = time.monotonic() + LEASE_SECONDS

    async def send(self, attr: AttrW[str, AotfRef], value: str) -> None:
        """Apply a named profile request and return its readback."""
        if attr.io_ref.field == "writer_command" and self.persistent:
            await self.writer_command(value)
            return
        if attr.io_ref.field != "profile":
            raise ValueError(f"{attr.io_ref.field} is read-only")
        if self.persistent:
            raise RuntimeError("Use the attached AOTF device to change profiles")
        async with self._lock:
            await self._core_call(self.aotf.apply_profile, value.strip())
        if isinstance(attr, AttrR):
            await self.update(attr)

    async def update(self, attr: AttrR[str, AotfRef]) -> None:
        """Publish the selected profile, service status or fault text."""
        async with self._lock:
            if attr.io_ref.field == "profile":
                value = self.aotf.active_profile
            elif attr.io_ref.field == "fault":
                value = self.aotf.fault
            else:
                value = json.dumps(self.aotf.snapshot(), separators=(",", ":"))
        await attr.update(value)


class AAAOTFController(Controller):
    """One persistent AA AOTF link with named shutter-profile commands."""

    def __init__(self, aotf: AAAOTFCore, *, persistent: bool = False) -> None:
        io = AotfProfileIO(aotf, persistent=persistent)
        super().__init__(ios=[io])
        self.aotf = aotf
        self.profile_io = io
        self.profile = AttrRW(
            String(),
            io_ref=AotfRef(field="profile", update_period=0.25 if persistent else None),
            description="Named shutter profile from the local instrument profile",
        )
        if persistent:
            self.writer_command = AttrW(
                String(), io_ref=AotfRef(field="writer_command")
            )
        self.active_profile = AttrR(
            String(), io_ref=AotfRef(field="profile", update_period=1.0)
        )
        self.status = AttrR(String(), io_ref=AotfRef(field="status", update_period=1.0))
        self.fault = AttrR(String(), io_ref=AotfRef(field="fault", update_period=1.0))


def _load_core(
    config: Path, adapter_path: Path, core_log: Path | None = None
) -> CMMCore:
    """Create one Core and load the explicit Micro-Manager configuration."""
    if not config.is_file():
        raise FileNotFoundError(config)
    if not (adapter_path / "mmgr_dal_SerialManager.dll").is_file():
        raise FileNotFoundError(adapter_path / "mmgr_dal_SerialManager.dll")
    if not (adapter_path / "mmgr_dal_AAAOTF.dll").is_file():
        raise FileNotFoundError(adapter_path / "mmgr_dal_AAAOTF.dll")
    core = CMMCore()
    try:
        if core_log is not None:
            core_log.parent.mkdir(parents=True, exist_ok=True)
            core.setPrimaryLogFile(str(core_log), True)
        core.setDeviceAdapterSearchPaths([str(adapter_path)])
        core.loadSystemConfiguration(str(config))
    except BaseException:
        logger.exception("AA AOTF MMCore failed while loading its configuration")
        try:
            core.unloadAllDevices()
        except Exception:
            logger.exception("Could not unload MMCore after failed AOTF startup")
        raise
    return core


def build_controller(
    config: Path,
    adapter_path: Path,
    profile_path: Path,
    *,
    core_log: Path | None = None,
    core: Any | None = None,
    persistent: bool = False,
) -> AAAOTFController:
    """Load the bundle-owned profile and hold its MMCore link for the service."""
    profile = AotfProfile.from_json(profile_path)
    owns_core = core is None
    active_core = (
        _load_core(config, adapter_path, core_log=core_log) if core is None else core
    )
    aotf = AAAOTFCore(active_core, profile)
    try:
        aotf.initialize()
    except BaseException:
        logger.exception("AA AOTF device failed during all-off initialization")
        if owns_core:
            try:
                active_core.unloadAllDevices()
            except Exception:
                logger.exception("Could not unload MMCore after failed AOTF init")
        raise
    logger.info(
        "AA AOTF service initialized with all eight slots OFF; "
        "Core connection remains open"
    )
    return AAAOTFController(aotf, persistent=persistent)


async def serve_aotf(options: argparse.Namespace) -> None:
    """Hold the AOTF connection until the standalone service is stopped."""
    controller = build_controller(
        options.config,
        options.adapter_path,
        options.profile,
        core_log=options.core_log,
        persistent=True,
    )
    watchdog = asyncio.create_task(controller.profile_io.watch_owner())
    try:
        await serve_controller(controller, EpicsPVATransport(), options.prefix)
    finally:
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)
        await controller.profile_io.close()


def main(argv: list[str] | None = None) -> int:
    """Run the PVA hardware-owner process for a local instrument profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="local MMCore config"
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        type=Path,
        help="matching MMCore adapter directory",
    )
    parser.add_argument(
        "--profile", required=True, type=Path, help="local AOTF slot/profile JSON"
    )
    parser.add_argument(
        "--core-log",
        type=Path,
        help="optional primary MMCore log file for startup and device diagnostics",
    )
    parser.add_argument("--prefix", default=PVA_PREFIX, help="local PVA prefix")
    options = parser.parse_args(argv)
    os.environ["EPICS_PVAS_INTF_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_PVAS_AUTO_BEACON_ADDR_LIST"] = "NO"
    configure_logging()
    with suppress(KeyboardInterrupt):
        asyncio.run(serve_aotf(options))
    return 0


if __name__ == "__main__":
    sys.exit(main())
