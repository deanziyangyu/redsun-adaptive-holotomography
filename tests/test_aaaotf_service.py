"""Hardware-free tests for the managed AA AOTF shutter service."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from redsun import Session
from redsun._config import SessionFile
from redsun.session._declarations import read_services

from redsun_aht.configurations import registry
from redsun_aht.configurations.profile import read_arh_profile, read_holospim_profile
from redsun_aht.services.aaaotf import (
    AAAOTFController,
    AAAOTFCore,
    AotfProfile,
    SlotSetting,
)


def test_holospim_attaches_aaaotf_by_default(tmp_path: Path) -> None:
    """A legacy host profile inherits the laser attachment on HoloSPIM only."""
    path = tmp_path / "host.yaml"
    path.write_text(
        "session: host\nservices:\n  transport: pv-access\n", encoding="utf-8"
    )
    config = read_holospim_profile(path)
    profile = SessionFile.model_validate(
        {**config, "services": {"laser_service": config["services"]["laser_service"]}}
    )
    service = read_services(Session, profile.services, "pv-access")["laser_service"]
    assert not service.launched
    assert service.prefix == "HOLOSPIM-AAAOTF:"
    assert service.address == "127.0.0.1"
    assert config["devices"]["laser"]["plugin_id"] == "aaaotf"
    assert "laser" not in read_arh_profile(path).get("devices", {})
    assert "aaaotf" in registry()[1]


class FakeCore:
    """Minimal MMCore property model for the multiplexed AAAOTF adapter."""

    def __init__(self) -> None:
        self.channel = 1
        self.writes: list[tuple[int, str, str]] = []
        self.slots = {
            slot: {
                "State": "0",
                "Frequency (MHz)": "0",
                "Maximum intensity (dB)": "0",
                "Power (% of max)": "0",
            }
            for slot in range(1, 9)
        }

    def getLoadedDevices(self) -> list[str]:
        return ["AAAOTF"]

    def getDeviceLibrary(self, label: str) -> str:
        assert label == "AAAOTF"
        return "AAAOTF"

    def getDevicePropertyNames(self, label: str) -> list[str]:
        assert label == "AAAOTF"
        return ["Name", "Description", "Port", "Channel", *self.slots[self.channel]]

    def setProperty(self, label: str, prop: str, value: str) -> None:
        assert label == "AAAOTF"
        if prop == "Channel":
            self.channel = int(value)
        else:
            self.writes.append((self.channel, prop, value))
            self.slots[self.channel][prop] = value

    def getProperty(self, label: str, prop: str) -> str:
        assert label == "AAAOTF"
        if prop in ("Name", "Description", "Port"):
            return {"Name": "AAAOTF", "Description": "Test AOTF", "Port": "TEST"}[prop]
        if prop == "Channel":
            return str(self.channel)
        return self.slots[self.channel][prop]

    def unloadAllDevices(self) -> None:
        pass


def test_startup_blanks_selected_live_slot_before_changing_channel() -> None:
    class SwitchingCore(FakeCore):
        def setProperty(self, label: str, prop: str, value: str) -> None:
            if prop == "Channel" and int(value) != self.channel:
                assert self.slots[self.channel]["State"] == "0"
            super().setProperty(label, prop, value)

    core = SwitchingCore()
    core.channel = 5
    core.slots[5]["State"] = "1"
    aotf = AAAOTFCore(core, make_profile())
    aotf.initialize()
    assert core.writes[0] == (5, "State", "0")
    assert set(aotf.states.values()) == {0}


def test_intensity_edit_preserves_selected_channels_power() -> None:
    core = FakeCore()
    aotf = AAAOTFCore(core, make_profile())
    aotf.initialize()
    aotf.apply_properties({"Channel": 1, "Power (% of max)": 25})
    aotf.apply_properties({"Channel": 2, "Power (% of max)": 75})
    aotf.apply_properties({"Channel": 1, "Maximum intensity (dB)": 1600})
    assert float(core.slots[1]["Power (% of max)"]) == 25


def make_profile() -> AotfProfile:
    return AotfProfile(
        device_label="AAAOTF",
        slots={
            1: SlotSetting(112.28, 1750, 100.0),
            2: SlotSetting(92.63, 1900, 100.0),
        },
        profiles={
            "all_off": frozenset(),
            "488_full": frozenset({1}),
            "561_full": frozenset({2}),
            "both_full": frozenset({1, 2}),
        },
    )


def test_initialize_turns_all_slots_off_and_sets_configured_slots() -> None:
    core = FakeCore()
    aotf = AAAOTFCore(core, make_profile())

    aotf.initialize()

    assert aotf.snapshot()["connected"] is True
    assert set(aotf.states.values()) == {0}
    assert core.slots[1]["Frequency (MHz)"] == "112.28"
    assert core.slots[2]["Maximum intensity (dB)"] == "1900"


def test_profiles_switch_break_before_make_and_report_readback() -> None:
    core = FakeCore()
    aotf = AAAOTFCore(core, make_profile())
    aotf.initialize()

    core.writes.clear()
    aotf.apply_profile("488_full")
    assert aotf.states[1] == 1 and aotf.states[2] == 0
    assert [entry[1] for entry in core.writes] == ["State"]
    core.writes.clear()
    aotf.apply_profile("561_full")
    assert aotf.states[1] == 0 and aotf.states[2] == 1
    state_writes = [
        (*entry[:2], int(entry[2])) for entry in core.writes if entry[1] == "State"
    ]
    assert state_writes == [(1, "State", 0), (2, "State", 1)]
    aotf.apply_profile("both_full")
    assert aotf.states[1] == 1 and aotf.states[2] == 1
    assert [entry[1] for entry in core.writes if entry[1] != "State"] == []
    assert aotf.last_transition["result"] == "completed"


def test_profile_json_rejects_missing_all_off(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "schema": "redsun.aaaotf-profiles/v1",
        "device_label": "AAAOTF",
        "slots": {
            "1": {
                "frequency_mhz": 112.28,
                "maximum_intensity": 1750,
                "power_percent": 100,
            }
        },
        "profiles": {"on": [1]},
    }
    source = tmp_path / "profiles.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="all_off"):
        AotfProfile.from_json(source)


def test_preserved_rf_profile_writes_only_state(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    payload = {
        "schema": "redsun.aaaotf-profiles/v1",
        "device_label": "AAAOTF",
        "preserve_rf_slots": [1],
        "profiles": {"all_off": [], "channel_1_on": [1]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    core = FakeCore()
    service = AAAOTFCore(core, AotfProfile.from_json(path))
    service.initialize()
    service.apply_profile("channel_1_on")
    service.confirm_all_off()
    assert set(service.states.values()) == {0}
    assert {name for _, name, _ in core.writes} == {"State"}


def test_shutdown_turns_every_slot_off_and_releases_connection() -> None:
    core = FakeCore()
    aotf = AAAOTFCore(core, make_profile())
    aotf.initialize()
    aotf.apply_profile("both_full")

    aotf.close()

    assert set(aotf.states.values()) == {0}
    assert aotf.connected is False


@pytest.mark.asyncio
async def test_live_properties_use_same_writer_and_leave_shutter_safe() -> None:
    core = FakeCore()
    aotf = AAAOTFCore(core, make_profile())
    aotf.initialize()
    io = AAAOTFController(aotf, persistent=True).profile_io
    await io.writer_command('{"action":"claim","token":"writer"}')
    values = {
        "Channel": 1,
        "Frequency (MHz)": 92.63,
        "Maximum intensity (dB)": 1900,
        "Power (% of max)": 100,
    }
    command = {"action": "properties", "token": "writer", "properties": values}
    core.writes.clear()
    await io.writer_command(json.dumps(command))
    assert core.writes == [
        (1, "Frequency (MHz)", "92.63"),
        (1, "Maximum intensity (dB)", "1900"),
        (1, "Power (% of max)", "100.0"),
    ]
    assert aotf.snapshot()["properties"]["Port"] == "TEST"
    assert set(aotf.snapshot()["writable_properties"]) == {*values, "State"}
    await io.writer_command(
        json.dumps({**command, "properties": {"State": 1}})
    )
    with pytest.raises(ValueError, match="Set all_off"):
        await io.writer_command(json.dumps({**command, "properties": {"Channel": 2}}))
    with pytest.raises(ValueError, match="read-only"):
        await io.writer_command(
            json.dumps({**command, "properties": {"Port": "OTHER"}})
        )
    assert aotf.states[1] == 1 and io.owner == "writer"

    core.writes.clear()
    await io.writer_command('{"action":"release","token":"writer"}')
    assert core.writes[0] == (1, "State", "0")
    assert aotf.connected and set(aotf.states.values()) == {0}


@pytest.mark.asyncio
async def test_persistent_writer_handoff_and_expiry_keep_one_core() -> None:
    core = FakeCore()
    aotf = AAAOTFCore(core, make_profile())
    aotf.initialize()
    controller = AAAOTFController(aotf, persistent=True)
    io = controller.profile_io

    async def command(action: str, token: str, **fields: str) -> None:
        await io.writer_command(json.dumps(dict(action=action, token=token, **fields)))

    await command("claim", "first")
    await command("profile", "first", profile="488_full")
    assert aotf.active_profile == "488_full"
    with pytest.raises(RuntimeError, match="already has a writer"):
        await command("claim", "second")
    with pytest.raises(RuntimeError, match="missing or expired"):
        await command("profile", "second", profile="all_off")

    await command("release", "first")
    assert aotf.connected and aotf.active_profile == "all_off"
    await command("claim", "second")
    await command("profile", "second", profile="561_full")
    io.deadline = 0
    watchdog = asyncio.create_task(io.watch_owner())
    try:
        async with asyncio.timeout(2):
            while aotf.active_profile != "all_off":
                await asyncio.sleep(0.01)
    finally:
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)
    assert aotf.connected and io.owner is None
    with pytest.raises(RuntimeError, match="missing or expired"):
        await command("renew", "second")


@pytest.mark.asyncio
async def test_persistent_profile_error_revokes_writer_and_attempts_off() -> None:
    class FaultingCore(FakeCore):
        def setProperty(self, label: str, prop: str, value: str) -> None:
            super().setProperty(label, prop, value)
            if prop == "State" and value == "1":
                raise OSError("link error after write")

    aotf = AAAOTFCore(FaultingCore(), make_profile())
    aotf.initialize()
    io = AAAOTFController(aotf, persistent=True).profile_io
    await io.writer_command('{"action":"claim","token":"writer"}')
    with pytest.raises(OSError, match="link error"):
        await io.writer_command(
            '{"action":"profile","token":"writer","profile":"488_full"}'
        )
    assert io.owner is None
    assert aotf.active_profile == "all_off"
    assert aotf.fault and aotf.last_transition["result"] == "outcome_unknown"
    with pytest.raises(RuntimeError, match="not ready"):
        await io.writer_command('{"action":"claim","token":"next"}')


@pytest.mark.asyncio
async def test_cancelled_profile_blanks_after_inflight_io_and_revokes_writer():
    from threading import Event

    entered, release = Event(), Event()

    class HeldCore(AAAOTFCore):
        def apply_profile(self, name):
            result = super().apply_profile(name)
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release the MMCore call")
            return result

    aotf = HeldCore(FakeCore(), make_profile())
    aotf.initialize()
    io = AAAOTFController(aotf, persistent=True).profile_io
    await io.writer_command('{"action":"claim","token":"writer"}')
    request = asyncio.create_task(io.writer_command(
        '{"action":"profile","token":"writer","profile":"488_full"}'
    ))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        request.cancel()
        await asyncio.sleep(0)
        assert aotf.states[1] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert io.owner is None
        assert set(aotf.states.values()) == {0}
        assert aotf.active_profile == "all_off"
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)
