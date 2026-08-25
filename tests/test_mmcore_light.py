from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from redsun_aht.device.light import (
    LightAdmissionError,
    MMCoreLightProfile,
    MMCoreShutteredLightBackend,
    MMPropertyKind,
    describe_mm_property,
    format_mm_property_value,
)


@dataclass
class FakeProp:
    value: str
    read_only: bool = False
    allowed: tuple[str, ...] = ()
    limits: tuple[float, float] | None = None


class FakeShutterCore:
    def __init__(
        self,
        *,
        label: str = "Laser-1",
        adapter: str = "LumencorSpectra",
        device_name: str = "Spectra",
        extra_devices: tuple[str, ...] = (),
        shutter_type: str = "Shutter",
        initial_open: bool = True,
    ) -> None:
        self.label = label
        self.adapter = adapter
        self.device_name = device_name
        self.extra_devices = extra_devices
        self.shutter_type = shutter_type
        self.auto_shutter = True
        self.open_state = initial_open
        self.loaded_config: str | None = None
        self.written: list[tuple[str, str]] = []
        self.unload_count = 0
        self.properties: dict[str, FakeProp] = {
            "Cyan_Enable": FakeProp("0", allowed=("0", "1")),
            "Cyan_Level": FakeProp("15", limits=(0.0, 100.0)),
            "LED_Mode": FakeProp("Internal", allowed=("Internal", "External")),
            "Interlock": FakeProp("1", read_only=True, allowed=("0", "1")),
            "Serial Number": FakeProp("spectra-test", read_only=True),
            "Label_Note": FakeProp("engine-a"),
        }

    def loadSystemConfiguration(self, path: str) -> None:
        self.loaded_config = path

    def getShutterDevice(self) -> str:
        return self.label

    def getShutterOpen(self, label: str) -> bool:
        assert label == self.label
        return self.open_state

    def setShutterOpen(self, label: str, state: bool) -> None:
        assert label == self.label
        self.open_state = state

    def getAutoShutter(self) -> bool:
        return self.auto_shutter

    def setAutoShutter(self, state: bool) -> None:
        self.auto_shutter = state

    def getLoadedDevices(self) -> tuple[str, ...]:
        return (self.label, *self.extra_devices)

    def getDeviceType(self, label: str) -> str:
        assert label == self.label
        return self.shutter_type

    def getDeviceLibrary(self, label: str) -> str:
        assert label == self.label
        return self.adapter

    def getDeviceName(self, label: str) -> str:
        assert label == self.label
        return self.device_name

    def getDeviceDescription(self, label: str) -> str:
        assert label == self.label
        return "Fake shuttered light engine"

    def getDevicePropertyNames(self, label: str) -> tuple[str, ...]:
        assert label == self.label
        return tuple(self.properties)

    def getProperty(self, label: str, name: str) -> str:
        assert label == self.label
        return self.properties[name].value

    def setProperty(self, label: str, name: str, value: object) -> None:
        assert label == self.label
        rendered = str(value)
        self.properties[name].value = rendered
        self.written.append((name, rendered))

    def isPropertyReadOnly(self, label: str, name: str) -> bool:
        assert label == self.label
        return self.properties[name].read_only

    def getAllowedPropertyValues(self, label: str, name: str) -> tuple[str, ...]:
        assert label == self.label
        return self.properties[name].allowed

    def hasPropertyLimits(self, label: str, name: str) -> bool:
        assert label == self.label
        return self.properties[name].limits is not None

    def getPropertyLowerLimit(self, label: str, name: str) -> float:
        assert name == "Cyan_Level"
        limits = self.properties[name].limits
        assert limits is not None
        return limits[0]

    def getPropertyUpperLimit(self, label: str, name: str) -> float:
        assert name == "Cyan_Level"
        limits = self.properties[name].limits
        assert limits is not None
        return limits[1]

    def unloadAllDevices(self) -> None:
        self.unload_count += 1


class ExplodingUnloadCore(FakeShutterCore):
    def unloadAllDevices(self) -> None:
        self.unload_count += 1
        raise RuntimeError("unload exploded")


class FailingIdentityQueryCore(FakeShutterCore):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.identity_queries = 0

    def getShutterDevice(self) -> str:
        self.identity_queries += 1
        if self.identity_queries == 2:
            raise RuntimeError("identity query exploded")
        return self.label


def _profile(tmp_path: Path, **changes: Any) -> MMCoreLightProfile:
    config = tmp_path / "laser.cfg"
    config.write_text("# fake", encoding="utf-8")
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir(exist_ok=True)
    values: dict[str, Any] = {
        "service_id": "laser",
        "config_path": config,
        "mm_path": mm_dir,
        "expected_label": "Laser-1",
        "adapter": "LumencorSpectra",
        "device_name": "Spectra",
        "wavelength_nm": 488,
        "interlock_property": "Interlock",
    }
    values.update(changes)
    return MMCoreLightProfile(**values)


def _connected(tmp_path: Path, core: FakeShutterCore) -> MMCoreShutteredLightBackend:
    backend = MMCoreShutteredLightBackend(
        _profile(tmp_path), core_factory=lambda _: core
    )
    backend.connect()
    return backend


def test_profile_rejects_incomplete_constraints(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="service_id"):
        MMCoreLightProfile(
            service_id="", config_path=tmp_path, mm_path=tmp_path, expected_label="L"
        )
    with pytest.raises(ValueError, match="expected_label"):
        MMCoreLightProfile(
            service_id="laser",
            config_path=tmp_path,
            mm_path=tmp_path,
            expected_label="",
        )
    with pytest.raises(ValueError, match="together"):
        _profile(tmp_path, device_name=None)
    with pytest.raises(ValueError, match="wavelength_nm"):
        _profile(tmp_path, wavelength_nm=0)
    with pytest.raises(ValueError, match="interlock_property"):
        _profile(tmp_path, interlock_property="")


def test_backend_admits_one_shutter_and_defaults_off(tmp_path: Path) -> None:
    core = FakeShutterCore()
    backend = MMCoreShutteredLightBackend(
        _profile(tmp_path), core_factory=lambda _: core
    )

    identity = backend.connect()

    assert identity.label == "Laser-1"
    assert identity.adapter == "LumencorSpectra"
    assert identity.device_name == "Spectra"
    assert identity.description == "Fake shuttered light engine"
    assert identity.wavelength_nm == 488
    assert core.auto_shutter is False
    assert core.open_state is False
    assert backend.connected
    assert backend.state() is False


def test_admitted_properties_are_typed_by_shared_rules(tmp_path: Path) -> None:
    backend = _connected(tmp_path, FakeShutterCore())

    props = {prop.name: prop for prop in backend.properties()}

    assert props["Cyan_Enable"].kind is MMPropertyKind.BOOLEAN
    assert props["Cyan_Level"].kind is MMPropertyKind.NUMERIC
    assert props["Cyan_Level"].limits == (0.0, 100.0)
    assert props["LED_Mode"].kind is MMPropertyKind.ENUMERABLE
    assert props["Serial Number"].kind is MMPropertyKind.TEXT
    assert props["Serial Number"].read_only
    assert backend.property_value("Cyan_Enable") is False
    assert backend.property_value("Cyan_Level") == 15.0
    assert backend.property_value("LED_Mode") == "Internal"
    assert backend.property_value("Serial Number") == "spectra-test"


def test_set_property_validates_type_range_and_readonly(tmp_path: Path) -> None:
    core = FakeShutterCore()
    backend = _connected(tmp_path, core)

    with pytest.raises(ValueError, match="unknown property"):
        backend.property_value("Missing")
    with pytest.raises(ValueError, match="unknown property"):
        backend.set_property("Missing", 1)
    with pytest.raises(ValueError, match="read-only"):
        backend.set_property("Interlock", True)
    with pytest.raises(ValueError, match="within"):
        backend.set_property("Cyan_Level", 100.5)
    with pytest.raises(ValueError, match="accepts"):
        backend.set_property("LED_Mode", "Strobe")

    backend.set_property("Cyan_Level", 42)
    backend.set_property("Cyan_Enable", True)
    backend.set_property("LED_Mode", "External")
    backend.set_property("Label_Note", "engine-b")
    assert ("Cyan_Level", "42.0") in core.written
    assert ("Cyan_Enable", "1") in core.written
    assert ("LED_Mode", "External") in core.written
    assert backend.property_value("Label_Note") == "engine-b"


def test_interlock_state_surfaces_declared_signal(tmp_path: Path) -> None:
    core = FakeShutterCore()
    guarded = _connected(tmp_path, core)
    unguarded = MMCoreShutteredLightBackend(
        _profile(tmp_path, interlock_property=None),
        core_factory=lambda _: core,
    )
    unguarded.connect()

    assert guarded.interlock_state() is True
    assert unguarded.interlock_state() is None


def test_missing_interlock_property_fails_admission(tmp_path: Path) -> None:
    core = FakeShutterCore()
    backend = MMCoreShutteredLightBackend(
        _profile(tmp_path, interlock_property="Bogus"),
        core_factory=lambda _: core,
    )

    with pytest.raises(LightAdmissionError, match="interlock"):
        backend.connect()
    assert not backend.connected
    assert core.unload_count == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"adapter": "OtherAdapter"}, "does not match the service profile"),
        ({"device_name": "Differ"}, "does not match the service profile"),
        ({"expected_label": "Other"}, "shutter label .* does not match"),
    ],
)
def test_identity_mismatch_fails_admission_and_unloads(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    core = FakeShutterCore()
    profile = _profile(tmp_path, **changes)
    backend = MMCoreShutteredLightBackend(profile, core_factory=lambda _: core)

    with pytest.raises(LightAdmissionError, match=message):
        backend.connect()
    assert not backend.connected
    assert core.unload_count == 1


def test_extra_loaded_devices_fail_admission(tmp_path: Path) -> None:
    core = FakeShutterCore(extra_devices=("Stray-1",))
    backend = MMCoreShutteredLightBackend(
        _profile(tmp_path), core_factory=lambda _: core
    )

    with pytest.raises(LightAdmissionError, match="only its shutter"):
        backend.connect()
    assert not backend.connected
    assert core.unload_count == 1


def test_non_shutter_device_fails_admission(tmp_path: Path) -> None:
    core = FakeShutterCore(shutter_type="Camera")
    backend = MMCoreShutteredLightBackend(
        _profile(tmp_path), core_factory=lambda _: core
    )

    with pytest.raises(LightAdmissionError, match="not a shutter"):
        backend.connect()
    assert not backend.connected
    assert core.unload_count == 1


def test_disconnected_backend_rejects_operations(tmp_path: Path) -> None:
    backend = MMCoreShutteredLightBackend(_profile(tmp_path))

    with pytest.raises(RuntimeError, match="not connected"):
        backend.state()
    with pytest.raises(RuntimeError, match="not connected"):
        _ = backend.identity
    with pytest.raises(RuntimeError, match="not connected"):
        backend.properties()
    with pytest.raises(RuntimeError, match="not connected"):
        backend.set_shutter(True)
    with pytest.raises(RuntimeError, match="not connected"):
        backend.property_value("Cyan_Enable")
    with pytest.raises(RuntimeError, match="not connected"):
        backend.set_property("Cyan_Enable", True)
    with pytest.raises(RuntimeError, match="not connected"):
        backend.interlock_state()
    assert not backend.connected


def test_disconnect_closes_shutter_and_is_idempotent(tmp_path: Path) -> None:
    core = FakeShutterCore(initial_open=False)
    backend = _connected(tmp_path, core)

    backend.set_shutter(True)
    assert backend.state() is True
    backend.disconnect()
    assert core.open_state is False
    assert core.auto_shutter is False
    assert core.unload_count == 1
    assert not backend.connected

    backend.disconnect()
    assert core.unload_count == 1

    reopened = MMCoreShutteredLightBackend(
        _profile(tmp_path), core_factory=lambda _: core
    )
    assert reopened.connect().label == "Laser-1"


def test_connect_requires_existing_paths(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    backend = MMCoreShutteredLightBackend(profile)

    profile.config_path.unlink()
    with pytest.raises(FileNotFoundError):
        backend.connect()

    profile.config_path.write_text("# fake", encoding="utf-8")
    shutil.rmtree(profile.mm_path)
    with pytest.raises(FileNotFoundError):
        backend.connect()


def test_reconnect_returns_admitted_identity_without_reload(tmp_path: Path) -> None:
    core = FakeShutterCore()
    backend = MMCoreShutteredLightBackend(
        _profile(tmp_path), core_factory=lambda _: core
    )
    first = backend.connect()
    loaded = core.loaded_config

    second = backend.connect()

    assert second == first
    assert core.loaded_config == loaded
    assert core.unload_count == 0


def test_disconnect_survives_failed_identity_query_and_unloads(
    tmp_path: Path,
) -> None:
    core = FailingIdentityQueryCore()
    backend = MMCoreShutteredLightBackend(
        _profile(tmp_path), core_factory=lambda _: core
    )
    backend.connect()

    with pytest.raises(ExceptionGroup, match="light cleanup failed"):
        backend.disconnect()
    still_connected = bool(backend.connected)
    assert still_connected
    assert core.unload_count == 1

    backend.disconnect()
    fully_disconnected = not backend.connected
    assert fully_disconnected
    assert not core.open_state
    assert core.unload_count == 2


def test_cleanup_failure_is_grouped_with_admission_error(tmp_path: Path) -> None:
    core = ExplodingUnloadCore()
    profile = _profile(tmp_path, interlock_property="Bogus")
    backend = MMCoreShutteredLightBackend(profile, core_factory=lambda _: core)

    with pytest.raises(BaseExceptionGroup) as excinfo:
        backend.connect()
    top = [
        str(error)
        for error in excinfo.value.exceptions
        if not isinstance(error, BaseExceptionGroup)
    ]
    nested = [
        str(inner)
        for error in excinfo.value.exceptions
        if isinstance(error, BaseExceptionGroup)
        for inner in error.exceptions
    ]
    assert any("interlock" in message for message in top + nested)
    assert any("unload exploded" in message for message in top + nested)


def test_shared_rules_classify_and_format_values() -> None:
    boolean = describe_mm_property(
        "Enable", value="0", read_only=False, allowed_values=("0", "1"), limits=None
    )
    enumerable = describe_mm_property(
        "Mode", value="A", read_only=False, allowed_values=("A", "B"), limits=None
    )
    numeric = describe_mm_property(
        "Level", value="7", read_only=False, allowed_values=(), limits=(0.0, 10.0)
    )
    inferred_numeric = describe_mm_property(
        "Gain", value="1.5", read_only=False, allowed_values=(), limits=None
    )
    text = describe_mm_property(
        "Note", value="hello", read_only=False, allowed_values=(), limits=None
    )

    assert boolean.kind is MMPropertyKind.BOOLEAN
    assert enumerable.kind is MMPropertyKind.ENUMERABLE
    assert numeric.kind is MMPropertyKind.NUMERIC
    assert inferred_numeric.kind is MMPropertyKind.NUMERIC
    assert text.kind is MMPropertyKind.TEXT
    assert boolean.typed_value() is False

    assert format_mm_property_value(boolean, False) == "0"
    assert format_mm_property_value(boolean, "1") == "1"
    assert format_mm_property_value(boolean, 1) == "1"
    with pytest.raises(ValueError, match="boolean"):
        format_mm_property_value(boolean, 2)
    assert format_mm_property_value(enumerable, "B") == "B"
    with pytest.raises(ValueError, match="accepts"):
        format_mm_property_value(enumerable, "C")
    assert format_mm_property_value(numeric, "3.5") == "3.5"
    with pytest.raises(ValueError, match="numeric"):
        format_mm_property_value(numeric, "fast")
    with pytest.raises(ValueError, match="within"):
        format_mm_property_value(numeric, -1)
