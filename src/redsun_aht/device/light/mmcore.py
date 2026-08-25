"""Process-local Micro-Manager shuttered-light backend.

One backend instance owns one ``CMMCorePlus`` instance and admits exactly one
shutter device, defaulting it to the safe closed state. Application processes
interact with it through a future illumination service; they never import
Micro-Manager or vendor SDKs directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from redsun_aht.device.light.properties import (
    MMProperty,
    describe_mm_property,
    format_mm_property_value,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


class MMCoreUnavailableError(RuntimeError):
    """Raised when the optional Micro-Manager runtime is unavailable."""


class LightAdmissionError(RuntimeError):
    """Raised when loaded hardware does not match the explicit profile."""


@dataclass(frozen=True, slots=True)
class MMCoreLightProfile:
    """Explicit identity and metadata constraints for one light service."""

    service_id: str
    config_path: Path
    mm_path: Path
    expected_label: str
    adapter: str | None = None
    device_name: str | None = None
    wavelength_nm: int | None = None
    interlock_property: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete or ambiguous admission constraints."""
        if not self.service_id:
            raise ValueError("service_id must not be empty")
        if not self.expected_label:
            raise ValueError("expected_label must not be empty")
        if (self.adapter is None) is not (self.device_name is None):
            raise ValueError("adapter and device_name must be provided together")
        if self.wavelength_nm is not None and self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive when provided")
        if self.interlock_property is not None and not self.interlock_property:
            raise ValueError("interlock_property must name a device property")


@dataclass(frozen=True, slots=True)
class ShutteredLightIdentity:
    """Hardware identity confirmed after loading one shutter configuration."""

    label: str
    adapter: str
    device_name: str
    description: str
    wavelength_nm: int | None


class ShutterCoreLike(Protocol):
    """Narrow CMMCore surface used by the light backend and its fakes."""

    def loadSystemConfiguration(self, path: str) -> None:
        """Load exactly one shutter-only system configuration."""
        ...

    def getShutterDevice(self) -> str:
        """Return the configured shutter device label, or empty if absent."""
        ...

    def getShutterOpen(self, label: str) -> bool:
        """Return the shutter open readback for one label."""
        ...

    def setShutterOpen(self, label: str, state: bool) -> None:
        """Command one open or close transition for one label."""
        ...

    def getAutoShutter(self) -> bool:
        """Return whether camera-driven auto-shutter logic is enabled."""
        ...

    def setAutoShutter(self, state: bool) -> None:
        """Enable or disable camera-driven auto-shutter logic."""
        ...

    def getLoadedDevices(self) -> Sequence[str]:
        """Return every currently loaded device label including ``Core``."""
        ...

    def getDeviceType(self, label: str) -> object:
        """Return the device type enum value for one label."""
        ...

    def getDeviceLibrary(self, label: str) -> str:
        """Return the device adapter library name for one label."""
        ...

    def getDeviceName(self, label: str) -> str:
        """Return the vendor device name for one label."""
        ...

    def getDeviceDescription(self, label: str) -> str:
        """Return the human-readable device description for one label."""
        ...

    def getDevicePropertyNames(self, label: str) -> Sequence[str]:
        """Return every discoverable property name for one label."""
        ...

    def getProperty(self, label: str, name: str) -> str:
        """Return one property's current string value."""
        ...

    def setProperty(self, label: str, name: str, value: object) -> None:
        """Write one property's string-encoded value."""
        ...

    def isPropertyReadOnly(self, label: str, name: str) -> bool:
        """Return whether one property rejects writes."""
        ...

    def getAllowedPropertyValues(self, label: str, name: str) -> Sequence[str]:
        """Return the enumerable values declared for one property."""
        ...

    def hasPropertyLimits(self, label: str, name: str) -> bool:
        """Return whether one property declares numeric limits."""
        ...

    def getPropertyLowerLimit(self, label: str, name: str) -> float:
        """Return one property's inclusive lower limit."""
        ...

    def getPropertyUpperLimit(self, label: str, name: str) -> float:
        """Return one property's inclusive upper limit."""
        ...

    def unloadAllDevices(self) -> None:
        """Unload every loaded device and release adapter resources."""
        ...


def _default_core_factory(mm_path: Path) -> ShutterCoreLike:
    try:
        from pymmcore_plus import CMMCorePlus
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise MMCoreUnavailableError(
            "install the 'camera-mmcore' extra to use a Micro-Manager light"
        ) from exc
    return cast("ShutterCoreLike", CMMCorePlus(mm_path=str(mm_path)))


class MMCoreShutteredLightBackend:
    """Own one shutter-only CMMCorePlus context inside one service process."""

    def __init__(
        self,
        profile: MMCoreLightProfile,
        *,
        core_factory: Callable[[Path], ShutterCoreLike] = _default_core_factory,
    ) -> None:
        self.profile = profile
        self._core_factory = core_factory
        self._core: ShutterCoreLike | None = None
        self._identity: ShutteredLightIdentity | None = None

    @property
    def connected(self) -> bool:
        """Return whether a validated core is currently owned."""
        return self._core is not None

    @property
    def identity(self) -> ShutteredLightIdentity:
        """Return the admitted identity after connection."""
        if self._identity is None:
            raise RuntimeError("shuttered light is not connected")
        return self._identity

    def connect(self) -> ShutteredLightIdentity:
        """Load, make shutter-safe, and validate exactly one light engine."""
        if self._core is not None:
            return self.identity
        if not self.profile.config_path.is_file():
            raise FileNotFoundError(self.profile.config_path)
        if not self.profile.mm_path.is_dir():
            raise FileNotFoundError(self.profile.mm_path)

        core = self._core_factory(self.profile.mm_path)
        try:
            core.loadSystemConfiguration(str(self.profile.config_path))
            # The service owns no camera. Auto-shutter must stay off so no
            # hidden camera-side logic can ever toggle this illumination.
            core.setAutoShutter(False)
            identity = self._validate_identity(core)
            # Admit only into the safe state: ordinary output stays off until
            # an explicit command opens the shutter.
            core.setShutterOpen(identity.label, False)
        except BaseException as admission_error:
            try:
                self._cleanup_core(core, self.profile.expected_label)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "light admission and cleanup both failed",
                    [admission_error, cleanup_error],
                ) from None
            raise
        self._core = core
        self._identity = identity
        return identity

    def properties(self) -> tuple[MMProperty, ...]:
        """Return typed descriptors for every admitted device property."""
        core = self._require_core()
        label = self.identity.label
        properties: list[MMProperty] = []
        for name in core.getDevicePropertyNames(label):
            limits = None
            if core.hasPropertyLimits(label, name):
                limits = (
                    float(core.getPropertyLowerLimit(label, name)),
                    float(core.getPropertyUpperLimit(label, name)),
                )
            properties.append(
                describe_mm_property(
                    name,
                    value=core.getProperty(label, name),
                    read_only=bool(core.isPropertyReadOnly(label, name)),
                    allowed_values=tuple(core.getAllowedPropertyValues(label, name)),
                    limits=limits,
                )
            )
        return tuple(properties)

    def state(self) -> bool:
        """Return the shutter open readback."""
        core = self._require_core()
        return bool(core.getShutterOpen(self.identity.label))

    def set_shutter(self, open_shutter: bool) -> None:
        """Command exactly one open or close transition of the shutter."""
        core = self._require_core()
        core.setShutterOpen(self.identity.label, open_shutter)

    def property_value(self, name: str) -> str | float | bool:
        """Return one discovered property coerced to its classified type."""
        prop = self._require_property(name)
        return prop.typed_value()

    def set_property(self, name: str, value: str | int | float | bool) -> None:
        """Write one writable property after classified validation."""
        prop = self._require_property(name)
        if prop.read_only:
            raise ValueError(f"property {name!r} is read-only")
        rendered = format_mm_property_value(prop, value)
        self._require_core().setProperty(self.identity.label, name, rendered)

    def interlock_state(self) -> str | float | bool | None:
        """Return the configured interlock readback, or ``None`` if unconfigured.

        Admission fails when the profile names an interlock property that the
        loaded device does not expose, so a connected backend always reports
        the declared signal. Disconnected backends reject like every other
        operation instead of reporting a missing signal.
        """
        if self.profile.interlock_property is None:
            return None
        return self.property_value(self.profile.interlock_property)

    def disconnect(self) -> None:
        """Close the shutter and release the process-local core."""
        core = self._core
        if core is None:
            return
        identity = self._identity
        label = identity.label if identity is not None else self.profile.expected_label
        self._cleanup_core(core, label)
        self._core = None
        self._identity = None

    def _validate_identity(self, core: ShutterCoreLike) -> ShutteredLightIdentity:
        label = core.getShutterDevice()
        if label != self.profile.expected_label:
            raise LightAdmissionError(
                f"shutter label {label!r} does not match "
                f"{self.profile.expected_label!r}"
            )
        owned_devices = [item for item in core.getLoadedDevices() if item != "Core"]
        if owned_devices != [label]:
            raise LightAdmissionError(
                f"service config must load only its shutter, got {owned_devices!r}"
            )
        if "shutter" not in str(core.getDeviceType(label)).lower():
            raise LightAdmissionError(f"loaded device {label!r} is not a shutter")
        adapter = core.getDeviceLibrary(label)
        device_name = core.getDeviceName(label)
        adapter_expected = self.profile.adapter
        device_expected = self.profile.device_name
        adapter_matches = adapter_expected is None or adapter == adapter_expected
        device_matches = device_expected is None or device_name == device_expected
        if not (adapter_matches and device_matches):
            raise LightAdmissionError(
                "loaded shutter adapter/device does not match the service "
                f"profile: {adapter}/{device_name}"
            )
        if self.profile.interlock_property is not None:
            known = tuple(core.getDevicePropertyNames(label))
            if self.profile.interlock_property not in known:
                raise LightAdmissionError(
                    f"loaded shutter {label!r} does not expose interlock property "
                    f"{self.profile.interlock_property!r}"
                )
        return ShutteredLightIdentity(
            label=label,
            adapter=adapter,
            device_name=device_name,
            description=core.getDeviceDescription(label),
            wavelength_nm=self.profile.wavelength_nm,
        )

    def _require_core(self) -> ShutterCoreLike:
        if self._core is None:
            raise RuntimeError("shuttered light is not connected")
        return self._core

    def _require_property(self, name: str) -> MMProperty:
        for prop in self.properties():
            if prop.name == name:
                return prop
        raise ValueError(f"unknown property {name!r}")

    @staticmethod
    def _cleanup_core(core: ShutterCoreLike, label_hint: str) -> None:
        errors: list[Exception] = []
        try:
            target = str(core.getShutterDevice()) or label_hint
        except Exception as exc:  # a missing shutter device must not skip unload
            errors.append(exc)
            target = None

        actions: list[Callable[[], None]] = [lambda: core.setAutoShutter(False)]
        if target is not None:
            actions.append(lambda: core.setShutterOpen(target, False))
        actions.append(core.unloadAllDevices)
        for action in actions:
            try:
                action()
            except Exception as exc:  # cleanup must continue through every action
                errors.append(exc)
        if errors:
            raise ExceptionGroup("light cleanup failed", errors)


__all__ = [
    "LightAdmissionError",
    "MMCoreLightProfile",
    "MMCoreShutteredLightBackend",
    "MMCoreUnavailableError",
    "ShutterCoreLike",
    "ShutteredLightIdentity",
]
