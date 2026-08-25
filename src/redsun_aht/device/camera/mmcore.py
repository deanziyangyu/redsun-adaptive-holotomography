"""Process-local Micro-Manager camera backend.

One backend instance owns one ``CMMCorePlus`` instance and admits exactly one
camera adapter. Application processes interact with it through the camera IOC;
they never import Micro-Manager or vendor SDKs directly.
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 - dataclass field annotation
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import numpy.typing as npt


class MMCoreUnavailableError(RuntimeError):
    """Raised when the optional Micro-Manager runtime is unavailable."""


class CameraAdmissionError(RuntimeError):
    """Raised when loaded hardware does not match the explicit profile."""


@dataclass(frozen=True, slots=True)
class MMCoreCameraProfile:
    """Explicit identity and adapter constraints for one service process."""

    service_id: str
    config_path: Path
    mm_path: Path
    adapter: str
    device_name: str
    expected_label: str
    serial_property: str | None = None
    expected_serial: str | None = None
    initial_properties: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject incomplete or ambiguous admission constraints."""
        if not self.service_id:
            raise ValueError("service_id must not be empty")
        if not self.adapter or not self.device_name or not self.expected_label:
            raise ValueError("camera adapter, device name, and label are required")
        if (self.serial_property is None) is not (self.expected_serial is None):
            raise ValueError(
                "serial_property and expected_serial must be provided together"
            )
        properties = {
            str(name): str(value) for name, value in self.initial_properties.items()
        }
        if self.adapter == "SpinnakerC":
            # SpinnakerC exposes this GenICam feature as ``Pixel Format``.
            # PyMMCore then returns the resulting Mono16 frames as uint16.
            properties.setdefault("Pixel Format", "Mono16")
        if any(not name or not value for name, value in properties.items()):
            raise ValueError("initial camera property names and values cannot be empty")
        object.__setattr__(self, "initial_properties", MappingProxyType(properties))


@dataclass(frozen=True, slots=True)
class CameraIdentity:
    """Hardware identity confirmed after loading one camera configuration."""

    label: str
    adapter: str
    device_name: str
    description: str
    serial: str | None


@dataclass(frozen=True, slots=True)
class CameraProperty:
    """Inspectable generic Micro-Manager property metadata."""

    name: str
    value: str
    read_only: bool
    allowed_values: tuple[str, ...]
    limits: tuple[float, float] | None


class CoreLike(Protocol):
    """Narrow CMMCore surface used by the backend and its fakes."""

    def loadSystemConfiguration(self, path: str) -> None: ...

    def getCameraDevice(self) -> str: ...

    def getLoadedDevices(self) -> Sequence[str]: ...

    def getDeviceType(self, label: str) -> object: ...

    def getDeviceLibrary(self, label: str) -> str: ...

    def getDeviceName(self, label: str) -> str: ...

    def getDeviceDescription(self, label: str) -> str: ...

    def getAutoShutter(self) -> bool: ...

    def setAutoShutter(self, state: bool) -> None: ...

    def getDevicePropertyNames(self, label: str) -> Sequence[str]: ...

    def getProperty(self, label: str, name: str) -> str: ...

    def setProperty(self, label: str, name: str, value: str) -> None: ...

    def isPropertyReadOnly(self, label: str, name: str) -> bool: ...

    def getAllowedPropertyValues(self, label: str, name: str) -> Sequence[str]: ...

    def hasPropertyLimits(self, label: str, name: str) -> bool: ...

    def getPropertyLowerLimit(self, label: str, name: str) -> float: ...

    def getPropertyUpperLimit(self, label: str, name: str) -> float: ...

    def snapImage(self) -> None: ...

    def getImage(self) -> object: ...

    def getExposure(self) -> float: ...

    def setExposure(self, exposure_ms: float) -> None: ...

    def getROI(self) -> Sequence[int]: ...

    def setROI(self, x: int, y: int, width: int, height: int) -> None: ...

    def clearCircularBuffer(self) -> None: ...

    def setCircularBufferMemoryFootprint(self, size_mb: int) -> None: ...

    def initializeCircularBuffer(self) -> None: ...

    def startSequenceAcquisition(
        self, frame_count: int, interval_ms: float, stop_on_overflow: bool
    ) -> None: ...

    def startContinuousSequenceAcquisition(self, interval_ms: float) -> None: ...

    def getRemainingImageCount(self) -> int: ...

    def popNextImage(self) -> object: ...

    def isBufferOverflowed(self) -> bool: ...

    def stopSequenceAcquisition(self) -> None: ...

    def unloadAllDevices(self) -> None: ...


def _default_core_factory(mm_path: Path) -> CoreLike:
    try:
        from pymmcore_plus import CMMCorePlus
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise MMCoreUnavailableError(
            "install the 'camera-mmcore' extra to use a Micro-Manager camera"
        ) from exc
    return cast("CoreLike", CMMCorePlus(mm_path=str(mm_path)))


class MMCoreCameraBackend:
    """Own one camera-only CMMCorePlus context inside one service process."""

    def __init__(
        self,
        profile: MMCoreCameraProfile,
        *,
        core_factory: Callable[[Path], CoreLike] = _default_core_factory,
    ) -> None:
        self.profile = profile
        self._core_factory = core_factory
        self._core: CoreLike | None = None
        self._identity: CameraIdentity | None = None
        self._armed = False
        self._sequence_active = False

    @property
    def connected(self) -> bool:
        """Return whether a validated core is currently owned."""
        return self._core is not None

    @property
    def identity(self) -> CameraIdentity:
        """Return the admitted identity after connection."""
        if self._identity is None:
            raise RuntimeError("camera is not connected")
        return self._identity

    def connect(self) -> CameraIdentity:
        """Load, make shutter-safe, and validate exactly one camera."""
        if self._core is not None:
            return self.identity
        if not self.profile.config_path.is_file():
            raise FileNotFoundError(self.profile.config_path)
        if not self.profile.mm_path.is_dir():
            raise FileNotFoundError(self.profile.mm_path)

        core = self._core_factory(self.profile.mm_path)
        try:
            core.loadSystemConfiguration(str(self.profile.config_path))
            # Camera services never own shutters. Override unsafe config files
            # before admitting the camera or accepting acquisition commands.
            core.setAutoShutter(False)
            identity = self._validate_identity(core)
            self._apply_initial_properties(core, identity.label)
        except BaseException as admission_error:
            try:
                self._cleanup_core(core)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "camera admission and cleanup both failed",
                    [admission_error, cleanup_error],
                ) from None
            raise
        self._core = core
        self._identity = identity
        return identity

    def properties(self) -> tuple[CameraProperty, ...]:
        """Return generic property descriptors for the admitted camera."""
        core = self._require_core()
        label = self.identity.label
        properties: list[CameraProperty] = []
        for name in core.getDevicePropertyNames(label):
            limits = None
            if core.hasPropertyLimits(label, name):
                limits = (
                    float(core.getPropertyLowerLimit(label, name)),
                    float(core.getPropertyUpperLimit(label, name)),
                )
            properties.append(
                CameraProperty(
                    name=name,
                    value=core.getProperty(label, name),
                    read_only=bool(core.isPropertyReadOnly(label, name)),
                    allowed_values=tuple(core.getAllowedPropertyValues(label, name)),
                    limits=limits,
                )
            )
        return tuple(properties)

    def configured_exposure_ms(self) -> str:
        """Return MMCore's admitted exposure setting as stable display text."""
        return f"{float(self._require_core().getExposure()):.12g}"

    def arm(self) -> None:
        """Permit explicit snapshots after identity admission."""
        self._require_core()
        self._armed = True

    def trigger(self) -> npt.NDArray[Any]:
        """Acquire one explicitly requested frame with ``snapImage``.

        This remains the validation and broad-compatibility path. Live view and
        list scans use :meth:`start_sequence` and :meth:`drain_sequence`.
        """
        core = self._require_core()
        if self._sequence_active:
            raise RuntimeError("cannot snap while sequence acquisition is active")
        if not self._armed:
            raise RuntimeError("camera is not armed")
        core.snapImage()
        return np.asarray(core.getImage()).copy()

    def start_sequence(
        self,
        *,
        interval_ms: float = 0.0,
        buffer_memory_mb: int = 128,
        frame_count: int | None = None,
    ) -> None:
        """Start one MMCore-backed continuous or finite sequence.

        ``frame_count=None`` is the live-view mode and uses MMCore's continuous
        sequence API. A positive frame count is reserved for list scans and
        uses ``startSequenceAcquisition``. The caller must drain frames through
        :meth:`drain_sequence` before MMCore's circular buffer overflows.
        """
        if interval_ms < 0 or not np.isfinite(interval_ms):
            raise ValueError("sequence interval_ms must be finite and non-negative")
        if buffer_memory_mb <= 0:
            raise ValueError("sequence buffer_memory_mb must be positive")
        if frame_count is not None and frame_count <= 0:
            raise ValueError("sequence frame_count must be positive when provided")
        core = self._require_core()
        if self._sequence_active:
            raise RuntimeError("sequence acquisition is already active")
        if self._armed:
            raise RuntimeError("disarm single-frame capture before starting a sequence")
        core.stopSequenceAcquisition()
        core.clearCircularBuffer()
        core.setCircularBufferMemoryFootprint(buffer_memory_mb)
        core.initializeCircularBuffer()
        if frame_count is None:
            core.startContinuousSequenceAcquisition(interval_ms)
        else:
            core.startSequenceAcquisition(frame_count, interval_ms, True)
        self._sequence_active = True

    def drain_sequence(self) -> tuple[npt.NDArray[Any], ...]:
        """Copy and remove every frame currently held by MMCore's ring."""
        core = self._require_core()
        if not self._sequence_active:
            raise RuntimeError("sequence acquisition is not active")
        remaining = int(core.getRemainingImageCount())
        if remaining < 0:
            raise RuntimeError("MMCore returned a negative image count")
        return tuple(np.asarray(core.popNextImage()).copy() for _ in range(remaining))

    def sequence_overflowed(self) -> bool:
        """Return whether MMCore has reported a circular-buffer overflow."""
        if not self._sequence_active:
            return False
        return bool(self._require_core().isBufferOverflowed())

    def stop(self) -> None:
        """Stop sequence activity if present and disarm single-frame capture."""
        core = self._require_core()
        core.stopSequenceAcquisition()
        self._armed = False
        self._sequence_active = False

    def disconnect(self) -> None:
        """Force shutter-safe cleanup and release the process-local core."""
        core = self._core
        if core is None:
            return
        self._cleanup_core(core)
        self._core = None
        self._identity = None
        self._armed = False
        self._sequence_active = False

    def _validate_identity(self, core: CoreLike) -> CameraIdentity:
        label = core.getCameraDevice()
        if label != self.profile.expected_label:
            raise CameraAdmissionError(
                f"camera label {label!r} does not match {self.profile.expected_label!r}"
            )
        owned_devices = [item for item in core.getLoadedDevices() if item != "Core"]
        if owned_devices != [label]:
            raise CameraAdmissionError(
                f"service config must load only its camera, got {owned_devices!r}"
            )
        if "camera" not in str(core.getDeviceType(label)).lower():
            raise CameraAdmissionError(f"loaded device {label!r} is not a camera")
        adapter = core.getDeviceLibrary(label)
        device_name = core.getDeviceName(label)
        if adapter != self.profile.adapter or device_name != self.profile.device_name:
            raise CameraAdmissionError(
                "loaded camera adapter/device does not match the service profile: "
                f"{adapter}/{device_name}"
            )
        serial = None
        if self.profile.serial_property is not None:
            serial = core.getProperty(label, self.profile.serial_property)
            if serial != self.profile.expected_serial:
                raise CameraAdmissionError(
                    f"camera serial {serial!r} does not match expected "
                    f"{self.profile.expected_serial!r}"
                )
        if core.getAutoShutter():
            raise CameraAdmissionError(
                "camera service failed to force auto-shutter off"
            )
        return CameraIdentity(
            label=label,
            adapter=adapter,
            device_name=device_name,
            description=core.getDeviceDescription(label),
            serial=serial,
        )

    def _apply_initial_properties(self, core: CoreLike, label: str) -> None:
        """Apply and read back explicitly requested camera properties."""
        available = set(core.getDevicePropertyNames(label))
        for name, value in self.profile.initial_properties.items():
            if name == "Exposure":
                try:
                    requested_exposure_ms = float(value)
                except ValueError as exc:
                    raise CameraAdmissionError(
                        f"camera exposure {value!r} is not numeric"
                    ) from exc
                if not np.isfinite(requested_exposure_ms) or requested_exposure_ms <= 0:
                    raise CameraAdmissionError(
                        "camera exposure must be finite and positive"
                    )
                core.setExposure(requested_exposure_ms)
                observed_exposure_ms = float(core.getExposure())
                if not np.isclose(
                    observed_exposure_ms, requested_exposure_ms, atol=0.01, rtol=0
                ):
                    raise CameraAdmissionError(
                        "camera exposure read back as "
                        f"{observed_exposure_ms:.12g}, not {requested_exposure_ms:.12g}"
                    )
                continue
            if name == "ROI":
                parts = value.split(",")
                if len(parts) != 4:
                    raise CameraAdmissionError(
                        "camera ROI must be x,y,width,height"
                    )
                try:
                    roi = tuple(int(part.strip()) for part in parts)
                except ValueError as exc:
                    raise CameraAdmissionError(
                        f"camera ROI {value!r} must contain integers"
                    ) from exc
                x, y, width, height = roi
                if x < 0 or y < 0 or width <= 0 or height <= 0:
                    raise CameraAdmissionError(
                        "camera ROI requires non-negative origin and positive size"
                    )
                core.setROI(x, y, width, height)
                observed_roi = tuple(int(item) for item in core.getROI())
                if observed_roi != roi:
                    raise CameraAdmissionError(
                        f"camera ROI read back as {observed_roi!r}, not {roi!r}"
                    )
                continue
            if name not in available:
                raise CameraAdmissionError(
                    f"camera property {name!r} is not available on {label!r}"
                )
            if core.isPropertyReadOnly(label, name):
                raise CameraAdmissionError(f"camera property {name!r} is read-only")
            core.setProperty(label, name, value)
            observed = core.getProperty(label, name)
            if observed != value:
                raise CameraAdmissionError(
                    f"camera property {name!r} read back as {observed!r}, not {value!r}"
                )

    def _require_core(self) -> CoreLike:
        if self._core is None:
            raise RuntimeError("camera is not connected")
        return self._core

    @staticmethod
    def _cleanup_core(core: CoreLike) -> None:
        errors: list[Exception] = []
        actions: tuple[Callable[[], None], ...] = (
            lambda: core.setAutoShutter(False),
            core.stopSequenceAcquisition,
            core.unloadAllDevices,
        )
        for action in actions:
            try:
                action()
            except Exception as exc:  # cleanup must continue through every action
                errors.append(exc)
        if errors:
            raise ExceptionGroup("camera cleanup failed", errors)


__all__ = [
    "CameraAdmissionError",
    "CameraIdentity",
    "CameraProperty",
    "MMCoreCameraBackend",
    "MMCoreCameraProfile",
    "MMCoreUnavailableError",
]
