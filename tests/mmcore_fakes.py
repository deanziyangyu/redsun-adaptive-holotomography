"""In-process doubles for the Micro-Manager camera and stage services.

Adapted from ``tests/test_services.py`` (``FakeCore``, ``MemoryFrameStore``) and
``tests/headless_fakes.py`` of ``redsun-searching-is-imaging`` at donor revision
``3a3bf06``.

Changed: there is one fake per device kind instead of one shared camera core
tidied for the searching bundle's broom, the Qt, napari, flyer and soft-axis
doubles are not needed here, and the stage double records the ``stop`` and
``unloadAllDevices`` calls the cleanup tests assert on. Nothing in this module
loads Micro-Manager, its device adapters, or opens a network transport.
"""

from __future__ import annotations

import threading
from queue import Queue
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ophyd_async.core import set_mock_value
from ophyd_async.fastcs.core import fastcs_connector

from redsun_aht.device.mmcore import MMAxis, MMStage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from pymmcore_plus import CMMCorePlus

#: Shape of the frames a fake camera takes, ``(height, width)``.
FRAME_SHAPE = (4, 6)

#: Label the camera is loaded under in a fake core.
LABEL = "Camera"

#: Data key a capture writes into its store.
DATA_KEY = "cam"

#: Position the stage service reports for each axis at startup.
START_POSITION = (100.0, 200.0)


class FakeCameraCore:
    """A camera whose frames the test produces one at a time.

    It answers the ``CMMCorePlus`` calls the camera service makes, refuses a
    setting while a sequence runs as some adapters do, and records every
    property read taken during one.
    """

    def __init__(
        self,
        *,
        exposure: float = 100.0,
        exposure_limits: tuple[float, float] | None = None,
        sequences: bool = True,
        frame_shape: tuple[int, int] = FRAME_SHAPE,
        pixel_type: str = "8bit",
    ) -> None:
        self.snapped = 0
        self.popped = 0
        self.exposure = exposure
        self.exposure_limits = exposure_limits
        self.frame_shape = frame_shape
        self.sensor_shape = frame_shape
        self.roi: tuple[int, ...] = (0, 0, frame_shape[1], frame_shape[0])
        self.sequences = sequences
        self.sequencing = False
        self.fault: Exception | None = None
        self.read_mid_sequence: list[str] = []
        self.properties = {
            "Gain": "0",
            "Photon Flux": "1",
            "CameraName": "fake",
            "Binning": "1x1",
            "PixelType": pixel_type,
            "ReadoutRate": "100MHz",
        }
        if exposure_limits is not None:
            self.properties["Exposure"] = str(exposure)
        self.loaded: tuple[str, str, str] | None = None
        self.initialized: str | None = None
        self.camera: str | None = None
        self.system_configuration: str | None = None
        self._buffer: Queue[NDArray[Any]] = Queue()

    def produce(self, frames: int = 1) -> None:
        """Put *frames* into the circular buffer, as the camera would."""
        for _ in range(frames):
            self._buffer.put(
                np.full(self.frame_shape, self.popped + 1, dtype=self.dtype)
            )

    def snap(self) -> NDArray[Any]:
        """Expose one frame, refusing while a sequence runs."""
        if self.sequencing:
            raise RuntimeError("cannot expose while a sequence acquisition runs")
        self.snapped += 1
        return np.full(self.frame_shape, self.snapped, dtype=self.dtype)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Numpy dtype of the frames this camera reads out."""
        return np.dtype(
            {"8bit": "uint8", "16bit": "uint16"}[self.properties["PixelType"]]
        )

    def loadSystemConfiguration(self, path: str) -> None:
        self.system_configuration = path

    def loadDevice(self, label: str, adapter: str, device: str) -> None:
        self.loaded = (label, adapter, device)

    def initializeDevice(self, label: str) -> None:
        self.initialized = label

    def setCameraDevice(self, label: str) -> None:
        self.camera = label

    def getCameraDevice(self) -> str:
        return self.camera or LABEL

    def hasProperty(self, label: str, name: str) -> bool:
        return name in self.properties

    def getAllowedPropertyValues(self, label: str, name: str) -> tuple[str, ...]:
        if name == "PixelType":
            return ("8bit", "16bit")
        if name == "Binning":
            return ("1x1", "2x2")
        if name == "Gain":
            return ("1-Gain", "2-Gain", "3-Sensitivity")
        if name == "ReadoutRate":
            return ("100MHz", "200MHz 11bit")
        return ()

    def startContinuousSequenceAcquisition(self, interval: float = 0) -> None:
        if not self.sequences:
            raise RuntimeError("this adapter takes no sequence")
        self.sequencing = True

    def stopSequenceAcquisition(self) -> None:
        self.sequencing = False

    def isBufferOverflowed(self) -> bool:
        return False

    def extra_frames_are_errors(self) -> bool:
        return True

    def getRemainingImageCount(self) -> int:
        if self.fault is not None:
            raise self.fault
        return self._buffer.qsize()

    def popNextImage(self) -> NDArray[Any]:
        frame = self._buffer.get_nowait()
        self.popped += 1
        return frame

    def popNextImageAndMD(self) -> tuple[NDArray[Any], dict[str, str]]:
        frame = self.popNextImage()
        return frame, {
            "ImageNumber": str(self.popped - 1),
            "ElapsedTime-ms": str(self.popped * 20),
        }

    def getDevicePropertyNames(self, label: str) -> tuple[str, ...]:
        return tuple(self.properties)

    def isPropertyReadOnly(self, label: str, name: str) -> bool:
        return name not in {"Gain", "Photon Flux", "Binning"}

    def getProperty(self, label: str, name: str) -> str:
        if self.sequencing:
            # some camera adapters stop delivering when a property is read
            # during a sequence; the fake records every such read
            self.read_mid_sequence.append(name)
        return self.properties[name]

    def setProperty(self, label: str, name: str, value: str) -> None:
        if self.sequencing:
            raise RuntimeError("Cannot set property while a sequence runs")
        self.properties[name] = str(value)
        if name == "Binning":
            binning = int(value.split("x", maxsplit=1)[0])
            width = self.sensor_shape[1] // binning
            height = self.sensor_shape[0] // binning
            self.frame_shape = (height, width)
            self.roi = (0, 0, width, height)

    def getExposure(self) -> float:
        return self.exposure

    def setExposure(self, exposure: float) -> None:
        if self.sequencing:
            raise RuntimeError("Cannot set exposure while a sequence runs")
        self.exposure = exposure

    def getROI(self) -> Sequence[int]:
        return self.roi

    def clearROI(self) -> None:
        if self.sequencing:
            raise RuntimeError("Cannot set ROI while a sequence runs")
        # some camera adapters write the size before the offsets, so one
        # clear zeroes the offsets and leaves the size they allowed
        self.roi = (
            0,
            0,
            self.frame_shape[1] - self.roi[0],
            self.frame_shape[0] - self.roi[1],
        )

    def setROI(self, *roi: int) -> None:
        if self.sequencing:
            raise RuntimeError("Cannot set ROI while a sequence runs")
        _, _, width, height = roi
        # a GenICam camera caps the width at what is left of the sensor
        # beyond the current offset
        room = (
            self.frame_shape[1] - self.roi[0],
            self.frame_shape[0] - self.roi[1],
        )
        if width > room[0] or height > room[1]:
            raise RuntimeError(
                f"Value = {max(width, height)} must be equal or smaller "
                f"than Max = {min(room)}"
            )
        self.roi = tuple(roi)

    def getImageWidth(self) -> int:
        return self.roi[2]

    def getImageHeight(self) -> int:
        return self.roi[3]

    def getPropertyLowerLimit(self, label: str, name: str) -> float:
        return self._exposure_limits()[0]

    def getPropertyUpperLimit(self, label: str, name: str) -> float:
        return self._exposure_limits()[1]

    def _exposure_limits(self) -> tuple[float, float]:
        """Return the limits the adapter declares, refusing when it declares none."""
        if self.exposure_limits is None:
            raise RuntimeError("this adapter reports no property limits")
        return self.exposure_limits


class InternalTriggerCameraCore(FakeCameraCore):
    """Fake camera with a writable trigger-mode readback."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.properties["TriggerMode"] = "Internal Trigger"

    def isPropertyReadOnly(self, label: str, name: str) -> bool:
        return name != "TriggerMode" and super().isPropertyReadOnly(label, name)


class MemoryFrameStore:
    """Keep written arrays observable without constructing a Zarr store."""

    def __init__(self, uri: str, data_key: str, frame: NDArray[Any]) -> None:
        self.uri = uri
        self.data_key = data_key
        self.frame_shape = frame.shape
        self.frames: list[NDArray[Any]] = []
        self.frames_written = 0
        self.closed = False

    def append(self, frame: NDArray[Any]) -> int:
        index = self.frames_written
        self.frames.append(frame.copy())
        self.frames_written += 1
        return index

    def close(self) -> None:
        self.closed = True


class FakeStageCore:
    """A two-axis stage plus one Z axis, with writable adapter properties.

    Every move lands at once, so a test asserts the request rather than waits
    for travel. ``stop`` and ``unloadAllDevices`` are recorded, and ``fail``
    makes either raise, so a cleanup path can be watched from the outside.
    """

    def __init__(
        self,
        *,
        position: tuple[float, float] = START_POSITION,
        z: float = 0.0,
        fail_on_stop: bool = False,
        fail_on_unload: bool = False,
    ) -> None:
        self.position = position
        self.z = z
        self.busy = False
        self.properties = {
            "Speed": "400",
            "StartSpeed": "100",
            "Acceleration": "4",
        }
        self.stop_calls: list[str] = []
        self.unloaded = False
        self.fail_on_stop = fail_on_stop
        self.fail_on_unload = fail_on_unload
        self.loaded: tuple[str, str, str] | None = None
        self.initialized: str | None = None
        self.system_configuration: str | None = None
        self._lock = threading.Lock()

    def loadSystemConfiguration(self, path: str) -> None:
        self.system_configuration = path

    def loadDevice(self, label: str, adapter: str, device: str) -> None:
        self.loaded = (label, adapter, device)

    def initializeDevice(self, label: str) -> None:
        self.initialized = label

    def getXYPosition(self, label: str) -> tuple[float, float]:
        return self.position

    def setXYPosition(self, label: str, x: float, y: float) -> None:
        self.position = (float(x), float(y))

    def getPosition(self, label: str) -> float:
        return self.z

    def setPosition(self, label: str, value: float) -> None:
        self.z = float(value)

    def waitForDevice(self, label: str) -> None:
        self.busy = False

    def deviceBusy(self, label: str) -> bool:
        return self.busy

    def stop(self, label: str) -> None:
        self.stop_calls.append(label)
        if self.fail_on_stop:
            raise RuntimeError("the stage refused to stop")

    def unloadAllDevices(self) -> None:
        self.unloaded = True
        if self.fail_on_unload:
            raise RuntimeError("the stage refused to unload")

    def getDevicePropertyNames(self, label: str) -> tuple[str, ...]:
        return tuple(self.properties)

    def isPropertyReadOnly(self, label: str, name: str) -> bool:
        return False

    def getProperty(self, label: str, name: str) -> str:
        return self.properties[name]

    def setProperty(self, label: str, name: str, value: str) -> None:
        self.properties[name] = value


def core_of(fake: object) -> CMMCorePlus:
    """Type *fake* as the ``CMMCorePlus`` the service is written against.

    The fakes answer the calls the service makes rather than inheriting from
    the Micro-Manager binding, which no headless test may load.
    """
    return cast("CMMCorePlus", fake)


def filled_axis(prefix: str) -> MMAxis:
    """Return an axis whose signals exist, as the stage connector builds them.

    A PVI connector turns a device's annotations into signals when the device
    is constructed; a hand-made ``MMAxis`` has none, so the same call is made
    here. The caller adds it to a stage's axis map and connects it in mock
    mode.
    """
    axis = MMAxis()
    fastcs_connector(prefix, axis)
    return axis


async def mock_stage(prefix: str, axes: tuple[str, ...] = ("x", "y")) -> MMStage:
    """Return a connected ``MMStage`` with mock signals instead of a service.

    The stage service discovers its axes through PVI, which a mock connection
    never reads, so the axes are registered by hand and named as the service
    would name them. Every axis is given limits of ``-1000..1000`` and a
    precision of ``0.001`` um, and the stage's declared axes are read once the
    axes are in place.
    """
    stage = MMStage(prefix, name="XY")
    await stage.connect(mock=True)
    set_mock_value(stage.provenance, "measured")
    stage.axis.set_name("XY-axis")
    for name in axes:
        axis = filled_axis(f"{prefix}Stage:Axis:{name}:")
        stage.axis[name] = axis
        axis.set_name(f"XY-axis-{name}")
        await axis.connect(mock=True)
        set_mock_value(axis.limits, np.array([-1000.0, 1000.0]))
        set_mock_value(axis.precision, 0.001)
    await stage.refresh_axes()
    return stage


__all__ = [
    "DATA_KEY",
    "FRAME_SHAPE",
    "LABEL",
    "START_POSITION",
    "FakeCameraCore",
    "FakeStageCore",
    "MemoryFrameStore",
    "core_of",
    "filled_axis",
    "mock_stage",
]
