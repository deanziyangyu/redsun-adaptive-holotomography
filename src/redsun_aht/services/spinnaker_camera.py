"""Optional direct-Spinnaker camera service, alongside the MMCore service.

PySpin is loaded only when this service is launched. Camera ownership is
exclusive: do not open the same camera in MMCore or another process at once.
The raw transport format and the converted, stored format are distinct. The
default selects a supported packed 12-bit format and converts with the SDK's
ImageProcessor to Mono16 before copying pixels into application-owned memory.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from typing import Any, ClassVar

import numpy as np
from fastcs.attributes import AttrRW, AttrW
from fastcs.datatypes import Bool
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.services import configure_logging, identity, serve_controller

from redsun_aht.services.mmcore_camera import MMCameraController, run


class SpinnakerCore:
    """Small camera-source adapter for the shared service capture contract.

    This is not an MMCore object. The small compatibility surface lets the
    existing camera service reuse its PVA signals and bounded storage while
    this class exclusively owns the actual SDK camera and image buffers.
    """

    _ENUMS: ClassVar[dict[str, str]] = {
        "PixelFormat": "PixelFormat",
        "TriggerMode": "TriggerMode",
        "TriggerSelector": "TriggerSelector",
        "TriggerSource": "TriggerSource",
        "TriggerActivation": "TriggerActivation",
        "TriggerOverlap": "TriggerOverlap",
        "LineSelector": "LineSelector",
        "ExposureAuto": "ExposureAuto",
        "GainAuto": "GainAuto",
    }
    _BOOLS: ClassVar[dict[str, str]] = {
        "AcquisitionFrameRateEnable": "AcquisitionFrameRateEnable",
        "LineInverter": "LineInverter",
    }
    _FLOATS: ClassVar[dict[str, str]] = {
        "AcquisitionFrameRate": "AcquisitionFrameRate",
        "Gain": "Gain",
        "TriggerDelay": "TriggerDelay",
    }

    def __init__(self, sdk: Any, camera: Any, *, label: str = "spinnaker") -> None:
        self.sdk = sdk
        self.camera = camera
        self.label = label
        self.nodes = camera.GetNodeMap()
        self.processor = sdk.ImageProcessor()
        self._pending: Any = None
        self._begun = False
        self._finite_count: int | None = None
        self._popped = 0
        self._last_frame_id: int | None = None
        self._owner: tuple[Any, Any] | None = None
        self._transport_format: str | None = None
        self._original_format: str | None = None
        self._original_acquisition_mode: str | None = None
        self._stream_counters: dict[str, int] = {}
        self._triggered_sequence = False

    def _node(self, name: str) -> Any:
        node = self.nodes.GetNode(name)
        if not self.sdk.IsAvailable(node):
            raise ValueError(f"camera feature unavailable: {name}")
        return node

    def _enum(self, name: str) -> Any:
        return self.sdk.CEnumerationPtr(self._node(name))

    def _set_enum(self, name: str, value: str) -> None:
        node = self._enum(name)
        if not self.sdk.IsWritable(node):
            raise RuntimeError(f"camera feature is not writable: {name}")
        entry = node.GetEntryByName(value)
        if not self.sdk.IsAvailable(entry) or not self.sdk.IsReadable(entry):
            raise ValueError(f"unsupported {name}: {value}")
        node.SetIntValue(entry.GetValue())

    def configure_pixel_format(self, requested: str = "auto") -> str:
        """Select a packed 12-bit transport, or a lossless Mono16 fallback."""
        choices = self.getAllowedPropertyValues(self.label, "PixelFormat")
        if requested == "auto":
            requested = next(
                (x for x in ("Mono12p", "Mono12Packed", "Mono16") if x in choices),
                "",
            )
            if not requested:
                raise ValueError("camera has no supported 12-bit or Mono16 format")
        allowed = {
            "Mono12p",
            "Mono12Packed",
            "Mono10p",
            "Mono10Packed",
            "Mono16",
            "Mono8",
        }
        if requested not in allowed or requested not in choices:
            raise ValueError(f"unsupported monochrome transport format: {requested}")
        self._set_enum("PixelFormat", requested)
        self._transport_format = requested
        return requested

    def getCameraDevice(self) -> str:
        """Return the service-local camera label."""
        return self.label

    def getImageWidth(self) -> int:
        """Return the current image width."""
        return int(self.sdk.CIntegerPtr(self._node("Width")).GetValue())

    def getImageHeight(self) -> int:
        """Return the current image height."""
        return int(self.sdk.CIntegerPtr(self._node("Height")).GetValue())

    def getExposure(self) -> float:
        """Read exposure in milliseconds."""
        return float(self.sdk.CFloatPtr(self._node("ExposureTime")).GetValue()) / 1000

    def setExposure(self, value: float) -> None:
        """Set exposure in milliseconds."""
        self.sdk.CFloatPtr(self._node("ExposureTime")).SetValue(float(value) * 1000)

    def getPropertyLowerLimit(self, label: str, name: str) -> float:
        """Return the exposure lower limit in milliseconds."""
        if name != "Exposure":
            raise ValueError("only exposure limits are available")
        return float(self.sdk.CFloatPtr(self._node("ExposureTime")).GetMin()) / 1000

    def getPropertyUpperLimit(self, label: str, name: str) -> float:
        """Return the exposure upper limit in milliseconds."""
        if name != "Exposure":
            raise ValueError("only exposure limits are available")
        return float(self.sdk.CFloatPtr(self._node("ExposureTime")).GetMax()) / 1000

    def getROI(self) -> tuple[int, int, int, int]:
        """Read the current sensor region."""
        integer = self.sdk.CIntegerPtr
        return (
            int(integer(self._node("OffsetX")).GetValue()),
            int(integer(self._node("OffsetY")).GetValue()),
            self.getImageWidth(),
            self.getImageHeight(),
        )

    def setROI(self, *_: int) -> None:
        """Set the readout region while idle, restoring the old ROI on failure."""
        if self._begun or len(_) != 4:
            raise RuntimeError("an idle camera and four ROI values are required")
        x, y, width, height = _
        if any(type(v) is not int or v < 0 for v in _) or width < 1 or height < 1:
            raise ValueError("invalid ROI")
        previous = self.getROI()
        try:
            self._apply_roi(x, y, width, height)
        except Exception:
            self._apply_roi(*previous)
            raise

    def _apply_roi(self, x: int, y: int, width: int, height: int) -> None:
        integer = self.sdk.CIntegerPtr
        offsets = [integer(self._node(name)) for name in ("OffsetX", "OffsetY")]
        sizes = [integer(self._node(name)) for name in ("Width", "Height")]
        for node in offsets:
            node.SetValue(node.GetMin())
        for node, value in zip(sizes, (width, height), strict=True):
            node.SetValue(value)
        for node, value in zip(offsets, (x, y), strict=True):
            node.SetValue(value)

    def clearROI(self) -> None:
        """Reset the region to the maximum supported at the active binning."""
        if self._begun:
            raise RuntimeError("ROI changes require an idle camera")
        integer = self.sdk.CIntegerPtr
        for name in ("OffsetX", "OffsetY"):
            node = integer(self._node(name))
            node.SetValue(node.GetMin())
        for name in ("Width", "Height"):
            node = integer(self._node(name))
            node.SetValue(node.GetMax())

    def hasProperty(self, label: str, name: str) -> bool:
        """Report whether the camera exposes a supported feature."""
        if name in ("PixelType", "Exposure"):
            return True
        if name == "Binning":
            name = "BinningHorizontal"
        mapped = self._ENUMS.get(
            name, self._BOOLS.get(name, self._FLOATS.get(name, name))
        )
        return bool(self.sdk.IsAvailable(self.nodes.GetNode(mapped)))

    def getDevicePropertyNames(self, label: str) -> list[str]:
        """List supported camera features."""
        return [
            name
            for name in (*self._ENUMS, *self._BOOLS, *self._FLOATS, "Binning")
            if self.hasProperty(label, name)
        ]

    def isPropertyReadOnly(self, label: str, name: str) -> bool:
        """Report whether the feature is currently writable."""
        if name in ("PixelType", "Exposure"):
            return True
        if name == "Binning":
            return not all(
                self.sdk.IsWritable(self._node(axis))
                for axis in ("BinningHorizontal", "BinningVertical")
            )
        node = self._ENUMS.get(
            name, self._BOOLS.get(name, self._FLOATS.get(name, name))
        )
        # Trigger features may be locked only because TriggerMode is currently
        # on. Expose readable features and let SDK write-time checks decide.
        return not bool(self.sdk.IsReadable(self._node(node)))

    def getAllowedPropertyValues(self, label: str, name: str) -> list[str]:
        """List advertised feature choices."""
        if name == "PixelType":
            return ["16bit"]
        if name == "Binning":
            horizontal = self.sdk.CIntegerPtr(self._node("BinningHorizontal"))
            vertical = self.sdk.CIntegerPtr(self._node("BinningVertical"))
            low = max(horizontal.GetMin(), vertical.GetMin())
            high = min(horizontal.GetMax(), vertical.GetMax())
            return [
                str(x)
                for x in range(int(low), int(high) + 1)
                if (x - horizontal.GetMin()) % horizontal.GetInc() == 0
                and (x - vertical.GetMin()) % vertical.GetInc() == 0
            ]
        if name in self._BOOLS:
            return ["0", "1"]
        if name not in self._ENUMS:
            return []
        node = self._enum(self._ENUMS[name])
        return [
            str(self.sdk.CEnumEntryPtr(entry).GetSymbolic())
            for entry in node.GetEntries()
            if self.sdk.IsAvailable(entry) and self.sdk.IsReadable(entry)
        ]

    def getProperty(self, label: str, name: str) -> str:
        """Read one camera feature as text."""
        if name == "PixelType":
            return "16bit"
        if name == "Binning":
            x = int(self.sdk.CIntegerPtr(self._node("BinningHorizontal")).GetValue())
            y = int(self.sdk.CIntegerPtr(self._node("BinningVertical")).GetValue())
            return str(x) if x == y else f"{x}x{y}"
        if name in self._ENUMS:
            return str(self._enum(self._ENUMS[name]).GetCurrentEntry().GetSymbolic())
        if name in self._BOOLS:
            return str(
                int(self.sdk.CBooleanPtr(self._node(self._BOOLS[name])).GetValue())
            )
        if name in self._FLOATS:
            return str(self.sdk.CFloatPtr(self._node(self._FLOATS[name])).GetValue())
        raise ValueError(f"unknown camera property: {name}")

    def setProperty(self, label: str, name: str, value: str) -> None:
        """Set one camera feature from text."""
        if name == "PixelType":
            if value != "16bit":
                raise ValueError("SDK conversion produces only 16bit arrays")
        elif name == "PixelFormat":
            self.configure_pixel_format(value)
        elif name == "Binning":
            if value not in self.getAllowedPropertyValues(label, name):
                raise ValueError("unsupported symmetric binning")
            horizontal = self.sdk.CIntegerPtr(self._node("BinningHorizontal"))
            vertical = self.sdk.CIntegerPtr(self._node("BinningVertical"))
            previous = (horizontal.GetValue(), vertical.GetValue())
            try:
                horizontal.SetValue(int(value))
                vertical.SetValue(int(value))
            except Exception:
                horizontal.SetValue(previous[0])
                vertical.SetValue(previous[1])
                raise
        elif name in self._ENUMS:
            self._set_enum(self._ENUMS[name], value)
        elif name in self._BOOLS:
            if value not in ("0", "1"):
                raise ValueError("boolean camera properties require 0 or 1")
            self.sdk.CBooleanPtr(self._node(self._BOOLS[name])).SetValue(value == "1")
        elif name in self._FLOATS:
            self.sdk.CFloatPtr(self._node(self._FLOATS[name])).SetValue(float(value))
        else:
            raise ValueError(f"unknown or read-only camera property: {name}")

    def clearCircularBuffer(self) -> None:
        """Clear cached state before a sequence."""
        if self._begun:
            raise RuntimeError("cannot clear a running sequence")
        self._release_pending()
        self._last_frame_id = None

    def startContinuousSequenceAcquisition(self, interval: float) -> None:
        """Begin one continuous SDK acquisition."""
        if self._begun:
            raise RuntimeError("camera sequence already running")
        self._set_enum("AcquisitionMode", "Continuous")
        self._last_frame_id = None
        self._popped = 0
        self._finite_count = None
        self._begin_acquisition()

    def startSequenceAcquisition(
        self, count: int, interval: float, stop_on_overflow: bool
    ) -> None:
        """Begin one finite SDK acquisition."""
        if self._begun or count < 1:
            raise RuntimeError("an idle camera and positive frame count are required")
        self._set_enum("AcquisitionMode", "MultiFrame")
        self.sdk.CIntegerPtr(self._node("AcquisitionFrameCount")).SetValue(count)
        self._last_frame_id = None
        self._popped = 0
        self._finite_count = count
        self._begin_acquisition()

    def _begin_acquisition(self) -> None:
        self._triggered_sequence = self.getProperty(self.label, "TriggerMode") == "On"
        self.camera.BeginAcquisition()
        self._begun = True
        try:
            self._stream_counters = self._read_stream_counters()
        except BaseException:
            self.stopSequenceAcquisition()
            raise

    def isSequenceRunning(self) -> bool:
        """Report whether the requested sequence can produce more frames."""
        return self._begun and (
            self._finite_count is None or self._popped < self._finite_count
        )

    def getRemainingImageCount(self) -> int:
        """Poll for at most one pending SDK frame."""
        if self._pending is not None:
            return 1
        if not self._begun:
            return 0
        try:
            self._pending = self.camera.GetNextImage(1)
        except self.sdk.SpinnakerException as error:
            if getattr(error, "errorcode", None) == self.sdk.SPINNAKER_ERR_TIMEOUT:
                return 0
            raise
        return 1

    def popNextImageAndMD(self) -> tuple[np.ndarray, dict[str, str]]:
        """Convert and copy one SDK frame and its metadata."""
        if self._pending is None and not self.getRemainingImageCount():
            raise RuntimeError("no camera frame is available")
        image, self._pending = self._pending, None
        try:
            if image.IsIncomplete():
                raise RuntimeError(f"incomplete camera frame: {image.GetImageStatus()}")
            frame_id = int(image.GetFrameID())
            if self._last_frame_id is not None and frame_id != self._last_frame_id + 1:
                raise RuntimeError("camera frame IDs are not contiguous")
            timestamp_ns = int(image.GetTimeStamp())
            # Convert packed wire data before exposing an owned NumPy image.
            converted = self.processor.Convert(image, self.sdk.PixelFormat_Mono16)
            frame = np.array(converted.GetNDArray(), dtype=np.uint16, copy=True)
            if frame.ndim != 2 or not frame.size:
                raise RuntimeError("converted camera frame is not a 2D image")
            self._last_frame_id = frame_id
            self._popped += 1
            return frame, {
                "frame_id": str(frame_id),
                "timestamp_ns": str(timestamp_ns),
                "raw_pixel_format": str(self._transport_format),
                "converted_pixel_format": "Mono16",
                "conversion": "PySpin.ImageProcessor.Convert",
            }
        finally:
            image.Release()

    def popNextImage(self) -> np.ndarray:
        """Convert and copy one SDK frame."""
        return self.popNextImageAndMD()[0]

    def isBufferOverflowed(self) -> bool:
        """Report an increase in available transport error counters."""
        current = self._read_stream_counters()
        return any(
            current.get(name, value) > value
            for name, value in self._stream_counters.items()
        )

    def extra_frames_are_errors(self) -> bool:
        """Require an exact count only for externally or software-triggered runs."""
        return self._triggered_sequence

    def _read_stream_counters(self) -> dict[str, int]:
        stream = self.camera.GetTLStreamNodeMap()
        counters: dict[str, int] = {}
        for name in (
            "StreamDroppedFrameCount",
            "StreamLostFrameCount",
            "StreamIncompleteFrameCount",
        ):
            node = stream.GetNode(name)
            if self.sdk.IsAvailable(node) and self.sdk.IsReadable(node):
                counters[name] = int(self.sdk.CIntegerPtr(node).GetValue())
        return counters

    def _release_pending(self) -> None:
        if self._pending is not None:
            self._pending.Release()
            self._pending = None

    def stopSequenceAcquisition(self) -> None:
        """End the active SDK acquisition."""
        if not self._begun:
            return
        self._release_pending()
        self.camera.EndAcquisition()
        self._begun = False
        self._finite_count = None

    def software_trigger(self) -> None:
        """Request one SDK software trigger during an active sequence."""
        if not self._begun:
            raise RuntimeError("software trigger requires an active sequence")
        command = self.sdk.CCommandPtr(self._node("TriggerSoftware"))
        if not self.sdk.IsWritable(command):
            raise RuntimeError("camera software trigger is not writable")
        command.Execute()

    def snap(self) -> np.ndarray:
        """Take one internal or software-triggered frame through the converter."""
        return self.snap_with_metadata()[0]

    def snap_with_metadata(self) -> tuple[np.ndarray, dict[str, str]]:
        """Take one snap while retaining the SDK frame ID and image timestamp."""
        if self._begun:
            raise RuntimeError("a snap requires an idle camera")
        trigger_mode = self.getProperty(self.label, "TriggerMode")
        if (
            trigger_mode == "On"
            and self.getProperty(self.label, "TriggerSource") != "Software"
        ):
            raise RuntimeError("snap requires internal or software triggering")
        previous = str(self._enum("AcquisitionMode").GetCurrentEntry().GetSymbolic())
        try:
            self._set_enum("AcquisitionMode", "SingleFrame")
            self._last_frame_id = None
            self.camera.BeginAcquisition()
            self._begun = True
            if trigger_mode == "On":
                self.sdk.CCommandPtr(self._node("TriggerSoftware")).Execute()
            self._pending = self.camera.GetNextImage(5000)
            return self.popNextImageAndMD()
        finally:
            try:
                self.stopSequenceAcquisition()
            finally:
                self._set_enum("AcquisitionMode", previous)

    def close(self) -> None:
        """Release the SDK camera and its system owner."""
        try:
            self.stopSequenceAcquisition()
        finally:
            if self._owner is not None:
                system, cameras = self._owner
                self._owner = None
                try:
                    if self._original_format is not None:
                        self._set_enum("PixelFormat", self._original_format)
                    if self._original_acquisition_mode is not None:
                        self._set_enum(
                            "AcquisitionMode", self._original_acquisition_mode
                        )
                finally:
                    try:
                        self.camera.DeInit()
                    finally:
                        self.camera = None
                        try:
                            cameras.Clear()
                        finally:
                            system.ReleaseInstance()


class SpinnakerCameraController(MMCameraController):
    """Separate direct-SDK service using the common camera PVA/storage contract."""

    software_trigger = AttrW(Bool(), description="Request one SDK software trigger")
    immediate_frame_count = AttrRW(
        Bool(), description="Publish frame counts as frames arrive"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_unbounded_capture", True)
        super().__init__(*args, **kwargs)
        # SDK settings can be changed by pausing a live stream without taking
        # an unsolicited snap. An open capture still fixes the image layout.
        self._property_io._is_busy = self._settings_busy
        self._core_io._is_busy = self._settings_busy
        self._software_trigger_lock = asyncio.Lock()
        self._count_task: asyncio.Task[None] | None = None
        self.software_trigger.set_on_put_callback(self._on_software_trigger)

    def _settings_busy(self) -> bool:
        return self._store is not None or self._snap_in_progress

    def _layout_template(self) -> np.ndarray:
        """Describe the converted SDK frame even while live acquisition runs."""
        with self._camera_request_lock, self._camera_lock:
            width = self._core.getImageWidth()
            height = self._core.getImageHeight()
            if width <= 0 or height <= 0:
                raise ValueError("camera reports an invalid image layout")
            return np.empty((height, width), dtype=np.uint16)

    def grab_once(self) -> np.ndarray | None:
        """Publish an optional low-latency count without slowing frame retrieval."""
        frame = super().grab_once()
        if frame is not None and self.immediate_frame_count.get() and self._loop:
            self._loop.call_soon_threadsafe(self._schedule_count_update)
        return frame

    def _schedule_count_update(self) -> None:
        if self._count_task is None or self._count_task.done():
            self._count_task = asyncio.create_task(self._publish_count_now())

    async def _publish_count_now(self) -> None:
        async with self._publish_frame_lock:
            if self.frame_count.get() < self._grabbed:
                await self.frame_count.update(self._grabbed)

    async def _on_software_trigger(self, attr: Any, value: bool) -> None:
        del attr
        if value is not True:
            raise ValueError("software trigger request must be true")
        if self._software_trigger_lock.locked():
            raise RuntimeError("a software trigger is already in flight")

        def issue() -> None:
            with self._camera_lock:
                self._core.software_trigger()

        async with self._software_trigger_lock:
            if not self._grabbing.is_set() or self._error is not None:
                raise RuntimeError("camera is not acquiring or has faulted")
            before = self._grabbed
            task = asyncio.create_task(asyncio.to_thread(issue))
            try:
                # Cancellation must not release the request lock while the SDK
                # command is still running in its worker thread.
                await asyncio.shield(task)
                async with asyncio.timeout(5):
                    while self._grabbed == before:
                        if self._error is not None or not self._grabbing.is_set():
                            raise RuntimeError(
                                "camera stopped before the triggered frame"
                            )
                        await asyncio.sleep(0.001)
                async with self._publish_frame_lock:
                    await self.frame_count.update(self._grabbed)
            except BaseException:
                self._error = "software trigger outcome uncertain; acquisition stopped"
                self._grabbing.clear()
                self._request_capture_end()
                if not task.done():
                    with contextlib.suppress(Exception):
                        await asyncio.shield(task)
                raise

    def _frame_metadata_semantics(self) -> str:
        return (
            "PySpin frame ID and image timestamp; timestamp event semantics and "
            "mapping to the host clock are not established"
        )

    def take_frame_with_metadata(
        self,
    ) -> tuple[np.ndarray | None, dict[str, str] | None]:
        """Retain native SDK metadata for both sequences and single snaps."""
        if self._sequencing:
            return super().take_frame_with_metadata()
        with self._camera_request_lock, self._camera_lock:
            if self._sequencing:
                if not self._core.getRemainingImageCount():
                    return None, None
                return self._core.popNextImageAndMD()
            return self._core.snap_with_metadata()

    def _frame_metadata_fields(self, metadata: dict[str, str] | None) -> dict[str, Any]:
        return {
            "camera_frame_counter": int(metadata["frame_id"]) if metadata else None,
            "camera_metadata_source": "PySpin.GetNextImage" if metadata else None,
            "camera_metadata": metadata,
        }

    def _stop_sequence(self) -> None:
        """End the SDK sequence directly, without MMCore's finite-stop wait."""
        self._sequence_started.clear()
        if not self._sequencing:
            return
        try:
            if self._capture_full and (
                self._core.isBufferOverflowed()
                or (
                    self._core.extra_frames_are_errors()
                    and self._core.getRemainingImageCount()
                )
            ):
                raise RuntimeError("camera transport loss or extra frames")
        finally:
            try:
                self._core.stopSequenceAcquisition()
            finally:
                self._sequencing = False
                self._sequence_is_finite = False

    async def disconnect(self) -> None:
        """Stop the service and release its camera."""
        try:
            await super().disconnect()
        finally:
            self._core.close()


def open_spinnaker(serial: str, *, pixel_format: str = "auto") -> SpinnakerCore:
    """Open one camera by serial; PySpin remains an ad-hoc dependency."""
    if not serial.strip():
        raise ValueError("an explicit camera serial is required")
    try:
        import PySpin  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("install the vendor PySpin package separately") from error
    system = PySpin.System.GetInstance()
    cameras = system.GetCameras()
    camera = None
    try:
        camera = cameras.GetBySerial(serial)
        if not camera.IsValid():
            raise ValueError(f"camera not found: {serial}")
        camera.Init()
        core = SpinnakerCore(PySpin, camera)
        core._original_format = core.getProperty(core.label, "PixelFormat")
        core._original_acquisition_mode = str(
            core._enum("AcquisitionMode").GetCurrentEntry().GetSymbolic()
        )
        core.configure_pixel_format(pixel_format)
        core._owner = (system, cameras)
        return core
    except BaseException:
        if camera is not None and camera.IsValid() and camera.IsInitialized():
            camera.DeInit()
        camera = None
        cameras.Clear()
        system.ReleaseInstance()
        raise


def main(argv: list[str] | None = None) -> int:
    """Launch the optional camera service."""
    service = identity()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="camera serial number")
    parser.add_argument(
        "--prefix", default=service.prefix if service else "", help="PVA service prefix"
    )
    parser.add_argument(
        "--name",
        default=service.name if service else "spinnaker",
        help="camera/data key",
    )
    parser.add_argument(
        "--pixel-format", default="auto", help="raw camera transport format"
    )
    parser.add_argument("--buffer-limit-mib", type=int, default=256)
    parser.add_argument("--finite-capture", action="store_true")
    options = parser.parse_args(argv)
    if not 1 <= options.buffer_limit_mib <= 4096:
        parser.error("--buffer-limit-mib must be 1..4096")
    configure_logging()
    core = open_spinnaker(options.serial, pixel_format=options.pixel_format)
    try:
        controller = SpinnakerCameraController(
            core,  # type: ignore[arg-type]
            core.label,
            options.name,
            defer_initial_frame=True,
            buffered_capture=True,
            finite_capture=options.finite_capture,
            buffer_limit_mib=options.buffer_limit_mib,
        )
        return run(
            serve_controller(controller, EpicsPVATransport(), options.prefix or None)
        )
    finally:
        core.close()


if __name__ == "__main__":
    sys.exit(main())
