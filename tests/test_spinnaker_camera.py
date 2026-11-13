"""Direct SDK source tests do not require the optional vendor package."""

import asyncio
import threading
from collections import deque

import numpy as np
import pytest
from mmcore_fakes import FakeCameraCore

from redsun_aht.services.spinnaker_camera import (
    SpinnakerCameraController,
    SpinnakerCore,
)


class FakeTimeout(Exception):
    errorcode = -1011


class FakeImage:
    def __init__(self, frame_id=1, *, incomplete=False):
        self.frame_id = frame_id
        self.incomplete = incomplete
        self.released = False
        self.data = np.array([[0, 1023, 2048, 4095]], dtype=np.uint16)

    def IsIncomplete(self):
        return self.incomplete

    def GetImageStatus(self):
        return 9

    def GetFrameID(self):
        return self.frame_id

    def GetTimeStamp(self):
        return 123456789

    def Release(self):
        self.released = True
        self.data[:] = 0


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def Convert(self, image, pixel_format):
        self.calls.append(pixel_format)
        return type("Converted", (), {"GetNDArray": lambda _: image.data})()


class FakeSdk:
    PixelFormat_Mono16 = "mono16"
    SPINNAKER_ERR_TIMEOUT = -1011
    SpinnakerException = FakeTimeout

    def __init__(self):
        self.processor = FakeProcessor()

    def ImageProcessor(self):
        return self.processor


class FakeCamera:
    def __init__(self, images=()):
        self.images = deque(images)
        self.ended = 0

    def GetNodeMap(self):
        return object()

    def GetNextImage(self, timeout):
        assert timeout == 1
        if not self.images:
            raise FakeTimeout()
        return self.images.popleft()

    def EndAcquisition(self):
        self.ended += 1


def test_trigger_delay_and_line_inversion_are_camera_properties():
    class Node:
        def __init__(self, value):
            self.value = value

        def GetValue(self):
            return self.value

        def SetValue(self, value):
            self.value = value

    nodes = {"TriggerDelay": Node(17.0), "LineInverter": Node(False)}
    sdk = FakeSdk()
    sdk.IsAvailable = lambda node: node is not None
    sdk.CFloatPtr = lambda node: node
    sdk.CBooleanPtr = lambda node: node
    source = SpinnakerCore(sdk, FakeCamera())
    source.nodes = type("Nodes", (), {"GetNode": lambda _, name: nodes.get(name)})()

    assert source.hasProperty(source.label, "TriggerDelay")
    assert source.hasProperty(source.label, "LineInverter")
    source.setProperty(source.label, "TriggerDelay", "20")
    source.setProperty(source.label, "LineInverter", "1")
    assert source.getProperty(source.label, "TriggerDelay") == "20.0"
    assert source.getProperty(source.label, "LineInverter") == "1"
    with pytest.raises(ValueError, match="0 or 1"):
        source.setProperty(source.label, "LineInverter", "yes")


def test_packed_frame_is_converted_copied_and_released():
    image = FakeImage()
    sdk = FakeSdk()
    source = SpinnakerCore(sdk, FakeCamera([image]))
    source._transport_format = "Mono12p"
    source._begun = True
    assert source.getRemainingImageCount() == 1
    frame, metadata = source.popNextImageAndMD()
    assert image.released
    assert sdk.processor.calls == ["mono16"]
    assert frame.dtype == np.uint16
    assert frame.tolist() == [[0, 1023, 2048, 4095]]
    assert metadata == {
        "frame_id": "1",
        "timestamp_ns": "123456789",
        "raw_pixel_format": "Mono12p",
        "converted_pixel_format": "Mono16",
        "conversion": "PySpin.ImageProcessor.Convert",
    }
    assert source.getRemainingImageCount() == 0


def test_incomplete_and_missing_frames_fail_and_release():
    bad = FakeImage(incomplete=True)
    source = SpinnakerCore(FakeSdk(), FakeCamera([bad]))
    source._begun = True
    with pytest.raises(RuntimeError, match="incomplete"):
        source.popNextImageAndMD()
    assert bad.released

    first, skipped = FakeImage(10), FakeImage(12)
    source = SpinnakerCore(FakeSdk(), FakeCamera([first, skipped]))
    source._begun = True
    source.popNextImageAndMD()
    with pytest.raises(RuntimeError, match="not contiguous"):
        source.popNextImageAndMD()
    assert skipped.released


def test_pending_frame_is_released_on_stop():
    image = FakeImage()
    camera = FakeCamera([image])
    source = SpinnakerCore(FakeSdk(), camera)
    source._begun = True
    assert source.getRemainingImageCount() == 1
    source.stopSequenceAcquisition()
    assert image.released and camera.ended == 1
    assert not source.isSequenceRunning()


def test_native_frame_metadata_is_not_labelled_mmcore():
    metadata = {"frame_id": "17", "timestamp_ns": "42"}
    fields = SpinnakerCameraController._frame_metadata_fields(None, metadata)
    assert fields["camera_frame_counter"] == 17
    assert fields["camera_metadata_source"] == "PySpin.GetNextImage"
    assert fields["camera_metadata"] == metadata
    assert "mmcore_metadata" not in fields


def test_default_format_prefers_twelve_bit_without_silent_ten_bit_downgrade():
    source = SpinnakerCore(FakeSdk(), FakeCamera())
    selected = []
    source.getAllowedPropertyValues = lambda *_: ["Mono10p", "Mono12Packed", "Mono16"]
    source._set_enum = lambda _, value: selected.append(value)
    assert source.configure_pixel_format() == "Mono12Packed"
    assert selected == ["Mono12Packed"]
    source.getAllowedPropertyValues = lambda *_: ["Mono10p", "Mono16"]
    assert source.configure_pixel_format() == "Mono16"
    source.getAllowedPropertyValues = lambda *_: ["Mono10p"]
    with pytest.raises(ValueError, match="12-bit"):
        source.configure_pixel_format()
    assert source.configure_pixel_format("Mono10p") == "Mono10p"


def test_failed_converter_releases_vendor_image():
    image = FakeImage()
    sdk = FakeSdk()

    def fail(*_):
        raise RuntimeError("conversion failed")

    sdk.processor.Convert = fail
    source = SpinnakerCore(sdk, FakeCamera([image]))
    source._begun = True
    with pytest.raises(RuntimeError, match="conversion failed"):
        source.popNextImageAndMD()
    assert image.released


async def test_software_trigger_waits_for_a_received_frame():
    class Camera(FakeCameraCore):
        callback = None

        def software_trigger(self):
            self.callback()

    camera = Camera(pixel_type="16bit")
    service = SpinnakerCameraController(
        camera, "Camera", "cam", defer_initial_frame=True, buffered_capture=True
    )
    service._grabbing.set()
    camera.callback = lambda: setattr(service, "_grabbed", service._grabbed + 1)
    await service._on_software_trigger(None, True)
    assert service._grabbed == 1
    with pytest.raises(ValueError, match="must be true"):
        await service._on_software_trigger(None, False)


async def test_new_live_sequence_forgets_a_completed_prior_capture(monkeypatch):
    service = SpinnakerCameraController(
        FakeCameraCore(pixel_type="16bit"),
        "Camera",
        "cam",
        defer_initial_frame=True,
        buffered_capture=True,
    )
    service._capture_full = True
    monkeypatch.setattr(service, "_grab_loop", lambda: None)
    await service._start_grabbing()
    await service._grab_task
    assert not service._capture_full
    service._grabbing.clear()


async def test_software_trigger_fails_if_camera_stops_before_frame():
    class Camera(FakeCameraCore):
        def software_trigger(self):
            pass

    service = SpinnakerCameraController(
        Camera(pixel_type="16bit"),
        "Camera",
        "cam",
        defer_initial_frame=True,
        buffered_capture=True,
    )
    with pytest.raises(RuntimeError, match="not acquiring"):
        await service._on_software_trigger(None, True)


async def test_cancelled_software_trigger_faults_without_allowing_a_retry():
    entered = threading.Event()
    release = threading.Event()

    class Camera(FakeCameraCore):
        calls = 0

        def software_trigger(self):
            self.calls += 1
            entered.set()
            assert release.wait(3)

    camera = Camera(pixel_type="16bit")
    service = SpinnakerCameraController(
        camera, "Camera", "cam", defer_initial_frame=True, buffered_capture=True
    )
    service._grabbing.set()
    task = asyncio.create_task(service._on_software_trigger(None, True))
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already in flight"):
        await service._on_software_trigger(None, True)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert camera.calls == 1
    assert service._error and "uncertain" in service._error
    with pytest.raises(RuntimeError, match="not acquiring"):
        await service._on_software_trigger(None, True)
