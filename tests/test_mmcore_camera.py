from __future__ import annotations

import asyncio
import hashlib
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from redsun_aht.buffers import BufferFullError
from redsun_aht.device.camera import (
    CameraAdmissionError,
    MMCoreCameraBackend,
    MMCoreCameraProfile,
)
from redsun_aht.services.camera_ioc import build_camera_ioc


class FakeCore:
    def __init__(
        self,
        *,
        label: str = "Camera-1",
        adapter: str = "PVCAM",
        device_name: str = "Camera-1",
        serial: str = "prime-test",
    ) -> None:
        self.label = label
        self.adapter = adapter
        self.device_name = device_name
        self.serial = serial
        self.auto_shutter = True
        self.loaded_config: str | None = None
        self.snap_count = 0
        self.stop_count = 0
        self.clear_buffer_count = 0
        self.initialize_buffer_count = 0
        self.buffer_memory_mb: int | None = None
        self.continuous_intervals: list[float] = []
        self.finite_sequences: list[tuple[int, float, bool]] = []
        self.sequence_frames: list[np.ndarray[Any, np.dtype[np.uint16]]] = []
        self.buffer_overflowed = False
        self.unload_count = 0
        self.exposure_ms = 10.0
        self.roi = (0, 0, 3, 2)
        self.properties = {"Serial Number": serial, "Exposure": "10.0"}

    def loadSystemConfiguration(self, path: str) -> None:
        self.loaded_config = path

    def getCameraDevice(self) -> str:
        return self.label

    def getLoadedDevices(self) -> tuple[str, ...]:
        return (self.label,)

    def getDeviceType(self, label: str) -> str:
        assert label == self.label
        return "Camera"

    def getDeviceLibrary(self, label: str) -> str:
        assert label == self.label
        return self.adapter

    def getDeviceName(self, label: str) -> str:
        assert label == self.label
        return self.device_name

    def getDeviceDescription(self, label: str) -> str:
        assert label == self.label
        return "Fake scientific camera"

    def getAutoShutter(self) -> bool:
        return self.auto_shutter

    def setAutoShutter(self, state: bool) -> None:
        self.auto_shutter = state

    def getDevicePropertyNames(self, label: str) -> tuple[str, ...]:
        assert label == self.label
        return tuple(self.properties)

    def getProperty(self, label: str, name: str) -> str:
        assert label == self.label
        return self.properties[name]

    def setProperty(self, label: str, name: str, value: str) -> None:
        assert label == self.label
        self.properties[name] = value

    def isPropertyReadOnly(self, label: str, name: str) -> bool:
        assert label == self.label
        return name == "Serial Number"

    def getAllowedPropertyValues(self, label: str, name: str) -> tuple[str, ...]:
        assert label == self.label
        return ()

    def hasPropertyLimits(self, label: str, name: str) -> bool:
        assert label == self.label
        return name == "Exposure"

    def getPropertyLowerLimit(self, label: str, name: str) -> float:
        assert label == self.label and name == "Exposure"
        return 0.1

    def getPropertyUpperLimit(self, label: str, name: str) -> float:
        assert label == self.label and name == "Exposure"
        return 1000.0

    def snapImage(self) -> None:
        self.snap_count += 1

    def getImage(self) -> np.ndarray[Any, np.dtype[np.uint16]]:
        return np.full((2, 3), self.snap_count, dtype=np.uint16)

    def getExposure(self) -> float:
        return self.exposure_ms

    def setExposure(self, exposure_ms: float) -> None:
        self.exposure_ms = exposure_ms
        self.properties["Exposure"] = str(exposure_ms)

    def getROI(self) -> tuple[int, int, int, int]:
        return self.roi

    def setROI(self, x: int, y: int, width: int, height: int) -> None:
        self.roi = (x, y, width, height)

    def clearCircularBuffer(self) -> None:
        self.clear_buffer_count += 1
        self.sequence_frames.clear()

    def setCircularBufferMemoryFootprint(self, size_mb: int) -> None:
        self.buffer_memory_mb = size_mb

    def initializeCircularBuffer(self) -> None:
        self.initialize_buffer_count += 1

    def startSequenceAcquisition(
        self, frame_count: int, interval_ms: float, stop_on_overflow: bool
    ) -> None:
        self.finite_sequences.append((frame_count, interval_ms, stop_on_overflow))

    def startContinuousSequenceAcquisition(self, interval_ms: float) -> None:
        self.continuous_intervals.append(interval_ms)

    def getRemainingImageCount(self) -> int:
        return len(self.sequence_frames)

    def popNextImage(self) -> np.ndarray[Any, np.dtype[np.uint16]]:
        return self.sequence_frames.pop(0)

    def isBufferOverflowed(self) -> bool:
        return self.buffer_overflowed

    def stopSequenceAcquisition(self) -> None:
        self.stop_count += 1

    def unloadAllDevices(self) -> None:
        self.unload_count += 1


def _profile(tmp_path: Path, **changes: Any) -> MMCoreCameraProfile:
    config = tmp_path / "camera.cfg"
    config.write_text("# fake", encoding="utf-8")
    values: dict[str, Any] = {
        "service_id": "fluorescence",
        "config_path": config,
        "mm_path": tmp_path,
        "adapter": "PVCAM",
        "device_name": "Camera-1",
        "expected_label": "Camera-1",
        "serial_property": "Serial Number",
        "expected_serial": "prime-test",
    }
    values.update(changes)
    return MMCoreCameraProfile(**values)


def test_backend_admits_one_camera_and_forces_auto_shutter_off(
    tmp_path: Path,
) -> None:
    core = FakeCore()
    backend = MMCoreCameraBackend(_profile(tmp_path), core_factory=lambda path: core)

    identity = backend.connect()
    assert identity.adapter == "PVCAM"
    assert identity.serial == "prime-test"
    assert backend.configured_exposure_ms() == "10"
    assert core.auto_shutter is False
    assert backend.properties()[1].limits == (0.1, 1000.0)

    with pytest.raises(RuntimeError, match="not armed"):
        backend.trigger()
    backend.arm()
    np.testing.assert_array_equal(backend.trigger(), np.ones((2, 3), dtype=np.uint16))
    backend.stop()
    backend.disconnect()
    assert core.auto_shutter is False
    assert core.unload_count == 1


def test_backend_keeps_single_snap_and_sequence_paths_separate(tmp_path: Path) -> None:
    core = FakeCore()
    backend = MMCoreCameraBackend(_profile(tmp_path), core_factory=lambda path: core)
    backend.connect()

    backend.arm()
    np.testing.assert_array_equal(backend.trigger(), np.ones((2, 3), dtype=np.uint16))
    backend.stop()
    core.sequence_frames.extend(
        (
            np.full((2, 3), 4, dtype=np.uint16),
            np.full((2, 3), 5, dtype=np.uint16),
        )
    )
    backend.start_sequence(interval_ms=1.5, buffer_memory_mb=64)

    assert core.snap_count == 1
    assert core.clear_buffer_count == 1
    assert core.initialize_buffer_count == 1
    assert core.buffer_memory_mb == 64
    assert core.continuous_intervals == [1.5]
    assert backend.drain_sequence() == ()
    core.sequence_frames.extend(
        (
            np.full((2, 3), 4, dtype=np.uint16),
            np.full((2, 3), 5, dtype=np.uint16),
        )
    )
    drained = backend.drain_sequence()
    assert len(drained) == 2
    np.testing.assert_array_equal(drained[-1], np.full((2, 3), 5, dtype=np.uint16))
    core.buffer_overflowed = True
    assert backend.sequence_overflowed()
    with pytest.raises(RuntimeError, match="sequence acquisition"):
        backend.trigger()


def test_backend_uses_finite_sequence_api_for_list_scans(tmp_path: Path) -> None:
    core = FakeCore()
    backend = MMCoreCameraBackend(_profile(tmp_path), core_factory=lambda path: core)
    backend.connect()

    backend.start_sequence(frame_count=7, interval_ms=2.5, buffer_memory_mb=32)

    assert core.finite_sequences == [(7, 2.5, True)]
    assert core.continuous_intervals == []


def test_backend_rejects_wrong_adapter_and_cleans_up(tmp_path: Path) -> None:
    core = FakeCore(adapter="SpinnakerC")
    backend = MMCoreCameraBackend(_profile(tmp_path), core_factory=lambda path: core)

    with pytest.raises(CameraAdmissionError, match="does not match"):
        backend.connect()
    assert not backend.connected
    assert core.auto_shutter is False
    assert core.unload_count == 1


def test_backend_applies_an_explicit_initial_camera_property(tmp_path: Path) -> None:
    core = FakeCore()
    core.properties["PixelBinning"] = "1"
    backend = MMCoreCameraBackend(
        _profile(tmp_path, initial_properties={"PixelBinning": "2"}),
        core_factory=lambda path: core,
    )

    backend.connect()

    assert core.properties["PixelBinning"] == "2"


def test_backend_applies_an_explicit_mmcore_exposure(tmp_path: Path) -> None:
    core = FakeCore()
    backend = MMCoreCameraBackend(
        _profile(tmp_path, initial_properties={"Exposure": "10"}),
        core_factory=lambda path: core,
    )

    backend.connect()

    assert backend.configured_exposure_ms() == "10"


def test_backend_applies_an_explicit_mmcore_roi(tmp_path: Path) -> None:
    core = FakeCore()
    backend = MMCoreCameraBackend(
        _profile(tmp_path, initial_properties={"ROI": "0,0,808,620"}),
        core_factory=lambda path: core,
    )

    backend.connect()

    assert core.roi == (0, 0, 808, 620)


def test_spinnaker_profile_defaults_pixel_type_to_mono16(tmp_path: Path) -> None:
    core = FakeCore(
        label="Blackfly S BFS-U3-20S4M",
        adapter="SpinnakerC",
        device_name="Blackfly S BFS-U3-20S4M",
        serial="20154173",
    )
    core.properties["Pixel Format"] = "Mono8"
    backend = MMCoreCameraBackend(
        _profile(
            tmp_path,
            adapter="SpinnakerC",
            device_name="Blackfly S BFS-U3-20S4M",
            expected_label="Blackfly S BFS-U3-20S4M",
            expected_serial="20154173",
        ),
        core_factory=lambda path: core,
    )

    backend.connect()

    assert core.properties["Pixel Format"] == "Mono16"


def test_pvcam_and_spinnakerc_profiles_are_process_separable(tmp_path: Path) -> None:
    pvcam = _profile(tmp_path)
    flir = _profile(
        tmp_path,
        service_id="dhm",
        adapter="SpinnakerC",
        device_name="Blackfly S BFS-U3-20S4M",
        expected_label="Blackfly S BFS-U3-20S4M",
        expected_serial="20154173",
    )

    assert pvcam.adapter == "PVCAM"
    assert flir.adapter == "SpinnakerC"
    assert pvcam.config_path == flir.config_path
    assert pvcam.service_id != flir.service_id


def test_ioc_delegates_lifecycle_to_mmcore_backend(tmp_path: Path) -> None:
    async def scenario() -> None:
        core = FakeCore()
        core.properties["Binning"] = "2"
        core.properties["Pixel Format"] = "Mono16"
        backend = MMCoreCameraBackend(
            _profile(tmp_path), core_factory=lambda path: core
        )
        ioc = build_camera_ioc(
            "fluorescence",
            "AHT:PVCAM:TEST:",
            backend=backend,
            acquisition_slots=1,
        )

        await ioc.command_connect.pvspec.put(ioc, ioc.command_connect, 1)
        assert ioc.camera_adapter.value == "PVCAM"
        assert ioc.camera_serial.value == "prime-test"
        assert ioc.camera_pixel_binning.value == "2"
        assert ioc.camera_pixel_type.value == "Mono16"
        assert ioc.camera_exposure_ms.value == "10"
        await ioc.command_arm.pvspec.put(ioc, ioc.command_arm, 1)
        await ioc.command_trigger.pvspec.put(ioc, ioc.command_trigger, 1)
        assert core.snap_count == 1
        assert ioc.descriptor_run_id.value == "unassigned"
        assert ioc.descriptor_shape.value == "2,3"
        memory_name = str(ioc.descriptor_name.value)
        memory = shared_memory.SharedMemory(name=memory_name)
        try:
            copied = np.ndarray(
                (2, 3),
                dtype=np.dtype(str(ioc.descriptor_dtype.value)),
                buffer=memory.buf,
            ).copy()
        finally:
            memory.close()
        np.testing.assert_array_equal(copied, np.ones((2, 3), dtype=np.uint16))
        checksum = hashlib.sha256(copied.tobytes(order="C")).hexdigest()
        assert checksum == (
            str(ioc.descriptor_checksum_hi.value)
            + str(ioc.descriptor_checksum_lo.value)
        )
        with pytest.raises(BufferFullError):
            await ioc.command_trigger.pvspec.put(ioc, ioc.command_trigger, 1)
        assert core.snap_count == 1
        assert ioc.frames_dropped.value == 1
        await ioc.command_acknowledge.pvspec.put(
            ioc, ioc.command_acknowledge, int(ioc.descriptor_sequence.value)
        )
        await ioc.command_trigger.pvspec.put(ioc, ioc.command_trigger, 1)
        assert core.snap_count == 2
        await ioc.command_acknowledge.pvspec.put(
            ioc, ioc.command_acknowledge, int(ioc.descriptor_sequence.value)
        )
        await ioc.command_stop.pvspec.put(ioc, ioc.command_stop, 1)
        await ioc.command_disconnect.pvspec.put(ioc, ioc.command_disconnect, 1)
        assert core.unload_count == 1
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=memory_name)

    import asyncio

    asyncio.run(scenario())


def test_ioc_live_view_drains_sequence_without_single_snaps(tmp_path: Path) -> None:
    async def scenario() -> None:
        core = FakeCore()
        backend = MMCoreCameraBackend(
            _profile(tmp_path), core_factory=lambda path: core
        )
        ioc = build_camera_ioc(
            "fluorescence",
            "AHT:PVCAM:LIVE:",
            backend=backend,
            live_publish_hz=60,
        )

        await ioc.command_connect.pvspec.put(ioc, ioc.command_connect, 1)
        await ioc.command_start_live.pvspec.put(ioc, ioc.command_start_live, 1)
        core.sequence_frames.extend(
            (
                np.full((2, 3), 4, dtype=np.uint16),
                np.full((2, 3), 5, dtype=np.uint16),
            )
        )
        await asyncio.sleep(0.04)

        assert core.snap_count == 0
        assert core.continuous_intervals == [0.0]
        assert ioc.live_state.value == "running"
        assert ioc.live_frames_drained.value == 2
        assert ioc.live_frames_published.value == 1
        assert ioc.live_frames_skipped.value == 1
        assert ioc.live_sequence.value == 1
        payload = str(ioc.live_image.value).encode("latin-1")
        image = np.frombuffer(payload, dtype=np.dtype(str(ioc.live_dtype.value)))
        np.testing.assert_array_equal(image.reshape(2, 3), np.full((2, 3), 5))

        await ioc.command_stop.pvspec.put(ioc, ioc.command_stop, 1)
        assert ioc.live_state.value == "idle"
        assert ioc.acquisition_state.value == "idle"

    asyncio.run(scenario())
