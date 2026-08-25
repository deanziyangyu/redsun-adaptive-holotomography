from __future__ import annotations

import asyncio

import numpy as np
import pytest

from redsun_aht.buffers import BufferFullError
from redsun_aht.configurations import DualDetectorSimulation
from redsun_aht.device import ProcessingDeviceAdapter
from redsun_aht.domain import AcquisitionState, ServiceConnectionState
from redsun_aht.protocols import Detector, SupervisedDetector
from redsun_aht.services import DetectorRegistry, MockCameraService


def test_dual_mock_services_acquire_independently_and_together() -> None:
    async def scenario() -> None:
        fluorescence = MockCameraService("fluorescence", "AHT:PVCAM:")
        dhm = MockCameraService("dhm", "AHT:FLIR:")
        registry = DetectorRegistry()
        registry.register(fluorescence)
        registry.register(dhm)
        assert isinstance(fluorescence, Detector)
        assert isinstance(fluorescence, SupervisedDetector)

        capabilities = await registry.discover()
        assert set(capabilities) == {"fluorescence", "dhm"}
        assert await registry.connect() == {}
        try:
            for detector in (fluorescence, dhm):
                await detector.configure({"exposure_s": 0.01})
                await detector.arm()

            await fluorescence.trigger()
            fluorescence_frame = await fluorescence.read()
            fluorescence_data = fluorescence.copy_latest()
            assert fluorescence_frame.detector_id == "fluorescence"
            assert dhm.status.frames_published == 0

            await asyncio.gather(fluorescence.trigger(), dhm.trigger())
            fluorescence_data_2 = fluorescence.copy_latest()
            dhm_data = dhm.copy_latest()
            assert fluorescence.status.frames_published == 2
            assert dhm.status.frames_published == 1
            assert not np.array_equal(fluorescence_data, dhm_data)
            assert not np.array_equal(fluorescence_data, fluorescence_data_2)
        finally:
            await registry.disconnect_all()

        assert all(
            status.connection_state is ServiceConnectionState.DISCONNECTED
            for status in registry.status().values()
        )

    asyncio.run(scenario())


def test_fault_and_reconnect_are_isolated_by_service() -> None:
    async def scenario() -> None:
        unstable = MockCameraService(
            "unstable",
            "AHT:CAM:A:",
            fail_connect_attempts=1,
        )
        peer = MockCameraService("peer", "AHT:CAM:B:")
        registry = DetectorRegistry()
        registry.register(unstable)
        registry.register(peer)

        failures = await registry.connect()
        assert set(failures) == {"unstable"}
        assert unstable.status.connection_state is ServiceConnectionState.FAULTED
        assert peer.status.connection_state is ServiceConnectionState.READY

        status = await registry.reconnect("unstable")
        assert status.connection_state is ServiceConnectionState.READY
        first_generation = status.schema_generation
        await unstable.inject_disconnect()
        assert peer.status.connection_state is ServiceConnectionState.READY

        status = await registry.reconnect("unstable")
        assert status.schema_generation > first_generation
        assert peer.status.schema_generation == 1
        await registry.disconnect_all()

    asyncio.run(scenario())


def test_lossless_service_buffer_backpressure_is_visible() -> None:
    async def scenario() -> None:
        detector = MockCameraService(
            "detector",
            "AHT:CAM:",
            acquisition_slots=1,
        )
        await detector.connect()
        try:
            await detector.arm()
            await detector.trigger()
            with pytest.raises(BufferFullError):
                await detector.trigger()
            assert detector.status.frames_dropped == 1
            assert detector.status.acquisition_state is AcquisitionState.ARMED

            detector.copy_latest()
            await detector.trigger()
            assert detector.status.frames_published == 2
        finally:
            await detector.disconnect()

    asyncio.run(scenario())


def test_registry_rejects_duplicate_identity_and_prefix() -> None:
    registry = DetectorRegistry()
    registry.register(MockCameraService("a", "AHT:A:"))

    with pytest.raises(ValueError, match="duplicate detector"):
        registry.register(MockCameraService("a", "AHT:B:"))
    with pytest.raises(ValueError, match="duplicate EPICS"):
        registry.register(MockCameraService("b", "AHT:A:"))


def test_processing_device_compatibility_seam_is_public() -> None:
    assert ProcessingDeviceAdapter.__name__ == "ProcessingDeviceAdapter"


def test_empty_selection_and_failed_composition_cleanup() -> None:
    async def scenario() -> None:
        failing = MockCameraService("failing", "AHT:FAIL:", fail_connect_attempts=1)
        peer = MockCameraService("peer", "AHT:PEER:")
        registry = DetectorRegistry()
        registry.register(failing)
        registry.register(peer)

        assert await registry.connect([]) == {}
        assert all(
            status.connection_state is ServiceConnectionState.DISCONNECTED
            for status in registry.status().values()
        )

        simulation = DualDetectorSimulation(registry, failing, peer)
        with pytest.raises(ExceptionGroup, match="connection failures"):
            await simulation.run()
        assert all(
            status.connection_state is ServiceConnectionState.DISCONNECTED
            for status in registry.status().values()
        )

    asyncio.run(scenario())
