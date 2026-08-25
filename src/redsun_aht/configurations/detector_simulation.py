"""Inspectable dual-detector simulation composition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from redsun_aht.services import DetectorRegistry, MockCameraService

if TYPE_CHECKING:
    from redsun_aht.domain import Frame


@dataclass(slots=True)
class DualDetectorSimulation:
    """Two isolated mock camera services and their shared supervisor."""

    registry: DetectorRegistry
    fluorescence: MockCameraService
    dhm: MockCameraService

    async def run(self) -> dict[str, Frame]:
        """Acquire once from both services and always release their resources."""
        try:
            failures = await self.registry.connect()
            if failures:
                raise ExceptionGroup(
                    "detector simulation connection failures", list(failures.values())
                )
            detectors = (self.fluorescence, self.dhm)
            for detector in detectors:
                await detector.configure({"exposure_s": 0.01})
                await detector.arm()
            await asyncio.gather(*(detector.trigger() for detector in detectors))
            frames = {
                detector.service_id: await detector.read() for detector in detectors
            }
            for detector in detectors:
                detector.copy_latest(acknowledge=True)
                await detector.stop()
            return frames
        finally:
            await self.registry.disconnect_all()


def build_dual_detector_simulation() -> DualDetectorSimulation:
    """Build two hardware-free camera services with isolated EPICS prefixes."""
    registry = DetectorRegistry()
    fluorescence = MockCameraService("fluorescence", "AHT:PVCAM:SIM:")
    dhm = MockCameraService("dhm", "AHT:FLIR:SIM:")
    registry.register(fluorescence)
    registry.register(dhm)
    return DualDetectorSimulation(registry, fluorescence, dhm)


def run_dual_detector_simulation() -> dict[str, Frame]:
    """Run the hardware-free dual-detector smoke scenario."""
    return asyncio.run(build_dual_detector_simulation().run())


__all__ = [
    "DualDetectorSimulation",
    "build_dual_detector_simulation",
    "run_dual_detector_simulation",
]
