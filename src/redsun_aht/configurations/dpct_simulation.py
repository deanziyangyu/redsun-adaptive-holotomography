"""Inspectable hardware-free DPCT composition and smoke workflow."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from redsun_aht import __version__
from redsun_aht.acquisition import (
    DpctRecipe,
    DpctRunner,
    DpctRunResult,
    panel91_dpct_quadrant_patterns,
)
from redsun_aht.device.mcu import McuHostClient
from redsun_aht.device.stage import (
    AxisSpec,
    CompositeStageDevice,
    SimulatedAxisBackend,
)
from redsun_aht.mcu import Capabilities, SimulatedMcu, SimulatedMcuTransport
from redsun_aht.motion import grid_scan
from redsun_aht.services import MockCameraService
from redsun_aht.storage import DpctZarrStore, RunManifest

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class DpctSimulation:
    """Complete in-process DPCT composition with no physical backends."""

    runner: DpctRunner
    recipe: DpctRecipe
    mcu: SimulatedMcu

    async def run(self) -> DpctRunResult:
        """Execute the configured simulation and return traceable shot records."""
        return await self.runner.run(self.recipe)


def build_dpct_simulation(
    run_id: str = "dpct-simulation",
    *,
    output_root: Path | None = None,
    pattern_ids: tuple[int, ...] = (10, 11),
) -> DpctSimulation:
    """Build a two-pose, configurable-pattern, dual-detector simulation."""
    stage = CompositeStageDevice(
        specs=(AxisSpec("z", "mm", 0, 1, 0.001),),
        backends={"z": SimulatedAxisBackend(settle_polls=1)},
        settle_poll_interval=0,
    )
    mcu = SimulatedMcu(
        Capabilities(
            firmware_version="1.0.0-sim",
            target="aht-simulation",
            panel_profile="aht-test-4",
            panel_revision=1,
            led_count=4,
            max_pattern_bytes=4,
            max_sequence_steps=8,
        )
    )
    illumination = McuHostClient(SimulatedMcuTransport(mcu), chunk_bytes=2)
    if not pattern_ids or len(pattern_ids) > mcu.capabilities.led_count:
        raise ValueError("DPCT simulation requires one to four pattern IDs")
    for index, pattern_id in enumerate(pattern_ids):
        payload = bytearray(mcu.capabilities.led_count)
        payload[index] = 0xFF
        illumination.upload_pattern(pattern_id, bytes(payload))
    detectors = (
        MockCameraService("fluorescence", "AHT:PVCAM:SIM:", run_id=run_id),
        MockCameraService("dhm", "AHT:FLIR:SIM:", run_id=run_id),
    )
    recipe = DpctRecipe(
        run_id=run_id,
        scan_points=grid_scan({"z": (0.1, 0.2)}),
        pattern_ids=pattern_ids,
        detector_settings={"exposure_s": 0.01},
    )
    sink = None
    if output_root is not None:
        run_manifest = RunManifest(
            schema_version=1,
            run_id=run_id,
            created_ns=time.time_ns(),
            software_version=__version__,
            hardware={
                "stage": {"backend": "simulation", "axes": ["z"]},
                "mcu": {
                    "backend": "simulation",
                    "protocol": "native-v1",
                    "panel_profile": mcu.capabilities.panel_profile,
                    "panel_revision": mcu.capabilities.panel_revision,
                },
                "detectors": [
                    {"id": detector.service_id, "backend": "simulation"}
                    for detector in detectors
                ],
            },
            reconstruction={},
            experiment={
                "recipe": "dpct",
                "scan_points": [dict(point.positions) for point in recipe.scan_points],
                "pattern_ids": list(recipe.pattern_ids),
                "detector_settings": dict(recipe.detector_settings),
            },
            deployment={"profile": "simulate-dpct", "python": ">=3.12"},
        )
        sink = DpctZarrStore(output_root, run_manifest)
    return DpctSimulation(
        DpctRunner(stage, illumination, detectors, sink=sink), recipe, mcu
    )


def run_dpct_simulation(
    *,
    output_root: Path | None = None,
    pattern_ids: tuple[int, ...] = (10, 11),
) -> DpctRunResult:
    """Run the inspectable hardware-free DPCT smoke workflow."""
    return asyncio.run(
        build_dpct_simulation(
            output_root=output_root,
            pattern_ids=pattern_ids,
        ).run()
    )


def build_panel91_dpct_simulation(
    run_id: str = "panel91-dpct-simulation",
    *,
    output_root: Path | None = None,
    pattern_ids: tuple[int, int, int, int] = (101, 102, 103, 104),
    intensity: int = 255,
) -> DpctSimulation:
    """Build a four-donor-group DPCT dry run for the active 91-LED panel.

    This confirms host pattern construction, transactional upload, MDA ordering,
    and durable reconstruction input shape. Its simulated camera data cannot
    determine the real panel's optical azimuth calibration.
    """
    profiles = panel91_dpct_quadrant_patterns(
        pattern_ids=pattern_ids,
        intensity=intensity,
    )
    stage = CompositeStageDevice(
        specs=(AxisSpec("z", "mm", 0, 1, 0.001),),
        backends={"z": SimulatedAxisBackend(settle_polls=1)},
        settle_poll_interval=0,
    )
    mcu = SimulatedMcu(
        Capabilities(
            firmware_version="1.0.0-sim",
            target="aht-simulation",
            panel_profile="panel91",
            panel_revision=1,
            led_count=91,
            max_pattern_bytes=91,
            max_sequence_steps=8,
        )
    )
    illumination = McuHostClient(SimulatedMcuTransport(mcu), chunk_bytes=91)
    for profile in profiles.values():
        illumination.upload_pattern(profile.pattern_id, profile.values)
    detectors = (
        MockCameraService("fluorescence", "AHT:PVCAM:SIM:", run_id=run_id),
        MockCameraService("dhm", "AHT:FLIR:SIM:", run_id=run_id),
    )
    recipe = DpctRecipe(
        run_id=run_id,
        scan_points=grid_scan({"z": (0.1, 0.2)}),
        pattern_ids=pattern_ids,
        detector_settings={"exposure_s": 0.01},
    )
    sink = None
    if output_root is not None:
        run_manifest = RunManifest(
            schema_version=1,
            run_id=run_id,
            created_ns=time.time_ns(),
            software_version=__version__,
            hardware={
                "stage": {"backend": "simulation", "axes": ["z"]},
                "mcu": {
                    "backend": "simulation",
                    "protocol": "native-v1",
                    "panel_profile": "panel91",
                    "panel_revision": 1,
                    "led_count": 91,
                    "channel": "red",
                },
                "detectors": [
                    {"id": detector.service_id, "backend": "simulation"}
                    for detector in detectors
                ],
            },
            reconstruction={"opticalAzimuthCalibration": "not-set"},
            experiment={
                "recipe": "dpct-panel91-four-group",
                "scan_points": [dict(point.positions) for point in recipe.scan_points],
                "pattern_ids": list(recipe.pattern_ids),
                "pattern_source": "panel91-dpct-four-group-donor",
                "detector_settings": dict(recipe.detector_settings),
            },
            deployment={"profile": "simulate-panel91-dpct", "python": ">=3.12"},
        )
        sink = DpctZarrStore(output_root, run_manifest)
    return DpctSimulation(
        DpctRunner(stage, illumination, detectors, sink=sink), recipe, mcu
    )


__all__ = [
    "DpctSimulation",
    "build_dpct_simulation",
    "build_panel91_dpct_simulation",
    "run_dpct_simulation",
]
