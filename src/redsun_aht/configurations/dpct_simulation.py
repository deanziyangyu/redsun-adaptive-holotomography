"""Inspectable hardware-free DPCT composition and smoke workflow."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from bluesky.run_engine import RunEngineResult
from bluesky.utils import FailedStatus
from redsun.containers import AppContainer, declare_device
from redsun.engine import RunEngine

from redsun_aht import __version__
from redsun_aht.acquisition import (
    DpctRecipe,
    DpctRunResult,
    dpct_plan,
    panel91_dpct_quadrant_patterns,
)
from redsun_aht.device import (
    DpctDetectorGroupDevice,
    DpctPatternDevice,
    DpctStageDevice,
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

from .profiles import profile_path

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class DpctSimulation:
    """Complete in-process DPCT composition with no physical backends."""

    container: AppContainer
    recipe: DpctRecipe
    mcu: SimulatedMcu

    async def run(self) -> DpctRunResult:
        """Execute the RedSun device plan and return traceable shot records."""
        engine = RunEngine({})
        self.container.build()
        try:
            stage = cast("DpctStageDevice", self.container.devices["stage"])
            illumination = cast(
                "DpctPatternDevice", self.container.devices["illumination"]
            )
            detectors = cast(
                "DpctDetectorGroupDevice", self.container.devices["detectors"]
            )
            future = engine(dpct_plan(stage, illumination, detectors, self.recipe))
            try:
                engine_result = await asyncio.wrap_future(future)
            except FailedStatus as error:
                cause = error.__cause__
                if cause is not None:
                    raise cause from error
                raise
            if not isinstance(engine_result, RunEngineResult):
                raise RuntimeError("RedSun RunEngine did not return a plan result")
            return cast("DpctRunResult", engine_result.plan_result)
        finally:
            self.container.shutdown()


def _build_container(
    *,
    stage: CompositeStageDevice,
    illumination: McuHostClient,
    detectors: tuple[MockCameraService, ...],
    recipe: DpctRecipe,
    sink: DpctZarrStore | None,
) -> AppContainer:
    """Declare runtime backends behind RedSun-compatible device components."""
    stage_backend = stage
    illumination_backend = illumination
    detector_backends = detectors
    acquisition_recipe = recipe
    frame_sink = sink

    class AHTDpctSimulationContainer(
        AppContainer, config=profile_path("simulate-dpct")
    ):
        stage = declare_device(DpctStageDevice, backend=stage_backend)
        illumination = declare_device(DpctPatternDevice, backend=illumination_backend)
        detectors = declare_device(
            DpctDetectorGroupDevice,
            detectors=detector_backends,
            recipe=acquisition_recipe,
            sink=frame_sink,
        )

    return AHTDpctSimulationContainer(
        session=f"AHT DPCT simulation: {recipe.run_id}",
        frontend="headless",
    )


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
    container = _build_container(
        stage=stage,
        illumination=illumination,
        detectors=detectors,
        recipe=recipe,
        sink=sink,
    )
    return DpctSimulation(container, recipe, mcu)


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
    container = _build_container(
        stage=stage,
        illumination=illumination,
        detectors=detectors,
        recipe=recipe,
        sink=sink,
    )
    return DpctSimulation(container, recipe, mcu)


__all__ = [
    "DpctSimulation",
    "build_dpct_simulation",
    "build_panel91_dpct_simulation",
    "run_dpct_simulation",
]
