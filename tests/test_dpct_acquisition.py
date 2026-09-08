from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from redsun_aht.acquisition import (
    DpctRecipe,
    DpctRunner,
    EpicsCameraDetector,
    MultishotListScanPlan,
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


def test_dpct_runner_associates_pose_pattern_and_lossless_frames() -> None:
    run_id = "dpct-run-1"
    z_backend = SimulatedAxisBackend(settle_polls=1)
    stage = CompositeStageDevice(
        specs=(AxisSpec("z", "mm", 0, 1, 0.001),),
        backends={"z": z_backend},
        settle_poll_interval=0,
    )
    endpoint = SimulatedMcu(
        Capabilities(
            firmware_version="sim",
            target="aht",
            panel_profile="test-4",
            panel_revision=1,
            led_count=4,
            max_pattern_bytes=4,
            max_sequence_steps=8,
        )
    )
    illumination = McuHostClient(SimulatedMcuTransport(endpoint), chunk_bytes=2)
    illumination.upload_pattern(10, b"\xff\x00\x00\x00")
    illumination.upload_pattern(11, b"\x00\xff\x00\x00")
    detectors = (
        MockCameraService("fluorescence", "AHT:PVCAM:SIM:", run_id=run_id),
        MockCameraService("dhm", "AHT:FLIR:SIM:", run_id=run_id),
    )
    recipe = DpctRecipe(
        run_id=run_id,
        scan_points=grid_scan({"z": (0.1, 0.2)}),
        pattern_ids=(10, 11),
        detector_settings={"exposure_s": 0.01},
    )

    result = asyncio.run(DpctRunner(stage, illumination, detectors).run(recipe))

    assert result.run_id == run_id
    assert result.detector_ids == ("fluorescence", "dhm")
    assert [(shot.scan_index, shot.pattern_id) for shot in result.shots] == [
        (0, 10),
        (0, 11),
        (1, 10),
        (1, 11),
    ]
    assert [shot.pose.joints for shot in result.shots] == [
        (0.1,),
        (0.1,),
        (0.2,),
        (0.2,),
    ]
    assert all(
        tuple(frame.detector_id for frame in shot.frames) == ("fluorescence", "dhm")
        for shot in result.shots
    )
    assert endpoint.displayed_pattern_id is None
    assert z_backend.stopped
    assert all(
        detector.status.command_state == "disconnected" for detector in detectors
    )


def test_multishot_list_scan_plan_builds_a_forty_position_four_shot_stack() -> None:
    stage = CompositeStageDevice(
        specs=(AxisSpec("z", "mm", 0, 1, 0.001),),
        backends={"z": SimulatedAxisBackend()},
        settle_poll_interval=0,
    )
    endpoint = SimulatedMcu(Capabilities("sim", "aht", "test", 1, 4, 4, 8))
    illumination = McuHostClient(SimulatedMcuTransport(endpoint))
    for pattern_id in (10, 11, 12, 13):
        illumination.upload_pattern(pattern_id, bytes((pattern_id,) * 4))
    detector = MockCameraService("dhm", "AHT:FLIR:SIM:", run_id="finite-plan")
    plan = MultishotListScanPlan(stage, illumination, (detector,))
    recipe = plan.four_shot_fixed_step_recipe(
        "finite-plan",
        fixed_positions={},
        z_start_mm=0,
        z_step_um=1,
        z_positions=40,
        pattern_ids=(10, 11, 12, 13),
        detector_settings={},
    )

    result = asyncio.run(plan.run(recipe))

    assert len(result.shots) == 160
    assert [shot.pattern_id for shot in result.shots[:4]] == [10, 11, 12, 13]
    assert [shot.pose.joints[0] for shot in result.shots[::4]] == pytest.approx(
        [index / 1000 for index in range(40)]
    )


def test_bare_parked_plan_does_not_move_between_grouped_four_shot_sequences() -> None:
    backend = SimulatedAxisBackend()
    stage = CompositeStageDevice(
        specs=(AxisSpec("z", "mm", 0, 1, 0.001),),
        backends={"z": backend},
        settle_poll_interval=0,
    )
    endpoint = SimulatedMcu(Capabilities("sim", "aht", "test", 1, 4, 4, 8))
    illumination = McuHostClient(SimulatedMcuTransport(endpoint))
    for pattern_id in (10, 11, 12, 13):
        illumination.upload_pattern(pattern_id, bytes((pattern_id,) * 4))
    detector = MockCameraService("dhm", "AHT:FLIR:SIM:", run_id="bare-plan")
    plan = MultishotListScanPlan(stage, illumination, (detector,))
    recipe = plan.four_shot_parked_recipe(
        "bare-plan",
        parked_positions={"z": 0.1},
        grouped_sequences=3,
        pattern_ids=(10, 11, 12, 13),
        detector_settings={},
    )

    result = asyncio.run(plan.run_bare(recipe))

    assert len(result.shots) == 12
    assert [shot.pose.joints for shot in result.shots] == [(0.1,)] * 12
    assert backend.commands == [0.1]
    assert backend.stopped


@pytest.mark.parametrize(
    "recipe",
    [
        lambda: DpctRecipe("", grid_scan({"z": (0,)}), (1,), {}),
        lambda: DpctRecipe("run", (), (1,), {}),
        lambda: DpctRecipe("run", grid_scan({"z": (0,)}), (), {}),
        lambda: DpctRecipe("run", grid_scan({"z": (0,)}), (1, 1), {}),
        lambda: DpctRecipe("run", grid_scan({"z": (0,)}), (0x10000,), {}),
    ],
)
def test_dpct_recipe_rejects_ambiguous_axes(
    recipe: Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        recipe()


def test_dpct_run_identity_mismatch_still_cleans_every_component() -> None:
    stage_backend = SimulatedAxisBackend()
    stage = CompositeStageDevice(
        specs=(AxisSpec("z", "mm", 0, 1, 0.001),),
        backends={"z": stage_backend},
        settle_poll_interval=0,
    )
    endpoint = SimulatedMcu(Capabilities("sim", "aht", "test", 1, 1, 1, 1))
    illumination = McuHostClient(SimulatedMcuTransport(endpoint))
    illumination.upload_pattern(1, b"\xff")
    detectors = (MockCameraService("camera-1", "AHT:C1:SIM:", run_id="wrong-run"),)
    recipe = DpctRecipe(
        "expected-run", grid_scan({"z": (0.5,)}), (1,), {"exposure_s": 0.01}
    )

    with pytest.raises(RuntimeError, match="run identity"):
        asyncio.run(DpctRunner(stage, illumination, detectors).run(recipe))

    assert endpoint.displayed_pattern_id is None
    assert stage_backend.stopped
    assert detectors[0].status.command_state == "disconnected"


def test_epics_camera_detector_requires_a_fixed_camera_configuration() -> None:
    detector = EpicsCameraDetector("camera", "AHT:CAM:")

    assert asyncio.run(detector.configure({})) == "camera-ioc-fixed-configuration-v1"
    with pytest.raises(ValueError, match="configuration is not implemented"):
        asyncio.run(detector.configure({"exposure_s": 0.01}))
    with pytest.raises(ValueError, match="poll interval"):
        EpicsCameraDetector("camera", "AHT:CAM:", poll_interval=0)


def test_epics_camera_detector_uses_a_redsun_camera_service_device() -> None:
    detector = EpicsCameraDetector("camera", "AHT:CAM:")

    assert detector.service_id == "camera"
    assert detector.prefix == "AHT:CAM:"
