"""Instrument profile path handling without hardware or a graphical session."""

import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock, patch

import pytest
from bluesky import RunEngine
from ophyd_async.core import get_mock_put
from redsun import AsPresenter, Session
from redsun.aio import run_coro
from redsun.engine import RunEngine as RedsunRunEngine
from redsun.presenter import PositionerPresenter

from redsun_aht.configurations.profile import (
    check_holospim_host,
    read_instrument_profile,
    run_profile,
)
from redsun_aht.presenter.acquisition import AcquisitionPresenter
from redsun_aht.presenter.detector import DetectorPresenter
from redsun_aht.presenter.holospim import HolospimPresenter


@pytest.mark.parametrize("instrument", ["arh", "holospim"])
def test_profile_and_overlay_resolve_paths_from_their_own_directories(
    instrument, tmp_path
):
    base = tmp_path / "base" / "profile.yaml"
    overlay = tmp_path / "overlay" / "profile.yaml"
    base.parent.mkdir()
    overlay.parent.mkdir()
    base.write_text(
        'services:\n  camera:\n    args:\n      mm-config: "./camera.cfg"\n'
    )
    overlay.write_text(
        'services:\n  stage:\n    args:\n      mmcore-config: "./stage.cfg"\n'
    )
    session = Mock()
    with patch(
        "redsun_aht.configurations.profile.Session.from_config", return_value=session
    ) as create:
        assert (
            run_profile(
                [
                    instrument,
                    "--profile",
                    str(base),
                    "--overlay",
                    str(overlay),
                    "--check",
                ]
            )
            == 0
        )
    layers = create.call_args.args[0]
    assert layers[0]["services"]["camera"]["args"]["mm-config"] == str(
        base.parent / "camera.cfg"
    )
    assert layers[1]["services"]["stage"]["args"]["mmcore-config"] == str(
        overlay.parent / "stage.cfg"
    )
    assert layers[-1]["mock"] is True
    session.build.assert_called_once()
    session.shutdown.assert_called_once()


def test_bad_service_args_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("services:\n  camera:\n    args: invalid\n")
    with pytest.raises(ValueError, match="args mapping"):
        read_instrument_profile(path)


def test_holospim_host_check_merges_overlay_and_reports_bad_settings(tmp_path, capsys):
    overlay = tmp_path / "host.yaml"
    overlay.write_text(
        """mock: false
services:
  survey_service:
    plugin_name: redsun-aht
    plugin_id: esp300-stage
    prefix: "HOLOSPIM-SURVEY:"
    args:
      port: COM1
      prefix: "HOLOSPIM-SURVEY:"
      axis-identities: [x-axis, y-axis, z-axis]
      limits: x=0:10,y=0:10
  camera_service:
    plugin_name: redsun-aht
    plugin_id: spinnaker-camera
    prefix: "HOLOSPIM-CAMERA:"
    args:
      serial: "123"
      pixel-format: Mono16
  pvcam_service:
    plugin_name: redsun-aht
    plugin_id: mmcore-camera
    prefix: "HOLOSPIM-PVCAM:"
    args:
      mm-config: ./host.cfg
      device: Camera-1
  mcu_service:
    plugin_name: redsun-aht
    plugin_id: unified-mcu
    prefix: "HOLOSPIM-MCU:"
    args:
      port: COM18
      prefix: "HOLOSPIM-MCU:"
""",
        encoding="utf-8",
    )
    args = ["holospim", "--overlay", str(overlay)]
    with patch("redsun_aht.configurations.profile.Session.from_config") as session:
        with pytest.raises(SystemExit) as error:
            run_profile(args)
        assert error.value.code == 2
        errors = capsys.readouterr().err
        assert "survey_service.args.limits" in errors
        assert "pvcam_service.args.mm-config" in errors
        overlay.write_text(
            overlay.read_text("utf-8").replace("y=0:10", "y=0:10,z=0:10"),
            "utf-8",
        )
        (tmp_path / "host.cfg").touch()
        assert run_profile([*args, "--check-host"]) == 0
        session.assert_not_called()


def test_static_host_check_rejects_partial_service_replacement(tmp_path, capsys):
    overlay = tmp_path / "partial.yaml"
    overlay.write_text("services:\n  mcu_service:\n    args:\n      port: COM18\n")
    with pytest.raises(SystemExit):
        run_profile(["holospim", "--overlay", str(overlay), "--check-host"])
    assert (
        "mcu_service supplies args without a service module or plugin"
        in capsys.readouterr().err
    )


def test_mock_profile_allows_placeholders_but_static_host_check_reports_them(capsys):
    with patch("redsun_aht.configurations.profile.Session.from_config") as session:
        assert run_profile(["holospim", "--check-profile"]) == 0
        session.assert_called_once()
        session.reset_mock()
        with pytest.raises(SystemExit) as error:
            run_profile(["holospim", "--check-host"])
        assert error.value.code == 2
        session.assert_not_called()
    assert (
        "survey_service.args.axis-identities[2] is still a placeholder"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize("profile", ["arh_system.yaml", "holospim.yaml"])
def test_packaged_device_profiles_build_and_shutdown_mock(profile, tmp_path):
    path = Path(__file__).resolve().parents[1] / "src/redsun_aht/profiles" / profile
    session = Session.from_config(
        [
            read_instrument_profile(path),
            {"mock": True, "strict": True, "storage": {"base_dir": str(tmp_path)}},
        ]
    )
    try:
        session.build()
        assert {"camera", "mcu"} <= session.devices.keys()
        if profile == "holospim.yaml":
            assert {
                "survey",
                "camera",
                "pvcam",
                "rotation",
                "mcu",
            } == session.devices.keys()
    finally:
        session.shutdown()


def test_holospim_operator_plan_targets_only_the_profile_mcu(tmp_path):
    class HeadlessSession(Session):
        config: ClassVar[dict[str, object]] = {
            "presenters": {
                "det_ctrl": {},
                "acq_ctrl": {},
                "holospim_ctrl": {},
                "positioner_ctrl": {},
            },
        }
        det_ctrl: AsPresenter[DetectorPresenter]
        acq_ctrl: AsPresenter[AcquisitionPresenter]
        holospim_ctrl: AsPresenter[HolospimPresenter]
        positioner_ctrl: AsPresenter[PositionerPresenter]

    path = Path(__file__).resolve().parents[1] / "src/redsun_aht/profiles/holospim.yaml"
    session = HeadlessSession(
        [
            read_instrument_profile(path),
            {
                "storage": {"base_dir": str(tmp_path)},
                "devices": {"mcu": {
                    "plugin_name": "redsun-aht",
                    "plugin_id": "unified-mcu",
                    "service": "mcu_service",
                    "x28_um_per_turn": -1000,
                    "x28_jog_bounds_tenths": [-36, 36],
                }},
                "presenters": {
                    "holospim_ctrl": {
                        "x28_bounds_tenths": [-180, 180],
                        "enable_dynamic_dpct": True,
                        "enable_x28_triggered_stack": True,
                        "hand_eye_roll_limits_deg": [-10, 10],
                    }
                },
            },
        ]
    )
    try:
        session.build()
        plan = session.acq_ctrl.plans["holospim_led_blank"]
        parameter = next(p for p in plan.parameters if p.name == "mcu")
        assert tuple(parameter.choices) == ("mcu",)
        assert "holospim_x28_readback" in session.acq_ctrl.plans
        stack = session.acq_ctrl.plans["holospim_x28_software_stack"]
        cameras = next(p for p in stack.parameters if p.name == "camera")
        assert set(cameras.choices) == {"camera", "pvcam"}
        triggered = session.acq_ctrl.plans["holospim_x28_triggered_stack"]
        camera_parameter = next(p for p in triggered.parameters if p.name == "camera")
        assert tuple(camera_parameter.choices) == ("camera",)
        assert "holospim_dynamic_dpct" in session.acq_ctrl.plans
        mcu = session.devices["mcu"]
        assert "x28" in session.positioner_ctrl.axes["mcu"]
        run_coro(session.positioner_ctrl.move("mcu", "x28", -5.0))
        assert run_coro(mcu.x28.locate())["readback"] == -5
        assert "rotation" in session.positioner_ctrl.axes["rotation"]
        survey = session.devices["survey"]
        assert set(session.positioner_ctrl.axes[survey.axis.name]) == {"x", "y", "z"}
        run_coro(session.positioner_ctrl.move_to(survey.axis.name, {"z": 5.0}))
        survey_reading = run_coro(survey.read_position())
        assert survey_reading.quality == "simulated"
        assert {axis.name: axis.value for axis in survey_reading.axes}["z"] == 5.0
        assert "holospim_hand_eye_preview" in session.acq_ctrl.plans
        starts = []
        rotation = session.devices["rotation"]
        camera = session.devices["camera"]
        preview = session.holospim_ctrl.hand_eye_preview(
            camera, survey, rotation, parked_x28_mm=0.25, x_step_um=10, roll_step_deg=2
        )
        engine = RedsunRunEngine({})
        engine(preview, {"start": lambda _, document: starts.append(document)}).result(
            10
        )
        path = json.loads(Path(starts[0]["hand_eye_path"]).read_text("utf-8"))
        assert path["origin_um"] == [0.0, 0.0, 5.0]
        assert path["parked_x28_mm"] == 0.25
        assert path["capture_waypoint_indices"] == [0, 1, 3, 5, 7]
        assert path["relative_waypoints"][1] == [-10, 0, 0, 0]
        assert path["relative_waypoints"][5][3] == -1.93359375
        RunEngine({})(session.holospim_ctrl.blank(mcu))
        get_mock_put(mcu.illumination).assert_awaited_once_with("00")
    finally:
        session.shutdown()


def test_holospim_x28_profile_requires_matching_bounded_service():
    service = {
        "plugin_name": "redsun-aht",
        "plugin_id": "unified-mcu",
        "prefix": "SYNTHETIC:",
        "args": {
            "prefix": "SYNTHETIC:",
            "port": "SYNTHETIC",
            "require-holospim-panel": True,
            "x28-jog-bounds-tenths": [-3600, 3600],
        },
    }
    device = {
        "service": "mcu_service",
        "x28_um_per_turn": -1000,
        "x28_jog_bounds_tenths": [-3600, 3600],
    }
    config = {"services": {"mcu_service": service}, "devices": {"mcu": device}}
    check_holospim_host([config])
    device["x28_jog_bounds_tenths"] = [-180, 180]
    with pytest.raises(ValueError, match="must match"):
        check_holospim_host([config])
    device["x28_jog_bounds_tenths"] = [-3600, 3600]
    device["x28_um_per_turn"] = 1001
    with pytest.raises(ValueError, match="exceed 1000"):
        check_holospim_host([config])


def test_holospim_live_roll_jog_requires_service_bounds():
    service = {
        "plugin_name": "redsun-aht",
        "plugin_id": "unified-mcu",
        "prefix": "SYNTHETIC:",
        "args": {
            "prefix": "SYNTHETIC:", "port": "SYNTHETIC",
            "require-holospim-panel": True,
        },
    }
    rotation = {"service": "mcu_service", "enable_controller_jog": True}
    config = {
        "services": {"mcu_service": service},
        "devices": {"rotation": rotation},
    }
    with pytest.raises(ValueError, match="commissioned HoloSPIM roll bounds"):
        check_holospim_host([config])
    service["args"]["roll-bounds-steps"] = [-50, 50]
    check_holospim_host([config])
    service["args"]["require-holospim-panel"] = False
    with pytest.raises(ValueError, match="commissioned HoloSPIM roll bounds"):
        check_holospim_host([config])
