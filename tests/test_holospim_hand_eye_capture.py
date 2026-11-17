"""Reviewed HoloSPIM path capture with a loopback camera and simulated joints."""

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
import zarr
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, InternalTriggerCameraCore, core_of
from redsun.engine import RunEngine
from redsun.path_provider import SessionPathProvider
from redsun.services import serve_controller

from redsun_aht.device.esp300_stage import Esp300Stage
from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.device.pva_rotation import PvaRotationMotor
from redsun_aht.presenter.holospim import HolospimPresenter
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.storage.writer import array_path


async def test_survey_stack_returns_on_success_and_holds_on_snap_failure(tmp_path):
    prefix = "SZ" + uuid4().hex[:10].upper()

    class FaultInjectingCore(InternalTriggerCameraCore):
        fail_second_plane = False

        def snap(self):
            if self.fail_second_plane and self.snapped == 3:
                raise RuntimeError("injected second-plane snap failure")
            return super().snap()

    core = FaultInjectingCore()
    core.properties["TriggerMode"] = "Edge Trigger"
    controller = MMCameraController(
        core_of(core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    server = asyncio.create_task(
        serve_controller(controller, EpicsPVATransport(), prefix)
    )
    camera = MMCamera(
        prefix + ":", path_provider=SessionPathProvider(base_dir=tmp_path), name="pvcam"
    )
    stage = Esp300Stage("SIM:", name="survey")
    presenter = HolospimPresenter("test")
    engine = RunEngine({})
    events = []

    def plan():
        return presenter.survey_stack(camera, stage, 5, 2, confirm_setup=True)

    try:
        await asyncio.wait_for(
            asyncio.gather(camera.connect(), stage.connect(mock=True)), 10
        )
        with pytest.raises(RuntimeError, match="internal trigger mode"):
            await asyncio.wrap_future(engine(plan()))
        assert core.snapped == 0
        await camera.properties["TriggerMode"].set("Internal Trigger")
        future = engine(plan(), {"event": lambda _, doc: events.append(doc["data"])})
        await asyncio.wait_for(asyncio.wrap_future(future), 20)
        assert core.snapped == 2
        assert [row["readback_z_um"] for row in events] == [0, 5]
        assert await stage.axis["z"].position.get_value() == 0

        core.fail_second_plane = True
        failed_events = []
        future = engine(
            plan(), {"event": lambda _, doc: failed_events.append(doc["data"])}
        )
        with pytest.raises(Exception, match="injected second-plane snap failure"):
            await asyncio.wait_for(asyncio.wrap_future(future), 20)
        assert [row["readback_z_um"] for row in failed_events] == [0]
        assert await stage.axis["z"].position.get_value() == 5
        assert await camera.capture.get_value(cached=False) is False
        assert await camera.acquire.get_value(cached=False) is False

        core.fail_second_plane = False
        recovery = presenter.survey_stack(camera, stage, -5, 2, confirm_setup=True)
        await asyncio.wait_for(asyncio.wrap_future(engine(recovery)), 20)
        assert core.snapped == 5
        assert await stage.axis["z"].position.get_value() == 5
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_capture_checks_review_and_preserves_frame_pose_association(tmp_path):
    prefix = "HE" + uuid4().hex[:10].upper()
    core = InternalTriggerCameraCore()
    controller = MMCameraController(
        core_of(core), LABEL, DATA_KEY, defer_initial_frame=True
    )
    server = asyncio.create_task(
        serve_controller(controller, EpicsPVATransport(), prefix)
    )
    camera = MMCamera(
        prefix + ":", path_provider=SessionPathProvider(base_dir=tmp_path), name="pvcam"
    )
    stage = Esp300Stage("SIM:", name="survey")
    roll = PvaRotationMotor("SIM:", name="rotation")
    presenter = HolospimPresenter("test", hand_eye_roll_limits_deg=(-10, 10))
    engine = RunEngine({})
    starts = []

    async def execute(plan):
        future = engine(plan, {"start": lambda _, doc: starts.append(doc)})
        await asyncio.wait_for(asyncio.wrap_future(future), 20)

    try:
        await asyncio.wait_for(
            asyncio.gather(
                camera.connect(),
                stage.connect(mock=True),
                roll.connect(mock=True),
            ),
            10,
        )
        await execute(
            presenter.hand_eye_preview(
                camera,
                stage,
                roll,
                parked_x28_mm=0.25,
                z_step_um=5,
                roll_step_deg=2,
            )
        )
        preview = Path(starts[-1]["hand_eye_path"])
        path = json.loads(preview.read_bytes())
        review = {
            "schema": "redsun.hand_eye.sampled_collision.v1",
            "path_sha256": "wrong",
            "instrument": "holospim",
            "focus_z_mm": 0.25,
            "checked_paths": ["nominal"],
            "sample_count": 90,
            "first_collision_sample": None,
            "mesh_sha256": {"fixture": "fixture-digest"},
            "profile_sha256": "profile-fixture",
            "capture_waypoint_indices": path["capture_waypoint_indices"],
        }
        review_file = tmp_path / "review.json"
        for digest in ("wrong", hashlib.sha256(preview.read_bytes()).hexdigest()):
            review_file.write_text(
                json.dumps({**review, "path_sha256": digest}), "utf-8"
            )
            plan = presenter.hand_eye_capture(
                camera,
                stage,
                roll,
                str(preview),
                str(review_file),
                True,
            )
            if digest == "wrong":
                with pytest.raises(ValueError, match="exact HoloSPIM preview"):
                    await execute(plan)
                assert core.snapped == 0
            else:
                await execute(plan)
        store = Path(starts[-1]["store_uri"])
        recorded = json.loads((store / "hand_eye_path.json").read_bytes())
        observations = recorded["observations"]
        assert [row["waypoint_index"] for row in observations] == [0, 1, 3, 5, 7]
        assert [row["camera_count"] for row in observations] == [1, 2, 3, 4, 5]
        assert all(
            row["target"] == row["before"] == row["after"] for row in observations
        )
        assert (store / "hand_eye_preview.json").read_bytes() == preview.read_bytes()
        array = zarr.open_array(str(array_path(str(store), DATA_KEY)), mode="r")
        assert array[:, 0, 0].tolist() == [1, 2, 3, 4, 5]
        assert await stage.axis["z"].position.get_value() == 0
        assert await roll.rotation.read_position() == 0
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
