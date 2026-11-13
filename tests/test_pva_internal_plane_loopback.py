"""Service-owned internal camera plane captures over loopback PVA."""

import asyncio
import json
import logging
import uuid

import pytest
import zarr
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from redsun.path_provider import SessionPathProvider
from redsun.services import serve_controller
from test_unified_mcu import HoldStagePort

from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.device.pva_internal_plane_capture import acquire_pva_internal_planes
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.spinnaker_camera import SpinnakerCamera
from redsun_aht.device.unified_mcu import AngularStack, UnifiedMcu
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.services.spinnaker_camera import SpinnakerCameraController
from redsun_aht.services.unified_mcu import UnifiedMcuController


@pytest.mark.parametrize("per", [4, 16])
@pytest.mark.parametrize("finite", [False, True])
async def test_internal_planes_through_camera_and_mcu_services(
    tmp_path, per, finite, caplog
):
    caplog.set_level(logging.INFO, logger="p4p.client.raw")

    class InternalCamera(FakeCameraCore):
        stop_calls = 0

        def startContinuousSequenceAcquisition(self, interval=0):
            super().startContinuousSequenceAcquisition(interval)
            self.produce(per)

        def clearCircularBuffer(self):
            while not self._buffer.empty():
                self._buffer.get_nowait()

        def startSequenceAcquisition(self, count, interval, stop_on_overflow):
            assert count == per and stop_on_overflow
            self.produce(count)
            self.sequencing = False

        def isSequenceRunning(self):
            return self.sequencing

        def isBufferOverflowed(self):
            return False

        def stopSequenceAcquisition(self):
            self.stop_calls += 1
            super().stopSequenceAcquisition()

    camera_core, port = InternalCamera(), HoldStagePort()
    camera_controller = MMCameraController(
        core_of(camera_core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
        finite_capture=finite,
    )
    suffix = uuid.uuid4().hex[:10].upper()
    cp, mp = "IC" + suffix, "IM" + suffix
    servers = [
        asyncio.create_task(
            serve_controller(camera_controller, EpicsPVATransport(), cp)
        ),
        asyncio.create_task(
            serve_controller(
                UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), mp
            )
        ),
    ]
    camera = MMCamera(
        cp + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="fake"),
        name="camera",
    )
    owner = PvaUnifiedMcu(mp + ":", name="mcu")
    planes = []

    async def sink(result):
        planes.append(result)

    try:
        await asyncio.wait_for(asyncio.gather(camera.connect(), owner.connect()), 10)
        await owner.reserve()
        await asyncio.wait_for(
            acquire_pva_internal_planes(
                owner,
                camera,
                AngularStack(9, 0, 18, 2, -180, 180),
                per,
                [1, 1],
                lambda plane: str(tmp_path / f"plane-{plane}"),
                sink,
                resources_exclusive=True,
                internal_trigger_configured=True,
                illumination_ready=True,
                bind_stage=False,
            ),
            30,
        )
        assert [item.plane for item in planes] == [0, 1]
        assert all(item.evidence.stored_frames == per for item in planes)
        assert camera_core.stop_calls == (0 if finite else 2)
        assert [cmd for cmd in port.commands if cmd.startswith("SHOT ")] == [
            "SHOT 9 0 1",
            "SHOT 9 1 1",
        ]
        for i in range(2):
            directory = tmp_path / f"plane-{i}"
            assert zarr.open_array(str(directory / DATA_KEY), mode="r").shape[0] == per
            timing = json.loads((directory / f"{DATA_KEY}.timing.json").read_text())
            assert timing["written_count"] == per
            assert timing["high_resolution_clock"] == "python.time.perf_counter_ns"
            assert all(
                record["service_perf_counter_ns"] > 0 for record in timing["records"]
            )
        await owner.release_reservation()
    finally:
        for task in servers:
            task.cancel()
        await asyncio.gather(*servers, return_exceptions=True)


@pytest.mark.parametrize("end_illumination", ["blank", "center"])
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("direct", [False, True], ids=["mmcore", "spinnaker"])
async def test_timelapse_plan_uses_service_owners_across_blocks(
    tmp_path, caplog, end_illumination, managed, direct
):
    from bluesky import RunEngine
    from bluesky import plan_stubs as bps

    from redsun_aht.presenter.pva_timelapse import (
        PvaTimeLapseBindings,
        PvaTimeLapsePresenter,
    )

    caplog.set_level(logging.INFO, logger="p4p.client.raw")

    class InternalCamera(FakeCameraCore):
        def startContinuousSequenceAcquisition(self, interval=0):
            super().startContinuousSequenceAcquisition(interval)
            self.produce(4)

        def isBufferOverflowed(self):
            return False

        def popNextImageAndMD(self):
            frame, metadata = super().popNextImageAndMD()
            if direct:
                metadata = {"frame_id": str(self.popped - 1), "timestamp_ns": "42"}
            return frame, metadata

        def close(self):
            pass

    port = HoldStagePort()
    controller_type = SpinnakerCameraController if direct else MMCameraController
    controller = controller_type(
        core_of(InternalCamera(pixel_type="16bit" if direct else "8bit")),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
    )
    suffix = uuid.uuid4().hex[:10].upper()
    cp, mp = "TC" + suffix, "TM" + suffix
    servers = [
        asyncio.create_task(serve_controller(controller, EpicsPVATransport(), cp)),
        asyncio.create_task(
            serve_controller(
                UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), mp
            )
        ),
    ]
    camera_type = SpinnakerCamera if direct else MMCamera
    camera = camera_type(
        cp + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="fake"),
        name="camera",
    )
    owner = PvaUnifiedMcu(mp + ":", name="mcu")
    try:
        await asyncio.wait_for(asyncio.gather(camera.connect(), owner.connect()), 10)
        if not managed:
            await owner.reserve()

        def run_plan():
            engine = RunEngine({}, context_managers=[])
            docs = []
            engine.subscribe(lambda name, doc: docs.append((name, doc)))

            def noop(*args):
                yield from bps.null()

            bindings = PvaTimeLapseBindings(
                owner=owner,
                camera=camera,
                stack_for_block=lambda i: AngularStack(9 + i, 0, 18, 2, -180, 180),
                store_for_plane=lambda block, plane: str(
                    tmp_path / f"b{block}p{plane}"
                ),
                move_position=noop,
                configure_channel=noop,
                open_gate=noop,
                close_gate=noop,
            )
            presenter = PvaTimeLapsePresenter("timelapse", bindings=bindings)
            plan = presenter.plan_map()[
                "pva_managed_timelapse" if managed else "pva_timelapse"
            ]["plan"]
            geometry = (
                dict(first=0, step=18, planes=2, lower=-180, upper=180)
                if managed
                else {}
            )
            engine(
                plan(
                    ["xy"],
                    2,
                    ["red"],
                    [1],
                    [100.0],
                    4,
                    resources_exclusive=True,
                    internal_trigger_configured=True,
                    illumination_ready=True,
                    bind_stage=False,
                    hold_illumination=True,
                    end_illumination=end_illumination,
                    **geometry,
                )
            )
            return docs

        docs = await asyncio.wait_for(asyncio.to_thread(run_plan), 45)
        assert len([doc for name, doc in docs if name == "event"]) == 4
        for block in range(2):
            for plane in range(2):
                directory = tmp_path / f"b{block}p{plane}"
                assert (
                    zarr.open_array(str(directory / DATA_KEY), mode="r").shape[0] == 4
                )
        first_id, second_id = (1, 2) if managed else (9, 10)
        assert [cmd for cmd in port.commands if cmd.startswith("SHOT ")] == [
            f"SHOT {first_id} 0 1",
            f"SHOT {first_id} 1 1",
            f"SHOT {second_id} 0 1",
            f"SHOT {second_id} 1 1",
        ]
        assert port.commands[-2:] == (
            ["01", "15"]
            if end_illumination == "center"
            else [f"SHOTFIN {second_id} 1", "00"]
        )
        if managed:
            assert not owner.reservation_held
            assert "LASTRUN?" not in port.commands
        else:
            await owner.release_reservation()
    finally:
        for task in servers:
            task.cancel()
        await asyncio.gather(*servers, return_exceptions=True)


async def test_headless_session_injects_real_pva_clients(tmp_path, caplog):
    from bluesky import RunEngine

    from redsun_aht.device.pva_timelapse_session import LedChannel, PvaTimeLapseSession

    caplog.set_level(logging.INFO, logger="p4p.client.raw")

    class InternalCamera(FakeCameraCore):
        def startContinuousSequenceAcquisition(self, interval=0):
            super().startContinuousSequenceAcquisition(interval)
            self.produce(4)

    port = HoldStagePort()
    camera_controller = MMCameraController(
        core_of(InternalCamera()),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
    )
    servers = [
        asyncio.create_task(
            serve_controller(camera_controller, EpicsPVATransport(), "TLCAM")
        ),
        asyncio.create_task(
            serve_controller(
                UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), "TLMCU"
            )
        ),
    ]

    def run_session():
        session = PvaTimeLapseSession().build()
        try:
            assert session.presenter.plan_map() == {}

            session.presenter.configure_stationary_led(
                store_for_plane=lambda b, p: str(tmp_path / f"session-{b}-{p}")
            )
            plan = session.presenter.plan_map()["pva_managed_timelapse"]["plan"]
            engine = RunEngine({}, context_managers=[])
            docs = []
            engine.subscribe(lambda name, doc: docs.append((name, doc)))
            engine(
                plan(
                    [None],
                    2,
                    [LedChannel("red", 2.0)],
                    [1],
                    [2.0],
                    4,
                    first=0,
                    step=18,
                    planes=2,
                    lower=-180,
                    upper=180,
                    resources_exclusive=True,
                    internal_trigger_configured=True,
                    illumination_ready=True,
                    bind_stage=False,
                    channel_frame_patterns=[[1, 2, 3, 4]],
                )
            )
            return docs
        finally:
            session.shutdown()

    try:
        docs = await asyncio.wait_for(asyncio.to_thread(run_session), 45)
        events = [doc["data"] for name, doc in docs if name == "event"]
        assert len(events) == 20
        assert [
            event["timelapse_pattern"] for event in events if "timelapse_frame" in event
        ] == [1, 2, 3, 4] * 4
        start = next(doc for name, doc in docs if name == "start")
        assert start["confirmation_id"].endswith("_PVA_MANAGED_TIMELAPSE")
        assert start["mcu_run_ids"] == [1, 2]
        assert start["uid"] != start["confirmation_id"]
        assert "LASTRUN?" not in port.commands and "SHOT 1 3 4" in port.commands
        assert port.commands.count("53") == 1 and port.commands.count("64") == 1
        for block in range(2):
            for plane in range(2):
                assert (
                    zarr.open_array(
                        str(tmp_path / f"session-{block}-{plane}" / DATA_KEY), mode="r"
                    ).shape[0]
                    == 4
                )
    finally:
        for task in servers:
            task.cancel()
        await asyncio.gather(*servers, return_exceptions=True)


@pytest.mark.parametrize("per", [4, 16])
async def test_pattern_shots_use_one_camera_store_per_plane(tmp_path, caplog, per):
    from redsun_aht.device.pva_pattern_shots import acquire_pva_pattern_shots

    caplog.set_level(logging.INFO, logger="p4p.client.raw")
    core, port = FakeCameraCore(), HoldStagePort()
    controller = MMCameraController(
        core_of(core),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
        finite_capture=True,
    )
    suffix = uuid.uuid4().hex[:10].upper()
    cp, mp = "XC" + suffix, "XM" + suffix
    servers = [
        asyncio.create_task(serve_controller(controller, EpicsPVATransport(), cp)),
        asyncio.create_task(
            serve_controller(
                UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), mp
            )
        ),
    ]
    camera = MMCamera(
        cp + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="fake"),
        name="camera",
    )
    owner = PvaUnifiedMcu(mp + ":", name="mcu")
    shots, planes = [], []

    async def shot(value):
        shots.append(value)

    async def plane(value):
        planes.append(value)

    try:
        await asyncio.wait_for(asyncio.gather(camera.connect(), owner.connect()), 10)
        await owner.reserve()
        patterns = [1, 2, 3, 4] * (per // 4)
        await asyncio.wait_for(
            acquire_pva_pattern_shots(
                owner,
                camera,
                AngularStack(7, 0, 18, 2, -180, 180),
                patterns,
                lambda plane: str(tmp_path / f"pattern-{plane}"),
                shot,
                plane,
                resources_exclusive=True,
                internal_trigger_configured=True,
                illumination_ready=True,
                bind_stage=False,
            ),
            30,
        )
        assert [item.pattern for item in shots] == patterns * 2
        assert [item.camera_count for item in shots] == list(range(1, 2 * per + 1))
        assert [item.evidence.stored_frames for item in planes] == [per, per]
        for p in range(2):
            frames = zarr.open_array(
                str(tmp_path / f"pattern-{p}" / DATA_KEY), mode="r"
            )
            assert [int(frames[i][0, 0]) for i in range(per)] == list(
                range(p * per + 1, (p + 1) * per + 1)
            )
        await owner.release_reservation()
    finally:
        for task in servers:
            task.cancel()
        await asyncio.gather(*servers, return_exceptions=True)
