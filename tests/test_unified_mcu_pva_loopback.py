"""Loopback PVA test of the unified service with fake serial only."""

import asyncio
import json
import logging
import uuid

import pytest
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from p4p.client.asyncio import Context
from redsun.services import serve_controller
from test_unified_mcu import FakePort

from redsun_aht.device.unified_mcu import UnifiedMcu
from redsun_aht.services.unified_mcu import UnifiedMcuController


@pytest.fixture(autouse=True)
def quiet_p4p_debug(caplog):
    """Avoid malformed third-party debug formatting during subscriptions."""
    caplog.set_level(logging.INFO, logger="p4p.client.raw")


async def test_loopback_stack_and_illumination():
    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = FakePort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    context = Context("pva")
    try:
        await asyncio.wait_for(context.get(prefix + ":PVI"), 10)
        await asyncio.wait_for(context.put(prefix + ":Illumination", "11"), 5)
        request = json.dumps(
            dict(run_id=7, first=0, step=18, count=3, lower=-180, upper=180, ready=True)
        )
        await asyncio.wait_for(context.put(prefix + ":Request", request), 10)
        result = json.loads(await asyncio.wait_for(context.get(prefix + ":Result"), 5))
        assert result["complete"] is True
        assert result["targets_tenths"] == [0, 18, 36]
        assert port.commands.count("START 7") == 1
        assert port.commands[:3] == ["identity", "00", "11"]
    finally:
        context.close()
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_typed_client_runs_against_loopback_service():
    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = FakePort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    try:
        await asyncio.wait_for(client.connect(), 10)
        result = await asyncio.wait_for(
            client.execute(AngularStack(7, 0, 18, 3, -180, 180), ready=True), 10
        )
        assert result.complete and result.targets_tenths == (0, 18, 36)
        assert port.commands.count("START 7") == 1
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_correlated_x_state_over_pva():
    from redsun.engine import RunEngine

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.pva_x_state import PvaXState
    from redsun_aht.presenter.holospim import HolospimPresenter

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = FakePort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaXState(prefix + ":", name="state")
    owner = PvaUnifiedMcu(prefix + ":", name="owner")
    try:
        await asyncio.wait_for(asyncio.gather(client.connect(), owner.connect()), 10)
        await owner.reserve()
        state = await client.read_state()
        assert state.position_tenths == 0 and state.motor_flags == 3
        docs = []
        engine = RunEngine({})
        plan = HolospimPresenter("test").x28_readback(owner)
        future = engine(plan, {"event": lambda _, doc: docs.append(doc["data"])})
        await asyncio.wait_for(asyncio.wrap_future(future), 10)
        assert docs == [{
            "target_tenths": 0, "position_tenths": 0,
            "speed_tenths_rpm": 0, "error_hundredths_degree": 0,
            "home_flags": 3, "motor_flags": 3, "raw_hex": state.raw_hex,
            "quality": "controller_reported",
        }]
        await owner.release_reservation()
        assert port.commands == ["identity", "00", "zasyncxstate", "zasyncxstate"]
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_one_typed_client_runs_two_increasing_stacks():
    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = FakePort(repeat=True)
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    try:
        await asyncio.wait_for(client.connect(), 10)
        for run_id in (7, 8):
            result = await asyncio.wait_for(
                client.execute(AngularStack(run_id, 0, 18, 3, -180, 180), ready=True),
                10,
            )
            assert result.complete and result.run_id == run_id
        assert port.commands.count("START 7") == 1
        assert port.commands.count("START 8") == 1
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_typed_client_selects_mask_on_capable_firmware():
    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    class MaskPort(FakePort):
        async def write(self, data):
            if data == b"identity\n":
                self.commands.append("identity")
                self.emit("ARH-UNIFIED-STACK/0.18")
                return len(data)
            return await super().write(data)

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = MaskPort(repeat=True)
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    try:
        await asyncio.wait_for(client.connect(), 10)
        result = await asyncio.wait_for(
            client.execute(
                AngularStack(7, 0, 18, 3, -180, 180, trigger_mask=6), ready=True
            ),
            10,
        )
        assert result.complete
        assert port.commands.count("TRIGMASK 7 6") == 1
        assert port.commands.count("START 7") == 1
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_cooperative_reservation_blocks_another_pva_client():
    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = FakePort(repeat=True)
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    owner = PvaUnifiedMcu(prefix + ":", name="owner")
    other = PvaUnifiedMcu(prefix + ":", name="other")
    try:
        await asyncio.wait_for(asyncio.gather(owner.connect(), other.connect()), 10)
        await owner.reserve()
        assert await owner.reservation_status.get_value(cached=False) == "held"
        with pytest.raises(Exception, match="reservation mismatch"):
            await other.execute(AngularStack(8, 0, 18, 3, -180, 180), ready=True)
        with pytest.raises(Exception, match="reserved"):
            await other.trigger_output.set(2)
        with pytest.raises(Exception, match="reserved"):
            await other.illumination.set("11")
        await owner.set_reserved_illumination("11")
        await owner.set_reserved_illumination("00")
        assert port.commands[-2:] == ["11", "00"]
        result = await owner.execute(AngularStack(7, 0, 18, 3, -180, 180), ready=True)
        assert result.complete and port.commands.count("START 7") == 1
        assert "START 8" not in port.commands and "d2" not in port.commands
        await owner.release_reservation()
        assert await owner.reservation_status.get_value(cached=False) == "free"
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_trigger_controls_share_the_loopback_owner():
    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = FakePort(repeat=True)
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    try:
        await asyncio.wait_for(client.connect(), 10)
        await client.auto_pulses.set(False)
        assert await client.auto_status.get_value(cached=False) == "disabled"
        for output in (1, 2, 3):
            await client.trigger_output.set(output)
        await client.auto_pulses.set(True)
        assert await client.auto_status.get_value(cached=False) == "enabled"
        assert [
            item for item in port.commands if item in {"c0", "c1", "d1", "d2", "d3"}
        ] == ["c0", "d1", "d2", "d3", "c1"]
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_headless_workflow_with_loopback_and_synthetic_camera():
    from test_triggered_stack import Camera

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.triggered_stack import acquire_stack
    from redsun_aht.device.unified_mcu import AngularStack

    camera = Camera()

    class TriggeringPort(FakePort):
        async def write(self, data):
            count = await super().write(data)
            if data == b"START 7\n":
                for plane in range(3):
                    camera.edge(plane)
                camera.finished.set()
            return count

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = TriggeringPort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    try:
        await asyncio.wait_for(client.connect(), 10)
        result = await acquire_stack(
            client,
            camera,
            AngularStack(7, 0, 18, 3, -180, 180),
            capture_exclusive=True,
            illumination_ready=True,
            timeout_s=10,
        )
        assert result.status == "complete_order_inferred"
        assert [p.camera_sequence for p in result.associations] == [101, 102, 103]
        assert port.commands.count("START 7") == 1 and camera.closed
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_loopback_abort_retains_partial_events():
    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = FakePort(complete=False)
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    context = Context("pva")
    task = None
    try:
        await asyncio.wait_for(context.get(prefix + ":PVI"), 10)
        request = json.dumps(
            dict(run_id=7, first=0, step=18, count=3, lower=-180, upper=180, ready=True)
        )
        task = asyncio.create_task(context.put(prefix + ":Request", request))
        await asyncio.wait_for(port.started.wait(), 5)
        port.emit("ZSTACK EDGE 0 0")
        await asyncio.wait_for(context.put(prefix + ":Abort", "7"), 5)
        with pytest.raises(Exception, match="unverified"):
            await asyncio.wait_for(task, 5)
        result = json.loads(await asyncio.wait_for(context.get(prefix + ":Result"), 5))
        assert result["targets_tenths"] == [0]
        assert result["complete"] is False
        assert port.commands.count("START 7") == 1
        assert port.commands.count("ABORT 7") == 1
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        context.close()
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


@pytest.mark.parametrize("per", [4, 16])
async def test_software_shots_through_typed_pva_client(per):
    from test_unified_mcu import ShotFakePort

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = ShotFakePort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    received = []

    async def snap(index, plane, pattern, target):
        received.append((index, plane, pattern, target))

    try:
        await asyncio.wait_for(client.connect(), 10)
        await client.reserve()
        await asyncio.wait_for(
            client.software_shots(
                AngularStack(9, 0, 18, 2, -180, 180),
                per,
                [1] * (per * 2),
                snap,
                camera_exclusive=True,
                illumination_ready=True,
            ),
            20,
        )
        assert received == [(i, i // per, 1, 18 * (i // per)) for i in range(per * 2)]
        assert port.commands[-1] == f"SHOTFIN 9 {per * 2 - 1}"
        await client.release_reservation()
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_software_shot_pva_snap_failure_aborts():
    from test_unified_mcu import ShotFakePort

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = ShotFakePort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")

    async def snap(index, plane, pattern, target):
        if index == 2:
            raise RuntimeError("camera failed")

    try:
        await asyncio.wait_for(client.connect(), 10)
        await client.reserve()
        with pytest.raises(RuntimeError, match="camera failed"):
            await asyncio.wait_for(
                client.software_shots(
                    AngularStack(9, 0, 18, 2, -180, 180),
                    4,
                    [1] * 8,
                    snap,
                    camera_exclusive=True,
                    illumination_ready=True,
                ),
                20,
            )
        assert "SHOTABORT 9" in port.commands
        assert "SHOT 9 3 1" not in port.commands
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


@pytest.mark.parametrize(
    ("auto_shutter", "per", "bound"),
    [
        (None, 4, True),
        (False, 4, True),
        (False, 16, True),
        (False, 4, False),
        (False, 16, False),
    ],
)
async def test_single_mda_run_uses_reserved_pva_software_shots(
    caplog, auto_shutter, per, bound
):
    import logging

    import numpy as np
    from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore
    from test_unified_mcu import OptionalStagePort

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack
    from redsun_aht.device.unimmcore_software_shots import run_software_shot_mda

    caplog.set_level(logging.INFO, logger="p4p.client.raw")

    class Camera(CameraDevice):
        def get_exposure(self):
            return 1.0

        def set_exposure(self, value):
            pass

        def shape(self):
            return (4, 4)

        def dtype(self):
            return np.uint16

        def start_sequence(self, n, get_buffer):
            for _ in range(n or 0):
                frame = get_buffer(self.shape(), self.dtype())
                frame[:] = 42
                yield {}

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = OptionalStagePort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    core = UniMMCore()
    core.loadPyDevice("camera", Camera())
    core.initializeDevice("camera")
    core.setCameraDevice("camera")
    core.setAutoShutter(False)
    frames = []
    external_planes = []
    try:
        await asyncio.wait_for(client.connect(), 10)
        await client.reserve()
        await asyncio.wait_for(
            run_software_shot_mda(
                client,
                core,
                AngularStack(9, 0, 18, 2, -180, 180),
                per,
                [1] * (2 * per),
                lambda shot, plane, pattern, target, image: frames.append(
                    (shot, plane, target, int(image[0, 0]))
                ),
                camera_exclusive=True,
                illumination_ready=True,
                auto_shutter=auto_shutter,
                bind_stage=bound,
                before_plane=None
                if bound
                else lambda plane, target: external_planes.append((plane, target)),
            ),
            20,
        )
        assert frames == [(i, i // per, 18 * (i // per), 42) for i in range(2 * per)]
        count = 2 if auto_shutter is False else 2 * per
        assert sum(cmd.startswith("SHOT ") for cmd in port.commands) == count
        assert port.commands[-1] == f"SHOTFIN 9 {count - 1}"
        assert external_planes == ([] if bound else [(0, 0), (1, 18)])
        await client.release_reservation()
    finally:
        core.unloadAllDevices()
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_unbound_shots_through_typed_pva_client(caplog):
    import logging

    from test_unified_mcu import OptionalStagePort

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    caplog.set_level(logging.INFO, logger="p4p.client.raw")
    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = OptionalStagePort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    frames = []

    async def snap(index, plane, pattern, target):
        frames.append((index, plane, target))

    try:
        await asyncio.wait_for(client.connect(), 10)
        await client.reserve()
        await asyncio.wait_for(
            client.software_shots(
                AngularStack(9, 0, 18, 2, -180, 180),
                4,
                [1] * 8,
                snap,
                camera_exclusive=True,
                illumination_ready=True,
                bind_stage=False,
            ),
            20,
        )
        assert frames == [(i, i // 4, 18 * (i // 4)) for i in range(8)]
        assert "SHOTBIND 0" in port.commands
        assert (
            "zasyncxconfig" not in port.commands and "zasyncxstate" not in port.commands
        )
        await client.release_reservation()
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_pva_shot_hold_reaches_firmware_owner(caplog):
    import logging

    from test_unified_mcu import HoldStagePort

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    caplog.set_level(logging.INFO, logger="p4p.client.raw")
    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = HoldStagePort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")

    async def snap(*args):
        pass

    try:
        await asyncio.wait_for(client.connect(), 10)
        await client.reserve()
        for run_id, hold in ((9, True), (10, False)):
            await asyncio.wait_for(
                client.software_shots(
                    AngularStack(run_id, 0, 18, 1, -180, 180),
                    1,
                    [1],
                    snap,
                    camera_exclusive=True,
                    illumination_ready=True,
                    bind_stage=False,
                    hold_illumination=hold,
                ),
                20,
            )
        assert [cmd for cmd in port.commands if cmd.startswith("SHOTHOLD")] == [
            "SHOTHOLD 1",
            "SHOTHOLD 0",
        ]
        await client.release_reservation()
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_reusable_id_policy_survives_new_pva_reservation():
    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    class Port(FakePort):
        async def write(self, data):
            if data == b"identity\n":
                self.commands.append("identity")
                self.emit("ARH-UNIFIED-STACK/0.18")
                return len(data)
            return await super().write(data)

    port = Port(repeat=True)
    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    try:
        await asyncio.wait_for(client.connect(), 10)
        for _ in range(2):
            await client.reserve()
            assert await client.allocate_run_ids(1) == (1,)
            await client.execute(
                AngularStack(1, 0, 18, 3, -180, 180),
                ready=True,
                require_reservation=True,
            )
            await client.release_reservation()
        assert port.commands.count("START 1") == 2
        assert "LASTRUN?" not in port.commands
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_autonomous_stack_through_reserved_pva_service():
    from test_unified_mcu import AutoShotFakePort

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
    from redsun_aht.device.unified_mcu import AngularStack

    prefix = "AUTO" + uuid.uuid4().hex[:10].upper()
    port = AutoShotFakePort()
    controller = UnifiedMcuController(UnifiedMcu(port))
    server = asyncio.create_task(
        serve_controller(controller, EpicsPVATransport(), prefix)
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    try:
        await asyncio.wait_for(client.connect(), 10)
        await client.reserve()
        checked = []

        async def ready():
            checked.append(True)

        result = await client.automatic_shots(
            AngularStack(9, 0, 18, 2, -180, 180),
            4,
            1,
            50,
            camera_armed=True,
            illumination_ready=True,
            before_start=ready,
        )
        assert checked == [True] and result.attempted_pulses == 8
        assert [c for c in port.commands if c.startswith("SHOTAUTO")] == [
            "SHOTAUTO 9 1 1 50"
        ]
        assert not any(c.startswith("SHOT ") for c in port.commands)
        await client.release_reservation()
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


@pytest.mark.parametrize("cancel", [None, "stop", "task"])
async def test_x28_point_move_uses_service_bounds_and_shared_owner(cancel):
    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu

    class JogPort(FakePort):
        async def write(self, data):
            replies = {
                b"IP?\n": "IP protocol=1 pixels=91 ids=6-13 loaded=0",
                b"DPCAPS?\n": "DPCAPS 1 91 5 20 1",
                b"zabs 18\n": "Z result=0",
            }
            if data in replies:
                self.commands.append(data.decode().strip())
                if data == b"zabs 18\n":
                    self.started.set()
                    if cancel:
                        return len(data)
                    self.x_position = 18
                self.emit(replies[data])
                return len(data)
            return await super().write(data)

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = JogPort()
    controller = UnifiedMcuController(
        UnifiedMcu(port), require_holospim_panel=True, x28_jog_bounds_tenths=(-36, 36),
    )
    server = asyncio.create_task(serve_controller(controller, EpicsPVATransport(), prefix))
    client = PvaUnifiedMcu(
        prefix + ":", name="owner", x28_um_per_turn=1000,
        x28_jog_bounds_tenths=(-36, 36),
    )
    context = Context("pva")
    request = {"request_id": uuid.uuid4().hex, "target": 37}
    try:
        await asyncio.wait_for(client.connect(), 10)
        with pytest.raises(Exception, match="target outside configured bounds"):
            await asyncio.wait_for(context.put(prefix + ":XMoveRequest", json.dumps(request)), 5)
        await client.reserve()
        request["target"] = 18
        with pytest.raises(Exception, match="busy or reserved"):
            await asyncio.wait_for(context.put(prefix + ":XMoveRequest", json.dumps(request)), 5)
        await client.release_reservation()
        await asyncio.wait_for(context.put(prefix + ":XMoveCancel", request["request_id"]), 5)
        with pytest.raises(Exception, match="cancelled before admission"):
            await asyncio.wait_for(context.put(prefix + ":XMoveRequest", json.dumps(request)), 5)
        if cancel:
            move = asyncio.create_task(client.move_x_absolute(18))
            await asyncio.wait_for(port.started.wait(), 3)
            if cancel == "task":
                move.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await move
            else:
                await client.stop_x_move()
                with pytest.raises(Exception, match="stop unverified"):
                    await move
            assert controller.owner.uncertain
            last_command = "zstop"
        else:
            await client.x28.set(5)
            last_command = "zasyncxstate"
        assert port.commands.count("zabs 18") == 1
        move_index = port.commands.index("zabs 18")
        assert port.commands[move_index - 2:move_index] == [
            "zasyncxconfig", "zasyncxstate",
        ]
        readbacks = port.commands[move_index + 1:]
        assert set(readbacks) == {last_command}
        assert len(readbacks) == 1 if cancel else len(readbacks) > 1
        if not cancel:
            assert await client.x28.locate() == {"setpoint": 5, "readback": 5}
    finally:
        context.close()
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_x28_mock_axis_is_discoverable_without_a_serial_connection():
    from redsun.presenter._positioner import boxes

    from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu

    owner = PvaUnifiedMcu(
        "UNUSED:", name="mcu", x28_um_per_turn=-1000,
        x28_jog_bounds_tenths=(-36, 72),
    )
    await owner.connect(mock=True)
    assert boxes({"mcu": owner})["mcu"][1] == {"x28": owner.x28}
    await owner.x28.set(-5)
    assert owner.x28._mock_position == 18
    assert await owner.x28.locate() == {"setpoint": -5, "readback": -5}
    with pytest.raises(ValueError, match="outside"):
        await owner.x28.set(10.001)
