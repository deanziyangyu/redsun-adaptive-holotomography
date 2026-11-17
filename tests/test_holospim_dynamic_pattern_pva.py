"""Reserved map upload through PVA and the HoloSPIM serial owner."""

import asyncio
import hashlib
import json
import zlib
from uuid import uuid4

import pytest
import zarr
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, InternalTriggerCameraCore, core_of
from redsun.engine import RunEngine
from redsun.path_provider import SessionPathProvider
from redsun.services import serve_controller
from test_unified_mcu import FakePort

from redsun_aht.device.esp300_stage import Esp300Stage
from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.device.pva_rotation import PvaRotationMotor
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu
from redsun_aht.device.unified_mcu import UnifiedMcu
from redsun_aht.presenter.holospim import HolospimPresenter
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.services.unified_mcu import UnifiedMcuController
from redsun_aht.storage.writer import array_path


class PatternPort(FakePort):
    def __init__(self):
        super().__init__()
        self.pixels = bytearray()
        self.committed = {}

    async def write(self, data):
        parts = data.decode("ascii").split()
        command = parts[0]
        if command == "DPCAPS?":
            response = "DPCAPS 1 91 5 20 1"
        elif command == "DPSLOT?":
            crc = self.committed.get(parts[1])
            state = f"1 {crc}" if crc is not None else "0 0"
            response = f"DPSLOT {parts[1]} {state}"
        elif command == "DPBEGIN":
            self.transaction, self.slot, self.generation, self.crc = parts[1:]
            self.pixels.clear()
            response = f"DPBEGIN {self.transaction}"
        elif command == "DPWRITE":
            _, transaction, offset, chunk = parts
            assert transaction == self.transaction and len(self.pixels) == int(offset)
            self.pixels.extend(bytes.fromhex(chunk))
            response = f"DPWRITE {transaction} {offset}"
        elif command == "DPCOMMIT":
            assert int(self.crc) == zlib.crc32(self.pixels)
            self.committed[self.slot] = self.crc
            response = f"DPCOMMIT {self.transaction} {self.slot} 1 {self.crc}"
        elif command == "DPABORT":
            response = f"DPABORT {self.transaction}"
        elif command == "DPCFG":
            self.total = int(parts[2]) * int(parts[3])
            self.steps = []
            response = f"DPCFG {parts[1]}"
        elif command == "DPSTEP":
            self.steps.append(parts[3])
            response = " ".join(parts[:3])
        elif command == "DPARM":
            response = f"DPARM {parts[1]} {self.total}"
        elif command == "DPNEXT":
            ordinal = int(parts[2])
            slot = self.steps[ordinal % len(self.steps)]
            response = f"DPREADY {parts[1]} {ordinal} {slot} 1 {ordinal + 1}"
        elif command == "DPSTATUS?":
            response = f"DPSTATUS {parts[1]} 4 {self.total} 0"
        elif command in {"DPACK", "DPSTOP"}:
            response = " ".join(parts)
        else:
            return await super().write(data)
        self.commands.append(" ".join(parts))
        self.emit(response)
        return len(data)


@pytest.mark.parametrize("fail_snap", (False, True))
async def test_pattern_upload_and_camera_acknowledgement(fail_snap):
    prefix = "DP" + uuid4().hex[:10].upper()
    port = PatternPort()
    controller = UnifiedMcuController(UnifiedMcu(port))
    server = asyncio.create_task(
        serve_controller(controller, EpicsPVATransport(), prefix)
    )
    client = PvaUnifiedMcu(prefix + ":", name="mcu")
    pixels = bytes(range(91))
    try:
        await asyncio.wait_for(client.connect(), 10)
        with pytest.raises(RuntimeError, match="reserved"):
            await client.upload_dynamic_pattern(2, pixels)
        with pytest.raises(RuntimeError, match="reservation mismatch"):
            await controller._upload_dynamic_pattern(
                controller.dynamic_pattern_request,
                json.dumps(
                    {
                        "id": uuid4().hex,
                        "token": uuid4().hex,
                        "slot": 2,
                        "pixels": pixels.hex(),
                    }
                ),
            )
        assert "DPCAPS?" not in port.commands
        await client.reserve()
        receipt = await client.upload_dynamic_pattern(2, pixels)
        assert port.pixels == pixels
        assert (receipt.slot, receipt.generation) == (2, 1)
        assert receipt.crc32 == zlib.crc32(pixels)
        assert receipt.sha256 == hashlib.sha256(pixels).hexdigest()
        seen = []

        async def snap(ordinal, pattern):
            assert pattern == receipt
            assert port.commands[-1].startswith("DPNEXT ")
            seen.append(ordinal)
            if fail_snap and ordinal == 1:
                raise RuntimeError("injected snap failure")

        acquire = client.dynamic_shots([receipt], 3, snap, ready=True)
        if fail_snap:
            with pytest.raises(RuntimeError, match="injected snap failure"):
                await acquire
            assert seen == [0, 1]
            assert sum(command.startswith("DPACK ") for command in port.commands) == 1
            assert await client.reservation_status.get_value() == "held"
        else:
            await acquire
            assert seen == [0, 1, 2]
            assert sum(command.startswith("DPACK ") for command in port.commands) == 3
            await client.release_reservation()
            assert await client.reservation_status.get_value() == "free"
        assert port.commands[-1].startswith("DPSTOP ")
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


@pytest.mark.parametrize("fail_snap", (False, True))
async def test_desktop_dynamic_recipe_cleans_up_and_stores_ordered_receipts(
    tmp_path, fail_snap
):
    suffix = uuid4().hex[:10].upper()
    mp, cp = "DP" + suffix, "DC" + suffix
    port = PatternPort()
    class RecipeCamera(InternalTriggerCameraCore):
        def snap(self):
            if fail_snap and self.snapped == 1:
                raise RuntimeError("injected DPCT snap failure")
            return super().snap()

    core = RecipeCamera()
    controllers = (
        (UnifiedMcuController(UnifiedMcu(port)), mp),
        (
            MMCameraController(
                core_of(core), LABEL, DATA_KEY, defer_initial_frame=True
            ),
            cp,
        ),
    )
    servers = [
        asyncio.create_task(serve_controller(controller, EpicsPVATransport(), prefix))
        for controller, prefix in controllers
    ]
    mcu = PvaUnifiedMcu(mp + ":", name="mcu")
    camera = MMCamera(
        cp + ":", path_provider=SessionPathProvider(base_dir=tmp_path), name="pvcam"
    )
    recipe = [list(range(91)), list(reversed(range(91)))]
    recipe_file = tmp_path / "recipe.json"
    recipe_file.write_text(json.dumps(recipe), encoding="utf-8")
    docs = []
    try:
        await asyncio.wait_for(asyncio.gather(camera.connect(), mcu.connect()), 10)
        plan = HolospimPresenter("test", enable_dynamic_dpct=True).dynamic_dpct(
            mcu,
            camera,
            Esp300Stage("UNUSED:", name="survey"),
            PvaRotationMotor(mp + ":", name="rotation"),
            str(recipe_file),
            "green",
            repeats=2,
            ready_for_internal_snaps=True,
        )
        engine = RunEngine({})
        future = engine(plan, lambda name, doc: docs.append((name, doc)))
        if fail_snap:
            with pytest.raises(Exception, match="injected DPCT snap failure"):
                await asyncio.wait_for(asyncio.wrap_future(future), 20)
            assert sum(cmd.startswith("DPACK ") for cmd in port.commands) == 1
            assert sum(cmd.startswith("DPNEXT ") for cmd in port.commands) == 2
            assert port.commands[-1].startswith("DPSTOP ")
            assert not any(name == "event" for name, _ in docs)
            assert await camera.capture.get_value(cached=False) is False
            assert await camera.acquire.get_value(cached=False) is False
            assert await mcu.reservation_status.get_value(cached=False) == "held"
            assert not mcu.reservation_held
            with pytest.raises(RuntimeError, match="uncertain"):
                await mcu.reserve()
            return
        await asyncio.wait_for(asyncio.wrap_future(future), 20)
        start = next(doc for name, doc in docs if name == "start")
        frames = [doc["data"] for name, doc in docs if name == "event"]
        assert start["association"] == "host_order_inferred"
        assert [frame["frame_index"] for frame in frames] == [0, 1, 2, 3]
        assert start["pattern_sha256"] == [
            hashlib.sha256(bytes(p)).hexdigest() for p in recipe
        ]
        assert [frame["slot"] for frame in frames] == [1, 2, 1, 2]
        assert [frame["crc32"] for frame in frames] == [
            zlib.crc32(bytes(p)) for p in recipe * 2
        ]
        pixels = zarr.open_array(
            str(array_path(start["store_uri"], DATA_KEY)), mode="r"
        )
        assert pixels[:, 0, 0].tolist() == [1, 2, 3, 4]
        assert port.commands[-1] == "00"
        assert await mcu.reservation_status.get_value() == "free"
    finally:
        for server in servers:
            server.cancel()
        await asyncio.gather(*servers, return_exceptions=True)
