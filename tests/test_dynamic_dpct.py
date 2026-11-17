import asyncio
import json
import os
import zlib
from collections import Counter
from uuid import uuid4

import pytest

from redsun_aht.device.unified_mcu import UnifiedMcu, UnifiedMcuError
from redsun_aht.robotic.services.unified_emmsh import EmmshController


class FirmwarePort:
    """Async serial substitute backed by the native firmware parser."""

    def __init__(self, process):
        self.process = process
        self.replies = asyncio.Queue()
        self.commands = Counter()
        self.drop = Counter()
        self.delay = set()
        self.delayed = {}

    async def exchange(self, command):
        self.process.stdin.write(command.encode("ascii") + b"\n")
        await self.process.stdin.drain()
        line = await asyncio.wait_for(self.process.stdout.readline(), 2)
        assert line
        return line

    async def state(self):
        return list(map(int, (await self.exchange("TEST STATE")).split()))

    async def write(self, payload):
        command = payload.decode("ascii").strip()
        verb = command.split()[0]
        self.commands[verb] += 1
        response = await self.exchange(command)
        if self.drop[verb]:
            self.drop[verb] -= 1
        elif verb in self.delay:
            self.delay.remove(verb)
            self.delayed[verb] = response
        else:
            if verb in self.delayed:
                self.replies.put_nowait(self.delayed.pop(verb))
            self.replies.put_nowait(response)
        return len(payload)

    async def send_unframed(self, payload):
        await self.write(payload)

    async def read_line(self):
        return await self.read_until(b"\n")

    async def flush(self):
        pass

    async def read_until(self, delimiter):
        assert delimiter == b"\n"
        return await self.replies.get()


@pytest.fixture
async def firmware(request):
    variable = getattr(request, "param", "ARH_DPCT_TEST_FIRMWARE")
    executable = os.getenv(variable)
    if executable is None:
        pytest.skip(f"set {variable} to the native dynamic_dpct harness")
    process = await asyncio.create_subprocess_exec(
        executable, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
    )
    try:
        yield FirmwarePort(process)
    finally:
        process.stdin.close()
        await asyncio.wait_for(process.wait(), 2)


@pytest.mark.asyncio
async def test_redsun_upload_and_frame_retries(firmware):
    owner = UnifiedMcu(firmware, line_timeout_s=0.1)
    await owner.identify(due=True)
    firmware.delay.add("DPWRITE")
    firmware.drop["DPCOMMIT"] = 2
    maps = [bytes((value * slot + slot) % 256 for value in range(61)) for slot in range(1, 6)]
    receipts = [
        await owner.upload_dynamic_pattern(slot, data) for slot, data in enumerate(maps, 1)
    ]
    assert await firmware.state() == [0, 0, *([0] * 61)]
    assert firmware.commands["DPWRITE"] == 21
    assert firmware.commands["DPCOMMIT"] == 6
    assert [item.crc32 for item in receipts] == [zlib.crc32(data) for data in maps]
    assert all(item.generation == 1 for item in receipts)

    captured = []

    async def snap(ordinal, pattern):
        state = await firmware.state()
        assert state == [ordinal + 1, 1, *maps[pattern.slot - 1]]
        captured.append((ordinal, pattern.slot))
        if ordinal == 9:
            firmware.drop["DPACK"] = 1

    firmware.delay.add("DPNEXT")
    firmware.delay.add("DPACK")
    await owner.dynamic_shots(
        receipts, 2, snap, camera_exclusive=True, illumination_ready=True
    )
    assert captured == [(i, i % 5 + 1) for i in range(10)]
    assert firmware.commands["DPNEXT"] == 11
    assert firmware.commands["DPACK"] == 12
    assert await firmware.state() == [10, 2, *([0] * 61)]
    replacement = await owner.upload_dynamic_pattern(1, bytes([250] * 61))
    assert replacement.generation == 2
    with pytest.raises(UnifiedMcuError):
        await owner.dynamic_shots(
            receipts, 1, snap, camera_exclusive=True, illumination_ready=True
        )
    assert firmware.commands["DPARM"] == 1


@pytest.mark.asyncio
async def test_firmware_rejects_bad_uploads_and_invalid_counter_steps(firmware):
    async def command(line, response):
        assert (await firmware.exchange(line)).decode().strip() == response

    data = bytes(range(61))
    crc = zlib.crc32(data)

    async def upload(tx, generation, checksum, size=61):
        await command(f"DPBEGIN {tx} 1 {generation} {checksum}", f"DPBEGIN {tx}")
        for offset in range(0, size, 20):
            chunk = data[offset : min(offset + 20, size)].hex()
            await command(f"DPWRITE {tx} {offset} {chunk}", f"DPWRITE {tx} {offset}")

    await upload(1, 0, crc)
    await command("DPCOMMIT 1", f"DPCOMMIT 1 1 1 {crc}")
    await command("DPCOMMIT 1", f"DPCOMMIT 1 1 1 {crc}")
    await upload(2, 1, crc, 60)
    await command("DPCOMMIT 2", "DPERR 2 coverage_or_crc")
    await upload(3, 1, crc ^ 1)
    await command("DPCOMMIT 3", "DPERR 3 coverage_or_crc")
    await command(f"DPBEGIN 4 1 1 {crc}", "DPBEGIN 4")
    await command("DPWRITE 4 0 00", "DPWRITE 4 0")
    await command(f"DPBEGIN 4 1 1 {crc}", "DPBEGIN 4")
    await command("DPWRITE 4 0 01", "DPERR 4 chunk")
    await command("DPCOMMIT 4", "DPERR 4 transaction")
    await command(f"DPBEGIN 5 1 1 {crc}", "DPBEGIN 5")
    await command("TEST EXPIRE", "EXPIRED")
    await command("DPCOMMIT 5", "DPERR 5 transaction")
    await command("DPSLOT? 1", f"DPSLOT 1 1 {crc}")
    assert await firmware.state() == [0, 0, *([0] * 61)]

    await command("DPCFG 6 1 2", "DPCFG 6")
    await command("DPARM 6", "DPERR 6 generation")
    await command("DPSTEP 6 0 1 1", "DPSTEP 6 0")
    await command("DPARM 6", "DPARM 6 2")
    await command(f"DPBEGIN 7 1 1 {crc}", "DPERR 7 busy")
    await command("DPACK 6 0 1", "DPERR 6 order_or_state")
    await command("DPNEXT 6 1", "DPERR 6 order_or_state")
    await command("DPNEXT 6 0", "DPREADY 6 0 1 1 1")
    await command("DPACK 6 0 2", "DPERR 6 order_or_state")
    await command("DPSTATUS? 6", "DPSTATUS 6 3 0 1")
    await command("DPACK 6 0 1", "DPACK 6 0 1")
    await command("DPNEXT 6 1", "DPREADY 6 1 1 1 2")
    await command("DPACK 6 0 1", "DPACK 6 0 1")
    await command("DPSTATUS? 6", "DPSTATUS 6 3 1 2")
    await command("TEST EXPIRE", "EXPIRED")
    await command("DPACK 6 1 2", "DPERR 6 order_or_state")
    assert await firmware.state() == [2, 2, *([0] * 61)]
    await command("DPSTATUS? 6", "DPSTATUS 6 6 1 0")
    await command("DPCFG 8 1 1", "DPCFG 8")
    await command("DPSTEP 8 0 1 1", "DPSTEP 8 0")
    await command("DPARM 8", "DPARM 8 1")
    await command("DPNEXT 8 0", "DPREADY 8 0 1 1 1")
    await command("TEST ABORT", "ABORTED")
    await command("DPSTATUS? 8", "DPSTATUS 8 5 0 0")
    assert await firmware.state() == [3, 4, *([0] * 61)]
    await command("TEST RESET", "RESET")
    await command("DPSLOT? 1", "DPSLOT 1 0 0")
    await command("DPSTATUS? 8", "DPERR 8 run")


@pytest.mark.asyncio
async def test_arh_service_upload_uses_live_slot_protocol(firmware):
    service = EmmshController(firmware, "fixture", led_array=True)
    service._led_pixels = 61
    request_id = uuid4().hex
    pixels = bytes(range(61))
    await service.set_led_array(json.dumps({
        "action": "dynamic_map",
        "value": {"id": request_id, "slot": 5, "pixels": list(pixels)},
    }))
    result = json.loads(service.dynamic_pattern_result.get())
    assert result["id"] == request_id
    assert result["slot"] == 5
    assert result["generation"] == 1
    assert result["crc32"] == zlib.crc32(pixels)
    assert (await firmware.exchange("DPSLOT? 5")).decode().strip() == (
        f"DPSLOT 5 1 {result['crc32']}"
    )
    assert (await firmware.exchange("DPCT?")).decode().strip() == "UNKNOWN"
    assert await firmware.state() == [0, 0, *([0] * 61)]


@pytest.mark.asyncio
async def test_failed_snap_never_acknowledges_a_frame(firmware):
    owner = UnifiedMcu(firmware, line_timeout_s=0.1)
    await owner.identify(due=True)
    pattern = await owner.upload_dynamic_pattern(1, bytes(range(61)))

    async def snap(ordinal, pattern):
        raise RuntimeError("camera did not return a frame")

    with pytest.raises(RuntimeError, match="camera did not return a frame"):
        await owner.dynamic_shots(
            [pattern], 1, snap, camera_exclusive=True, illumination_ready=True
        )
    assert firmware.commands["DPACK"] == 0
    assert firmware.commands["DPSTOP"] == 1
    assert owner.uncertain
    assert await firmware.state() == [1, 2, *([0] * 61)]


@pytest.mark.asyncio
async def test_desktop_service_recipe_uses_firmware_frame_counter(firmware):
    service = EmmshController(firmware, "fixture", led_array=True)
    service._led_pixels = 61
    pixels = list(range(61))

    async def request(action, value):
        await service.set_led_array(json.dumps({"action": action, "value": value}))

    await request("dynamic_map", {"id": uuid4().hex, "slot": 1, "pixels": pixels})
    receipt = json.loads(service.dynamic_pattern_result.get())
    await request("dynamic_arm", [7, [1, receipt["generation"], receipt["crc32"]]])
    await request("dynamic_next", [7, 0, 1, 1])
    assert await firmware.state() == [1, 1, *pixels]
    assert (await firmware.exchange("DPSTATUS? 7")).strip() == b"DPSTATUS 7 3 0 1"
    await request("dynamic_ack", [7, 0])
    await request("dynamic_finish", [7, 1])
    assert await firmware.state() == [1, 2, *([0] * 61)]


@pytest.mark.asyncio
@pytest.mark.parametrize("firmware", ["ARH_DPCT_TEST_ESP32_FIRMWARE"], indirect=True)
async def test_esp32_pattern_covers_all_91_leds_and_counts_ack_retries(firmware):
    owner = UnifiedMcu(firmware, line_timeout_s=0.1)
    await owner.identify()
    pixels = bytes(range(1, 92))
    receipt = await owner.upload_dynamic_pattern(5, pixels)
    assert receipt.crc32 == zlib.crc32(pixels)
    assert firmware.commands["DPWRITE"] == 5
    assert await firmware.state() == [0, 0, *([0] * 91)]
    frames = []

    async def snap(ordinal, pattern):
        assert await firmware.state() == [1, 1, *pixels]
        frames.append((ordinal, pattern.slot))

    firmware.drop["DPACK"] = 1
    await owner.dynamic_shots(
        [receipt], 1, snap, camera_exclusive=True, illumination_ready=True
    )
    assert frames == [(0, 5)]
    assert firmware.commands["DPACK"] == 2
    assert await firmware.state() == [1, 2, *([0] * 91)]
