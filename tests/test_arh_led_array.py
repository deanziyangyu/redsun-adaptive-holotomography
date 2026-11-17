"""Headless checks for LED Array plan routing and serialized map upload."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bluesky import RunEngine
from ophyd_async.core import get_mock_put

from redsun_aht.presenter.arh import ArhPresenter
from redsun_aht.robotic.device.auxiliary import UnifiedEmmsh
from redsun_aht.robotic.services.unified_emmsh import EmmshController


def request(action, value):
    return json.dumps({"action": action, "value": value})


@pytest.mark.asyncio
async def test_map_is_chunked_and_discarded_on_rejection():
    commands = []

    async def exchange(data, *, terminator):
        assert terminator == b"\n"
        commands.append(data)
        return b"ERR invalid_led_command\n" if data.startswith(b"lmw20 ") else data

    service = EmmshController(SimpleNamespace(exchange=exchange), "fixture")
    service._led_pixels = 61
    service._led_map = True

    with pytest.raises(ValueError, match="61 integer"):
        await service.set_led_array(request("map", [255] * 60 + [True]))
    assert commands == []

    with pytest.raises(RuntimeError, match="Due rejected LED Array command lmw20"):
        await service.set_led_array(request("map", list(range(61))))
    assert commands == [b"lmb\n", b"lmw0 " + bytes(range(20)).hex().encode() + b"\n",
                        b"lmw20 " + bytes(range(20, 40)).hex().encode() + b"\n", b"lmx\n"]


@pytest.mark.asyncio
async def test_verified_firmware_enables_and_commits_map():
    serial = SimpleNamespace(exchange=AsyncMock(side_effect=[
        b"LED compact=1 pixels=61 map=8 ttl=software\n",
        b"0\n", b"0\n", b"0\n", b"Panel blanked\n",
    ]))
    service = EmmshController(serial, "fixture", led_array=True)
    await service.connect()
    assert service.led_pixels.get() == 61
    assert service.led_map_supported.get() is True

    async def echo(data, *, terminator):
        return data

    serial.exchange.reset_mock(side_effect=True)
    serial.exchange.side_effect = echo
    await service.set_led_array(request("map", [0] * 60 + [255]))
    assert [call.args[0] for call in serial.exchange.await_args_list] == [
        b"lmb\n", b"lmw0 " + b"00" * 20 + b"\n", b"lmw20 " + b"00" * 20 + b"\n",
        b"lmw40 " + b"00" * 20 + b"\n", b"lmw60 ff\n", b"lmc\n",
    ]


def test_led_array_plan_routes_one_validated_service_request():
    from bluesky import plan_stubs as bps

    def plan(mcu):
        yield from bps.wait_for([lambda: mcu.connect(mock=True)])
        yield from ArhPresenter.led_array(None, mcu, request("intensity", 127))

    mcu = UnifiedEmmsh("TEST:", name="due")
    RunEngine({})(plan(mcu))
    payload, = get_mock_put(mcu.led_array_request).await_args.args
    assert json.loads(payload) == {"action": "intensity", "value": 127}
