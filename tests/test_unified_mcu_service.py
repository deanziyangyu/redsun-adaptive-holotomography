"""Hardware-free tests of the experimental unified MCU PVA controller."""

import asyncio
import json

import pytest
from test_unified_mcu import FakePort

from redsun_aht.device.unified_mcu import UnifiedMcu, UnifiedMcuError
from redsun_aht.services.unified_mcu import UnifiedMcuController


def request(**overrides):
    payload = dict(
        run_id=7, first=0, step=18, count=3, lower=-180, upper=180, ready=True
    )
    payload.update(overrides)
    return json.dumps(payload)


async def test_stack_and_led_share_one_owner():
    port = FakePort()
    service = UnifiedMcuController(UnifiedMcu(port))
    await service.initialise()
    await service.illumination.put("11")
    await service.request.put(request())
    result = json.loads(service.result.get())
    assert result["targets_tenths"] == [0, 18, 36]
    assert result["complete"] and result["quality"] == "controller_reported"
    assert port.commands.count("START 7") == 1
    assert not service.busy.get()
    await service.request.put(request())
    assert port.commands.count("START 7") == 2


async def test_busy_abort_retains_partial_result():
    port = FakePort(complete=False)
    service = UnifiedMcuController(UnifiedMcu(port))
    await service.initialise()
    task = asyncio.create_task(service.request.put(request()))
    await asyncio.wait_for(port.started.wait(), 1)
    port.emit("ZSTACK EDGE 0 0")
    with pytest.raises(RuntimeError, match="busy"):
        await service.illumination.put("11")
    with pytest.raises(RuntimeError, match="busy"):
        await service.trigger_output.put(2)
    with pytest.raises(RuntimeError, match="busy"):
        await service.auto_pulses.put(False)
    with pytest.raises(RuntimeError, match="busy"):
        await service.x_state_request.put("a" * 32)
    with pytest.raises(RuntimeError, match="busy"):
        await service.request.put(request(run_id=8))
    await service.abort.put("7")
    with pytest.raises(UnifiedMcuError, match="unverified"):
        await task
    result = json.loads(service.result.get())
    assert result["targets_tenths"] == [0]
    assert result["complete"] is False
    assert not service.busy.get()
    assert port.commands.count("ABORT 7") == 1
    assert "11" not in port.commands


async def test_x_state_request_is_correlated_and_read_only():
    port = FakePort()
    service = UnifiedMcuController(UnifiedMcu(port))
    await service.initialise()
    await service.x_state_request.put("a" * 32)
    result = json.loads(service.x_state_result.get())
    assert result["request_id"] == "a" * 32
    assert result["state"]["position_tenths"] == 0
    assert result["state"]["quality"] == "controller_reported"
    assert port.commands == ["identity", "00", "zasyncxstate"]
    with pytest.raises(ValueError, match="request ID"):
        await service.x_state_request.put("invalid")


async def test_readiness_and_input_validation_precede_motion():
    port = FakePort()
    service = UnifiedMcuController(UnifiedMcu(port))
    await service.initialise()
    with pytest.raises(ValueError, match="explicitly assert"):
        await service.request.put(request(ready=False))
    with pytest.raises(ValueError, match="integer"):
        await service.request.put(request(count=True))
    with pytest.raises(ValueError, match="numeric"):
        await service.abort.put("-7")
    assert port.commands == ["identity", "00"]


async def test_auto_pulses_and_individual_trigger_requests():
    port = FakePort()
    service = UnifiedMcuController(UnifiedMcu(port))
    await service.initialise()
    assert service.auto_status.get() == "unknown"
    await service.auto_pulses.put(False)
    assert service.auto_status.get() == "disabled"
    with pytest.raises(RuntimeError, match="pulses disabled"):
        await service.request.put(request())
    assert not any(command.startswith("COMMIT") for command in port.commands)
    for device in (1, 2, 3):
        await service.trigger_output.put(device)
        assert service.trigger_result.get() == f"output {device} acknowledged"
    await service.auto_pulses.put(True)
    assert service.auto_status.get() == "enabled"
    await service.request.put(request())
    assert port.commands[-1] == "START 7"
    assert [
        command
        for command in port.commands
        if command in {"c0", "c1", "d1", "d2", "d3"}
    ] == ["c0", "d1", "d2", "d3", "c1"]


async def test_reservation_blocks_other_requests_until_idle_release():
    port = FakePort(complete=False, repeat=True)
    service = UnifiedMcuController(UnifiedMcu(port))
    await service.initialise()
    token, other = "a" * 32, "b" * 32
    await service.reservation.put(f"acquire {token}")
    assert service.reservation_status.get() == "held"
    with pytest.raises(RuntimeError, match="reserved"):
        await service.reservation.put(f"acquire {other}")
    with pytest.raises(RuntimeError, match="reservation mismatch"):
        await service.request.put(request())
    with pytest.raises(RuntimeError, match="reserved"):
        await service.illumination.put("11")
    with pytest.raises(RuntimeError, match="reserved"):
        await service.trigger_output.put(2)
    with pytest.raises(RuntimeError, match="reserved"):
        await service.auto_pulses.put(False)
    task = asyncio.create_task(service.request.put(request(reservation_token=token)))
    await asyncio.wait_for(port.started.wait(), 1)
    with pytest.raises(RuntimeError, match="busy"):
        await service.reservation.put(f"release {token}")
    await service.abort.put("7")
    with pytest.raises(UnifiedMcuError, match="unverified"):
        await task
    assert service.reservation_status.get() == "held"
    with pytest.raises(RuntimeError, match="mismatch"):
        await service.reservation.put(f"release {other}")
    with pytest.raises(RuntimeError, match="uncertain"):
        await service.reservation.put(f"release {token}")
    assert service.reservation_status.get() == "held"


async def test_reservation_releases_after_success():
    service = UnifiedMcuController(UnifiedMcu(FakePort(repeat=True)))
    await service.initialise()
    token = "a" * 32
    await service.reservation.put(f"acquire {token}")
    await service.request.put(request(reservation_token=token))
    await service.reservation.put(f"release {token}")
    assert service.reservation_status.get() == "free"


async def test_reserved_illumination_requires_matching_idle_token():
    port = FakePort(repeat=True)
    service = UnifiedMcuController(UnifiedMcu(port))
    await service.initialise()
    token = "a" * 32
    await service.reservation.put(f"acquire {token}")
    with pytest.raises(RuntimeError, match="mismatch"):
        await service.reserved_illumination.put(
            json.dumps({"token": "b" * 32, "command": "11"})
        )
    with pytest.raises(ValueError, match="invalid"):
        await service.reserved_illumination.put(json.dumps({"command": "11"}))
    await service.reserved_illumination.put(
        json.dumps({"token": token, "command": "11"})
    )
    await service.reserved_illumination.put(
        json.dumps({"token": token, "command": "00"})
    )
    assert port.commands[-2:] == ["11", "00"]
    await service.reservation.put(f"release {token}")


@pytest.mark.parametrize("per", [4, 16])
async def test_reserved_software_shots_require_ordered_confirmations(per):
    from test_unified_mcu import ShotFakePort

    port = ShotFakePort()
    service = UnifiedMcuController(UnifiedMcu(port))
    await service.initialise()
    token = "a" * 32
    await service.reservation.put(f"acquire {token}")
    payload = {
        "stack": dict(run_id=9, first=0, step=18, count=2, lower=-180, upper=180),
        "shots_per_plane": per,
        "patterns": [1] * (2 * per),
        "reservation_token": token,
        "camera_exclusive": True,
        "illumination_ready": True,
        "bind_stage": True,
    }
    task = asyncio.create_task(service.shot_request.put(json.dumps(payload)))
    try:
        for index in range(2 * per):
            async with asyncio.timeout(2):
                while True:
                    text = service.shot_ready.get()
                    event = json.loads(text) if text else {}
                    if event.get("shot") == index:
                        break
                    await asyncio.sleep(0.001)
            assert event == {
                "run_id": 9,
                "shot": index,
                "plane": index // per,
                "pattern": 1,
                "target_tenths": 18 * (index // per),
            }
            assert port.commands[-1] == f"SHOT 9 {index} 1"
            with pytest.raises(RuntimeError, match="does not match"):
                await service.shot_confirm.put(
                    json.dumps({"token": "b" * 32, "run_id": 9, "shot": index})
                )
            await service.shot_confirm.put(
                json.dumps({"token": token, "run_id": 9, "shot": index})
            )
        await asyncio.wait_for(task, 3)
        assert port.commands[-1] == f"SHOTFIN 9 {2 * per - 1}"
        assert not service.busy.get()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_shot_cancellation_requests_owner_abort():
    from test_unified_mcu import ShotFakePort

    port = ShotFakePort()
    service = UnifiedMcuController(UnifiedMcu(port))
    await service.initialise()
    token = "a" * 32
    await service.reservation.put(f"acquire {token}")
    payload = {
        "stack": dict(run_id=9, first=0, step=18, count=2, lower=-180, upper=180),
        "shots_per_plane": 4,
        "patterns": [1] * 8,
        "reservation_token": token,
        "camera_exclusive": True,
        "illumination_ready": True,
        "bind_stage": True,
    }
    task = asyncio.create_task(service.shot_request.put(json.dumps(payload)))
    async with asyncio.timeout(2):
        while not service.shot_ready.get():
            await asyncio.sleep(0.001)
    await service.shot_cancel.put(json.dumps({"token": token, "run_id": 9}))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "SHOTABORT 9" in port.commands and service.owner.uncertain
    assert not service.busy.get()
