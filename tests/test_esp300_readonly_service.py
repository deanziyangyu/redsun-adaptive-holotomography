"""Hardware-free read-only ESP300 PVA service tests."""

import asyncio
import json
import threading

import pytest
from test_esp300_readonly import Port

from redsun_aht.device.esp300_readonly import Esp300ReadOnly
from redsun_aht.services.esp300_readonly import Esp300ReadOnlyController


async def test_snapshot_refresh_and_uncertain_failure():
    port = Port()
    controller = Esp300ReadOnlyController(Esp300ReadOnly(port))
    await controller.initialise()
    result = json.loads(controller.snapshot.get())
    assert result["schema"] == "redsun.esp300-readonly/v1"
    assert result["end_ns"] >= result["start_ns"]
    assert result["axes"][2]["actual_um"] == 2
    assert controller.status.get() == "ready"
    await controller.refresh.put(True)
    assert len(port.commands) == 50
    port.bad = "1TP"
    with pytest.raises(RuntimeError, match="missing"):
        await controller.refresh.put(True)
    assert controller.snapshot.get() == "" and controller.status.get() == "uncertain"
    with pytest.raises(RuntimeError, match="uncertain"):
        await controller.refresh.put(True)


async def test_cancelled_snapshot_thread_is_drained_before_close():
    entered, release = threading.Event(), threading.Event()

    class BlockedPort(Port):
        def readline(self, size=-1):
            if not entered.is_set():
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release")
            return super().readline(size)

    port = BlockedPort()
    controller = Esp300ReadOnlyController(Esp300ReadOnly(port))
    task = asyncio.create_task(controller.initialise())
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    draining = asyncio.create_task(controller.drain())
    await asyncio.sleep(0.02)
    assert not draining.done()
    release.set()
    await asyncio.wait_for(draining, 2)
    assert len(port.commands) == 25
