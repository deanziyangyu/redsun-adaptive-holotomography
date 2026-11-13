"""Managed shutdown while fake MCU stack is active."""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from p4p.client.asyncio import Context


@pytest.mark.parametrize("blocked", [False, True])
async def test_active_stack_shutdown_attempts_abort_before_port_close(
    tmp_path, blocked
):
    prefix = "AB" + uuid.uuid4().hex[:12].upper()
    trace = tmp_path / "trace.txt"
    env = os.environ.copy()
    env["REDSUN_SERVICE_NAME"] = "fake-active"
    env["REDSUN_SERVICE_PREFIX"] = prefix
    helper = Path(__file__).with_name("fake_unified_shutdown.py")
    child = subprocess.Popen(
        [
            sys.executable,
            str(helper),
            prefix,
            str(trace),
            "blocked" if blocked else "ok",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    context = Context("pva")
    put = None
    try:
        await asyncio.wait_for(context.get(prefix + ":PVI"), 10)
        request = json.dumps(
            {
                "run_id": 7,
                "first": 0,
                "step": 18,
                "count": 3,
                "lower": -180,
                "upper": 180,
                "ready": True,
            }
        )
        put = asyncio.create_task(context.put(prefix + ":Request", request))
        for _ in range(300):
            if trace.exists() and "WRITE START 7" in trace.read_text():
                break
            await asyncio.sleep(0.01)
        assert "WRITE START 7" in trace.read_text()
        child.stdin.close()
        assert await asyncio.wait_for(asyncio.to_thread(child.wait), 10) == 0
        lines = trace.read_text().splitlines()
        assert lines.count("WRITE ABORT 7") == 1
        assert lines.index("WRITE ABORT 7") < lines.index("CLOSE")
        assert ("DONE ABORT 7" in lines) is not blocked
    finally:
        if put is not None:
            put.cancel()
            await asyncio.gather(put, return_exceptions=True)
        context.close()
        if child.poll() is None:
            child.kill()
            await asyncio.to_thread(child.wait)
