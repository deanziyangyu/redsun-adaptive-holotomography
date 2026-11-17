import json
import zlib
from types import SimpleNamespace

import numpy as np
from bluesky import RunEngine
from bluesky import plan_stubs as bps
from ophyd_async.core import callback_on_mock_put

from redsun_aht.device.arh_dpct_patterns import demo_phases
from redsun_aht.presenter.arh import ArhPresenter
from redsun_aht.robotic.device.auxiliary import UnifiedEmmsh


def test_dynamic_dpct_desktop_plan_routes_three_recipes_and_captures_before_ack(
    monkeypatch, tmp_path
):
    presenter = ArhPresenter("arh")
    mcu = UnifiedEmmsh("TEST:", name="due")
    camera = SimpleNamespace(
        name="dcam", buffer=SimpleNamespace(name="dcam-buffer"),
        dpct_store_uri=lambda _: str(tmp_path / "frames.zarr"),
    )
    log = []
    frames = np.ones((4, 4), dtype=np.uint16)
    count = 0

    def snap(_camera):
        nonlocal count
        count += 1
        log.append(("snap", count))
        yield from bps.null()
        return count, frames

    async def request(raw):
        body = json.loads(raw)
        log.append((body["action"], body["value"]))
        if body["action"] == "dynamic_map":
            value = body["value"]
            from hashlib import sha256
            from ophyd_async.core import set_mock_value

            data = bytes(value["pixels"])
            set_mock_value(mcu.dynamic_pattern_result, json.dumps({
                "id": value["id"], "slot": value["slot"], "generation": 1,
                "crc32": zlib.crc32(data), "sha256": sha256(data).hexdigest(),
            }))

    monkeypatch.setattr(presenter, "_snap", snap)
    monkeypatch.setattr(presenter, "_require_idle_camera", lambda _: bps.null())
    monkeypatch.setattr(bps, "sleep", lambda _: bps.null())

    def plan():
        yield from bps.wait_for([lambda: mcu.connect(mock=True)])
        callback_on_mock_put(mcu.led_array_request, request)
        callback_on_mock_put(mcu.blank, lambda _: log.append(("blank", None)))
        yield from presenter.dynamic_dpct(camera, mcu, confirm_exclusive_camera=True)

    documents = []
    engine = RunEngine({}, context_managers=[])
    engine.subscribe(lambda name, doc: documents.append((name, doc)))
    engine(plan())
    assert log[0][0] == log[-1][0] == "blank"
    assert [value[-1] for action, value in log if action == "dynamic_finish"] == [2, 4, 4]
    assert [bytes(value["pixels"]) for action, value in log if action == "dynamic_map"] == [
        pattern for phase in demo_phases() for pattern in phase
    ]
    for index, (action, _) in enumerate(log):
        if action == "dynamic_ack":
            assert log[index - 1][0] == "snap"
    events = [doc for name, doc in documents if name == "event"]
    assert len(events) == 10
    assert events[-1]["data"]["dynamic_slot"] == 4
