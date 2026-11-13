"""Session-owned fake controller services and plan lifecycle."""

import json
import os
from pathlib import Path

import pytest
from bluesky import RunEngine

from redsun_aht.device.controller_evidence_session import managed_controller_session


def test_explicit_distinct_ports_required():
    for ports in (("", "COM2"), ("COM1", "com1"), ("<port>", "COM2")):
        with pytest.raises(ValueError):
            managed_controller_session(*ports)


def test_session_launches_and_stops_two_fake_services(monkeypatch):
    tests_dir = str(Path(__file__).parent)
    monkeypatch.setenv(
        "PYTHONPATH", tests_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    session = managed_controller_session(
        "fake-coarse",
        "fake-fast",
        coarse_module="fake_managed_controller",
        fast_module="fake_managed_controller",
    )
    try:
        session.build()
        services = list(session.services.values())
        assert len(services) == 2 and all(
            service.launched and service.running for service in services
        )
        processes = [service._process for service in services]
        plan = session.presenter.plan_map()["controller_stack_report"]["plan"]
        documents = []
        engine = RunEngine({})
        engine.subscribe(lambda name, doc: documents.append((name, doc)))
        engine(plan(7, 0, 18, 3, -180, 180, ready=True))
        events = [doc["data"] for name, doc in documents if name == "event"]
        assert json.loads(events[1]["fast_stack_json"])["targets_tenths"] == [0, 18, 36]
    finally:
        session.shutdown()
    assert all(not service.running for service in services)
    assert all(process is not None and process.poll() == 0 for process in processes)
