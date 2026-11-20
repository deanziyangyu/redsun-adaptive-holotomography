"""Headless admission and command checks for HoloSPIM's hardware dock."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.engine import RunEngine
from redsun.services import serve_controller
from test_aaaotf_service import FakeCore, make_profile

from redsun_aht.device.aaaotf import AttachedAAAOTF
from redsun_aht.presenter.holospim import HolospimPresenter
from redsun_aht.services.aaaotf import AAAOTFController, AAAOTFCore


def test_hardware_launch_resolves_session_laser_and_rejects_busy():
    from redsun_aht.configurations.holospim import HolospimSession

    session = SimpleNamespace(
        _laser_name="laser",
        hardware_widget=Mock(),
        acq_ctrl=SimpleNamespace(
            plans={"holospim_laser_profile": object()},
            accepting_plan=True,
            launch=Mock(),
        ),
    )
    HolospimSession._launch_hardware_plan(
        session, "holospim_laser_profile", {"profile": "488 nm", "laser": "OTHER"}
    )
    session.acq_ctrl.launch.assert_called_once_with(
        "holospim_laser_profile", {"profile": "488 nm", "laser": "laser"}
    )
    session.acq_ctrl.launch.reset_mock()
    session.acq_ctrl.accepting_plan = False
    HolospimSession._launch_hardware_plan(
        session, "holospim_laser_profile", {"profile": "All Off"}
    )
    session.acq_ctrl.launch.assert_not_called()
    session.hardware_widget.plan_failed.assert_called()
    session.acq_ctrl.accepting_plan = True
    session._laser_name = None
    HolospimSession._launch_hardware_plan(session, "holospim_laser_profile", {})
    session.acq_ctrl.launch.assert_not_called()


def test_hardware_status_and_lockout_without_constructing_qt():
    from redsun_aht.view.holospim import HolospimControls

    view = Mock(spec=HolospimControls)
    view._readback = Mock()
    view._off = Mock()
    view._apply_profile = Mock()
    view._apply_settings = Mock()
    view._available = True
    view._active_plan = None
    view._locked = False
    state = {
        "schema": "redsun.aaaotf-status/v1",
        "connected": True,
        "fault": "",
        "active_profile": "all_off",
        "slot_states": {"1": 0},
        "properties": {},
    }
    HolospimControls.on_status(view, {"laser-status": {"value": json.dumps(state)}})
    assert view._healthy
    HolospimControls._update_enabled(view)
    view._apply_profile.setEnabled.assert_called_with(True)
    view._locked = True
    HolospimControls._update_enabled(view)
    view._off.setEnabled.assert_called_with(False)
    HolospimControls.on_status(view, {})
    assert not view._healthy


@pytest.mark.asyncio
async def test_hardware_plans_use_the_leased_aotf_service():
    core = FakeCore()
    owner = AAAOTFCore(core, make_profile())
    owner.initialize()
    controller = AAAOTFController(owner, persistent=True)
    prefix = "HLA" + uuid4().hex[:10].upper()
    server = asyncio.create_task(
        serve_controller(controller, EpicsPVATransport(), prefix)
    )
    laser = AttachedAAAOTF(prefix + ":", name="laser")
    presenter = HolospimPresenter("hardware")
    engine = RunEngine({})
    try:
        await asyncio.wait_for(laser.connect(), 10)
        await asyncio.wrap_future(engine(presenter.laser_profile(laser, "488 nm")))
        assert owner.states[1] == 1
        await asyncio.wrap_future(
            engine(presenter.laser_settings(laser, 1, 92.63, 1900, 25))
        )
        assert set(owner.states.values()) == {0}
        assert float(core.getProperty("AAAOTF", "Power (% of max)")) == 25
        await asyncio.wrap_future(engine(presenter.laser_profile(laser, "561 nm")))
        assert owner.states[1] == 1
        await asyncio.wrap_future(engine(presenter.laser_profile(laser, "All Off")))
        assert set(owner.states.values()) == {0}
        with pytest.raises(ValueError, match="adapter ranges"):
            await asyncio.wrap_future(
                engine(presenter.laser_settings(laser, 9, 92.63, 1900, 25))
            )
    finally:
        await laser.shutdown()
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
