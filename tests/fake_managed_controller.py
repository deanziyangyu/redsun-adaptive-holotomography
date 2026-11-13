"""Fake endpoints for session-managed subprocess lifecycle tests."""

import argparse
import asyncio

from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.services import serve_controller
from test_esp300_readonly import Port
from test_unified_mcu import FakePort

from redsun_aht.device.esp300_readonly import Esp300ReadOnly
from redsun_aht.device.unified_mcu import UnifiedMcu
from redsun_aht.services.esp300_readonly import Esp300ReadOnlyController
from redsun_aht.services.unified_mcu import UnifiedMcuController

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", required=True)
parser.add_argument("--port", required=True)
options = parser.parse_args()
controller = (
    Esp300ReadOnlyController(Esp300ReadOnly(Port()))
    if options.port == "fake-coarse"
    else UnifiedMcuController(UnifiedMcu(FakePort(repeat=True)))
)
asyncio.run(serve_controller(controller, EpicsPVATransport(), options.prefix))
