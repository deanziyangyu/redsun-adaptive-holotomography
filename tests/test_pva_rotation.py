"""Open-loop rotation conversion and the complete fake-serial PVA path."""

import asyncio
import uuid

import pytest
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from redsun.services import serve_controller
from test_unified_mcu import FakePort

from redsun_aht.device.pva_rotation import PvaRotationAxis, PvaRotationMotor
from redsun_aht.device.unified_mcu import UnifiedMcu
from redsun_aht.services.unified_mcu import UnifiedMcuController


class RotationPort(FakePort):
    def __init__(self):
        super().__init__()
        self.position = 0
        self.enabled = False

    async def write(self, data):
        command = data.decode().strip()
        if not command.startswith("rot"):
            return await super().write(data)
        self.commands.append(command)
        if command == "rot?":
            self.emit(
                f"ROT pos={self.position} target={self.position} "
                f"moving=0 enabled={int(self.enabled)}"
            )
        elif command == "rot enable":
            self.enabled = True
            self.emit("ROT enabled")
        elif command == "rot disable":
            self.enabled = False
            self.emit("ROT disabled")
        elif command == "rot zero":
            self.position = 0
            self.emit("ROT zeroed")
        elif command == "rot stop":
            self.emit("ROT stopping")
        elif command.startswith("rot move ") and self.enabled:
            self.position = int(command.split()[-1])
            self.emit(f"ROT target={self.position}")
        else:
            raise AssertionError(command)
        return len(data)


def test_assumed_scale_and_reversed_direction():
    axis = PvaRotationAxis("TEST:")
    assert axis.to_steps(360) == -2048
    assert axis.to_steps(1) == -6
    assert axis.to_degrees(-6) == pytest.approx(1.0546875)
    assert PvaRotationAxis("TEST:", direction=1, gear_ratio=1).to_steps(360) == 2048
    with pytest.raises(ValueError):
        axis.to_steps(float("nan"))


async def test_rotation_axis_over_pva():
    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = RotationPort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )
    motor = PvaRotationMotor(prefix + ":", name="rotation_motor")
    axis = motor.axis["rotation"]
    try:
        await asyncio.wait_for(motor.connect(), 10)
        await axis.set(1.0)
        assert port.position == -6
        assert await axis.read_position() == pytest.approx(1.0546875)
        assert (await axis.locate())["readback"] == pytest.approx(1.0546875)
        await axis.set(0.0)
        await axis.disable()
        assert port.position == 0 and not port.enabled
        assert "rot move -6" in port.commands
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_positioner_jogs_rotation_in_degrees():
    """Exercise the same presenter used by the motor widget without a GUI."""
    from redsun.aio import run_coro
    from redsun.presenter import PositionerPresenter

    prefix = "UT" + uuid.uuid4().hex[:12].upper()
    port = RotationPort()
    server = asyncio.create_task(
        serve_controller(
            UnifiedMcuController(UnifiedMcu(port)), EpicsPVATransport(), prefix
        )
    )

    def run_presenter():
        motor = PvaRotationMotor(prefix + ":", name="rotation_motor")
        run_coro(motor.connect())
        presenter = PositionerPresenter("motor_ctrl", devices={"rotation": motor})
        readings = []
        presenter.sig_readback.connect(
            lambda device, axis, value: readings.append(value)
        )
        try:
            info = presenter.axes["rotation_motor"]["rotation"]
            assert info.readback.units == "deg"
            run_coro(presenter.move("rotation_motor", "rotation", 1.0))
            assert port.position == -6
            assert any(value == pytest.approx(1.0546875) for value in readings)
            run_coro(presenter.move("rotation_motor", "rotation", -1.0546875))
            assert port.position == 0
            run_coro(motor.rotation.disable())
        finally:
            presenter.shutdown()

    try:
        await asyncio.wait_for(asyncio.to_thread(run_presenter), 20)
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
