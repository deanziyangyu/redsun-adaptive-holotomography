"""Fake-port tests for the serialized Arduino LED service.

No test in this module opens a serial port.  The fake records every write so
exact bytes, ordering, one-write-per-pattern and blank-on-close can be asserted
without hardware.
"""

from __future__ import annotations

import asyncio

import pytest
from oxiserial import SerialTimeoutException

from redsun_aht.device.arduino import service as service_module
from redsun_aht.device.arduino.commands import (
    SUPPORTED_ORDER,
    CommandStatus,
    pattern_command,
)
from redsun_aht.device.arduino.service import (
    ArduinoLedProfile,
    ArduinoLedService,
    open_serial_port,
)


class FakePort:
    """An in-memory port that records writes and can fail deliberately."""

    def __init__(
        self,
        *,
        echoes: list[bytes] | None = None,
        write_error: BaseException | None = None,
        flush_error: BaseException | None = None,
        short_write: bool = False,
        write_delay_s: float = 0.0,
    ) -> None:
        self.writes: list[bytes] = []
        self.write_attempts = 0
        self.flushes = 0
        self.closed = False
        self._open = True
        self._echoes = list(echoes or [])
        self._write_error = write_error
        self._flush_error = flush_error
        self._short_write = short_write
        self._write_delay_s = write_delay_s
        self._active_writes = 0
        self.max_active_writes = 0

    @property
    def is_open(self) -> bool:
        return self._open and not self.closed

    async def write(self, data: bytes) -> int:
        self.write_attempts += 1
        if self._write_error is not None:
            raise self._write_error
        self._active_writes += 1
        self.max_active_writes = max(self.max_active_writes, self._active_writes)
        try:
            if self._write_delay_s:
                await asyncio.sleep(self._write_delay_s)
            self.writes.append(data)
        finally:
            self._active_writes -= 1
        if self._short_write:
            return max(0, len(data) - 1)
        return len(data)

    async def flush(self) -> None:
        self.flushes += 1
        if self._flush_error is not None:
            raise self._flush_error

    async def read_until(self, expected: bytes = b"\n") -> bytes:
        del expected
        if not self._echoes:
            return b""
        return self._echoes.pop(0)

    def close(self) -> None:
        self.closed = True
        self._open = False


def make_service(port: FakePort) -> ArduinoLedService:
    """Bind the service to a fake endpoint without opening anything."""
    return ArduinoLedService(port, ArduinoLedProfile(port="loop://"))


async def test_sends_exactly_the_validated_bytes_in_order() -> None:
    port = FakePort()
    service = make_service(port)

    outcomes = [await service.send(command) for command in SUPPORTED_ORDER]

    assert [outcome.status for outcome in outcomes] == [CommandStatus.SENT] * len(
        SUPPORTED_ORDER
    )
    assert port.writes == [
        b"00\r\n",
        b"01\r\n",
        b"11\r\n",
        b"12\r\n",
        b"13\r\n",
        b"14\r\n",
        b"15\r\n",
    ]
    assert port.write_attempts == len(SUPPORTED_ORDER)


async def test_one_write_per_pattern_command() -> None:
    port = FakePort()
    service = make_service(port)

    outcomes = [await service.send(pattern_command(number)) for number in range(1, 6)]

    assert port.writes == [
        b"11\r\n",
        b"12\r\n",
        b"13\r\n",
        b"14\r\n",
        b"15\r\n",
    ]
    assert port.write_attempts == 5
    assert all(outcome.is_pattern for outcome in outcomes)
    assert outcomes[-1].command == "15"
    assert service.last_pattern == 5


async def test_pattern_echo_is_accepted_as_diagnostic_evidence() -> None:
    port = FakePort(echoes=[b"11\r\n"])
    service = make_service(port)

    outcome = await service.send("11")

    assert outcome.status is CommandStatus.ECHOED
    assert outcome.echo == "11"
    assert port.writes == [b"11\r\n"]


async def test_missing_echo_is_still_a_sent_command() -> None:
    port = FakePort()
    service = make_service(port)

    outcome = await service.send("12")

    assert outcome.status is CommandStatus.SENT
    assert outcome.echo is None
    assert outcome.delivered


async def test_unexpected_reply_is_tolerated_and_reported() -> None:
    port = FakePort(echoes=[b"boot\r\n"])
    service = make_service(port)

    outcome = await service.send("13")

    assert outcome.status is CommandStatus.SENT
    assert "unexpected" in outcome.detail
    assert port.writes == [b"13\r\n"]


async def test_short_write_is_uncertain_and_is_not_resent() -> None:
    port = FakePort(short_write=True)
    service = make_service(port)

    outcome = await service.send("11")

    assert outcome.status is CommandStatus.UNCERTAIN
    assert outcome.uncertain
    assert "short serial write" in outcome.detail
    assert port.writes == [b"11\r\n"]
    assert port.write_attempts == 1


async def test_write_timeout_is_uncertain_and_is_not_resent() -> None:
    port = FakePort(write_error=SerialTimeoutException("timed out"))
    service = make_service(port)

    outcome = await service.send("11")

    assert outcome.status is CommandStatus.UNCERTAIN
    assert "not retried" in outcome.detail
    assert port.writes == []
    assert port.write_attempts == 1
    assert service.last_command == "11"
    assert service.last_pattern is None


async def test_flush_timeout_after_accepted_bytes_is_uncertain() -> None:
    port = FakePort(flush_error=SerialTimeoutException("flush timed out"))
    service = make_service(port)

    outcome = await service.send("12")

    assert outcome.status is CommandStatus.UNCERTAIN
    assert port.writes == [b"12\r\n"]
    assert port.write_attempts == 1


async def test_unsupported_command_fails_without_writing() -> None:
    port = FakePort()
    service = make_service(port)

    outcome = await service.send("10")

    assert outcome.status is CommandStatus.FAILED
    assert port.writes == []
    assert port.write_attempts == 0


async def test_capabilities_are_exactly_the_advertised_commands() -> None:
    service = make_service(FakePort())

    assert service.capabilities() == (
        "00",
        "01",
        "11",
        "12",
        "13",
        "14",
        "15",
    )
    assert all(service.supports(command) for command in SUPPORTED_ORDER)
    assert service.supports("15")
    assert not service.supports("10")
    assert not service.supports("16")
    assert not service.supports("24")


async def test_panel_profile_is_read_only_state() -> None:
    service = make_service(FakePort())

    assert service.panel_profile.name == "panel91"
    assert service.panel_profile.revision == 1
    assert service.panel_profile.pattern_count == 5
    assert service.panel_profile.centre_pattern == 5
    assert service.panel_profile.dpct_patterns == (1, 2, 3, 4)
    assert service.led_count == 91


async def test_close_blanks_outputs_before_releasing_the_port() -> None:
    port = FakePort()
    service = make_service(port)
    await service.send("11")

    await service.close()

    assert port.writes[-1] == b"00\r\n"
    assert port.writes.count(b"00\r\n") == 1
    assert port.closed
    assert service.is_closed
    assert service.last_command == "00"
    assert service.last_outcome is not None
    assert service.last_outcome.delivered


async def test_close_is_idempotent() -> None:
    port = FakePort()
    service = make_service(port)

    await service.close()
    attempts = port.write_attempts
    await service.close()

    assert port.write_attempts == attempts
    assert port.writes == [b"00\r\n"]


async def test_close_records_a_failed_blank_but_still_releases_the_port() -> None:
    port = FakePort(write_error=SerialTimeoutException("timed out"))
    service = make_service(port)

    await service.send("11")
    await service.close()

    assert port.closed
    assert service.last_command == "00"
    assert service.last_outcome is not None
    assert service.last_outcome.status is CommandStatus.UNCERTAIN
    assert port.write_attempts == 2


async def test_send_after_close_fails_without_writing() -> None:
    port = FakePort()
    service = make_service(port)
    await service.close()
    port.writes.clear()

    outcome = await service.send("14")

    assert outcome.status is CommandStatus.FAILED
    assert port.writes == []
    assert port.write_attempts == 1


async def test_one_lock_serialises_concurrent_plans() -> None:
    port = FakePort(write_delay_s=0.02)
    service = make_service(port)

    await asyncio.gather(service.send("11"), service.send("12"))

    assert port.max_active_writes == 1
    assert port.writes == [b"11\r\n", b"12\r\n"]


async def test_reconfigure_blanks_both_endpoints_and_releases_the_old_one() -> None:
    old_port = FakePort()
    new_port = FakePort(echoes=[b"00\n", b"AHT-MM-ASCII-91/1\n"])
    service = ArduinoLedService(
        old_port,
        ArduinoLedProfile(port="loop://old"),
        opener=lambda profile: new_port,
    )

    await service.reconfigure(port="loop://new")

    assert old_port.writes == [b"00\r\n"]
    assert new_port.writes == [b"00\r\n", b"90\r\n"]
    assert old_port.closed
    assert not new_port.closed
    assert service.port_name == "loop://new"
    assert service.baudrate == 9600
    assert service.last_command == "00"


async def test_fastcs_controller_exposes_required_attributes() -> None:
    from redsun_aht.services.arduino_led import ArduinoLedController

    port = FakePort(echoes=[b"00\n", b"AHT-MM-ASCII-91/1\n"])
    controller = ArduinoLedController(make_service(port))
    controller.post_initialise()

    assert set(controller.attributes) >= {
        "port",
        "baudrate",
        "pattern",
        "all_off",
        "open_shutter",
        "last_command",
        "last_outcome",
        "supported_commands",
        "panel_profile",
        "led_count",
    }
    await controller.io.update(controller.supported_commands)
    rows = controller.supported_commands.get()
    assert list(rows["command"]) == list(SUPPORTED_ORDER)
    assert list(rows["command"])[-1] == "15"

    await controller.io.update(controller.panel_profile)
    await controller.io.update(controller.led_count)
    assert controller.panel_profile.get() == "panel91"
    assert controller.led_count.get() == 91

    await controller.pattern.put(3)
    assert port.writes == [b"13\r\n"]
    assert controller.pattern.get() == 3
    assert controller.last_command.get() == "13"
    assert controller.last_outcome.get() == "sent"

    await controller.pattern.put(5)
    assert port.writes[-1] == b"15\r\n"
    assert controller.pattern.get() == 5
    assert controller.last_command.get() == "15"
    assert controller.last_outcome.get() == "sent"

    await controller.all_off.put(True)
    assert port.writes[-1] == b"00\r\n"
    assert controller.last_command.get() == "00"

    await controller.open_shutter.put(True)
    assert port.writes[-1] == b"01\r\n"
    assert controller.last_command.get() == "01"


async def test_fastcs_controller_blanks_outputs_at_startup() -> None:
    from redsun_aht.services.arduino_led import ArduinoLedController

    port = FakePort(echoes=[b"00\n", b"AHT-MM-ASCII-91/1\n"])
    controller = ArduinoLedController(make_service(port))

    await controller.initialise()

    assert port.writes == [b"00\r\n", b"90\r\n"]
    assert controller.service.last_outcome is not None
    assert controller.service.last_outcome.command == "00"


@pytest.mark.parametrize("answer", [b"", b"OTHER-PANEL\n"])
async def test_startup_refuses_an_unidentified_firmware(answer: bytes) -> None:
    """A matching blank echo alone does not prove the panel dialect."""
    from redsun_aht.services.arduino_led import ArduinoLedController

    port = FakePort(echoes=[b"00\n", answer])
    controller = ArduinoLedController(make_service(port))
    with pytest.raises(RuntimeError, match="firmware identity mismatch"):
        await controller.initialise()
    assert port.writes == [b"00\r\n", b"90\r\n"]


@pytest.mark.parametrize(
    "port",
    [
        FakePort(write_error=SerialTimeoutException("write timed out")),
        FakePort(short_write=True),
    ],
)
async def test_fastcs_controller_refuses_uncertain_startup_blank(
    port: FakePort,
) -> None:
    from redsun_aht.services.arduino_led import ArduinoLedController

    controller = ArduinoLedController(make_service(port))
    with pytest.raises(RuntimeError, match="startup blank was not echoed"):
        await controller.initialise()

    assert controller.service.last_outcome is not None
    assert not controller.service.last_outcome.delivered


async def test_fastcs_controller_rejects_out_of_range_pattern() -> None:
    from redsun_aht.services.arduino_led import ArduinoLedController

    port = FakePort()
    controller = ArduinoLedController(make_service(port))
    controller.post_initialise()

    with pytest.raises(ValueError):
        await controller.pattern.put(6)
    assert port.writes == []


class _FakeSerial:
    """Records the settings that ``open_serial_port`` applied."""

    def __init__(self) -> None:
        self.dtr = True
        self.rts = True
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_open_serial_port_uses_the_validated_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake = _FakeSerial()

    def fake_serial_for_url(url: str, **kwargs: object) -> _FakeSerial:
        captured["url"] = url
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(service_module, "serial_for_url", fake_serial_for_url)

    open_serial_port(ArduinoLedProfile(port="loop://arduino"))

    assert captured == {
        "url": "loop://arduino",
        "baudrate": 9600,
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "timeout": 0.25,
        "write_timeout": 2.0,
        "xonxoff": False,
        "rtscts": False,
        "dsrdtr": False,
    }
    assert fake.dtr is False
    assert fake.rts is False
