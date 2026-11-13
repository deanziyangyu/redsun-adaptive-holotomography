"""Device-level tests for the PVA 91-LED v1 Arduino client.

The device is never connected here: the tests prove that it refuses patterns
outside the five panel patterns before touching a signal, and that the valid
ids (including the centre) are forwarded exactly once.
"""

from __future__ import annotations

from typing import cast

import pytest

from redsun_aht.contracts import (
    CommandStatus as ContractCommandStatus,
)
from redsun_aht.contracts import (
    Illumination,
    IlluminationAction,
    IlluminationCapability,
)
from redsun_aht.device.arduino import ArduinoIllumination
from redsun_aht.device.arduino.commands import (
    CENTER_PATTERN,
    PANEL_PROFILE,
    pattern_command,
)
from redsun_aht.device.arduino.device import (
    PATTERN_MAX,
    PATTERN_MIN,
    ArduinoLedDevice,
    pva_prefix,
)


def test_pva_prefix_is_explicit_and_has_one_trailing_colon() -> None:
    assert pva_prefix("SIM:ARDUINO") == "pva://SIM:ARDUINO:"
    assert pva_prefix("SIM:ARDUINO:") == "pva://SIM:ARDUINO:"
    assert pva_prefix("pva://SIM:ARDUINO:") == "pva://SIM:ARDUINO:"


def test_device_can_be_built_without_opening_a_connection() -> None:
    device = ArduinoLedDevice("SIM:ARDUINO:", name="led")

    assert device.name == "led"
    assert PATTERN_MIN == 1
    assert PATTERN_MAX == 5
    assert PATTERN_MAX == CENTER_PATTERN


@pytest.mark.parametrize("number", [0, 6, -1, 7, 100])
async def test_set_pattern_refuses_out_of_range_patterns(number: int) -> None:
    device = ArduinoLedDevice("SIM:ARDUINO:")

    with pytest.raises(ValueError, match="between 1 and 5"):
        await device.set_pattern(number)


@pytest.mark.parametrize("number", [True, False, "2", 2.0, None])
async def test_set_pattern_refuses_non_integers(number: object) -> None:
    device = ArduinoLedDevice("SIM:ARDUINO:")

    with pytest.raises(ValueError, match="integer"):
        await device.set_pattern(number)  # type: ignore[arg-type]


async def test_set_pattern_and_set_centre_forward_the_panel_ids_once() -> None:
    device = ArduinoLedDevice("SIM:ARDUINO:")
    written: list[int] = []

    async def fake_set(value: int) -> None:
        written.append(value)

    object.__setattr__(device.pattern, "set", fake_set)

    await device.set_pattern(1)
    await device.set_pattern(CENTER_PATTERN)
    await device.set_centre()

    assert written == [1, 5, 5]


class _FakeWriteSignal:
    """Record values written through the PVA signal surface."""

    def __init__(self) -> None:
        self.writes: list[bool | int] = []

    async def set(self, value: bool | int) -> None:
        self.writes.append(value)


class _FakeReadSignal:
    """Expose one mutable readback string."""

    def __init__(self, value: str = "") -> None:
        self.value = value

    async def get_value(self) -> str:
        return self.value


class _FakePvaDevice:
    """Structural stand-in for ``ArduinoLedDevice`` in adapter tests."""

    def __init__(self, outcome: str = "sent") -> None:
        self.pattern = _FakeWriteSignal()
        self.all_off = _FakeWriteSignal()
        self.open_shutter = _FakeWriteSignal()
        self.last_command = _FakeReadSignal()
        self.last_outcome = _FakeReadSignal(outcome)
        self.last_detail = _FakeReadSignal()


def make_adapter(device: _FakePvaDevice) -> ArduinoIllumination:
    """Give the adapter the real device type while using the structural fake."""
    return ArduinoIllumination(cast(ArduinoLedDevice, device))


def test_illumination_adapter_satisfies_the_contract_protocol() -> None:
    adapter = make_adapter(_FakePvaDevice())

    assert isinstance(adapter, Illumination)


def test_illumination_adapter_accepts_the_pva_device() -> None:
    adapter = ArduinoIllumination(ArduinoLedDevice("SIM:ARDUINO:"))

    assert isinstance(adapter, Illumination)


def test_illumination_capability_matches_panel_profile() -> None:
    adapter = make_adapter(_FakePvaDevice())

    capability = adapter.capability

    assert capability == IlluminationCapability(
        pattern_ids=(1, 2, 3, 4, 5),
        center_pattern_id=5,
        panel_profile="panel91",
        panel_revision=1,
        led_count=91,
        blank_supported=True,
        open_supported=True,
        echo_expected=False,
        dialect="mm-ascii",
    )
    assert capability.pattern_ids == (1, 2, 3, 4, 5)
    assert capability.center_pattern_id == PANEL_PROFILE.centre_pattern
    assert capability.panel_profile == PANEL_PROFILE.name
    assert capability.panel_revision == PANEL_PROFILE.revision
    assert capability.led_count == PANEL_PROFILE.led_count


async def test_blank_sends_00_and_returns_a_blank_receipt() -> None:
    device = _FakePvaDevice()
    adapter = make_adapter(device)

    receipt = await adapter.blank()

    assert device.all_off.writes == [True]
    assert receipt.action == IlluminationAction.BLANK
    assert receipt.command == "00"
    assert receipt.status == ContractCommandStatus.SENT
    assert receipt.pattern_id is None
    assert receipt.sent_monotonic_ns is None


async def test_open_sends_01_and_returns_an_open_receipt() -> None:
    device = _FakePvaDevice()
    adapter = make_adapter(device)

    receipt = await adapter.open()

    assert device.open_shutter.writes == [True]
    assert receipt.action == IlluminationAction.OPEN
    assert receipt.command == "01"
    assert receipt.status == ContractCommandStatus.SENT
    assert receipt.pattern_id is None


async def test_pattern_sends_each_panel_command_exactly_once() -> None:
    device = _FakePvaDevice()
    adapter = make_adapter(device)

    receipts = []
    for pattern_id in range(1, 6):
        device.last_command.value = pattern_command(pattern_id)
        receipts.append(await adapter.pattern(pattern_id))

    assert device.pattern.writes == [1, 2, 3, 4, 5]
    assert [receipt.command for receipt in receipts] == [
        "11",
        "12",
        "13",
        "14",
        "15",
    ]
    assert [receipt.pattern_id for receipt in receipts] == [1, 2, 3, 4, 5]
    assert all(receipt.action == IlluminationAction.PATTERN for receipt in receipts)


async def test_centre_pattern_receipt_carries_pattern_id_five() -> None:
    device = _FakePvaDevice()
    device.last_command.value = "15"
    adapter = make_adapter(device)

    receipt = await adapter.pattern(5)

    assert device.pattern.writes == [5]
    assert receipt.command == "15"
    assert receipt.pattern_id == 5
    assert receipt.status == ContractCommandStatus.SENT


async def test_uncertain_readback_is_reported_and_not_retried() -> None:
    device = _FakePvaDevice(outcome="uncertain")
    device.last_command.value = "12"
    device.last_detail.value = "write timed out; delivery unknown"
    adapter = make_adapter(device)

    receipt = await adapter.pattern(2)

    assert receipt.status == ContractCommandStatus.UNCERTAIN
    assert "delivery unknown" in receipt.detail
    assert device.pattern.writes == [2]


async def test_echoed_readback_carries_the_matching_echo() -> None:
    device = _FakePvaDevice(outcome="echoed")
    device.last_command.value = "13"
    adapter = make_adapter(device)

    receipt = await adapter.pattern(3)

    assert receipt.status == ContractCommandStatus.ECHOED
    assert receipt.echo == "13"


@pytest.mark.parametrize("pattern_id", [0, 6, -1, 7, 100])
async def test_out_of_range_pattern_is_refused_without_a_write(
    pattern_id: int,
) -> None:
    device = _FakePvaDevice()
    adapter = make_adapter(device)

    with pytest.raises(ValueError, match="pattern"):
        await adapter.pattern(pattern_id)

    assert device.pattern.writes == []
