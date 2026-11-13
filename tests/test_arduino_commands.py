"""Pure tests for the MMStudio-compatible 91-LED v1 command encoding."""

from __future__ import annotations

import pytest

from redsun_aht.device.arduino.commands import (
    ARRAY_OFF,
    BLANK,
    CENTER_PATTERN,
    DPCT_PATTERN_NUMBERS,
    OPEN_SHUTTER,
    PANEL_PROFILE,
    PATTERN_COMMANDS,
    PATTERN_NUMBERS,
    SUPPORTED,
    CommandOutcome,
    CommandStatus,
    blank,
    centre,
    decode,
    drain_echo,
    encode,
    is_echo,
    open_shutter,
    pattern,
    pattern_command,
)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (BLANK, b"00\r\n"),
        (OPEN_SHUTTER, b"01\r\n"),
        ("11", b"11\r\n"),
        ("12", b"12\r\n"),
        ("13", b"13\r\n"),
        ("14", b"14\r\n"),
        ("15", b"15\r\n"),
    ],
)
def test_encode_emits_exact_ascii_crlf(command: str, expected: bytes) -> None:
    assert encode(command) == expected


def test_helpers_emit_their_validated_commands() -> None:
    assert blank() == b"00\r\n"
    assert open_shutter() == b"01\r\n"
    assert centre() == b"15\r\n"


@pytest.mark.parametrize(
    ("number", "command"), list(zip((1, 2, 3, 4, 5), PATTERN_COMMANDS, strict=True))
)
def test_pattern_maps_one_based_group_to_two_digit_command(
    number: int, command: str
) -> None:
    assert pattern_command(number) == command
    assert pattern(number) == f"{command}\r\n".encode("ascii")


def test_centre_is_pattern_five_and_emits_15() -> None:
    assert CENTER_PATTERN == 5
    assert pattern_command(CENTER_PATTERN) == "15"
    assert pattern(CENTER_PATTERN) == b"15\r\n"


def test_pattern_number_table_distinguishes_dpct_from_centre() -> None:
    assert PATTERN_NUMBERS == (1, 2, 3, 4, 5)
    assert DPCT_PATTERN_NUMBERS == (1, 2, 3, 4)
    assert CENTER_PATTERN not in DPCT_PATTERN_NUMBERS
    assert PATTERN_COMMANDS == ("11", "12", "13", "14", "15")


def test_panel_profile_describes_the_installed_panel() -> None:
    assert PANEL_PROFILE.name == "panel91"
    assert PANEL_PROFILE.revision == 1
    assert PANEL_PROFILE.led_count == 91
    assert PANEL_PROFILE.pattern_count == 5
    assert PANEL_PROFILE.centre_pattern == 5
    assert PANEL_PROFILE.dpct_patterns == (1, 2, 3, 4)
    assert PANEL_PROFILE.pattern_commands == ("11", "12", "13", "14", "15")


def test_supported_set_is_exactly_this_panel_dialect() -> None:
    assert frozenset({"00", "01", "11", "12", "13", "14", "15"}) == SUPPORTED


def test_array_off_with_trigger_is_deliberately_unsupported() -> None:
    assert ARRAY_OFF == "10"
    assert ARRAY_OFF not in SUPPORTED
    with pytest.raises(ValueError, match="unsupported"):
        encode(ARRAY_OFF)


@pytest.mark.parametrize("number", [0, 6, -1, 7, True, 1.0, "1"])
def test_pattern_rejects_anything_outside_the_five_patterns(number: object) -> None:
    with pytest.raises(ValueError, match="pattern"):
        pattern_command(number)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pattern"):
        pattern(number)  # type: ignore[arg-type]


@pytest.mark.parametrize("command", ["", "1", "10", "16", "20", "30", "51", "61", "ab"])
def test_encode_rejects_commands_outside_the_advertised_set(command: str) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        encode(command)


def test_decode_strips_crlf_and_surrounding_whitespace() -> None:
    assert decode(b"11\r\n") == "11"
    assert decode(b"  12\r\n") == "12"
    assert decode(b"\r\n") == ""


def test_decode_tolerates_unreadable_bytes_for_diagnostics() -> None:
    assert "\ufffd" in decode(b"\xff\xfe")


def test_is_echo_matches_only_the_same_command() -> None:
    assert is_echo(b"13\r\n", "13")
    assert is_echo(b" 13 ", "13")
    assert not is_echo(b"12\r\n", "13")
    assert not is_echo(b"", "13")


def test_outcome_status_properties() -> None:
    sent = CommandOutcome("11", CommandStatus.SENT)
    centre_outcome = CommandOutcome("15", CommandStatus.SENT)
    echoed = CommandOutcome("11", CommandStatus.ECHOED, echo="11")
    uncertain = CommandOutcome("11", CommandStatus.UNCERTAIN, "timeout")
    failed = CommandOutcome("10", CommandStatus.FAILED, "unsupported")

    assert sent.delivered and sent.is_pattern and not sent.uncertain
    assert centre_outcome.delivered and centre_outcome.is_pattern
    assert echoed.delivered and echoed.is_pattern
    assert uncertain.uncertain and not uncertain.delivered and uncertain.is_pattern
    assert not failed.delivered and not failed.uncertain and not failed.is_pattern


class _LineReader:
    """Return queued lines, then behave like an idle serial read."""

    def __init__(self, *lines: bytes | BaseException) -> None:
        self._lines = list(lines)
        self.reads = 0

    async def read_line(self) -> bytes:
        self.reads += 1
        if not self._lines:
            return b""
        item = self._lines.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.mark.asyncio
async def test_drain_echo_accepts_the_matching_echo() -> None:
    reader = _LineReader(b"11\r\n")

    drained = await drain_echo(reader.read_line, "11")

    assert drained.matched
    assert drained.line == "11"
    assert drained.unexpected == ()
    assert reader.reads == 1


@pytest.mark.asyncio
async def test_drain_echo_tolerates_unexpected_lines_before_its_echo() -> None:
    reader = _LineReader(b"boot\r\n", b"11\r\n")

    drained = await drain_echo(reader.read_line, "11")

    assert drained.matched
    assert drained.unexpected == ("boot",)


@pytest.mark.asyncio
async def test_drain_echo_treats_a_silent_sketch_as_no_echo() -> None:
    reader = _LineReader()

    drained = await drain_echo(reader.read_line, "12")

    assert not drained.matched
    assert drained.line is None
    assert drained.unexpected == ()


@pytest.mark.asyncio
async def test_drain_echo_tolerates_a_read_timeout() -> None:
    reader = _LineReader(TimeoutError())

    drained = await drain_echo(reader.read_line, "13")

    assert not drained.matched


@pytest.mark.asyncio
async def test_drain_echo_is_bounded_for_chatter() -> None:
    reader = _LineReader(*[b"noise\r\n"] * 20)

    drained = await drain_echo(reader.read_line, "14")

    assert not drained.matched
    assert len(drained.unexpected) == 4
    assert reader.reads == 4
