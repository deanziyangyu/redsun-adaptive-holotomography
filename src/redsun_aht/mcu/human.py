"""Human-readable commands for exercising the semantic AHT MCU API."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from redsun_aht.mcu.commands import SequenceStep


class HumanMcuCommandError(ValueError):
    """Raised when one human-readable MCU command is invalid."""


@dataclass(frozen=True, slots=True)
class GetCapabilitiesCommand:
    """Read endpoint and panel capabilities."""


@dataclass(frozen=True, slots=True)
class UploadPatternCommand:
    """Upload and commit one raw unsigned 8-bit pattern."""

    pattern_id: int
    values: bytes


@dataclass(frozen=True, slots=True)
class ShowPatternCommand:
    """Display one committed pattern."""

    pattern_id: int


@dataclass(frozen=True, slots=True)
class DefineSequenceCommand:
    """Define one sequence from readable pattern/dwell/trigger steps."""

    sequence_id: int
    steps: tuple[SequenceStep, ...]


@dataclass(frozen=True, slots=True)
class ArmSequenceCommand:
    """Arm one defined sequence."""

    sequence_id: int


@dataclass(frozen=True, slots=True)
class StartSequenceCommand:
    """Start the armed sequence."""


@dataclass(frozen=True, slots=True)
class StopCommand:
    """Stop normal sequence execution."""


@dataclass(frozen=True, slots=True)
class AllOffCommand:
    """Stop execution and clear ordinary illumination output."""


type HumanMcuCommand = (
    GetCapabilitiesCommand
    | UploadPatternCommand
    | ShowPatternCommand
    | DefineSequenceCommand
    | ArmSequenceCommand
    | StartSequenceCommand
    | StopCommand
    | AllOffCommand
)

HUMAN_MCU_COMMAND_HELP = """Human-readable MCU v1 test commands:
  get-capabilities
  upload-pattern PATTERN_ID VALUE [VALUE ...]
  show-pattern PATTERN_ID
  define-sequence SEQUENCE_ID PATTERN_ID:DWELL_US[:TRIGGER] [...]
  arm-sequence SEQUENCE_ID
  start-sequence
  stop
  all-off

Integers accept decimal or 0x-prefixed notation. Pattern values may be
space- or comma-separated unsigned bytes. TRIGGER defaults to true and accepts
true/false, on/off, yes/no, or 1/0. Blank lines and # comments are ignored."""


def parse_human_mcu_command(line: str) -> HumanMcuCommand | None:
    """Parse one readable line into an existing semantic MCU operation."""
    try:
        words = shlex.split(line, comments=True, posix=True)
    except ValueError as error:
        raise HumanMcuCommandError(str(error)) from error
    if not words:
        return None
    name, *arguments = words
    if name == "get-capabilities":
        _require_argument_count(name, arguments, 0)
        return GetCapabilitiesCommand()
    if name == "upload-pattern":
        if len(arguments) < 2:
            raise HumanMcuCommandError(
                "upload-pattern requires PATTERN_ID and at least one VALUE"
            )
        pattern_id = _parse_u16(arguments[0], "pattern ID")
        values = _parse_pattern_values(arguments[1:])
        return UploadPatternCommand(pattern_id, values)
    if name == "show-pattern":
        _require_argument_count(name, arguments, 1)
        return ShowPatternCommand(_parse_u16(arguments[0], "pattern ID"))
    if name == "define-sequence":
        if len(arguments) < 2:
            raise HumanMcuCommandError(
                "define-sequence requires SEQUENCE_ID and at least one step"
            )
        sequence_id = _parse_u16(arguments[0], "sequence ID")
        steps = tuple(_parse_sequence_step(value) for value in arguments[1:])
        return DefineSequenceCommand(sequence_id, steps)
    if name == "arm-sequence":
        _require_argument_count(name, arguments, 1)
        return ArmSequenceCommand(_parse_u16(arguments[0], "sequence ID"))
    if name == "start-sequence":
        _require_argument_count(name, arguments, 0)
        return StartSequenceCommand()
    if name == "stop":
        _require_argument_count(name, arguments, 0)
        return StopCommand()
    if name == "all-off":
        _require_argument_count(name, arguments, 0)
        return AllOffCommand()
    raise HumanMcuCommandError(f"unknown MCU command: {name}")


def _parse_pattern_values(arguments: list[str]) -> bytes:
    tokens: list[str] = []
    for argument in arguments:
        parts = argument.split(",")
        if any(not part for part in parts):
            raise HumanMcuCommandError("pattern values contain an empty item")
        tokens.extend(parts)
    return bytes(_parse_uint(value, "pattern value", maximum=0xFF) for value in tokens)


def _parse_sequence_step(value: str) -> SequenceStep:
    parts = value.split(":")
    if len(parts) not in {2, 3} or any(not part for part in parts):
        raise HumanMcuCommandError("sequence steps use PATTERN_ID:DWELL_US[:TRIGGER]")
    trigger = True if len(parts) == 2 else _parse_bool(parts[2])
    return SequenceStep(
        _parse_u16(parts[0], "step pattern ID"),
        _parse_uint(parts[1], "step dwell", maximum=0xFFFFFFFF),
        trigger,
    )


def _parse_bool(value: str) -> bool:
    normalized = value.casefold()
    if normalized in {"1", "true", "on", "yes"}:
        return True
    if normalized in {"0", "false", "off", "no"}:
        return False
    raise HumanMcuCommandError(f"invalid trigger boolean: {value}")


def _parse_u16(value: str, name: str) -> int:
    return _parse_uint(value, name, maximum=0xFFFF)


def _parse_uint(value: str, name: str, *, maximum: int) -> int:
    try:
        base = 16 if value.casefold().startswith("0x") else 10
        parsed = int(value, base)
    except ValueError as error:
        raise HumanMcuCommandError(f"{name} is not an integer: {value}") from error
    if not 0 <= parsed <= maximum:
        raise HumanMcuCommandError(f"{name} must be in the range 0..{maximum}")
    return parsed


def _require_argument_count(name: str, arguments: list[str], count: int) -> None:
    if len(arguments) != count:
        raise HumanMcuCommandError(
            f"{name} requires {count} argument{'s' if count != 1 else ''}"
        )


__all__ = [
    "HUMAN_MCU_COMMAND_HELP",
    "AllOffCommand",
    "ArmSequenceCommand",
    "DefineSequenceCommand",
    "GetCapabilitiesCommand",
    "HumanMcuCommand",
    "HumanMcuCommandError",
    "ShowPatternCommand",
    "StartSequenceCommand",
    "StopCommand",
    "UploadPatternCommand",
    "parse_human_mcu_command",
]
