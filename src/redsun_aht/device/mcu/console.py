"""Readable test console backed by the typed MCU host client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from redsun_aht.mcu.commands import pattern_checksum
from redsun_aht.mcu.human import (
    AllOffCommand,
    ArmSequenceCommand,
    DefineSequenceCommand,
    GetCapabilitiesCommand,
    HumanMcuCommand,
    ShowPatternCommand,
    StartSequenceCommand,
    StopCommand,
    UploadPatternCommand,
    parse_human_mcu_command,
)

if TYPE_CHECKING:
    from redsun_aht.device.mcu.adapter import McuHostClient


@dataclass(slots=True)
class McuCommandConsole:
    """Execute readable commands through one typed host client."""

    host: McuHostClient

    def execute_line(self, line: str) -> str | None:
        """Parse and execute one line, returning a stable readable result."""
        command = parse_human_mcu_command(line)
        if command is None:
            return None
        return self.execute(command)

    def execute(self, command: HumanMcuCommand) -> str:
        """Execute one parsed command without bypassing host validation."""
        if isinstance(command, GetCapabilitiesCommand):
            capabilities = self.host.get_capabilities()
            return (
                "OK get-capabilities "
                f"firmware={capabilities.firmware_version} "
                f"target={capabilities.target} "
                f"panel={capabilities.panel_profile} "
                f"panel-revision={capabilities.panel_revision} "
                f"led-count={capabilities.led_count} "
                f"max-pattern-bytes={capabilities.max_pattern_bytes} "
                f"max-sequence-steps={capabilities.max_sequence_steps} "
                f"encodings=0x{capabilities.supported_encodings:x}"
            )
        if isinstance(command, UploadPatternCommand):
            self.host.upload_pattern(command.pattern_id, command.values)
            return (
                f"OK upload-pattern id={command.pattern_id} "
                f"values={len(command.values)} "
                f"crc32=0x{pattern_checksum(command.values):08x}"
            )
        if isinstance(command, ShowPatternCommand):
            self.host.show_pattern(command.pattern_id)
            return f"OK show-pattern id={command.pattern_id}"
        if isinstance(command, DefineSequenceCommand):
            self.host.define_sequence(command.sequence_id, command.steps)
            return (
                f"OK define-sequence id={command.sequence_id} "
                f"steps={len(command.steps)}"
            )
        if isinstance(command, ArmSequenceCommand):
            self.host.arm_sequence(command.sequence_id)
            return f"OK arm-sequence id={command.sequence_id}"
        if isinstance(command, StartSequenceCommand):
            self.host.start_sequence()
            return "OK start-sequence"
        if isinstance(command, StopCommand):
            self.host.stop()
            return "OK stop"
        if isinstance(command, AllOffCommand):
            self.host.all_off()
            return "OK all-off"
        raise TypeError(f"unsupported human MCU command: {type(command).__name__}")


__all__ = ["McuCommandConsole"]
