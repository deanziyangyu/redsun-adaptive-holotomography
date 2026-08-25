from __future__ import annotations

from pathlib import Path

import pytest

from redsun_aht.__main__ import main
from redsun_aht.configurations import build_simulated_mcu_test_session
from redsun_aht.device.mcu import McuCommandFailure
from redsun_aht.mcu import (
    AllOffCommand,
    ArmSequenceCommand,
    DefineSequenceCommand,
    GetCapabilitiesCommand,
    HumanMcuCommandError,
    SequenceStep,
    ShowPatternCommand,
    StartSequenceCommand,
    StopCommand,
    UploadPatternCommand,
    parse_human_mcu_command,
)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", None),
        ("# comment", None),
        ("get-capabilities", GetCapabilitiesCommand()),
        (
            "upload-pattern 0x11 0,0x20 64 255 # readable values",
            UploadPatternCommand(17, bytes((0, 32, 64, 255))),
        ),
        ("show-pattern 0017", ShowPatternCommand(17)),
        (
            "define-sequence 2 17:250 18:500:off 19:750:YES",
            DefineSequenceCommand(
                2,
                (
                    SequenceStep(17, 250, True),
                    SequenceStep(18, 500, False),
                    SequenceStep(19, 750, True),
                ),
            ),
        ),
        ("arm-sequence 2", ArmSequenceCommand(2)),
        ("start-sequence", StartSequenceCommand()),
        ("stop", StopCommand()),
        ("all-off", AllOffCommand()),
    ],
)
def test_human_mcu_parser_maps_to_typed_commands(line: str, expected: object) -> None:
    assert parse_human_mcu_command(line) == expected


def test_human_mcu_console_exercises_existing_host_command_set() -> None:
    session = build_simulated_mcu_test_session(led_count=4, panel_revision=3)
    console = session.console

    capabilities = console.execute_line("get-capabilities")
    assert capabilities is not None
    assert "panel-revision=3" in capabilities
    assert "led-count=4" in capabilities
    assert console.execute_line("upload-pattern 17 0,32,64,255") == (
        "OK upload-pattern id=17 values=4 crc32=0xc4723974"
    )
    assert console.execute_line("show-pattern 17") == "OK show-pattern id=17"
    assert console.execute_line("define-sequence 2 17:250:on") == (
        "OK define-sequence id=2 steps=1"
    )
    assert console.execute_line("arm-sequence 2") == "OK arm-sequence id=2"
    assert console.execute_line("start-sequence") == "OK start-sequence"
    running_after_start = session.endpoint.running
    assert running_after_start
    assert console.execute_line("stop") == "OK stop"
    running_after_stop = session.endpoint.running
    assert not running_after_stop
    assert console.execute_line("all-off") == "OK all-off"
    assert session.endpoint.displayed_pattern_id is None


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("unknown", "unknown MCU command"),
        ("show-pattern", "requires 1 argument"),
        ("show-pattern -1", "range"),
        ("upload-pattern 1 0,,2", "empty item"),
        ("upload-pattern 1 256", "range"),
        ("define-sequence 1 malformed", "sequence steps use"),
        ("define-sequence 1 2:5:maybe", "invalid trigger"),
        ('show-pattern "unterminated', "No closing quotation"),
    ],
)
def test_human_mcu_parser_reports_readable_errors(line: str, message: str) -> None:
    with pytest.raises(HumanMcuCommandError, match=message):
        parse_human_mcu_command(line)


def test_human_mcu_console_preserves_typed_endpoint_failures() -> None:
    console = build_simulated_mcu_test_session().console

    with pytest.raises(McuCommandFailure, match="PATTERN_NOT_FOUND"):
        console.execute_line("show-pattern 404")


def test_mcu_test_cli_runs_readable_script(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = tmp_path / "mcu-test.txt"
    script.write_text(
        """# native-v1 simulator test
get-capabilities
upload-pattern 17 0,32,64,255
show-pattern 17
define-sequence 2 17:250:on
arm-sequence 2
start-sequence
stop
all-off
""",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "mcu-test",
                "--led-count",
                "4",
                "--panel-revision",
                "3",
                "--script",
                str(script),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "OK get-capabilities" in output
    assert "OK upload-pattern id=17 values=4" in output
    assert output.rstrip().endswith("OK all-off")


def test_mcu_test_cli_reports_command_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["mcu-test", "--command", "show-pattern 404"]) == 1
    assert "PATTERN_NOT_FOUND" in capsys.readouterr().err


def test_mcu_test_cli_prints_command_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["mcu-test", "--show-command-help"]) == 0
    assert "upload-pattern PATTERN_ID" in capsys.readouterr().out


def test_mcu_test_cli_reports_invalid_simulator_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["mcu-test", "--led-count", "0", "--command", "stop"]) == 2
    assert "ERROR configuration" in capsys.readouterr().err
