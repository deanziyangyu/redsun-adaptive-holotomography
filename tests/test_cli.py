from __future__ import annotations

from pathlib import Path

from redsun_aht.__main__ import main


def test_cli_lists_profiles(capsys: object) -> None:
    assert main(["profiles"]) == 0


def test_cli_runs_hardware_free_simulation(tmp_path: Path, capsys: object) -> None:
    journal = tmp_path / "cli-events.jsonl"

    assert main(["simulate", "--journal", str(journal)]) == 0
    assert journal.read_text(encoding="utf-8").count("\n") == 2
