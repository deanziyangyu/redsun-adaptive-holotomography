"""Tests for the ``aht`` command line and its lazy registry."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

import redsun_aht.__main__ as launcher

if TYPE_CHECKING:
    from collections.abc import Sequence


class Paths(NamedTuple):
    """Launcher paths pointed at one writable temporary tree."""

    repo: Path
    data: Path
    cache: Path
    temp: Path


@pytest.fixture()
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Paths:
    """Point every launcher path at a fresh temporary tree."""
    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    temp = tmp_path / "temp"
    monkeypatch.setattr(launcher, "MAX_PVA_TEMP", 1024)
    monkeypatch.setenv("REDSUN_AHT_REPO", str(repo))
    monkeypatch.setenv("REDSUN_DATA_ROOT", str(data))
    monkeypatch.setenv("REDSUN_NAPARI_CACHE", str(cache))
    monkeypatch.setenv("TEMP", str(temp))
    monkeypatch.setenv("TMP", str(temp))
    return Paths(repo=repo, data=data, cache=cache, temp=temp)


def patch_redsun_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Make Redsun's resolver report *root*, as the launcher's patch does."""

    def fake_user_data_dir(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return str(root)

    monkeypatch.setattr("redsun.log.user_data_dir", fake_user_data_dir)


def test_check_reports_every_resolved_path(
    paths: Paths,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--check`` prints all five paths and succeeds on writable roots."""
    patch_redsun_root(monkeypatch, paths.data)

    assert launcher.main(["--check"]) == 0

    output = capsys.readouterr().out
    for expected in (paths.repo, paths.data, paths.temp, paths.cache):
        assert str(expected) in output
    assert "NOT WRITABLE" not in output
    for directory in (paths.repo, paths.data, paths.cache, paths.temp):
        assert directory.is_dir()
        assert not list(directory.glob(".aht-write-probe-*"))


def test_check_names_the_repository_fallback(
    paths: Paths,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unwritable default data root falls back and says so."""
    monkeypatch.delenv("REDSUN_DATA_ROOT")
    blocked = paths.repo / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(launcher, "DEFAULT_DATA_ROOT", blocked / "data")
    fallback = paths.repo / ".redsun_data"
    patch_redsun_root(monkeypatch, fallback)

    assert launcher.main(["--check"]) == 0

    output = capsys.readouterr().out
    assert "repository fallback" in output
    assert str(fallback) in output


def test_check_fails_on_an_unwritable_root(
    paths: Paths,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A path that cannot accept a probe file makes ``--check`` exit non-zero."""
    patch_redsun_root(monkeypatch, paths.data)
    blocked = paths.repo / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    unwritable = blocked / "cache"
    monkeypatch.setenv("REDSUN_NAPARI_CACHE", str(unwritable))

    assert launcher.main(["--check"]) == 1

    output = capsys.readouterr().out
    assert str(unwritable) in output
    assert "NOT WRITABLE" in output


def test_list_works_without_a_registry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--list`` reports empty sections instead of raising when none exist."""
    configurations = import_module("redsun_aht.configurations")
    monkeypatch.delattr(configurations, "registry", raising=False)

    assert launcher.main(["--list"]) == 0

    output = capsys.readouterr().out
    assert "no sessions registered yet" in output
    assert "no services registered yet" in output


def test_unknown_session_names_the_registered_ones(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown name exits non-zero and lists the names that do exist."""

    def first(arguments: Sequence[str]) -> int:
        del arguments
        return 0

    def second(arguments: Sequence[str]) -> int:
        del arguments
        return 0

    registry = launcher.Registry(sessions={"first": first, "second": second})
    monkeypatch.setattr(launcher, "load_registry", lambda: registry)

    assert launcher.main(["session", "missing"]) == 2

    error = capsys.readouterr().err
    assert "missing" in error
    assert "first" in error
    assert "second" in error


def test_registry_function_drives_list_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The coordinator's ``registry()`` is read lazily and used for dispatch."""
    configurations = import_module("redsun_aht.configurations")
    seen: list[list[str]] = []

    def run_demo(arguments: Sequence[str]) -> int:
        """Run the demo session."""
        seen.append(list(arguments))
        return 0

    def registry() -> tuple[dict[str, launcher.Entry], dict[str, launcher.Entry]]:
        return {"demo": run_demo}, {}

    monkeypatch.setattr(configurations, "registry", registry, raising=False)

    assert launcher.main(["--list"]) == 0
    listed = capsys.readouterr().out
    assert "demo" in listed
    assert "Run the demo session." in listed

    assert launcher.main(["session", "demo", "--", "--flag"]) == 0
    assert seen == [["--flag"]]


def test_main_without_arguments_prints_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main([])`` prints usage and returns a non-zero status."""
    assert launcher.main([]) == 2

    captured = capsys.readouterr()
    assert "usage:" in (captured.out + captured.err).lower()


def test_session_receives_arguments_after_the_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arguments after ``--`` reach the registered session unchanged."""
    seen: list[list[str]] = []

    def capture(arguments: Sequence[str]) -> int:
        seen.append(list(arguments))
        return 0

    registry = launcher.Registry(sessions={"demo": capture})
    monkeypatch.setattr(launcher, "load_registry", lambda: registry)

    status = launcher.main(["session", "demo", "--", "--prefix", "AHT:", "7"])

    assert status == 0
    assert seen == [["--prefix", "AHT:", "7"]]


def test_session_return_code_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session's integer return value becomes the CLI exit status."""

    def failing(arguments: Sequence[str]) -> int:
        del arguments
        return 7

    registry = launcher.Registry(sessions={"failing": failing})
    monkeypatch.setattr(launcher, "load_registry", lambda: registry)

    assert launcher.main(["session", "failing"]) == 7


def test_import_does_not_load_gui_or_camera_stack() -> None:
    """Importing the command line must stay free of napari, Qt and pymmcore."""
    code = (
        "import sys\n"
        "import redsun_aht.__main__\n"
        "heavy = ('napari', 'qtpy', 'PyQt6', 'PySide6', 'pymmcore_plus')\n"
        "print(','.join(name for name in heavy if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", result.stdout
