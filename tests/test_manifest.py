from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from redsun_aht.storage import RunManifest, RunManifestStore


def test_run_manifest_recursively_copies_and_freezes_configuration() -> None:
    hardware: dict[str, Any] = {"detectors": [{"id": "camera-a"}]}
    manifest = RunManifest(
        schema_version=1,
        run_id="run-1",
        created_ns=1,
        software_version="0.1.0.dev0",
        hardware=hardware,
        reconstruction={},
        experiment={},
        deployment={},
    )

    hardware["detectors"][0]["id"] = "mutated"

    detectors = cast(tuple[Any, ...], manifest.hardware["detectors"])
    assert detectors[0]["id"] == "camera-a"
    with pytest.raises(TypeError):
        cast(Any, manifest.hardware)["new"] = "value"


def test_run_manifest_store_atomically_round_trips(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path / "run-manifest.json")
    manifest = RunManifest(
        schema_version=1,
        run_id="run-1",
        created_ns=1,
        software_version="0.1.0.dev0",
        hardware={"detector": {"backend": "mock"}},
        reconstruction={},
        experiment={"recipe": "simulation"},
        deployment={"profile": "test"},
    )

    store.write(manifest)

    assert store.read() == manifest
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"run_id": ""},
        {"software_version": ""},
        {"created_ns": -1},
    ],
)
def test_run_manifest_rejects_invalid_identity(
    overrides: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "run-1",
        "created_ns": 1,
        "software_version": "0.1.0.dev0",
        "hardware": {},
        "reconstruction": {},
        "experiment": {},
        "deployment": {},
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        RunManifest(**values)


def test_run_manifest_store_rejects_wrong_fields(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    path.write_text('{"schema_version":1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="fields do not match"):
        RunManifestStore(path).read()


def test_run_manifest_store_rejects_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    path.write_text(
        '{"created_ns":1,"deployment":{},"experiment":{},"hardware":{},'
        '"reconstruction":{},"run_id":"run-1","schema_version":2,'
        '"software_version":"0.1.0.dev0"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains invalid values"):
        RunManifestStore(path).read()
