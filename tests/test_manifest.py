from __future__ import annotations

from typing import Any, cast

import pytest

from redsun_aht.storage import RunManifest


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
