from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from importlib.resources import files

import yaml


def test_redsun_manifest_has_only_supported_sections() -> None:
    manifest_path = files("redsun_aht").joinpath("redsun.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest == {
        "devices": {},
        "presenters": {
            "simulation-lifecycle": (
                "redsun_aht.presenter:SimulationLifecyclePresenter"
            )
        },
        "views": {},
    }
    for group in ("devices", "presenters", "views"):
        for class_path in manifest[group].values():
            module_name, attribute = class_path.split(":", maxsplit=1)
            assert getattr(import_module(module_name), attribute) is not None


def test_redsun_entry_point_resolves_package_manifest() -> None:
    entry_point = next(
        entry
        for entry in entry_points(group="redsun.plugins")
        if entry.name == "redsun-aht"
    )

    assert entry_point.value == "redsun.yaml"
    assert (
        files(entry_point.name.replace("-", "_")).joinpath(entry_point.value).is_file()
    )
