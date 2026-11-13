"""Tests for the R02c session bundle: registry, manifest, session, profiles.

Nothing here constructs the session, a ``QApplication`` or a viewer. The
session class is inspected through its declarations and annotations, which is
what Redsun 0.16 reads before it builds anything, and the manifest and profile
files are resolved with ``importlib`` so a rename breaks a test rather than
startup.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_type_hints

import numpy as np
import pytest
import yaml

import redsun_aht
import redsun_aht.configurations as configurations

if TYPE_CHECKING:
    from collections.abc import Mapping


def _package_root() -> Path:
    """Return the installed package directory."""
    assert redsun_aht.__file__ is not None
    return Path(redsun_aht.__file__).resolve().parent


PACKAGE = _package_root()
MANIFEST = PACKAGE / "redsun.yaml"
PROFILES = PACKAGE / "profiles"


def _manifest() -> dict[str, Any]:
    """Return the plugin manifest as a mapping."""
    data: Any = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _profile(name: str) -> dict[str, Any]:
    """Return one packaged session profile as a mapping."""
    data: Any = yaml.safe_load((PROFILES / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _resolve(path: str) -> Any:
    """Import the ``module:attribute`` path *path* names."""
    module_name, separator, attribute = path.partition(":")
    assert separator == ":" and module_name and attribute, path
    return getattr(importlib.import_module(module_name), attribute)


def _readings(values: Mapping[str, Any]) -> Any:
    """Return readings shaped as the presenter emits them."""
    return {key: {"value": value, "timestamp": 0.0} for key, value in values.items()}


def test_registry_names_the_camera_session_and_services() -> None:
    """Expose exactly the session and services the bundle dispatches."""
    sessions, services = configurations.registry()

    assert set(sessions) == {"camera", "dpct", "rotation", "profile"}
    assert set(services) == {
        "mmcore-camera",
        "spinnaker-camera",
        "mmcore-stage",
        "arduino-led",
        "asi-lx4000",
        "lumencor-spectra",
        "unified-emmsh",
        "unified-mcu",
    }
    for entry in (*sessions.values(), *services.values()):
        assert callable(entry)
        assert (entry.__doc__ or "").strip(), entry


def test_importing_configurations_loads_no_gui_or_camera_stack() -> None:
    """Import the registry package without Qt, napari or pymmcore."""
    code = (
        "import sys\n"
        "import redsun_aht.configurations\n"
        "heavy = (\n"
        "    'napari', 'qtpy', 'PyQt6', 'PySide6',\n"
        "    'pymmcore', 'pymmcore_plus', 'PySpin', 'redsun.qt',\n"
        ")\n"
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


def test_the_lazy_surface_exports_the_session_and_wiring() -> None:
    """Expose the session and helpers through ``__getattr__``, and nothing else."""
    assert "CameraSession" in dir(configurations)
    assert "DetectorPreview" in dir(configurations)

    session_type = configurations.CameraSession
    assert session_type.__name__ == "CameraSession"
    presenter_type = configurations.DetectorPresenter
    assert presenter_type.__name__ == "DetectorPresenter"
    with pytest.raises(AttributeError):
        _ = configurations.no_such_entry


def test_manifest_declares_the_camera_bundle() -> None:
    """Name the required sections and ids, and the same services as the registry."""
    data = _manifest()

    assert set(data["services"]) == {
        "mmcore-camera",
        "spinnaker-camera",
        "mmcore-stage",
        "arduino-led",
        "asi-lx4000",
        "lumencor-spectra",
        "unified-emmsh",
        "unified-mcu",
    }
    assert set(data["devices"]) == {
        "mmcore-camera",
        "spinnaker-camera",
        "mmcore-stage",
        "arduino-led",
        "asi-lx4000",
        "lumencor-spectra",
        "unified-emmsh",
        "unified-mcu",
        "unified-rotation",
    }
    assert set(data["presenters"]) == {"detector", "acquisition", "dpct"}
    assert set(data["views"]) == {"detector", "acquisition", "image"}

    _, services = configurations.registry()
    assert set(services) == set(data["services"])


def test_rotation_session_declares_positioner_and_axis() -> None:
    """Inspect the widget session without constructing Qt or opening a port."""
    from redsun._config import validate_session
    from redsun.qt import Qt
    from redsun.session._declarations import read, read_services

    from redsun_aht.configurations.rotation import (
        RotationSession,
        build_rotation_session,
    )

    config = validate_session([], RotationSession.config)
    declarations = read(RotationSession, config, Qt)
    assert set(declarations) == {"rotation", "motor_ctrl", "motor_widget"}
    assert all(item.refusal is None for item in declarations.values())
    managed = build_rotation_session("COM18")
    managed_services = read_services(type(managed), {}, "pv-access")
    assert "mcu_service" in managed_services


def test_manifest_paths_resolve() -> None:
    """Resolve every manifest class path and service module with importlib."""
    data = _manifest()

    for name, path in data["devices"].items():
        assert isinstance(_resolve(path), type), f"device {name}: {path}"
    for name, path in data["presenters"].items():
        assert isinstance(_resolve(path), type), f"presenter {name}: {path}"
    for name, path in data["views"].items():
        assert isinstance(_resolve(path), type), f"view {name}: {path}"
    for name, entry in data["services"].items():
        module = importlib.import_module(entry["module"])
        assert callable(getattr(module, "main", None)), f"service {name}"
        assert (entry.get("ready") or "").strip(), f"service {name} has no ready line"


@pytest.mark.parametrize(
    ("name", "cameras"),
    [("camera.yaml", 1), ("camera-dual.yaml", 2)],
)
def test_profiles_match_the_manifest(name: str, cameras: int) -> None:
    """Declare only manifest plugin ids, a PVA transport and one service per camera."""
    profile = _profile(name)
    manifest = _manifest()
    services = {
        key: value for key, value in profile["services"].items() if key != "transport"
    }

    assert profile["schema_version"] == 1.0
    assert profile["session"]
    assert profile["services"]["transport"] == "pv-access"
    assert len(services) == cameras
    for service in services.values():
        assert service["plugin_name"] == "redsun-aht"
        assert service["plugin_id"] in manifest["services"]
        assert service["prefix"].endswith(":")
    assert "det_ctrl" in profile["presenters"]
    assert "det_widget" in profile["views"]


def test_the_sessions_point_at_the_packaged_profiles() -> None:
    """Carry a config path per session class and keep those files present."""
    from redsun_aht.configurations.camera import (
        CAMERA_PROFILE,
        DUAL_CAMERA_PROFILE,
        CameraSession,
        DualCameraSession,
    )

    assert CameraSession.config == CAMERA_PROFILE
    assert DualCameraSession.config == DUAL_CAMERA_PROFILE
    assert CAMERA_PROFILE == PROFILES / "camera.yaml"
    assert DUAL_CAMERA_PROFILE == PROFILES / "camera-dual.yaml"
    assert CAMERA_PROFILE.is_file()
    assert DUAL_CAMERA_PROFILE.is_file()


def test_the_session_declares_the_presenter_and_view_layers() -> None:
    """Declare ``det_ctrl`` as a presenter and ``det_widget`` as a view."""
    from redsun import Layer
    from redsun.qt import QtSession

    from redsun_aht.configurations.camera import CameraSession
    from redsun_aht.presenter.detector import DetectorPresenter
    from redsun_aht.view.detector import DetectorView

    hints = get_type_hints(CameraSession, include_extras=True)

    assert issubclass(CameraSession, QtSession)
    assert Layer.PRESENTER in get_args(hints["det_ctrl"])[1:]
    assert Layer.VIEW in get_args(hints["det_widget"])[1:]
    assert get_args(hints["det_ctrl"])[0] is DetectorPresenter
    assert get_args(hints["det_widget"])[0] is DetectorView


def test_the_view_constructor_matches_the_qt_session_contract() -> None:
    """Start with ``(name: str, parent: QWidget)`` and ask for a placement."""
    from qtpy import QtWidgets
    from redsun.qt import Dock, Qt

    from redsun_aht.view.detector import DetectorView

    parameters = list(
        inspect.signature(DetectorView, eval_str=True).parameters.values()
    )

    assert [(param.name, param.kind, param.annotation) for param in parameters] == [
        ("name", inspect.Parameter.POSITIONAL_OR_KEYWORD, str),
        ("parent", inspect.Parameter.POSITIONAL_OR_KEYWORD, QtWidgets.QWidget),
    ]
    assert DetectorView.placement == Dock("right")
    assert Qt.check_view(DetectorView, "det_widget") is None


def test_the_presenter_is_injectable_by_redsun() -> None:
    """Annotate the constructor and ``setup`` the way Redsun 0.16 reads them."""
    from redsun import Layer
    from redsun.qt import Qt
    from redsun.session._declarations import check

    from redsun_aht.presenter.detector import DetectorPresenter

    constructor = inspect.signature(DetectorPresenter, eval_str=True)
    setup = inspect.signature(DetectorPresenter.setup, eval_str=True)

    assert "detectors" in constructor.parameters
    assert "deferrals" in setup.parameters
    assert (
        check(DetectorPresenter, Layer.PRESENTER, "det_ctrl", Qt) is DetectorPresenter
    )


@pytest.mark.parametrize(
    ("session_name", "profile_name", "expected"),
    [
        (
            "CameraSession",
            "camera.yaml",
            {
                "acq_ctrl",
                "acq_widget",
                "camera1",
                "det_ctrl",
                "det_widget",
                "img_widget",
            },
        ),
        (
            "DualCameraSession",
            "camera-dual.yaml",
            {
                "acq_ctrl",
                "acq_widget",
                "camera1",
                "camera2",
                "det_ctrl",
                "det_widget",
                "img_widget",
            },
        ),
    ],
)
def test_a_profile_declares_every_annotated_component(
    session_name: str, profile_name: str, expected: set[str]
) -> None:
    """Read the session's declarations from its profile without building it."""
    from redsun._config import load, validate_session
    from redsun.qt import Qt
    from redsun.session._declarations import read

    session_type = getattr(configurations, session_name)
    profile = PROFILES / profile_name
    config = validate_session([profile], load(profile))
    declarations = read(session_type, config, Qt)

    assert set(declarations) == expected
    for name, declaration in declarations.items():
        assert declaration.refusal is None, f"{name}: {declaration.refusal}"


def test_the_factory_refuses_a_detector_count_other_than_one_or_two() -> None:
    """Refuse an unsupported detector count before deciding on a session class."""
    from redsun_aht.configurations.camera import build_camera_session

    with pytest.raises(ValueError, match="1 or 2"):
        build_camera_session(detectors=3)


def test_the_preview_keeps_only_the_newest_frame_of_each_detector() -> None:
    """Replace one detector's frame on every offer and leave the other alone."""
    from redsun_aht.configurations.wiring import DetectorPreview

    preview = DetectorPreview()
    started = preview.detectors
    assert len(started) == 0
    assert preview.latest("camera1") is None

    first = np.zeros((2, 2), dtype=np.uint8)
    second = np.ones((2, 2), dtype=np.uint16)
    preview.update_layers(_readings({"camera1-buffer": first}))
    preview.update_layers(_readings({"camera1-buffer": second}))
    preview.update_layers(_readings({"camera2-buffer": first}))
    preview.update_layers(_readings({"camera1-roi": "0,0,2,2"}))
    preview.update_layers(_readings({"camera1-buffer": None}))

    assert preview.detectors == ("camera1", "camera2")
    latest = preview.latest("camera1")
    assert latest is not None
    frame = latest.snapshot()
    assert frame is not None
    assert frame.sequence == 1
    np.testing.assert_array_equal(frame.image, second)

    other = preview.latest("camera2")
    assert other is not None
    other_frame = other.snapshot()
    assert other_frame is not None
    np.testing.assert_array_equal(other_frame.image, first)
