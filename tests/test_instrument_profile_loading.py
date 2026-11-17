"""Instrument profile path handling without hardware or a graphical session."""

from unittest.mock import Mock, patch

import pytest

from redsun_aht.configurations.profile import read_instrument_profile, run_profile
from redsun import Session
from pathlib import Path


@pytest.mark.parametrize("instrument", ["arh", "holospim"])
def test_profile_and_overlay_resolve_paths_from_their_own_directories(instrument, tmp_path):
    base = tmp_path / "base" / "profile.yaml"
    overlay = tmp_path / "overlay" / "profile.yaml"
    base.parent.mkdir()
    overlay.parent.mkdir()
    base.write_text('services:\n  camera:\n    args:\n      mm-config: "./camera.cfg"\n')
    overlay.write_text('services:\n  stage:\n    args:\n      mmcore-config: "./stage.cfg"\n')
    session = Mock()
    with patch("redsun_aht.configurations.profile.Session.from_config", return_value=session) as create:
        assert run_profile([instrument, "--profile", str(base), "--overlay", str(overlay), "--check"]) == 0
    layers = create.call_args.args[0]
    assert layers[0]["services"]["camera"]["args"]["mm-config"] == str(base.parent / "camera.cfg")
    assert layers[1]["services"]["stage"]["args"]["mmcore-config"] == str(overlay.parent / "stage.cfg")
    assert layers[-1]["mock"] is True
    session.build.assert_called_once()
    session.shutdown.assert_called_once()


def test_bad_service_args_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("services:\n  camera:\n    args: invalid\n")
    with pytest.raises(ValueError, match="args mapping"):
        read_instrument_profile(path)


@pytest.mark.parametrize("profile", ["arh_system.yaml", "holospim.yaml"])
def test_packaged_device_profiles_build_and_shutdown_mock(profile, tmp_path):
    path = Path(__file__).resolve().parents[1] / "src/redsun_aht/profiles" / profile
    session = Session.from_config([
        read_instrument_profile(path),
        {"mock": True, "strict": True, "storage": {"base_dir": str(tmp_path)}},
    ])
    try:
        session.build()
        assert {"camera", "mcu"} <= session.devices.keys()
    finally:
        session.shutdown()
