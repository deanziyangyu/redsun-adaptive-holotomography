"""RedSun-shaped execution profiles loaded from packaged YAML."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

ProfileConfig = Mapping[str, Any]

_PROFILE_FILES = MappingProxyType(
    {
        "simulate": "simulate.yaml",
        "detector-simulate": "detector-simulate.yaml",
        "camera-gui": "camera-gui.yaml",
        "simulate-dpct": "simulate-dpct.yaml",
        "process-headless": "process-headless.yaml",
        "process-gui": "process-gui.yaml",
    }
)


def profile_path(name: str) -> Path:
    """Return the packaged RedSun configuration path for one profile."""
    try:
        filename = _PROFILE_FILES[name]
    except KeyError as error:
        raise KeyError(f"unknown RedSun profile: {name}") from error
    return Path(__file__).resolve().parents[1] / "profiles" / filename


def load_profile(name: str) -> ProfileConfig:
    """Load one immutable RedSun application configuration."""
    path = profile_path(name)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"RedSun profile {name!r} must be a YAML mapping")
    required = {"schema_version", "frontend", "session", "metadata"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"RedSun profile {name!r} is missing: {', '.join(sorted(missing))}"
        )
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or metadata.get("profile") != name:
        raise ValueError(f"RedSun profile {name!r} has invalid metadata identity")
    return MappingProxyType(cast("dict[str, Any]", payload))


def profile_metadata(profile: ProfileConfig) -> Mapping[str, Any]:
    """Return the immutable application metadata section of a profile."""
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):  # pragma: no cover - guarded by loader
        raise TypeError("RedSun profile metadata must be a mapping")
    return MappingProxyType(metadata)


PROFILES = MappingProxyType({name: load_profile(name) for name in _PROFILE_FILES})

__all__ = [
    "PROFILES",
    "ProfileConfig",
    "load_profile",
    "profile_metadata",
    "profile_path",
]
