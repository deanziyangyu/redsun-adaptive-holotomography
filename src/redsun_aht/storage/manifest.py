"""Immutable run configuration snapshot contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from redsun_aht.domain.models import JsonValue, freeze_json_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Resolved hardware, recipe, reconstruction, and deployment snapshot."""

    schema_version: int
    run_id: str
    created_ns: int
    software_version: str
    hardware: Mapping[str, JsonValue]
    reconstruction: Mapping[str, JsonValue]
    experiment: Mapping[str, JsonValue]
    deployment: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        """Recursively detach and freeze all configuration domains."""
        object.__setattr__(self, "hardware", freeze_json_mapping(self.hardware))
        object.__setattr__(
            self, "reconstruction", freeze_json_mapping(self.reconstruction)
        )
        object.__setattr__(self, "experiment", freeze_json_mapping(self.experiment))
        object.__setattr__(self, "deployment", freeze_json_mapping(self.deployment))
