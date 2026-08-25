"""Immutable run configuration snapshot contract."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from redsun_aht.domain.models import JsonValue, freeze_json_mapping, thaw_json

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import ClassVar


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
        if self.schema_version != 1:
            raise ValueError("only run manifest schema version 1 is supported")
        if not self.run_id or not self.software_version:
            raise ValueError("run_id and software_version must be non-empty")
        if self.created_ns < 0:
            raise ValueError("created_ns must be non-negative")
        object.__setattr__(self, "hardware", freeze_json_mapping(self.hardware))
        object.__setattr__(
            self, "reconstruction", freeze_json_mapping(self.reconstruction)
        )
        object.__setattr__(self, "experiment", freeze_json_mapping(self.experiment))
        object.__setattr__(self, "deployment", freeze_json_mapping(self.deployment))


class RunManifestStore:
    """Atomically persist and recover one immutable run manifest."""

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "run_id",
            "created_ns",
            "software_version",
            "hardware",
            "reconstruction",
            "experiment",
            "deployment",
        }
    )

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, manifest: RunManifest) -> None:
        """Atomically replace the manifest with a flushed JSON snapshot."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        payload = {
            "schema_version": manifest.schema_version,
            "run_id": manifest.run_id,
            "created_ns": manifest.created_ns,
            "software_version": manifest.software_version,
            "hardware": thaw_json(manifest.hardware),
            "reconstruction": thaw_json(manifest.reconstruction),
            "experiment": thaw_json(manifest.experiment),
            "deployment": thaw_json(manifest.deployment),
        }
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self) -> RunManifest:
        """Validate and reconstruct the immutable manifest snapshot."""
        with self.path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or set(payload) != self._FIELDS:
            raise ValueError("run manifest fields do not match schema version 1")
        try:
            return RunManifest(**payload)
        except (TypeError, ValueError) as error:
            raise ValueError("run manifest contains invalid values") from error
