"""Typed provider keys shared through RedSun virtual containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dependency_injector.providers as dip
from redsun.storage import BaseStorage

if TYPE_CHECKING:
    from typing import Any

    from redsun.virtual import ProviderKey

    from redsun_aht.domain import CalibrationArtifact, ProcessingJob

DETECTOR_REGISTRY: ProviderKey[dict[str, Any]] = dip.Dependency(instance_of=dict)
PLAN_REGISTRY: ProviderKey[dict[str, Any]] = dip.Dependency(instance_of=dict)
CALIBRATION_SELECTION: ProviderKey[dict[str, CalibrationArtifact]] = dip.Dependency(
    instance_of=dict
)
PROCESSING_JOBS: ProviderKey[dict[str, ProcessingJob]] = dip.Dependency(
    instance_of=dict
)
DEVICE_PROPERTIES: ProviderKey[dict[str, Any]] = dip.Dependency(instance_of=dict)

#: The application-owned storage orchestrator used by the active run profile.
RUN_STORAGE: ProviderKey[BaseStorage] = dip.Dependency(instance_of=BaseStorage)

__all__ = [
    "CALIBRATION_SELECTION",
    "DETECTOR_REGISTRY",
    "DEVICE_PROPERTIES",
    "PLAN_REGISTRY",
    "PROCESSING_JOBS",
    "RUN_STORAGE",
]
