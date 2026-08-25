"""Isolated hardware-service contracts and supervisors."""

from .mock_camera import CameraServiceStateError, MockCameraService
from .registry import DetectorRegistry

__all__ = [
    "CameraServiceStateError",
    "DetectorRegistry",
    "MockCameraService",
]
