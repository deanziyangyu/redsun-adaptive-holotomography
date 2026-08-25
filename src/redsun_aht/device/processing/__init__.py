"""Optional device-compatible adapters for detached processing."""

from .contracts import ProcessingDeviceAdapter
from .flyer import ProcessingFlyerDevice

__all__ = ["ProcessingDeviceAdapter", "ProcessingFlyerDevice"]
