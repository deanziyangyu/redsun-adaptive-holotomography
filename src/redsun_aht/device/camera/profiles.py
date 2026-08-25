"""Verified camera-property profiles applied after MMCore config admission."""

from types import MappingProxyType

PVCAM_FCS_PROPERTIES = MappingProxyType(
    {
        "Binning": "2x2",
        "CircularBufferAutoSize": "ON",
        "ClearMode": "Never",
        "ClearCycles": "0",
        "DiskStreamingEnabled": "No",
        "ROI": "0,0,64,64",
        "Exposure": "0.5",
        "FrameSummingEnabled": "No",
        "ReadoutRate": "200MHz 12bit",
        "Gain": "3-Sensitivity",
        "MetadataEnabled": "No",
        "PMode": "Normal",
        "PP  1   ENABLED": "No",
        "PP  2   ENABLED": "No",
        "PP  3   ENABLED": "No",
        "PP  4   ENABLED": "No",
        "PP  5   ENABLED": "No",
        "SMARTStreamingEnabled": "No",
        "ShutterCloseDelay": "0",
        "TriggerMode": "Internal Trigger",
    }
)
"""Device API 75 PVCAM FCS profile verified on 2026-08-25.

Use with the admitted camera-only base configuration, not the legacy FCS
MMStudio config file: that file applies gain before the matching readout mode
is admitted by the newer adapter.
"""


__all__ = ["PVCAM_FCS_PROPERTIES"]
