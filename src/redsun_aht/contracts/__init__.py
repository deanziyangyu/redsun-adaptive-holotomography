"""CA-free plan, device, frame and asset contracts.

This package is the narrow interface between the acquisition application's
device, serial, live, plan and storage packages. It imports nothing but the
standard library and ``numpy``: no Qt, no napari, no Channel Access, no
``redsun`` and no other ``redsun_aht`` module. A device or a store may therefore
depend on it without pulling an application in.

The four modules divide the surface by question, not by owner:

``devices``
    What a camera, a stage and an MM-compatible illumination service declare,
    and what a caller may ask them for.
``frames``
    How one frame is identified, what its exposure and setting evidence is, and
    how honestly its pixels are labelled.
``plans``
    What a run is asked for, which roles it selects, and how a run reports the
    frames it did and did not produce.
``assets``
    How a committed plane is referenced, refused and receipted.
"""

from .assets import (
    ASSET_CONTRACT_VERSION,
    AssetContractViolation,
    AssetRef,
    AssetRefusal,
    PlaneReceipt,
)
from .devices import (
    CENTER_PATTERN_ID,
    DPCT_PATTERN_IDS,
    ILLUMINATION_PATTERN_IDS,
    AxisReading,
    AxisSpec,
    Camera,
    CameraDescription,
    CameraReadout,
    CameraSettings,
    CommandReceipt,
    CommandStatus,
    Illumination,
    IlluminationAction,
    IlluminationCapability,
    Roi,
    SettingRefusal,
    SettingRefused,
    Stage,
    StageReading,
    TriggerMode,
)
from .frames import (
    MAX_RUN_ID_BYTES,
    ExposureWindow,
    FrameIdentity,
    FrameRecord,
    QualityLabel,
    RunIdentity,
    SettingChange,
)
from .plans import (
    AxisGrid,
    DeviceRole,
    PartialRun,
    PlanAbortReason,
    PlanOutcome,
    PlanRequest,
    PlanStatus,
    ScanPoint,
)

__all__ = [
    "ASSET_CONTRACT_VERSION",
    "CENTER_PATTERN_ID",
    "DPCT_PATTERN_IDS",
    "ILLUMINATION_PATTERN_IDS",
    "MAX_RUN_ID_BYTES",
    "AssetContractViolation",
    "AssetRef",
    "AssetRefusal",
    "AxisGrid",
    "AxisReading",
    "AxisSpec",
    "Camera",
    "CameraDescription",
    "CameraReadout",
    "CameraSettings",
    "CommandReceipt",
    "CommandStatus",
    "DeviceRole",
    "ExposureWindow",
    "FrameIdentity",
    "FrameRecord",
    "Illumination",
    "IlluminationAction",
    "IlluminationCapability",
    "PartialRun",
    "PlanAbortReason",
    "PlanOutcome",
    "PlanRequest",
    "PlanStatus",
    "PlaneReceipt",
    "QualityLabel",
    "Roi",
    "RunIdentity",
    "ScanPoint",
    "SettingChange",
    "SettingRefusal",
    "SettingRefused",
    "Stage",
    "StageReading",
    "TriggerMode",
]
