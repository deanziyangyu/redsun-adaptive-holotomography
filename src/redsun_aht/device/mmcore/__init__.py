"""Micro-Manager camera and stage devices, over the PVA services that own them.

Ported from ``redsun_searching_is_imaging/device/mmcore/__init__.py`` at donor
revision ``3a3bf06`` (``redsun-searching-is-imaging``).

Changed: the modules are ``camera.py`` and ``stage.py`` rather than ``_camera``
and ``_stage``; the ROI is the settled ``redsun_aht.contracts.Roi`` rather than
a private copy, re-exported here so a caller of this package needs one import;
and ``ReadableDeviceMap`` is exported until ``redsun_aht.device`` or ophyd-async
owns it. ``MMAxis``, ``MMCamera`` and ``MMStage`` are the donor's names.
"""

from __future__ import annotations

from ...contracts import Roi
from ._containers import ReadableDeviceMap
from .camera import MMCamera, frame_shape_and_dtype
from .stage import AXIS_UNITS, MOVE_TIMEOUT, POSITION_TOLERANCE, MMAxis, MMStage

__all__ = [
    "AXIS_UNITS",
    "MOVE_TIMEOUT",
    "POSITION_TOLERANCE",
    "MMAxis",
    "MMCamera",
    "MMStage",
    "ReadableDeviceMap",
    "Roi",
    "frame_shape_and_dtype",
]
