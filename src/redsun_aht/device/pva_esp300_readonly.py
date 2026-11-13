"""Typed Redsun view of the read-only ESP300 PVA snapshot service."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from ophyd_async.core import Device, SignalR, SignalW
from ophyd_async.fastcs.core import fastcs_connector

from redsun_aht.device.esp300_readonly import AxisSnapshot, ControllerSnapshot


@dataclass(frozen=True, slots=True)
class Esp300Observation:
    """Sequential controller-reported axes and their service-clock bracket."""

    start_ns: int
    end_ns: int
    snapshot: ControllerSnapshot
    clock: str = "host_monotonic_ns"


def parse_snapshot(value: str) -> Esp300Observation:
    """Validate an untrusted service snapshot without asserting calibration."""
    doc = json.loads(value)
    if not isinstance(doc, dict) or set(doc) != {
        "schema",
        "clock",
        "start_ns",
        "end_ns",
        "version",
        "axes",
        "quality",
    }:
        raise ValueError("invalid snapshot fields")
    if (
        doc["schema"] != "redsun.esp300-readonly/v1"
        or doc["clock"] != "host_monotonic_ns"
        or doc["quality"] != "controller_reported"
        or type(doc["start_ns"]) is not int
        or type(doc["end_ns"]) is not int
        or doc["start_ns"] < 0
        or doc["end_ns"] < doc["start_ns"]
        or not isinstance(doc["version"], str)
        or "ESP300" not in doc["version"].upper()
    ):
        raise ValueError("invalid snapshot provenance")
    records = doc["axes"]
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("expected three axes")
    axes = []
    numeric = ("resolution_um", "lower_um", "upper_um", "actual_um", "target_um")
    for expected, item in enumerate(records, 1):
        if (
            not isinstance(item, dict)
            or set(item) != {"axis", "identity", "motion_done", *numeric}
            or type(item["axis"]) is not int
            or item["axis"] != expected
            or not isinstance(item["identity"], str)
            or not item["identity"].strip()
            or type(item["motion_done"]) is not bool
        ):
            raise ValueError("invalid axis identity or status")
        if any(
            type(item[key]) not in (int, float) or not math.isfinite(item[key])
            for key in numeric
        ):
            raise ValueError("invalid axis numeric value")
        if (
            item["resolution_um"] <= 0
            or item["lower_um"] >= item["upper_um"]
            or not item["lower_um"] <= item["actual_um"] <= item["upper_um"]
        ):
            raise ValueError("invalid axis limits or position")
        axes.append(AxisSnapshot(**item))
    return Esp300Observation(
        doc["start_ns"], doc["end_ns"], ControllerSnapshot(doc["version"], tuple(axes))
    )


class PvaEsp300ReadOnly(Device):
    """Observe a shared service; intentionally provides no motion methods."""

    snapshot: SignalR[str]
    status: SignalR[str]
    refresh: SignalW[bool]

    def __init__(self, prefix: str, *, name: str = "") -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))

    async def read_observation(self, *, refresh: bool = True) -> Esp300Observation:
        """Optionally request a new snapshot and validate its service provenance."""
        if refresh:
            await self.refresh.set(True)
        if await self.status.get_value(cached=False) != "ready":
            raise RuntimeError("ESP300 snapshot is not ready")
        value = await self.snapshot.get_value(cached=False)
        observation = parse_snapshot(value)
        if await self.status.get_value(cached=False) != "ready":
            raise RuntimeError("ESP300 snapshot changed during read")
        return observation
