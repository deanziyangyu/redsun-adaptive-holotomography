"""Plan-side contracts: what was asked for, and what was actually produced.

A plan is the only thing that knows both halves of an acquisition: the frames it
requested and the frames that arrived. This module freezes those halves so a
partial run is reported by naming the frames that are missing, rather than by
returning ``False`` and leaving the store to guess. It also freezes the scan
request itself: selected devices by role, the axes and the grid over them, the
illumination patterns and the detectors that are expected to answer.

Nothing here commands a device. The types are what a plan takes in and what it
must hand back, so the live, device, serial and storage packages can agree about
a run without importing the plan that ran it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise, product
from types import MappingProxyType
from typing import Any

from .devices import DPCT_PATTERN_IDS, AxisSpec
from .frames import FrameIdentity, RunIdentity


class DeviceRole(StrEnum):
    """The three roles a plan selects devices by."""

    CAMERA = "camera"
    STAGE = "stage"
    ILLUMINATION = "illumination"


@dataclass(frozen=True, slots=True)
class AxisGrid:
    """One scanned axis and the positions requested on it.

    Parameters
    ----------
    axis
        The axis and its declared limits.
    positions
        The requested positions, in visit order. They are all within the axis
        limits and run strictly in one direction, so a traversed line is
        reproducible.
    """

    axis: AxisSpec
    positions: tuple[float, ...]

    def __post_init__(self) -> None:
        """Refuse a grid that cannot be traversed deterministically."""
        if not isinstance(self.axis, AxisSpec):
            raise ValueError("an axis grid needs an axis spec")
        if not isinstance(self.positions, tuple) or not self.positions:
            raise ValueError("an axis grid needs at least one position")
        values = tuple(self.axis.check(position) for position in self.positions)
        if len(set(values)) != len(values):
            raise ValueError("axis grid positions must be distinct")
        ascending = all(later > earlier for earlier, later in pairwise(values))
        descending = all(later < earlier for earlier, later in pairwise(values))
        if not (ascending or descending):
            raise ValueError("axis grid positions must run strictly in one direction")
        object.__setattr__(self, "positions", values)

    @property
    def direction(self) -> int:
        """Return ``1`` for an ascending grid and ``-1`` for a descending one."""
        if len(self.positions) < 2:
            return 1
        return 1 if self.positions[1] > self.positions[0] else -1


@dataclass(frozen=True, slots=True)
class ScanPoint:
    """One requested position of the grid, in visit order.

    Parameters
    ----------
    index
        The point's ordinal in the run, counting from zero.
    positions
        The axis positions as ``(name, value)`` pairs, in grid order.
    """

    index: int
    positions: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        """Refuse a point that does not name one value per axis."""
        if type(self.index) is not int or self.index < 0:
            raise ValueError("scan point index must be a non-negative integer")
        if not isinstance(self.positions, tuple):
            raise ValueError("scan point positions must be a tuple")
        names = []
        for item in self.positions:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("scan point positions are (name, value) pairs")
            name, value = item
            if not isinstance(name, str) or not name.strip():
                raise ValueError("scan point axis names must be non-empty text")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("scan point values must be real numbers")
            if not math.isfinite(float(value)):
                raise ValueError("scan point values must be finite")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("scan point axis names must be unique")

    def as_mapping(self) -> Mapping[str, float]:
        """Return the positions by axis name."""
        return MappingProxyType({name: value for name, value in self.positions})


@dataclass(frozen=True, slots=True)
class PlanRequest:
    """Everything one plan run is asked to do, before it starts.

    Parameters
    ----------
    run
        The run the plan will produce.
    devices
        The selected device names by role. Every selected role has at least one
        device, and no device appears twice in a role.
    detector_ids
        The detector identities the plan expects frames from. Each one is a
        selected camera device, because a detector identity that no selected
        camera declares could not produce a frame. Device names and detector
        identities are equal for this wave; a mapping from detector identity to
        device name is the extension point if one device ever serves more than
        one logical detector.
    grid
        The scanned axes. The first grid entry is the outermost, slowest axis
        and the last is the innermost, fastest one.
    pattern_ids
        The illumination patterns the plan will render, in visit order. The
        DPCT plan acquires the four semicircle patterns, so its request is
        limited to :data:`~redsun_aht.contracts.devices.DPCT_PATTERN_IDS`; the
        91-LED panel's centre pattern is a valid illumination pattern but not
        part of this plan's frame set, and a request for it is refused before
        the run.
    snake
        Whether the innermost axis reverses direction on alternate lines, so a
        grid is traversed without returning to the start of every line.
    """

    run: RunIdentity
    devices: Mapping[DeviceRole, tuple[str, ...]]
    detector_ids: tuple[str, ...]
    grid: tuple[AxisGrid, ...] = ()
    pattern_ids: tuple[int, ...] = ()
    snake: bool = False

    def __post_init__(self) -> None:
        """Refuse a request that no plan could carry out."""
        if not isinstance(self.run, RunIdentity):
            raise ValueError("plan request needs a run identity")
        if not isinstance(self.devices, Mapping) or not self.devices:
            raise ValueError("plan request needs at least one selected device")
        frozen: dict[DeviceRole, tuple[str, ...]] = {}
        for role, names in self.devices.items():
            if not isinstance(role, DeviceRole):
                raise ValueError("plan request device roles must be DeviceRole members")
            if not isinstance(names, tuple) or not names:
                raise ValueError(f"plan request role {role} needs at least one device")
            for name in names:
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("plan request device names must be text")
            if len(set(names)) != len(names):
                raise ValueError(f"plan request role {role} names must be unique")
            frozen[role] = tuple(names)
        object.__setattr__(self, "devices", MappingProxyType(frozen))
        cameras = frozen.get(DeviceRole.CAMERA, ())
        if not cameras:
            raise ValueError("plan request must select at least one camera")
        if not isinstance(self.detector_ids, tuple) or not self.detector_ids:
            raise ValueError("plan request needs at least one detector identity")
        for detector_id in self.detector_ids:
            if not isinstance(detector_id, str) or not detector_id.strip():
                raise ValueError("plan request detector identities must be text")
            # Device names and detector identities are equal for this wave. When
            # one device ever serves more than one logical detector, this is the
            # line that grows a detector-id-to-device-name mapping.
            if detector_id not in cameras:
                raise ValueError(
                    f"detector identity {detector_id!r} is not a selected camera"
                )
        if len(set(self.detector_ids)) != len(self.detector_ids):
            raise ValueError("plan request detector identities must be unique")
        if not isinstance(self.grid, tuple):
            raise ValueError("plan request grid must be a tuple")
        if any(not isinstance(item, AxisGrid) for item in self.grid):
            raise ValueError("plan request grid entries must be AxisGrid values")
        if self.grid and DeviceRole.STAGE not in frozen:
            raise ValueError("a scanned plan must select a stage")
        axis_names = [item.axis.name for item in self.grid]
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("plan request grid axes must be unique")
        if not isinstance(self.pattern_ids, tuple):
            raise ValueError("plan request pattern ids must be a tuple")
        if any(type(item) is not int for item in self.pattern_ids):
            raise ValueError("plan request pattern ids must be integers")
        if len(set(self.pattern_ids)) != len(self.pattern_ids):
            raise ValueError("plan request pattern ids must be unique")
        unknown = tuple(
            item for item in self.pattern_ids if item not in DPCT_PATTERN_IDS
        )
        if unknown:
            raise ValueError(
                f"plan request pattern ids outside {DPCT_PATTERN_IDS}: {unknown}"
            )
        if self.pattern_ids and DeviceRole.ILLUMINATION not in frozen:
            raise ValueError("a requested pattern needs an illumination device")
        if not isinstance(self.snake, bool):
            raise ValueError("plan request snake flag must be a boolean")

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        """Return the scanned axes, outermost first."""
        return tuple(item.axis for item in self.grid)

    @property
    def cameras(self) -> tuple[str, ...]:
        """Return the selected camera device names."""
        return self.devices.get(DeviceRole.CAMERA, ())

    def device_names(self, role: DeviceRole) -> tuple[str, ...]:
        """Return the selected device names for *role*, empty when unselected."""
        return self.devices.get(role, ())

    def scan_points(self) -> tuple[ScanPoint, ...]:
        """Enumerate the requested positions in visit order.

        The first grid axis varies slowest and the last varies fastest. With
        ``snake`` set, the innermost axis is traversed in reverse on alternate
        lines, where a line's parity is the sum of the outer axes' indices.
        """
        if not self.grid:
            return (ScanPoint(index=0, positions=()),)
        names = tuple(item.axis.name for item in self.grid)
        counts = tuple(len(item.positions) for item in self.grid)
        points: list[ScanPoint] = []
        for combination in product(*(range(count) for count in counts)):
            chosen = list(combination)
            if self.snake and len(chosen) > 1 and sum(chosen[:-1]) % 2:
                chosen[-1] = counts[-1] - 1 - chosen[-1]
            positions = tuple(
                (name, item.positions[index])
                for name, item, index in zip(names, self.grid, chosen, strict=True)
            )
            points.append(ScanPoint(index=len(points), positions=positions))
        return tuple(points)


class PlanStatus(StrEnum):
    """How a plan run ended."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    ABORTED = "aborted"


class PlanAbortReason(StrEnum):
    """Why a plan run stopped before its requested frames arrived."""

    OPERATOR_STOP = "operator_stop"
    DEVICE_FAULT = "device_fault"
    ILLUMINATION_UNCERTAIN = "illumination_uncertain"
    CAMERA_TIMEOUT = "camera_timeout"
    STORAGE_REFUSED = "storage_refused"
    UNEXPECTED_ERROR = "unexpected_error"


@dataclass(frozen=True, slots=True)
class PartialRun:
    """The requested frames of one run, and the ones that were produced.

    Parameters
    ----------
    requested
        Every frame the plan asked for, in visit order.
    produced
        The frames that were produced and identified. Each one must be one of
        the requested frames.

    Notes
    -----
    ``missing`` is derived, never supplied, so a report cannot claim a frame is
    absent while listing it as produced. Frames compare on
    :attr:`~redsun_aht.contracts.frames.FrameIdentity.identity_key`, which
    ignores the producer's readout sequence: a requested frame has none yet.
    """

    requested: tuple[FrameIdentity, ...]
    produced: tuple[FrameIdentity, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a report whose two halves contradict each other."""
        if not isinstance(self.requested, tuple) or not self.requested:
            raise ValueError("a partial run needs the requested frames")
        if any(not isinstance(item, FrameIdentity) for item in self.requested):
            raise ValueError("requested frames must be frame identities")
        if not isinstance(self.produced, tuple):
            raise ValueError("produced frames must be a tuple")
        if any(not isinstance(item, FrameIdentity) for item in self.produced):
            raise ValueError("produced frames must be frame identities")
        requested_keys = [item.identity_key for item in self.requested]
        if len(set(requested_keys)) != len(requested_keys):
            raise ValueError("requested frames must be unique")
        produced_keys = [item.identity_key for item in self.produced]
        if len(set(produced_keys)) != len(produced_keys):
            raise ValueError("produced frames must be unique")
        unexpected = [key for key in produced_keys if key not in set(requested_keys)]
        if unexpected:
            raise ValueError(f"produced frames outside the request: {unexpected}")

    @property
    def missing(self) -> tuple[FrameIdentity, ...]:
        """Return the requested frames that were never produced, in request order."""
        produced = {item.identity_key for item in self.produced}
        return tuple(
            item for item in self.requested if item.identity_key not in produced
        )

    @property
    def is_complete(self) -> bool:
        """Whether every requested frame was produced."""
        return not self.missing

    @property
    def produced_count(self) -> int:
        """Return how many requested frames were produced."""
        return len(self.produced)

    @property
    def requested_count(self) -> int:
        """Return how many frames the plan asked for."""
        return len(self.requested)

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document naming both halves."""
        return {
            "requested": [item.as_payload() for item in self.requested],
            "produced": [item.as_payload() for item in self.produced],
            "missing": [item.as_payload() for item in self.missing],
        }


@dataclass(frozen=True, slots=True)
class PlanOutcome:
    """How a plan run ended, with a named reason and the exact frame accounting.

    Parameters
    ----------
    run
        The run the outcome belongs to.
    status
        Whether the run completed, completed partially, or was aborted.
    reason
        Why an aborted run stopped. It is absent for a complete or partial run,
        which ended by itself.
    partial
        The frame accounting. A partial run always carries it; a complete run
        never does; an aborted run carries it when any frame was requested.
    detail
        One line for the operator, for a log rather than for a decision.
    finished_monotonic_ns
        When the plan returned control, on a monotonic clock.
    """

    run: RunIdentity
    status: PlanStatus
    reason: PlanAbortReason | None = None
    partial: PartialRun | None = None
    detail: str = ""
    finished_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        """Refuse an outcome that hides what happened."""
        if not isinstance(self.run, RunIdentity):
            raise ValueError("plan outcome needs a run identity")
        if not isinstance(self.status, PlanStatus):
            raise ValueError("plan outcome needs a plan status")
        if self.reason is not None and not isinstance(self.reason, PlanAbortReason):
            raise ValueError("plan outcome reason must be a PlanAbortReason")
        if self.partial is not None and not isinstance(self.partial, PartialRun):
            raise ValueError("plan outcome partial report must be a PartialRun")
        if not isinstance(self.detail, str):
            raise ValueError("plan outcome detail must be text")
        if self.finished_monotonic_ns is not None and (
            type(self.finished_monotonic_ns) is not int
            or self.finished_monotonic_ns < 0
        ):
            raise ValueError("plan outcome time must be non-negative")
        if self.status is PlanStatus.COMPLETE:
            if self.reason is not None:
                raise ValueError("a complete plan outcome carries no abort reason")
            if self.partial is not None:
                raise ValueError("a complete plan outcome carries no partial report")
        elif self.status is PlanStatus.PARTIAL:
            if self.reason is not None:
                raise ValueError("a partial plan outcome carries no abort reason")
            if self.partial is None:
                raise ValueError("a partial plan outcome needs its partial report")
            if self.partial.is_complete:
                raise ValueError(
                    "a partial plan outcome must name at least one missing frame"
                )
        elif self.reason is None:
            raise ValueError("an aborted plan outcome must name its reason")

    @property
    def missing(self) -> tuple[FrameIdentity, ...]:
        """Return the requested frames that were not produced."""
        return () if self.partial is None else self.partial.missing

    @property
    def produced(self) -> tuple[FrameIdentity, ...]:
        """Return the requested frames that were produced."""
        return () if self.partial is None else self.partial.produced

    @property
    def is_complete(self) -> bool:
        """Whether the run produced every requested frame."""
        return self.status is PlanStatus.COMPLETE

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {
            "run": self.run.as_payload(),
            "status": str(self.status),
            "reason": None if self.reason is None else str(self.reason),
            "partial": None if self.partial is None else self.partial.as_payload(),
            "detail": self.detail,
            "finished_monotonic_ns": self.finished_monotonic_ns,
        }


__all__ = [
    "AxisGrid",
    "DeviceRole",
    "PartialRun",
    "PlanAbortReason",
    "PlanOutcome",
    "PlanRequest",
    "PlanStatus",
    "ScanPoint",
]
