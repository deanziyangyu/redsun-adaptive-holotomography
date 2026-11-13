"""Finite simulated MCU stack and frame reconciliation.

This model does not operate hardware or model timing. It is intentionally not
an implementation of a Redsun device, service, or Bluesky plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class Axis:
    """Synthetic axis bounds and command increment, in micrometres."""

    name: str
    lower_um: float
    upper_um: float
    increment_um: float

    def __post_init__(self) -> None:
        """Validate the synthetic configuration."""
        if not self.name or not all(
            map(math.isfinite, (self.lower_um, self.upper_um, self.increment_um))
        ):
            raise ValueError("axis name and finite bounds/increment required")
        if self.lower_um > self.upper_um or self.increment_um <= 0:
            raise ValueError("invalid axis limits or increment")

    def check(self, value: float) -> float:
        """Refuse unrepresentable or out-of-range absolute positions."""
        if not math.isfinite(value) or not self.lower_um <= value <= self.upper_um:
            raise ValueError("target outside simulated limits")
        units = value / self.increment_um
        if not math.isclose(units, round(units), rel_tol=0, abs_tol=1e-9):
            raise ValueError("target is not on the simulated command grid")
        return value


@dataclass(frozen=True, slots=True)
class Stack:
    """A finite committed set of simulated targets and one pattern per plane."""

    run_id: str
    targets_um: tuple[float, ...]
    pattern: int

    def __post_init__(self) -> None:
        """Validate sequence identity, length, and pattern."""
        if not self.run_id or not self.targets_um or len(self.targets_um) > 1024:
            raise ValueError("run identity and 1..1024 targets required")
        if not 1 <= self.pattern <= 5:
            raise ValueError("invalid simulated illumination pattern")


@dataclass(frozen=True, slots=True)
class Edge:
    """A simulated emitted trigger with target and sequence identity."""

    run_id: str
    plane: int
    target_um: float
    pattern: int


@dataclass(frozen=True, slots=True)
class Frame:
    """Simulated camera frame bearing the trigger identity."""

    run_id: str
    plane: int


class State(StrEnum):
    """The model lifecycle, not a physical device status."""

    EMPTY = "empty"
    COMMITTED = "committed"
    ARMED = "armed"
    RUNNING = "running"
    COMPLETE = "complete"
    ABORTED = "aborted"


class SimulatedMcu:
    """One-shot local sequencer with exclusive simulated resource ownership."""

    def __init__(self, fast_z: Axis, *, stage: SimulatedStage | None = None) -> None:
        if stage is not None and stage.axes.get(fast_z.name) != fast_z:
            raise ValueError("fast-Z must match an axis in the simulated stage")
        self.fast_z = fast_z
        self.stage = stage
        self.state = State.EMPTY
        self.owner: str | None = None
        self.stack: Stack | None = None
        self.edges: list[Edge] = []
        self.start_count = 0

    def reserve(self, owner: str) -> None:
        """Reserve this simulated sequencer for one owner."""
        if not owner or self.owner is not None:
            raise RuntimeError("resources already reserved or owner missing")
        if self.stage is not None:
            self.stage.reserve(owner)
        if self.state in (State.COMPLETE, State.ABORTED):
            self.state = State.EMPTY
            self.stack = None
            self.edges = []
        self.owner = owner

    def commit(self, owner: str, stack: Stack) -> None:
        """Validate every target before accepting a stack."""
        if self.owner != owner or self.state is not State.EMPTY:
            raise RuntimeError("commit refused")
        for target in stack.targets_um:
            self.fast_z.check(target)
        self.stack = stack
        self.state = State.COMMITTED

    def arm(self, owner: str, *, camera_ready: bool, referenced: bool) -> None:
        """Arm only with simulated camera readiness and reference."""
        if (
            self.owner != owner
            or self.state is not State.COMMITTED
            or not camera_ready
            or not referenced
        ):
            raise RuntimeError("arm refused")
        self.state = State.ARMED

    def begin(self, owner: str, run_id: str) -> None:
        """Accept one start for the currently armed transaction."""
        if (
            self.owner != owner
            or self.state is not State.ARMED
            or self.stack is None
            or self.stack.run_id != run_id
        ):
            raise RuntimeError("start refused")
        self.start_count += 1
        self.state = State.RUNNING

    def step(
        self,
        owner: str,
        *,
        camera_ready: bool = True,
        on_edge: Callable[[Edge], None] | None = None,
    ) -> Edge:
        """Advance one internal fake plane; this is not a host command."""
        if self.owner != owner or self.state is not State.RUNNING or self.stack is None:
            raise RuntimeError("step refused")
        if not camera_ready:
            self.state = State.ABORTED
            raise RuntimeError("simulated camera not ready")
        index = len(self.edges)
        target = self.stack.targets_um[index]
        try:
            if self.stage is not None:
                self.stage.move({self.fast_z.name: target}, owner=owner)
            edge = Edge(self.stack.run_id, index, target, self.stack.pattern)
            self.edges.append(edge)
            if on_edge is not None:
                on_edge(edge)
        except Exception:
            self.state = State.ABORTED
            raise
        if len(self.edges) == len(self.stack.targets_um):
            self.state = State.COMPLETE
        return edge

    def start(
        self,
        owner: str,
        run_id: str,
        *,
        fail_before_plane: int | None = None,
        on_edge: Callable[[Edge], None] | None = None,
    ) -> tuple[Edge, ...]:
        """Execute locally after one start; failure preserves emitted edges."""
        self.begin(owner, run_id)
        while self.state is State.RUNNING:
            if len(self.edges) == fail_before_plane:
                self.state = State.ABORTED
                break
            self.step(owner, on_edge=on_edge)
        return tuple(self.edges)

    def cancel(self, owner: str) -> None:
        """End an uncompleted synthetic sequence before releasing ownership."""
        if self.owner != owner:
            raise RuntimeError("cancel refused")
        if self.state is not State.COMPLETE:
            self.state = State.ABORTED

    def release(self, owner: str) -> None:
        """Release ownership after simulated completion or abort."""
        if self.owner != owner or self.state not in (State.COMPLETE, State.ABORTED):
            raise RuntimeError("release refused")
        if self.stage is not None:
            self.stage.release(owner)
        self.owner = None


class SimulatedTriggeredCamera:
    """Independently count received synthetic triggers and produced frames.

    Frame numbers are camera-local counters, not plane IDs or physical timing.
    The optional drop models one trigger that produced no camera frame.
    """

    def __init__(self, *, drop_trigger: int | None = None) -> None:
        self.drop_trigger = drop_trigger
        self.armed = False
        self.triggers = 0
        self.frames: list[int] = []

    def arm(self) -> None:
        """Arm the synthetic camera before accepting trigger callbacks."""
        if self.armed:
            raise RuntimeError("camera already armed")
        self.armed = True

    def receive(self, edge: Edge) -> None:
        """React to a fake electrical edge without copying its plane identity."""
        del edge
        if not self.armed:
            raise RuntimeError("camera not armed")
        trigger = self.triggers
        self.triggers += 1
        if trigger != self.drop_trigger:
            self.frames.append(len(self.frames))


def reconcile(
    edges: tuple[Edge, ...], frames: tuple[Frame, ...], expected: Stack
) -> tuple[tuple[Edge, Frame], ...]:
    """Match only exact ordered one-to-one simulated evidence; reject ambiguity."""
    if len(edges) != len(expected.targets_um) or len(frames) != len(edges):
        raise ValueError("incomplete or extra trigger/frame")
    paired = []
    for index, (edge, frame) in enumerate(zip(edges, frames, strict=True)):
        if (edge.run_id, edge.plane, edge.target_um, edge.pattern) != (
            expected.run_id,
            index,
            expected.targets_um[index],
            expected.pattern,
        ) or (frame.run_id, frame.plane) != (expected.run_id, index):
            raise ValueError("trigger/frame identity or ordering mismatch")
        paired.append((edge, frame))
    return tuple(paired)


class SimulatedStage:
    """Synthetic composite axes; prevalidate all targets before movement."""

    def __init__(self, axes: tuple[Axis, ...]) -> None:
        self.axes = {axis.name: axis for axis in axes}
        if not axes or len(self.axes) != len(axes):
            raise ValueError("unique axes required")
        self.positions = {name: axis.check(0.0) for name, axis in self.axes.items()}
        self.owner: str | None = None

    def reserve(self, owner: str) -> None:
        """Reserve the whole synthetic composite for one in-process owner."""
        if not owner or self.owner is not None:
            raise RuntimeError("simulated stage already reserved")
        self.owner = owner

    def release(self, owner: str) -> None:
        """Release a matching in-process reservation."""
        if self.owner != owner or not owner:
            raise RuntimeError("simulated stage release refused")
        self.owner = None

    def move(
        self, targets: dict[str, float], *, owner: str | None = None
    ) -> dict[str, float]:
        """Apply validated positions only when the synthetic stage is available."""
        if self.owner != owner:
            raise RuntimeError("simulated stage owned by another caller")
        for name, value in targets.items():
            if name not in self.axes:
                raise ValueError("unknown simulated axis")
            self.axes[name].check(value)
        self.positions.update(targets)
        return dict(self.positions)


def camera_frames(edges: tuple[Edge, ...]) -> tuple[Frame, ...]:
    """Generate ideal fake frame identities; makes no timing or optical claim."""
    return tuple(Frame(edge.run_id, edge.plane) for edge in edges)
