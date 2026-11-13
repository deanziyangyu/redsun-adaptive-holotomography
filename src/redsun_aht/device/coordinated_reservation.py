"""Cooperative reservation grouping for the experimental motion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from redsun_aht.device.motion_coordinator import CompositeMotionCoordinator


class GroupState(StrEnum):
    """Local knowledge, not proof of a remote service's state."""

    NEW = "new"
    ACQUIRING = "acquiring"
    HELD = "held"
    RELEASING = "releasing"
    RELEASED = "released"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class ReservationResource:
    """Named asynchronous reservation operations supplied by a service client."""

    name: str
    acquire: Callable[[], Awaitable[None]]
    release: Callable[[], Awaitable[None]]


class CoordinatedReservation:
    """Acquire local coordinator then services in fixed order; release in reverse.

    This is not atomic across services. An exception or cancellation after a
    remote operation starts retains local ownership and reports uncertainty;
    the caller must reconcile it instead of blindly retrying or releasing.
    """

    def __init__(
        self,
        coordinator: CompositeMotionCoordinator,
        resources: Sequence[ReservationResource],
    ) -> None:
        names = [item.name for item in resources]
        if (
            not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("unique named reservation resources required")
        self.coordinator = coordinator
        self.resources = tuple(resources)
        self.state = GroupState.NEW
        self.token: str | None = None
        self.acquired: list[str] = []
        self.attempting: str | None = None

    async def acquire(self) -> str:
        """Acquire each resource once; retain state if an outcome is unknown."""
        if self.state is not GroupState.NEW:
            raise RuntimeError("reservation group is not new")
        self.token = self.coordinator.reserve()
        self.state = GroupState.ACQUIRING
        try:
            for resource in self.resources:
                self.attempting = resource.name
                await resource.acquire()
                self.acquired.append(resource.name)
                self.attempting = None
        except BaseException:
            self.state = GroupState.UNCERTAIN
            raise
        self.state = GroupState.HELD
        return self.token

    async def release(self) -> None:
        """Release remote resources in reverse, then release the coordinator."""
        if self.state is not GroupState.HELD or self.token is None:
            raise RuntimeError("reservation group not certainly held")
        self.state = GroupState.RELEASING
        try:
            for resource in reversed(self.resources):
                self.attempting = resource.name
                await resource.release()
                self.acquired.remove(resource.name)
                self.attempting = None
            self.coordinator.release(self.token)
        except BaseException:
            self.state = GroupState.UNCERTAIN
            raise
        self.token = None
        self.state = GroupState.RELEASED
