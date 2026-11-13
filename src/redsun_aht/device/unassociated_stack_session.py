"""Experimental headless Redsun session for unassociated stack evidence.

Services are started separately; this session does not enforce ownership,
configure triggering, qualify camera readiness, or establish a physical stop.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Protocol,
    cast,
    runtime_checkable,
)

from ophyd_async.core import SignalR, SignalRW  # noqa: TC002
from redsun import AsDevice, AsPresenter, Declare, DevicesOf, Session

from redsun_aht.device.controller_evidence_session import FastStack
from redsun_aht.device.mmcore.camera import MMCamera  # noqa: TC001
from redsun_aht.device.pva_capture_evidence import PvaCaptureEvidence
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu  # noqa: TC001
from redsun_aht.device.unassociated_stack_plan import unassociated_stack_plan
from redsun_aht.device.unified_mcu import AngularStack

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from redsun.registry import PlanEntry


@runtime_checkable
class CaptureDevice(Protocol):
    """Signals needed by the unassociated capture adapter."""

    acquire: SignalRW[bool]
    capture: SignalRW[bool]
    sequence_started: SignalR[bool]
    file_path: SignalRW[str]
    num_capture: SignalRW[int]
    record_frame_timing: SignalRW[bool]
    captured: SignalR[int]


@runtime_checkable
class ReservedFastStack(FastStack, Protocol):
    """Fast stack with a cooperative service reservation."""

    async def reserve(self) -> None:
        """Acquire the cooperative MCU reservation."""
        ...

    async def release_reservation(self) -> None:
        """Release the cooperative MCU reservation."""
        ...


class UnassociatedStackPresenter:
    """Expose the plan when one camera and one MCU service are present."""

    def __init__(
        self,
        name: str,
        *,
        cameras: DevicesOf[CaptureDevice],
        stacks: DevicesOf[ReservedFastStack],
    ) -> None:
        self.name = name
        self.cameras = dict(cameras)
        self.stacks = dict(stacks)

    def plan_map(self) -> Mapping[str, PlanEntry]:
        """Return the plan only for an unambiguous pair of devices."""
        if len(self.cameras) != 1 or len(self.stacks) != 1:
            return {}
        return {"unassociated_mcu_stack": {"plan": self.run_stack}}

    def run_stack(
        self,
        run_id: int,
        first: int,
        step: int,
        count: int,
        lower: int,
        upper: int,
        store_uri: str,
        *,
        resources_exclusive: bool,
        external_trigger_configured: bool,
        illumination_ready: bool,
        trigger_mask: int = 1,
        timeout_s: float = 20,
    ) -> Generator[Any, Any, None]:
        """Submit one explicitly admitted stack and retain unassociated evidence."""
        if len(self.cameras) != 1 or len(self.stacks) != 1:
            raise RuntimeError("expected one camera and one MCU service")
        camera = PvaCaptureEvidence(
            cast("MMCamera", next(iter(self.cameras.values()))), timeout_s=timeout_s
        )
        stack = AngularStack(run_id, first, step, count, lower, upper, trigger_mask)
        yield from unassociated_stack_plan(
            next(iter(self.stacks.values())),
            camera,
            stack,
            store_uri,
            resources_exclusive=resources_exclusive,
            external_trigger_configured=external_trigger_configured,
            illumination_ready=illumination_ready,
            timeout_s=timeout_s,
        )


class UnassociatedStackSession(Session):
    """Connect externally started camera and MCU PVA services without a GUI."""

    config: ClassVar[dict[str, object]] = {
        "session": "unassociated-mcu-stack",
        "strict": True,
    }
    camera: Annotated[AsDevice[MMCamera], Declare(prefix="CAMERA:")]
    mcu: Annotated[AsDevice[PvaUnifiedMcu], Declare(prefix="FASTZ:")]
    presenter: AsPresenter[UnassociatedStackPresenter]
