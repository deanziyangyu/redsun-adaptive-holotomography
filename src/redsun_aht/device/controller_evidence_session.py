"""Experimental headless session: coarse readback plus MCU fast-Z reports.

This is not a camera acquisition, a cross-device lease, or a physical stop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Protocol,
    cast,
    runtime_checkable,
)

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from redsun import AsDevice, AsPresenter, AsService, Declare, DevicesOf, Launch, Session

from redsun_aht.device.pva_esp300_readonly import PvaEsp300ReadOnly  # noqa: TC001
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu  # noqa: TC001
from redsun_aht.device.unified_mcu import AngularStack, StackProgress

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from redsun.registry import PlanEntry

    from redsun_aht.device.pva_esp300_readonly import Esp300Observation

logger = logging.getLogger(__name__)


@runtime_checkable
class CoarseObservation(Protocol):
    """Read-only coarse axis capability."""

    async def read_observation(self, *, refresh: bool = True) -> Esp300Observation:
        """Return a controller-reported observation."""
        ...


@runtime_checkable
class FastStack(Protocol):
    """One finite MCU stack with explicit host readiness."""

    async def execute(self, stack: AngularStack, *, ready: bool) -> StackProgress:
        """Submit one explicitly admitted stack."""
        ...

    async def abort_stack(self, run_id: int) -> None:
        """Request a run-scoped abort."""
        ...


class _Evidence:
    name = "controller_report"

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def read(self) -> dict[str, dict[str, Any]]:
        """Return evidence with a Bluesky host timestamp."""
        now = time.time()
        return {
            key: {"value": value, "timestamp": now} for key, value in self.data.items()
        }

    def describe(self) -> dict[str, dict[str, Any]]:
        """Describe the serialized report without assigning camera provenance."""
        return {
            key: {"source": "ARH:PVA:controller-report", "dtype": "string", "shape": []}
            for key in self.data
        }


def controller_stack_plan(
    coarse: CoarseObservation,
    fast: FastStack,
    stack: AngularStack,
    *,
    ready: bool,
    timeout_s: float = 20.0,
) -> Generator[Any, Any, None]:
    """Read coarse state, submit one stack, and emit controller reports only."""
    if ready is not True:
        raise ValueError(
            "explicit camera and illumination readiness assertion required"
        )
    if not 0 < timeout_s <= 300:
        raise ValueError("invalid timeout")

    @bpp.run_decorator(
        md={
            "stack_id": stack.run_id,
            "evidence": "controller_reported",
            "camera_frames": False,
        }
    )  # type: ignore[untyped-decorator]
    def run() -> Generator[Any, Any, None]:
        coarse_task = yield from bps.wait_for(
            [lambda: asyncio.wait_for(coarse.read_observation(), timeout=timeout_s)]
        )
        observation = coarse_task[0].result()
        source = _Evidence({"coarse_snapshot_json": json.dumps(asdict(observation))})
        yield from bps.create(name="coarse_observation")
        yield from bps.read(source)
        yield from bps.save()
        try:
            tasks = yield from bps.wait_for(
                [
                    lambda: asyncio.wait_for(
                        fast.execute(stack, ready=True), timeout=timeout_s
                    )
                ]
            )
            progress = tasks[0].result()
        except Exception:
            try:
                abort = yield from bps.wait_for(
                    [
                        lambda: asyncio.wait_for(
                            fast.abort_stack(stack.run_id), timeout=2
                        )
                    ]
                )
                abort[0].result()
            except Exception:
                logger.exception("Abort was not confirmed; physical stop is unverified")
            raise
        source = _Evidence({"fast_stack_json": json.dumps(asdict(progress))})
        yield from bps.create(name="fast_stack_report")
        yield from bps.read(source)
        yield from bps.save()

    yield from run()


class ControllerEvidencePresenter:
    """Expose a deliberate controller-report plan, not a camera acquisition."""

    def __init__(
        self,
        name: str,
        *,
        coarse: DevicesOf[CoarseObservation],
        fast: DevicesOf[FastStack],
    ) -> None:
        self.name = name
        self.coarse = dict(coarse)
        self.fast = dict(fast)

    def plan_map(self) -> Mapping[str, PlanEntry]:
        """Offer the plan only with one device of each distinct role."""
        return (
            {"controller_stack_report": {"plan": self.run_stack}}
            if len(self.coarse) == len(self.fast) == 1
            else {}
        )

    def run_stack(
        self,
        run_id: int,
        first: int,
        step: int,
        count: int,
        lower: int,
        upper: int,
        *,
        ready: bool,
        timeout_s: float = 20,
    ) -> Generator[Any, Any, None]:
        """Run a caller-specified angular stack after reading coarse XYZ."""
        if len(self.coarse) != 1 or len(self.fast) != 1:
            raise RuntimeError("expected one device for each role")
        stack = AngularStack(run_id, first, step, count, lower, upper)
        yield from controller_stack_plan(
            next(iter(self.coarse.values())),
            next(iter(self.fast.values())),
            stack,
            ready=ready,
            timeout_s=timeout_s,
        )


class ControllerEvidenceSession(Session):
    """Connect separately started coarse and fast services; no GUI required."""

    config: ClassVar[dict[str, object]] = {
        "session": "controller-evidence",
        "strict": True,
    }
    coarse: Annotated[AsDevice[PvaEsp300ReadOnly], Declare(prefix="COARSE:")]
    fast: Annotated[AsDevice[PvaUnifiedMcu], Declare(prefix="FASTZ:")]
    presenter: AsPresenter[ControllerEvidencePresenter]


def managed_controller_session(
    coarse_port: str,
    fast_port: str,
    *,
    coarse_module: str = "redsun_aht.services.esp300_readonly",
    fast_module: str = "redsun_aht.services.unified_mcu",
) -> ControllerEvidenceSession:
    """Declare two explicit-port services owned by the returned unbuilt session.

    Idle shutdown closes services; this does not qualify stopping active motion.
    """
    for port in (coarse_port, fast_port):
        if not isinstance(port, str) or not port.strip() or "<" in port or ">" in port:
            raise ValueError("explicit serial endpoint names are required")
    if coarse_port.strip().casefold() == fast_port.strip().casefold():
        raise ValueError("coarse and fast services require separate endpoints")
    declarations = {
        "coarse_service": Annotated[
            AsService,
            Launch(
                coarse_module,
                prefix="COARSE:",
                args=("--prefix", "COARSE:", "--port", coarse_port),
                stop_timeout=30,
            ),
        ],
        "fast_service": Annotated[
            AsService,
            Launch(
                fast_module,
                prefix="FASTZ:",
                args=("--prefix", "FASTZ:", "--port", fast_port),
                stop_timeout=30,
            ),
        ],
    }
    session_type = type(
        "ManagedControllerEvidenceSession",
        (ControllerEvidenceSession,),
        {"__annotations__": declarations},
    )
    return cast(
        "ControllerEvidenceSession",
        session_type({"services": {"transport": "pv-access"}}),
    )
