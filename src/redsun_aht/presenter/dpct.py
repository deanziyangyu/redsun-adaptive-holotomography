"""The four-pattern DPCT plan as a session offers it.

Modelled on ``redsun_searching_is_imaging/presenter/detector.py`` at donor
revision ``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed;
copyright the Redsun contributors. The plan itself is not the donor's:
:func:`redsun_aht.plans.dpct_software.software_dpct_plan` is called with the devices
the session declares, and this module only decides which devices those are and
what the operator is told afterwards.

The illumination service is a session question, like a camera. A profile that
declares an ``arduino-led`` device is what makes a run possible; one that does
not still builds this presenter, which then offers no plan and names the missing
role instead of refusing to start. The panel is the installed 91-LED v1 layout,
so a pattern is named by its id (``1``..``4`` are the four semicircles) and the
command the service writes is ``10 + id``.

The plan opens its own run, so it is launched on its own and never from inside
another run; the acquisition page's Run button does exactly that, which is why
this presenter offers one parameterless plan. Two launches are independent,
because :func:`new_run_id` mints a fresh store key for each one.

The stage is optional: the four-pattern sequence is the same without one, and a
frame only records a position when the session declared a stage to read.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable
from uuid import uuid4

from bluesky.utils import MsgGenerator  # noqa: TC002 - the plan's return annotation
from psygnal import Signal
from redsun import DevicesOf  # noqa: TC002 - the constructor annotation resolves
from redsun.log import Loggable

from redsun_aht.contracts import (
    PlanAbortReason,
    PlanOutcome,
    PlanStatus,
    RunIdentity,
)
from redsun_aht.device.arduino.illumination import ArduinoIllumination
from redsun_aht.plans.dpct import (  # noqa: TC001 - Redsun resolves DevicesOf annotations
    DpctDetector,
    MeasuredStage,
)
from redsun_aht.plans.dpct_software import SoftwareDpctDetector, software_dpct_plan
from redsun_aht.presenter.acquisition import CAMERA_ROLE_NOTICE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ophyd_async.core import SignalR, SignalRW, SignalW
    from redsun.registry import PlanEntry

    from redsun_aht.device.arduino.device import ArduinoLedDevice

__all__ = [
    "DPCT_PLAN",
    "ILLUMINATION_ROLE_NOTICE",
    "DpctPresenter",
    "IlluminationDevice",
    "describe_outcome",
    "new_run_id",
]

DPCT_PLAN: Final = "dpct"
"""The plan name this presenter offers, and the name its runs record."""

ILLUMINATION_ROLE_NOTICE: Final = (
    "no illumination declared in this session - declare an arduino-led device in "
    "the session profile"
)
"""What the presenter says when the session declares no illumination device."""

_NOT_RUN_PURPOSE: Final = "four-pattern DPCT (not run)"
"""Purpose recorded by a run that stopped before a device could be used."""


@runtime_checkable
class IlluminationDevice(Protocol):
    """The Arduino LED device surface a DPCT run drives.

    It is the part of :class:`~redsun_aht.device.arduino.device.ArduinoLedDevice`
    that :class:`~redsun_aht.device.arduino.illumination.ArduinoIllumination`
    reads and writes, declared here because a session question is asked with a
    protocol: ``DevicesOf`` takes a protocol rather than a class, and this
    module does not need the concrete device to name what it drives.
    """

    pattern: SignalRW[int]
    """The pattern select; the camera is read separately after this write."""

    all_off: SignalW[bool]
    """The blanking command, which triggers no frame."""

    open_shutter: SignalW[bool]
    """The command clearing the shutter-closed flag, which triggers no frame."""

    last_command: SignalR[str]
    """The two-digit command most recently attempted."""

    last_outcome: SignalR[str]
    """``sent``, ``echoed``, ``uncertain`` or ``failed``; never optical evidence."""

    last_detail: SignalR[str]
    """Diagnostics for the most recent command attempt."""


def new_run_id(*, now: datetime | None = None) -> str:
    """Return the store key of one DPCT run.

    Parameters
    ----------
    now : datetime | None
        The moment to stamp the id with; ``None`` uses the current UTC time. It
        exists so a test can pin the stamp.

    Returns
    -------
    str
        ``dpct-<UTC stamp>-<random token>``: a whitespace-free token that names
        one run, fits
        :data:`~redsun_aht.contracts.MAX_RUN_ID_BYTES` and stays unique for two
        runs started in the same second.
    """
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return f"{DPCT_PLAN}-{moment:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def describe_outcome(outcome: PlanOutcome) -> str:
    """Return the one-line report of a finished DPCT run.

    A complete run says so with the plan's own summary. A run that stopped early
    names what it produced beside the ordinals and pattern ids that never
    arrived, because the operator has to know which illumination is missing
    rather than only that the run was short.

    Parameters
    ----------
    outcome : PlanOutcome
        What :func:`~redsun_aht.plans.dpct.dpct_plan` returned.

    Returns
    -------
    str
        The report, for a log line or a page. A pattern is named by its id, so
        ``1``..``4`` are the four semicircles and the command the panel is sent
        is ``10 + id``.
    """
    run_id = outcome.run.run_id
    if outcome.status is PlanStatus.COMPLETE:
        summary = f"dpct run {run_id} complete"
        return f"{summary}: {outcome.detail}" if outcome.detail else summary
    partial = outcome.partial
    requested = 0 if partial is None else partial.requested_count
    produced = len(outcome.produced)
    head = f"dpct run {run_id} {outcome.status}"
    if outcome.reason is not None:
        head = f"{head} ({outcome.reason})"
    accounting = f"{produced} of {requested} frames produced"
    missing = outcome.missing
    if missing:
        ordinals = _numbers(sorted({frame.plane_ordinal for frame in missing}))
        patterns = _numbers(sorted({frame.pattern_id for frame in missing}))
        accounting = f"{accounting}; missing ordinals {ordinals} (patterns {patterns})"
    report = f"{head}: {accounting}"
    return f"{report}; {outcome.detail}" if outcome.detail else report


def _numbers(values: Sequence[int]) -> str:
    """Return integers as a comma-separated list, or ``none`` when empty."""
    return ", ".join(str(value) for value in values) if values else "none"


class DpctPresenter(Loggable):
    """Offer the four-pattern DPCT plan over the session's cameras and panel.

    Parameters
    ----------
    name : str
        The presenter's name in its session, used in log records.
    detectors : DevicesOf[DpctDetector]
        The cameras the run expects a frame from; the session fills it with
        devices that can stage and provide a fresh free-running frame.
    stage : DevicesOf[MeasuredStage]
        The stages that can report a position. A run records the first by name,
        and runs without one when the session declares none.
    illumination : DevicesOf[IlluminationDevice]
        The Arduino LED devices; the first by name is the panel a run drives.

    Notes
    -----
    A session with no camera or no illumination device still builds this
    presenter: :meth:`missing_notice` names what is absent and :meth:`plan_map`
    offers nothing, so a profile that declares only part of the bundle is a
    readable absence rather than a refusal to start.
    """

    sig_outcome = Signal(object)
    """Emitted with the :class:`~redsun_aht.contracts.PlanOutcome` of a run."""

    def __init__(
        self,
        name: str,
        *,
        detectors: DevicesOf[DpctDetector],
        stage: DevicesOf[MeasuredStage],
        illumination: DevicesOf[IlluminationDevice],
    ) -> None:
        self.name = name
        # sorted, so two sessions that declare the same devices record them in
        # one order and a run is reproducible from its start document
        self._detectors = {key: detectors[key] for key in sorted(detectors)}
        self._stages = {key: stage[key] for key in sorted(stage)}
        self._illuminations = {key: illumination[key] for key in sorted(illumination)}

    def missing_notice(self) -> str | None:
        """Return the sentence naming what a run lacks, or ``None`` when it can run.

        Returns
        -------
        str | None
            One sentence per missing role, joined with ``"; "``: the camera
            role first, then the illumination role. ``None`` when the session
            declares both, which is the only state in which a plan is offered.
        """
        missing: list[str] = []
        if not self._detectors:
            missing.append(CAMERA_ROLE_NOTICE)
        if not self._illuminations:
            missing.append(ILLUMINATION_ROLE_NOTICE)
        return "; ".join(missing) if missing else None

    def plan_map(self) -> Mapping[str, PlanEntry]:
        """Return the ``dpct`` plan, or nothing when a device is missing.

        Returns
        -------
        Mapping[str, PlanEntry]
            ``{"dpct": {"plan": self.dpct}}``, or an empty mapping when
            :meth:`missing_notice` names something. An empty mapping keeps the
            plan off the acquisition page instead of offering one that cannot
            run, and the notice is logged so the absence is visible.
        """
        notice = self.missing_notice()
        if notice is not None:
            self.logger.warning(f"Not offering {DPCT_PLAN!r}: {notice}")
            return {}
        entry: PlanEntry = {"plan": self.dpct}
        return {DPCT_PLAN: entry}

    def dpct(self) -> MsgGenerator[PlanOutcome]:
        """Acquire one four-pattern DPCT frame set and report its outcome.

        The plan opens its own run, so it is launched on its own and never from
        inside another run. One pattern command is written per expected frame,
        the camera supplies a fresh frame after each pattern, and the run is
        closed with the panel blanked.

        Returns
        -------
        PlanOutcome
            ``COMPLETE`` when every requested frame arrived, otherwise
            ``ABORTED`` with the reason and a report naming the missing frames.
            A session that has no camera or no illumination device gets an
            aborted outcome carrying the notice, rather than an exception.

        Notes
        -----
        The illumination device is wrapped in
        :class:`~redsun_aht.device.arduino.illumination.ArduinoIllumination`
        here, so the plan sees the contract it declares and this presenter is
        the only place the concrete device is named.
        """
        run_id = new_run_id()
        notice = self.missing_notice()
        if notice is not None:
            outcome = PlanOutcome(
                run=RunIdentity(
                    run_id=run_id,
                    plan_name=DPCT_PLAN,
                    purpose=_NOT_RUN_PURPOSE,
                    scan_id=0,
                ),
                status=PlanStatus.ABORTED,
                reason=PlanAbortReason.DEVICE_FAULT,
                detail=notice,
            )
            self.report(outcome)
            return outcome
        stage = next(iter(self._stages.values()), None)
        device = next(iter(self._illuminations.values()))
        outcome = yield from software_dpct_plan(
            cast("tuple[SoftwareDpctDetector, ...]", tuple(self._detectors.values())),
            stage,
            ArduinoIllumination(cast("ArduinoLedDevice", device)),
            run_id=run_id,
        )
        self.report(outcome)
        return outcome

    def report(self, outcome: PlanOutcome) -> str:
        """Log a finished run's outcome and announce it on :attr:`sig_outcome`.

        Parameters
        ----------
        outcome : PlanOutcome
            What the plan returned.

        Returns
        -------
        str
            The same sentence :func:`describe_outcome` renders, so a caller can
            show it without reading the log.
        """
        notice = describe_outcome(outcome)
        if outcome.status is PlanStatus.COMPLETE:
            self.logger.info(notice)
        else:
            self.logger.warning(notice)
        self.sig_outcome.emit(outcome)
        return notice
