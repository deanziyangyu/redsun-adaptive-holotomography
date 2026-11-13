"""The four-pattern DPCT plan for an externally triggered camera.

This legacy contract assumes firmware that fires the camera TTL when a pattern
command is written, so this plan never triggers the camera. It arms the
detector for an external-trigger capture, blanks and opens the illumination,
then writes one pattern command per expected frame and waits for that frame
before advancing. A second write would take a second frame, so an uncertain
command ends the run instead of being retried.

One run emits this message order:

1. ``open_run`` with the run identity, the panel profile and the pattern
   sequence in the start document;
2. ``stage`` for the detectors, the stage and the illumination device;
3. ``prepare`` with ``TriggerInfo(number_of_events=len(sequence))`` for every
   detector, and no trigger of the plan's own;
4. ``00`` (blank) and then ``01`` (open) on the illumination;
5. for each pattern step, in sequence order: one pattern command, one bounded
   wait per detector for one new frame, then one Event per detector carrying the
   frame identity, the command receipt and the measured position;
6. ``00`` again, then one ``unstage`` per staged device, in reverse order;
7. ``close_run`` and the :class:`~redsun_aht.contracts.PlanOutcome`.

The plan does not kick the detector off, collect it or store pixels. It admits
only a detector that explicitly reports an external-trigger mode and a durable
capture path. Current MMCamera reports free-run/software modes and has no such
path, so it cannot run this plan. A future detector's own writing path must own
the frame data; the Event carries the identity and producer sequence that pair
a stored frame with the pattern which lit it. The plan accounts for every
requested frame and names the frames that did not arrive as missing.

A stopped or aborted run cannot return an outcome, because the RunEngine raises
``RunEngineInterrupted`` instead. The accounting is still recoverable: the start
document lists the requested sequence and every Event names the frame it
produced, so ``requested - produced`` is the same partial report.

The device surface is the frozen contracts plus three narrow protocols declared
here, so a fake, a simulated session and the real PVA-backed devices are all
acceptable without this module importing a device class and without this module
ever issuing a camera trigger.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from bluesky import plan_stubs as bps
from bluesky.protocols import Stageable
from bluesky.utils import FailedStatus, RunEngineControlException
from ophyd_async.core import TriggerInfo

from redsun_aht.contracts import (
    DPCT_PATTERN_IDS,
    AxisSpec,
    CameraDescription,
    CommandReceipt,
    CommandStatus,
    FrameIdentity,
    Illumination,
    IlluminationAction,
    IlluminationCapability,
    PartialRun,
    PlanAbortReason,
    PlanOutcome,
    PlanStatus,
    RunIdentity,
    TriggerMode,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from bluesky.protocols import Reading, Status
    from bluesky.utils import MsgGenerator
    from event_model import DataKey

    from redsun_aht.contracts import StageReading

__all__ = [
    "DEVICE_TIMEOUT_S",
    "DPCT_PATTERN_SEQUENCE",
    "FRAME_POLL_S",
    "FRAME_TIMEOUT_S",
    "STREAM_NAME",
    "UNKNOWN_CLOCK_NS",
    "UNKNOWN_PRODUCER_SEQUENCE",
    "DpctCaptureAdmission",
    "DpctDetector",
    "DpctError",
    "FrameSource",
    "FrameTimeoutError",
    "IlluminationFailed",
    "IlluminationUncertain",
    "MeasuredStage",
    "PatternStep",
    "dpct_plan",
]

logger = logging.getLogger(__name__)

STREAM_NAME: str = "dpct"
"""Name of the Event stream a DPCT run writes; one Event per produced frame."""

FRAME_TIMEOUT_S: float = 5.0
"""Seconds one detector may take to deliver the frame a pattern command started."""

FRAME_POLL_S: float = 0.02
"""Seconds between two reads of a detector's frame counter.

The counter is polled rather than monitored because the PVA camera service does
not hold a monitor on it reliably; the service publishes every 0.02 s, so
polling faster only adds traffic.
"""

DEVICE_TIMEOUT_S: float = 3.0
"""Upper bound on one illumination command or one position read."""

UNKNOWN_CLOCK_NS: int = -1
"""Recorded send time of a command whose service reports no clock."""

UNKNOWN_PRODUCER_SEQUENCE: int = -1
"""Event value for a frame with no provable producer sequence."""

#: The ``DataKey`` dtypes the DPCT Event fields use.
_Dtype = Literal["string", "integer", "number"]

#: Field name and ``DataKey`` dtype of every identity value an Event carries.
_IDENTITY_FIELDS: tuple[tuple[str, _Dtype], ...] = (
    ("dpct-run-id", "string"),
    ("dpct-detector-id", "string"),
    ("dpct-sequence", "integer"),
    ("dpct-plane-ordinal", "integer"),
    ("dpct-pattern-id", "integer"),
    ("dpct-scan-index", "integer"),
    ("dpct-producer-sequence", "integer"),
)

#: Field name and ``DataKey`` dtype of every command receipt value.
_COMMAND_FIELDS: tuple[tuple[str, _Dtype], ...] = (
    ("dpct-command", "string"),
    ("dpct-command-status", "string"),
    ("dpct-command-echo", "string"),
    ("dpct-command-sent-ns", "integer"),
)


class DpctError(RuntimeError):
    """A DPCT run could not continue with the devices it was given."""


class FrameTimeoutError(TimeoutError, DpctError):
    """A detector did not deliver the frame a pattern command started in time."""


class IlluminationUncertain(DpctError):
    """A command's delivery is unknown, so the sequence must not be resumed.

    In the external-trigger contract a pattern command fires the camera, so writing
    it again after an uncertain outcome would take a second frame. The plan
    therefore stops and reports what it did produce.
    """


class IlluminationFailed(DpctError):
    """The illumination transport refused a command without writing it."""


@runtime_checkable
class DpctDetector(Protocol):
    """A detector that an external trigger starts and that a run may arm.

    The protocol is deliberately narrower than a Bluesky flyer. The illumination
    service fires the camera, so a DPCT run never triggers, kicks off or collects
    the detector itself; it stages it, prepares it for the number of external
    triggers, and releases it. The counter it waits on is a separate
    :class:`FrameSource`, so a detector that declares none is refused by name
    rather than being mistaken for a detector that never answers.
    """

    @property
    def name(self) -> str:
        """Return the detector identity a frame is recorded under."""
        ...

    def stage(self) -> Status:
        """Stage the detector for the capture."""
        ...

    def unstage(self) -> Status:
        """Release the detector after the capture."""
        ...

    def prepare(self, value: TriggerInfo) -> Status:
        """Arm the detector for ``value.number_of_events`` external triggers."""
        ...


@runtime_checkable
class FrameSource(Protocol):
    """A readable whose value counts the frames one detector has delivered.

    The installed camera service publishes a counter that rises once per frame,
    so a plan can poll it for the arrival of the frame a pattern command started.
    ``SignalR[int]`` satisfies this; so does any test double with a ``read``.
    """

    def read(self) -> dict[str, Reading[Any]]:
        """Return one reading whose single value is the delivered-frame count."""
        ...


@runtime_checkable
class MeasuredStage(Protocol):
    """The position surface a DPCT frame records.

    It is the part of :class:`redsun_aht.contracts.Stage` this plan needs: the
    declared axes and one reading of where the sample was when the frame was
    taken. The plan does not move the stage, because Z/Y/X scans are a separate
    plan over the same sequence.
    """

    @property
    def axes(self) -> Mapping[str, AxisSpec]:
        """Return the declared axes by name."""
        ...

    async def read_position(self) -> StageReading:
        """Return the current axis positions and their evidence."""
        ...


@dataclass(frozen=True, slots=True)
class PatternStep:
    """One illuminated frame of a DPCT sequence.

    Parameters
    ----------
    pattern_id
        The illumination pattern rendered for this frame. It is one of
        :data:`~redsun_aht.contracts.DPCT_PATTERN_IDS`, so extending a sequence
        never changes what ``11``..``14`` mean.
    label
        Free text naming the step in the start document; it is for a log rather
        than for a decision.
    """

    pattern_id: int
    label: str = ""

    def __post_init__(self) -> None:
        """Refuse a step the installed panel cannot render as a DPCT frame."""
        if type(self.pattern_id) is not int or self.pattern_id not in DPCT_PATTERN_IDS:
            raise ValueError(
                f"DPCT pattern {self.pattern_id!r} is outside {DPCT_PATTERN_IDS}; "
                "the four semicircles are the only patterns this plan acquires"
            )
        if not isinstance(self.label, str):
            raise ValueError("a DPCT pattern step label must be text")


DPCT_PATTERN_SEQUENCE: tuple[PatternStep, ...] = tuple(
    PatternStep(pattern_id=pattern_id, label=f"group{ordinal}")
    for ordinal, pattern_id in enumerate(DPCT_PATTERN_IDS, start=1)
)
"""The editable four-pattern sequence: ``11``, ``12``, ``13`` and ``14``.

A later adaptive plan may pass its own sequence of :class:`PatternStep` values.
The pattern ids stay in :data:`~redsun_aht.contracts.DPCT_PATTERN_IDS`, so one
plan's sequence cannot redefine what a recorded pattern means.
"""


@dataclass(frozen=True, slots=True)
class DpctCaptureAdmission:
    """Evidence supplied by a detector with an armed, durable TTL capture path.

    A counter alone is insufficient: free-running frames can satisfy the frame
    wait without any TTL response, and metadata Events do not store pixels.
    Current MMCamera does not provide this admission and cannot run DPCT.
    """

    detector_id: str
    trigger_mode: TriggerMode
    store_uri: str
    writes_external_frames: bool


def dpct_plan(
    detectors: Sequence[DpctDetector],
    stage: MeasuredStage | None,
    illumination: Illumination,
    *,
    run_id: str,
    plan_name: str = "dpct",
    purpose: str = "four-pattern DPCT",
    scan_id: int = 0,
    sequence: Sequence[PatternStep] = DPCT_PATTERN_SEQUENCE,
    stream_name: str = STREAM_NAME,
    frame_timeout_s: float = FRAME_TIMEOUT_S,
    frame_poll_s: float = FRAME_POLL_S,
    device_timeout_s: float = DEVICE_TIMEOUT_S,
) -> MsgGenerator[PlanOutcome]:
    """Acquire one four-pattern DPCT frame set from externally triggered cameras.

    The plan opens its own run, so it is not meant to be called inside an open
    run. The illumination service is the trigger: nothing here triggers a
    camera.

    Parameters
    ----------
    detectors
        The cameras to expect a frame from. At least one is required, each must
        expose a readable frame counter, and their names must be unique because
        a name is the frame identity's detector id.
    stage
        The stage whose measured position is recorded with every frame, or
        ``None`` when the run has no position to report. The plan never moves it.
    illumination
        The MM-compatible illumination service. Its capability must advertise
        every pattern of *sequence*, blank and open.
    run_id
        The store key of the run. It is a whitespace-free token of at most
        :data:`~redsun_aht.contracts.MAX_RUN_ID_BYTES` UTF-8 bytes.
    plan_name
        The plan name recorded in the run identity.
    purpose
        One line describing what the operator acquired the run for.
    scan_id
        The plan's own scan number; it is independent of Bluesky's counters.
    sequence
        The pattern steps, in visit order. The default is the four semicircles,
        and a later adaptive plan can pass a longer or reordered sequence without
        redefining the individual patterns.
    stream_name
        The Event stream the frames are recorded in.
    frame_timeout_s
        Seconds to wait for one frame before the run is aborted. It bounds every
        detector separately, because each one answers the same trigger.
    frame_poll_s
        Seconds between two reads of a frame counter while waiting.
    device_timeout_s
        Seconds one illumination command or one position read may take.

    Returns
    -------
    PlanOutcome
        ``COMPLETE`` when every requested frame arrived. Otherwise ``ABORTED``
        with the reason and a :class:`~redsun_aht.contracts.PartialRun` naming
        the frames that were and were not produced.

    Raises
    ------
    ValueError
        Before any message is emitted, when a detector, the stage, the
        illumination capability, the pattern sequence or a bound cannot support
        the requested run.

    Notes
    -----
    An uncertain illumination command aborts the sequence without a resend, and
    a missing frame aborts the run with
    :attr:`~redsun_aht.contracts.PlanAbortReason.CAMERA_TIMEOUT`. Both still
    blank the illumination and unstage the devices. A Stop or an Abort runs the
    same cleanup; the RunEngine then raises instead of returning an outcome, so
    the partial report is read from the start document and the Events.
    """
    detector_list = list(detectors)
    sources = _validate_detectors(detector_list)
    detector_names = tuple(source.name for source in sources)
    steps = _validate_sequence(sequence)
    capability = _validate_illumination(illumination, steps)
    axes = _validate_stage(stage)
    _validate_bound(frame_timeout_s, "frame_timeout_s")
    _validate_bound(frame_poll_s, "frame_poll_s")
    _validate_bound(device_timeout_s, "device_timeout_s")
    if not isinstance(stream_name, str) or not stream_name.strip():
        raise ValueError("stream_name must be non-empty text")
    run = RunIdentity(
        run_id=run_id, plan_name=plan_name, purpose=purpose, scan_id=scan_id
    )
    yield from _preflight_capture(sources, device_timeout_s)
    trigger_info = TriggerInfo(number_of_events=len(steps))
    requested = _requested_frames(run, detector_names, steps)
    frames = _FrameSlot(axes)

    produced: list[FrameIdentity] = []
    failure: list[tuple[PlanAbortReason, str]] = []

    def acquire() -> MsgGenerator[None]:
        yield from bps.open_run(
            md=_start_document(run, detector_names, steps, capability, axes)
        )
        staged: list[Any] = []
        try:
            for device in _stageable(detector_list, stage, illumination):
                yield from bps.stage(device)
                staged.append(device)
            for detector in detector_list:
                yield from bps.prepare(detector, trigger_info, wait=True)
            yield from _blank_and_open(illumination, device_timeout_s)
            for ordinal, step in enumerate(steps):
                yield from _acquire_step(
                    sources,
                    stage,
                    illumination,
                    run=run,
                    step=step,
                    ordinal=ordinal,
                    frames=frames,
                    stream_name=stream_name,
                    frame_timeout_s=frame_timeout_s,
                    frame_poll_s=frame_poll_s,
                    device_timeout_s=device_timeout_s,
                    produced=produced,
                )
        except RunEngineControlException:
            raise
        except FrameTimeoutError as error:
            failure.append((PlanAbortReason.CAMERA_TIMEOUT, str(error)))
        except IlluminationUncertain as error:
            failure.append((PlanAbortReason.ILLUMINATION_UNCERTAIN, str(error)))
        except (DpctError, FailedStatus) as error:
            failure.append((PlanAbortReason.DEVICE_FAULT, str(error)))
        except Exception as error:
            failure.append(
                (PlanAbortReason.UNEXPECTED_ERROR, f"{type(error).__name__}: {error}")
            )
        finally:
            yield from _finalize(illumination, staged, device_timeout_s)
            if failure:
                logger.warning(
                    "DPCT run %s stopped after %d of %d frames: %s",
                    run.run_id,
                    len(produced),
                    len(requested),
                    failure[0][1],
                )
        if failure:
            yield from bps.close_run(exit_status="abort", reason=failure[0][1])
        else:
            yield from bps.close_run()

    yield from acquire()

    if failure:
        reason, detail = failure[0]
        return PlanOutcome(
            run=run,
            status=PlanStatus.ABORTED,
            reason=reason,
            partial=PartialRun(requested=requested, produced=tuple(produced)),
            detail=detail,
            finished_monotonic_ns=time.monotonic_ns(),
        )
    return PlanOutcome(
        run=run,
        status=PlanStatus.COMPLETE,
        detail=f"{len(steps)} patterns, {len(produced)} frames",
        finished_monotonic_ns=time.monotonic_ns(),
    )


def _acquire_step(
    sources: Sequence[_DetectorSource],
    stage: MeasuredStage | None,
    illumination: Illumination,
    *,
    run: RunIdentity,
    step: PatternStep,
    ordinal: int,
    frames: _FrameSlot,
    stream_name: str,
    frame_timeout_s: float,
    frame_poll_s: float,
    device_timeout_s: float,
    produced: list[FrameIdentity],
) -> MsgGenerator[None]:
    """Write one pattern command once, wait for its frames and record them.

    The frame counter is sampled before the command so the wait is for one *new*
    frame, not for a total: a detector that is already ahead cannot satisfy the
    wait with a frame an earlier pattern produced. The counter value at the
    frame becomes the identity's ``producer_sequence``, which is the detector's
    own account of which readout this was.
    """
    baselines: list[int] = []
    for source in sources:
        seen = yield from bps.rd(source.counter)
        baselines.append(_as_count(seen, source.name))
    result = yield from _bounded_call(
        partial(illumination.pattern, step.pattern_id),
        timeout_s=device_timeout_s,
        what=f"pattern {step.pattern_id}",
    )
    receipt = _receipt(
        result,
        action=IlluminationAction.PATTERN,
        pattern_id=step.pattern_id,
    )
    _confirmed(receipt, what=f"pattern {step.pattern_id}")
    counts: list[int] = []
    for source, baseline in zip(sources, baselines, strict=True):
        count = yield from _wait_for_frame(
            source,
            expected=baseline + 1,
            ordinal=ordinal,
            pattern_id=step.pattern_id,
            timeout_s=frame_timeout_s,
            poll_s=frame_poll_s,
        )
        counts.append(count)
    position: StageReading | None = None
    if stage is not None:
        position = yield from _bounded_call(
            stage.read_position,
            timeout_s=device_timeout_s,
            what="reading the stage position",
        )
    for source, count in zip(sources, counts, strict=True):
        identity = FrameIdentity(
            run=run,
            detector_id=source.name,
            sequence=ordinal,
            plane_ordinal=ordinal,
            pattern_id=step.pattern_id,
            scan_index=0,
            producer_sequence=count,
        )
        yield from _emit_frame(stream_name, frames, identity, receipt, position)
        produced.append(identity)


def _blank_and_open(
    illumination: Illumination, device_timeout_s: float
) -> MsgGenerator[None]:
    """Blank and then open the illumination, refusing either failure."""
    for action, call, what in (
        (IlluminationAction.BLANK, illumination.blank, "blanking the illumination"),
        (IlluminationAction.OPEN, illumination.open, "opening the shutter state"),
    ):
        result = yield from _bounded_call(call, timeout_s=device_timeout_s, what=what)
        _confirmed(_receipt(result, action=action, pattern_id=None), what=what)


def _finalize(
    illumination: Illumination, staged: Sequence[Any], device_timeout_s: float
) -> MsgGenerator[None]:
    """Blank the illumination, then unstage every staged device in reverse.

    This runs on success, on a failure and while the RunEngine unwinds a Stop or
    an Abort, so the panel is never left lit and no device is left staged. A
    failing cleanup is logged rather than raised: it must not replace the reason
    the run stopped.
    """
    try:
        result = yield from _bounded_call(
            illumination.blank,
            timeout_s=device_timeout_s,
            what="blanking the illumination on finalize",
        )
        receipt = _receipt(result, action=IlluminationAction.BLANK, pattern_id=None)
        if receipt.status not in (CommandStatus.SENT, CommandStatus.ECHOED):
            logger.warning(
                "DPCT finalize blank was %s: %s", receipt.status, receipt.detail
            )
    except Exception:
        logger.exception("The DPCT plan could not blank the illumination")
    for device in reversed(list(staged)):
        try:
            yield from bps.unstage(device)
        except Exception:
            logger.exception("The DPCT plan could not unstage %r", device)


def _wait_for_frame(
    source: _DetectorSource,
    *,
    expected: int,
    ordinal: int,
    pattern_id: int,
    timeout_s: float,
    poll_s: float,
) -> MsgGenerator[int]:
    """Poll one detector's counter until it reaches *expected* or time out."""
    deadline = time.monotonic() + timeout_s
    count = 0
    while True:
        seen = yield from bps.rd(source.counter)
        count = _as_count(seen, source.name)
        if count >= expected:
            return count
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FrameTimeoutError(
                f"detector {source.name!r} delivered no frame for pattern "
                f"{pattern_id} (ordinal {ordinal}) within {timeout_s:g} s: its "
                f"frame counter stayed at {count}, expected {expected}"
            )
        yield from bps.sleep(min(poll_s, remaining))


def _emit_frame(
    stream_name: str,
    frames: _FrameSlot,
    identity: FrameIdentity,
    receipt: CommandReceipt,
    position: StageReading | None,
) -> MsgGenerator[None]:
    """Publish one frame to the slot and write the Event that records it."""
    frames.publish(identity, receipt, position)
    yield from bps.create(name=stream_name)
    yield from bps.read(frames)
    yield from bps.save()


def _bounded_call[T](
    factory: Callable[[], Awaitable[T]],
    *,
    timeout_s: float,
    what: str,
) -> MsgGenerator[T]:
    """Await one device coroutine on the RunEngine loop, bounded in time.

    A plan generator cannot await, and the RunEngine's own ``wait_for`` timeout
    leaves a pending coroutine running, so the bound is applied to the coroutine
    itself: a timeout cancels the device call before it raises.
    """
    tasks = yield from bps.wait_for(
        [lambda: _bounded(factory(), timeout_s=timeout_s, what=what)]
    )
    task: Any = tasks[0]
    if task.cancelled():
        raise DpctError(f"{what} was cancelled before it answered")
    return cast("T", task.result())


async def _bounded[T](awaitable: Awaitable[T], *, timeout_s: float, what: str) -> T:
    """Await *awaitable* for at most *timeout_s*, naming what timed out."""
    try:
        return await asyncio.wait_for(awaitable, timeout_s)
    except TimeoutError as error:
        raise DpctError(f"{what} did not answer within {timeout_s:g} s") from error


def _receipt(
    result: object,
    *,
    action: IlluminationAction,
    pattern_id: int | None,
) -> CommandReceipt:
    """Return a plan-facing receipt from whatever the illumination answered.

    The frozen protocol already returns a
    :class:`~redsun_aht.contracts.CommandReceipt`; a service that answers with its
    own transport dataclass is adapted through
    :meth:`~redsun_aht.contracts.CommandReceipt.from_transport`, which is the one
    named seam between the two.
    """
    if isinstance(result, CommandReceipt):
        return result
    command = getattr(result, "command", "")
    if not isinstance(command, str) or len(command) != 2 or not command.isdigit():
        raise IlluminationFailed(
            f"the illumination service answered {result!r} instead of a receipt; "
            "no command could be identified"
        )
    try:
        return CommandReceipt.from_transport(
            action=action,
            command=command,
            status=getattr(result, "status", ""),
            detail=getattr(result, "detail", ""),
            echo=getattr(result, "echo", None),
            pattern_id=pattern_id,
            sent_monotonic_ns=getattr(result, "sent_monotonic_ns", None),
        )
    except ValueError as error:
        raise IlluminationFailed(
            f"the illumination service answered an unusable receipt: {error}"
        ) from error


def _confirmed(receipt: CommandReceipt, *, what: str) -> CommandReceipt:
    """Return *receipt* when the transport delivered it, otherwise refuse."""
    if receipt.status is CommandStatus.UNCERTAIN:
        raise IlluminationUncertain(
            f"{what}: the outcome is uncertain ({receipt.detail or 'no detail'}); "
            "it is not resent, because a second pattern command would take a "
            "second frame"
        )
    if receipt.status is CommandStatus.FAILED:
        raise IlluminationFailed(
            f"{what} failed without being written ({receipt.detail or 'no detail'})"
        )
    return receipt


def _as_count(value: object, detector_name: str) -> int:
    """Return a frame counter reading as an integer, refusing anything else."""
    if isinstance(value, bool):
        raise DpctError(f"detector {detector_name!r} reported a boolean frame count")
    try:
        count = int(cast("Any", value))
    except (TypeError, ValueError) as error:
        raise DpctError(
            f"detector {detector_name!r} reported {value!r} as its frame count; "
            "a DPCT run needs an integer count of delivered frames"
        ) from error
    return count


@dataclass(frozen=True, slots=True)
class _DetectorSource:
    """One detector, the name its frames are recorded under, and its counter."""

    detector: DpctDetector
    name: str
    counter: FrameSource


def _validate_detectors(
    detectors: Sequence[DpctDetector],
) -> tuple[_DetectorSource, ...]:
    """Refuse an empty list, a nameless or unarmable detector, or no counter."""
    if not detectors:
        raise ValueError("a DPCT run needs at least one detector")
    sources: list[_DetectorSource] = []
    for index, detector in enumerate(detectors):
        name = getattr(detector, "name", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"detector {index} has no name; a frame identity needs the "
                "detector identity it is recorded under"
            )
        if not isinstance(detector, DpctDetector):
            raise ValueError(
                f"detector {name!r} cannot be armed for an external-trigger "
                "capture; a DPCT detector exposes name, stage, unstage and prepare"
            )
        counter = getattr(detector, "frame_count", None)
        if counter is None:
            raise ValueError(
                f"detector {name!r} has no frame source; a DPCT run waits on "
                "the frame counter of every detector it expects a frame from"
            )
        if not isinstance(counter, FrameSource):
            raise ValueError(
                f"detector {name!r} has a frame source that cannot be read; "
                "a DPCT run polls the counter of delivered frames"
            )
        sources.append(_DetectorSource(detector=detector, name=name, counter=counter))
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError(
            "detector names must be unique; a frame identity is keyed by the "
            "detector id, so two detectors under one name would collide"
        )
    return tuple(sources)


def _preflight_capture(
    sources: Sequence[_DetectorSource], timeout_s: float
) -> MsgGenerator[None]:
    """Refuse a detector before illumination unless TTL frames are stored.

    The admission is a narrow integration seam for a future capture device.
    A detector must implement it explicitly; the present MMCore camera does
    not, and its free-running frame counter cannot establish TTL causality.
    """
    for source in sources:
        describe_settings = getattr(source.detector, "describe_settings", None)
        if not callable(describe_settings):
            raise DpctError(
                f"detector {source.name!r} has no camera capability description"
            )
        description = yield from _bounded_call(
            describe_settings,
            timeout_s=timeout_s,
            what=f"detector {source.name!r} description",
        )
        if not isinstance(description, CameraDescription):
            raise DpctError(f"detector {source.name!r} gave no camera description")
        if description.detector_id != source.name:
            raise DpctError(f"detector {source.name!r} description identity differs")
        if TriggerMode.EXTERNAL not in description.trigger_modes:
            raise DpctError(
                f"detector {source.name!r} does not support external trigger"
            )
        admission_method = getattr(source.detector, "dpct_capture_admission", None)
        if not callable(admission_method):
            raise DpctError(
                f"detector {source.name!r} has no verified durable TTL capture path"
            )
        admission = yield from _bounded_call(
            admission_method,
            timeout_s=timeout_s,
            what=f"detector {source.name!r} capture admission",
        )
        if (
            not isinstance(admission, DpctCaptureAdmission)
            or admission.detector_id != source.name
            or admission.trigger_mode is not TriggerMode.EXTERNAL
            or not isinstance(admission.store_uri, str)
            or not admission.store_uri.strip()
            or admission.writes_external_frames is not True
        ):
            raise DpctError(
                f"detector {source.name!r} has no verified durable TTL capture path"
            )


def _validate_sequence(sequence: Sequence[PatternStep]) -> tuple[PatternStep, ...]:
    """Refuse an empty sequence, a foreign step or a repeated pattern."""
    steps = tuple(sequence)
    if not steps:
        raise ValueError("a DPCT sequence needs at least one pattern step")
    for index, step in enumerate(steps):
        if not isinstance(step, PatternStep):
            raise ValueError(f"DPCT sequence step {index} is not a PatternStep")
    pattern_ids = [step.pattern_id for step in steps]
    if len(set(pattern_ids)) != len(pattern_ids):
        raise ValueError(
            "a DPCT sequence must request each pattern once; a repeated pattern "
            "would produce two frames of the same illumination"
        )
    return steps


def _validate_illumination(
    illumination: Illumination, steps: Sequence[PatternStep]
) -> IlluminationCapability:
    """Refuse an illumination service that cannot drive the requested sequence."""
    if not isinstance(illumination, Illumination):
        raise ValueError(
            "the illumination device does not expose the MM-compatible command "
            "surface (capability, blank, open, pattern)"
        )
    capability = illumination.capability
    if not isinstance(capability, IlluminationCapability):
        raise ValueError(
            "the illumination device must declare an IlluminationCapability"
        )
    requested = [step.pattern_id for step in steps]
    missing = [
        pattern for pattern in requested if pattern not in capability.pattern_ids
    ]
    if missing:
        raise ValueError(
            f"the illumination service advertises {capability.pattern_ids} but "
            f"the DPCT sequence asks for {missing}"
        )
    if not capability.blank_supported:
        raise ValueError(
            "the illumination service does not advertise blank; a DPCT run must "
            "be able to blank the panel before and after the sequence"
        )
    if not capability.open_supported:
        raise ValueError(
            "the illumination service does not advertise open; the validated "
            "sequence clears the shutter-closed flag before the first pattern"
        )
    return capability


def _validate_stage(stage: MeasuredStage | None) -> tuple[AxisSpec, ...]:
    """Return the stage axes, refusing a stage that cannot report a position."""
    if stage is None:
        return ()
    if not isinstance(stage, MeasuredStage):
        raise ValueError(
            "the stage does not expose the position surface a frame records "
            "(axes and read_position)"
        )
    axes = tuple(stage.axes.values())
    if any(not isinstance(axis, AxisSpec) for axis in axes):
        raise ValueError("stage axes must be AxisSpec values")
    return axes


def _validate_bound(value: float, name: str) -> None:
    """Refuse a wait bound that is not finite and positive."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _stageable(
    detectors: Sequence[DpctDetector],
    stage: MeasuredStage | None,
    illumination: Illumination,
) -> tuple[Any, ...]:
    """Return the devices of this run that are worth staging, in stage order."""
    devices: list[Any] = [*detectors]
    if stage is not None:
        devices.append(stage)
    devices.append(illumination)
    return tuple(device for device in devices if isinstance(device, Stageable))


def _requested_frames(
    run: RunIdentity, detector_names: Sequence[str], steps: Sequence[PatternStep]
) -> tuple[FrameIdentity, ...]:
    """Return every frame this run asks for, in visit order."""
    return tuple(
        FrameIdentity(
            run=run,
            detector_id=detector_name,
            sequence=ordinal,
            plane_ordinal=ordinal,
            pattern_id=step.pattern_id,
            scan_index=0,
        )
        for ordinal, step in enumerate(steps)
        for detector_name in detector_names
    )


def _start_document(
    run: RunIdentity,
    detector_names: Sequence[str],
    steps: Sequence[PatternStep],
    capability: IlluminationCapability,
    axes: Sequence[AxisSpec],
) -> dict[str, Any]:
    """Return the start document of one DPCT run.

    The requested sequence is recorded here, so a run interrupted before its
    outcome could be returned still reports which frames it asked for; every
    Event then names the frame it produced.
    """
    return {
        "purpose": run.purpose,
        "plan_name": run.plan_name,
        "run_id": run.run_id,
        "run_identity": run.as_payload(),
        "detector_ids": list(detector_names),
        "axes": [{"name": axis.name, "units": axis.units} for axis in axes],
        "frame_trigger": "illumination-pattern-command",
        "dpct": {
            "pattern_ids": [step.pattern_id for step in steps],
            "sequence": [
                {
                    "ordinal": ordinal,
                    "pattern_id": step.pattern_id,
                    "label": step.label,
                }
                for ordinal, step in enumerate(steps)
            ],
            "panel": {
                "profile": capability.panel_profile,
                "led_count": capability.led_count,
                "dialect": capability.dialect,
                "pattern_ids": list(capability.pattern_ids),
                "center_pattern_id": capability.center_pattern_id,
                "echo_expected": capability.echo_expected,
            },
        },
    }


class _FrameSlot:
    """The mutable readable one DPCT stream publishes every frame through.

    A plan cannot invent a device reading, so the frame identity travels as a
    readable object: the RunEngine describes and reads it inside the Event the
    plan opens for the frame. It is one long-lived object rather than one per
    frame, because a stream's descriptor is keyed to the objects that were read;
    a fresh object per frame is refused as a mismatched read. The declared keys
    come from the axes alone, so every Event of a run carries the same set.
    """

    def __init__(self, axes: Sequence[AxisSpec]) -> None:
        self._axes = tuple(axes)
        self._identity: FrameIdentity | None = None
        self._receipt: CommandReceipt | None = None
        self._position: StageReading | None = None

    @property
    def name(self) -> str:
        """Return the stream-local name of this readable."""
        return "dpct-frames"

    def publish(
        self,
        identity: FrameIdentity,
        receipt: CommandReceipt,
        position: StageReading | None,
    ) -> None:
        """Hold one frame's values until the next frame is published."""
        if self._axes and position is None:
            raise DpctError(
                "the run declares stage axes but the frame has no position; "
                "a frame records every declared axis"
            )
        if position is not None:
            measured = dict(position.positions)
            missing = [axis.name for axis in self._axes if axis.name not in measured]
            if missing:
                raise DpctError(
                    f"the stage reported no position for {missing}; a frame "
                    "records every declared axis"
                )
        self._identity = identity
        self._receipt = receipt
        self._position = position

    def describe(self) -> dict[str, DataKey]:
        """Return one data key per value this stream publishes."""
        return {
            key: {"source": source, "dtype": dtype, "shape": []}
            for key, (source, dtype) in self._declared_keys().items()
        }

    def read(self) -> dict[str, Reading[Any]]:
        """Return the frame most recently published, with one timestamp."""
        identity = self._identity
        receipt = self._receipt
        if identity is None or receipt is None:
            raise DpctError("no frame has been published to the DPCT frame stream")
        stamp = time.time()
        values: dict[str, Any] = {
            "dpct-run-id": identity.run.run_id,
            "dpct-detector-id": identity.detector_id,
            "dpct-sequence": identity.sequence,
            "dpct-plane-ordinal": identity.plane_ordinal,
            "dpct-pattern-id": identity.pattern_id,
            "dpct-scan-index": identity.scan_index,
            "dpct-producer-sequence": (
                UNKNOWN_PRODUCER_SEQUENCE
                if identity.producer_sequence is None
                else identity.producer_sequence
            ),
            "dpct-command": receipt.command,
            "dpct-command-status": str(receipt.status),
            "dpct-command-echo": "" if receipt.echo is None else receipt.echo,
            "dpct-command-sent-ns": (
                UNKNOWN_CLOCK_NS
                if receipt.sent_monotonic_ns is None
                else receipt.sent_monotonic_ns
            ),
        }
        if self._axes:
            position = self._position
            if position is None:
                raise DpctError(
                    "the run declares stage axes but no position was published"
                )
            values["dpct-position-quality"] = str(position.quality)
            measured = dict(position.positions)
            for axis in self._axes:
                values[f"dpct-position-{axis.name}"] = float(measured[axis.name])
        return {
            key: {"value": values[key], "timestamp": stamp}
            for key in self._declared_keys()
        }

    def _declared_keys(self) -> dict[str, tuple[str, _Dtype]]:
        """Return every data key with its source and dtype, in a stable order."""
        identity = self._identity
        detector_id = "unknown" if identity is None else identity.detector_id
        keys: dict[str, tuple[str, _Dtype]] = {
            key: (f"dpct:frame:{detector_id}", dtype) for key, dtype in _IDENTITY_FIELDS
        }
        keys.update(
            {
                key: (f"dpct:command:{detector_id}", dtype)
                for key, dtype in _COMMAND_FIELDS
            }
        )
        if self._axes:
            keys["dpct-position-quality"] = ("dpct:position", "string")
            for axis in self._axes:
                keys[f"dpct-position-{axis.name}"] = (
                    f"dpct:position:{axis.name}:{axis.units}",
                    "number",
                )
        return keys
