"""Plan-level tests for the four-pattern DPCT plan.

A real :class:`bluesky.RunEngine` drives a fake externally triggered camera, a
fake stage and a fake MM-compatible illumination service, so these tests assert
the pattern-command order, the one-command-per-frame rule, the absence of a
software camera trigger, the recorded frame identity and the partial-run report.
No hardware, no serial port and no Qt.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import pytest
from bluesky import RunEngine
from bluesky.utils import RunEngineInterrupted

from redsun_aht.contracts import (
    AxisReading,
    AxisSpec,
    CameraDescription,
    CommandReceipt,
    CommandStatus,
    Illumination,
    IlluminationAction,
    IlluminationCapability,
    PlanAbortReason,
    PlanOutcome,
    PlanStatus,
    QualityLabel,
    StageReading,
    TriggerMode,
)
from redsun_aht.plans import dpct_plan
from redsun_aht.plans.dpct import (
    DpctCaptureAdmission,
    DpctDetector,
    MeasuredStage,
    PatternStep,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from bluesky.protocols import Reading, Status
    from bluesky.utils import Msg
    from event_model import DataKey


class _CompletedStatus:
    """A Bluesky status that is already complete when it is returned."""

    def __init__(self, *, success: bool = True) -> None:
        self._success = success

    def add_callback(self, callback: Callable[[Status], None]) -> None:
        callback(self)

    def exception(self, timeout: float | None = 0.0) -> BaseException | None:
        return None

    @property
    def done(self) -> bool:
        return True

    @property
    def success(self) -> bool:
        return self._success


class _FrameCounter:
    """A readable integer counter of delivered frames."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.value = 0

    def read(self) -> dict[str, Reading[Any]]:
        return {self.name: {"value": self.value, "timestamp": time.time()}}

    def describe(self) -> dict[str, DataKey]:
        return {
            self.name: {
                "source": f"fake://{self.name}",
                "dtype": "integer",
                "shape": [],
            }
        }


class FakeCamera:
    """An externally triggered camera that answers one pattern with one frame.

    The frame counter only moves when the fake illumination writes a pattern
    command, which is the installed sketch's behaviour: a software trigger would
    never produce a frame here, so a plan that issued one would time out.
    """

    def __init__(self, name: str = "dhm", *, log: list[str] | None = None) -> None:
        self.name = name
        self.log = log if log is not None else []
        self.counter = _FrameCounter(f"{name}-frame-count")
        self.prepare_values: list[Any] = []
        self.patterns: list[str] = []
        self.stage_calls = 0
        self.unstage_calls = 0
        self.trigger_calls = 0
        self.staged = False
        self.drop_ordinals: set[int] = set()
        self.trigger_modes: tuple[TriggerMode, ...] = (TriggerMode.EXTERNAL,)
        self.store_uri = f"fake://{name}/frames"

    async def describe_settings(self) -> CameraDescription:
        return CameraDescription(
            detector_id=self.name,
            shape=(2, 3),
            pixel_dtypes=("uint16",),
            exposure_range_s=(0.001, 1.0),
            roi_supported=False,
            binning=(1,),
            trigger_modes=self.trigger_modes,
            revision=0,
        )

    async def dpct_capture_admission(self) -> DpctCaptureAdmission:
        return DpctCaptureAdmission(
            detector_id=self.name,
            trigger_mode=TriggerMode.EXTERNAL,
            store_uri=self.store_uri,
            writes_external_frames=True,
        )

    @property
    def frame_count(self) -> _FrameCounter:
        """Return the counter the plan polls."""
        return self.counter

    def stage(self) -> Status:
        self.stage_calls += 1
        self.staged = True
        self.log.append(f"{self.name}:stage")
        return _CompletedStatus()

    def unstage(self) -> Status:
        self.unstage_calls += 1
        self.staged = False
        self.log.append(f"{self.name}:unstage")
        return _CompletedStatus()

    def prepare(self, value: Any) -> Status:
        self.prepare_values.append(value)
        self.log.append(f"{self.name}:prepare")
        return _CompletedStatus()

    def trigger(self) -> Status:
        """Count a software trigger; a DPCT run must never call this."""
        self.trigger_calls += 1
        self.counter.value += 1
        return _CompletedStatus()

    def frame(self, ordinal: int, command: str) -> None:
        """Answer one pattern command with one frame, or with none."""
        self.patterns.append(command)
        self.log.append(f"{self.name}:frame")
        if ordinal not in self.drop_ordinals:
            self.counter.value += 1


class FakeIllumination:
    """An MM-compatible illumination service that records its command order.

    After an uncertain outcome it refuses every further pattern command, because
    a second pattern write would take a second frame.
    """

    def __init__(
        self,
        cameras: list[FakeCamera] | None = None,
        *,
        capability: IlluminationCapability | None = None,
        log: list[str] | None = None,
    ) -> None:
        self.cameras = list(cameras or [])
        self.log = log if log is not None else []
        self.commands: list[str] = []
        self.uncertain_pattern: int | None = None
        self.stage_calls = 0
        self.unstage_calls = 0
        self.staged = False
        self._capability = (
            capability if capability is not None else (IlluminationCapability())
        )
        self._uncertain = False
        self._ordinal = 0

    @property
    def capability(self) -> IlluminationCapability:
        """Return the panel this service advertises."""
        return self._capability

    def stage(self) -> Status:
        self.stage_calls += 1
        self.staged = True
        self.log.append("led:stage")
        return _CompletedStatus()

    def unstage(self) -> Status:
        self.unstage_calls += 1
        self.staged = False
        self.log.append("led:unstage")
        return _CompletedStatus()

    async def blank(self) -> CommandReceipt:
        return self._record("00", IlluminationAction.BLANK)

    async def open(self) -> CommandReceipt:
        return self._record("01", IlluminationAction.OPEN)

    async def pattern(self, pattern_id: int) -> CommandReceipt:
        command = f"{10 + pattern_id:02d}"
        if self._uncertain:
            raise AssertionError(
                f"refused a second pattern command ({command}) after an "
                "uncertain outcome"
            )
        uncertain = pattern_id == self.uncertain_pattern
        self._uncertain = self._uncertain or uncertain
        return self._record(
            command,
            IlluminationAction.PATTERN,
            pattern_id,
            status=CommandStatus.UNCERTAIN if uncertain else CommandStatus.SENT,
        )

    def _record(
        self,
        command: str,
        action: IlluminationAction,
        pattern_id: int | None = None,
        *,
        status: CommandStatus = CommandStatus.SENT,
    ) -> CommandReceipt:
        self.commands.append(command)
        self.log.append(f"led:{command}")
        if action is IlluminationAction.PATTERN:
            self._fire(command)
        detail = (
            "fake serial timeout; delivery unknown"
            if status is CommandStatus.UNCERTAIN
            else ""
        )
        return CommandReceipt(
            action=action,
            command=command,
            status=status,
            pattern_id=pattern_id,
            detail=detail,
        )

    def _fire(self, command: str) -> None:
        for camera in self.cameras:
            camera.frame(self._ordinal, command)
        self._ordinal += 1


class FakeStage:
    """A stage that reports a measured position and is never moved."""

    def __init__(
        self,
        positions: dict[str, float] | None = None,
        *,
        log: list[str] | None = None,
    ) -> None:
        self.log = log if log is not None else []
        self.positions = dict(positions if positions is not None else {"z": 1.25})
        self.axes = {"z": AxisSpec("z", "um", -50.0, 50.0, 0.1)}
        self.reads = 0
        self.stage_calls = 0
        self.unstage_calls = 0

    async def read_position(self) -> StageReading:
        self.reads += 1
        self.log.append("stage:read")
        return StageReading(
            axes=tuple(
                AxisReading(name=name, value=value, setpoint=value, measured=True)
                for name, value in self.positions.items()
            ),
            monotonic_ns=time.monotonic_ns(),
            quality=QualityLabel.MEASURED,
        )

    async def move(self, position: Any) -> StageReading:
        raise AssertionError("a DPCT run does not move the stage")

    async def stop(self) -> None:
        self.log.append("stage:stop")

    def stage(self) -> Status:
        self.stage_calls += 1
        self.log.append("stage:stage")
        return _CompletedStatus()

    def unstage(self) -> Status:
        self.unstage_calls += 1
        self.log.append("stage:unstage")
        return _CompletedStatus()


def _documents(engine: RunEngine) -> list[tuple[str, Any]]:
    documents: list[tuple[str, Any]] = []

    def record(name: str, doc: Any) -> None:
        documents.append((name, doc))

    engine.subscribe(record)  # type: ignore[no-untyped-call]
    return documents


def _messages(engine: RunEngine) -> list[str]:
    commands: list[str] = []

    def record(msg: Msg) -> None:
        commands.append(msg.command)

    engine.msg_hook = record  # type: ignore[assignment]
    return commands


def _run_plan(engine: RunEngine, **kwargs: Any) -> PlanOutcome:
    """Run the DPCT plan and return what it reported."""
    outcomes: list[PlanOutcome] = []

    def plan() -> Any:
        outcomes.append((yield from dpct_plan(**kwargs)))

    engine(plan())
    return outcomes[0]


def _events(documents: list[tuple[str, Any]]) -> list[Any]:
    return [doc for name, doc in documents if name == "event"]


def _stop_statuses(documents: list[tuple[str, Any]]) -> list[Any]:
    return [doc["exit_status"] for name, doc in documents if name == "stop"]


@pytest.mark.parametrize("failure", ["free_run", "no_store", "no_admission"])
def test_capture_preflight_refuses_before_any_illumination_command(
    failure: str,
) -> None:
    camera = FakeCamera()
    if failure == "free_run":
        camera.trigger_modes = (TriggerMode.FREE_RUN, TriggerMode.SOFTWARE)
    elif failure == "no_store":
        camera.store_uri = ""
    else:
        camera.dpct_capture_admission = None  # type: ignore[assignment]
    led = FakeIllumination([camera])
    engine = RunEngine({})
    documents = _documents(engine)

    with pytest.raises(RuntimeError, match=r"external trigger|durable TTL capture"):
        _run_plan(
            engine,
            detectors=[camera],
            stage=None,
            illumination=led,
            run_id="preflight-refusal",
        )

    assert led.commands == []
    assert camera.stage_calls == 0
    assert not _events(documents)


def test_the_plan_writes_one_pattern_per_frame_and_never_triggers_the_camera() -> None:
    log: list[str] = []
    camera = FakeCamera(log=log)
    stage = FakeStage(log=log)
    led = FakeIllumination([camera], log=log)
    assert isinstance(camera, DpctDetector)
    assert isinstance(stage, MeasuredStage)
    assert isinstance(led, Illumination)
    engine = RunEngine({})
    documents = _documents(engine)
    messages = _messages(engine)

    outcome = _run_plan(
        engine,
        detectors=[camera],
        stage=stage,
        illumination=led,
        run_id="run-20261103-a",
    )

    assert outcome.status is PlanStatus.COMPLETE
    assert outcome.is_complete
    assert outcome.partial is None
    assert outcome.missing == ()

    # 00 blanks, 01 opens, one command per frame, 00 again on finalize.
    assert led.commands == ["00", "01", "11", "12", "13", "14", "00"]
    assert camera.patterns == ["11", "12", "13", "14"]
    # The sketch fires the camera, so the plan issues no trigger of its own.
    assert camera.trigger_calls == 0
    assert "trigger" not in messages
    # One new frame per pattern, and one prepare for the whole sequence.
    assert camera.counter is not None and camera.counter.value == 4
    assert len(camera.prepare_values) == 1
    assert camera.prepare_values[0].number_of_events == 4

    assert log[:3] == ["dhm:stage", "stage:stage", "led:stage"]
    assert log[3:5] == ["dhm:prepare", "led:00"]
    assert log[5] == "led:01"
    for ordinal, pattern in enumerate(["11", "12", "13", "14"]):
        start = 6 + 3 * ordinal
        assert log[start : start + 3] == [f"led:{pattern}", "dhm:frame", "stage:read"]
    assert log[18:] == ["led:00", "led:unstage", "stage:unstage", "dhm:unstage"]
    assert camera.unstage_calls == 1
    assert stage.unstage_calls == 1
    assert led.unstage_calls == 1
    assert _stop_statuses(documents) == ["success"]


def test_the_start_document_names_the_run_the_panel_and_the_sequence() -> None:
    camera = FakeCamera()
    led = FakeIllumination([camera])
    stage = FakeStage()
    engine = RunEngine({})
    documents = _documents(engine)

    _run_plan(
        engine,
        detectors=[camera],
        stage=stage,
        illumination=led,
        run_id="run-20261103-a",
        plan_name="dpct-basic",
        purpose="four semicircles",
    )

    start = next(doc for name, doc in documents if name == "start")
    assert start["run_id"] == "run-20261103-a"
    assert start["plan_name"] == "dpct-basic"
    assert start["purpose"] == "four semicircles"
    assert start["run_identity"]["run_id"] == "run-20261103-a"
    assert start["run_identity"]["plan_name"] == "dpct-basic"
    assert start["detector_ids"] == ["dhm"]
    assert start["axes"] == [{"name": "z", "units": "um"}]
    assert start["frame_trigger"] == "illumination-pattern-command"
    assert start["dpct"]["pattern_ids"] == [1, 2, 3, 4]
    assert [entry["pattern_id"] for entry in start["dpct"]["sequence"]] == [
        1,
        2,
        3,
        4,
    ]
    assert start["dpct"]["panel"]["profile"] == "panel91"
    assert start["dpct"]["panel"]["led_count"] == 91
    assert start["dpct"]["panel"]["pattern_ids"] == [1, 2, 3, 4, 5]


def test_each_event_carries_the_frame_identity_and_its_receipt() -> None:
    camera = FakeCamera()
    stage = FakeStage()
    led = FakeIllumination([camera])
    engine = RunEngine({})
    documents = _documents(engine)

    _run_plan(
        engine,
        detectors=[camera],
        stage=stage,
        illumination=led,
        run_id="run-20261103-a",
    )

    events = _events(documents)
    assert len(events) == 4
    assert events[0]["descriptor"] == events[3]["descriptor"]
    for ordinal, event in enumerate(events):
        data = event["data"]
        assert data["dpct-run-id"] == "run-20261103-a"
        assert data["dpct-detector-id"] == "dhm"
        assert data["dpct-sequence"] == ordinal
        assert data["dpct-plane-ordinal"] == ordinal
        assert data["dpct-pattern-id"] == ordinal + 1
        assert data["dpct-scan-index"] == 0
        assert data["dpct-producer-sequence"] == ordinal + 1
        assert data["dpct-command"] == f"{11 + ordinal}"
        assert data["dpct-command-status"] == "sent"
        assert data["dpct-command-echo"] == ""
        assert data["dpct-command-sent-ns"] == -1
        assert data["dpct-position-z"] == pytest.approx(1.25)
        assert data["dpct-position-quality"] == "measured"


def test_a_missing_frame_aborts_with_a_named_partial_report() -> None:
    camera = FakeCamera()
    camera.drop_ordinals = {2}
    stage = FakeStage()
    led = FakeIllumination([camera])
    engine = RunEngine({})
    documents = _documents(engine)

    outcome = _run_plan(
        engine,
        detectors=[camera],
        stage=stage,
        illumination=led,
        run_id="run-20261103-b",
        frame_timeout_s=0.05,
        frame_poll_s=0.005,
    )

    assert outcome.status is PlanStatus.ABORTED
    assert outcome.reason is PlanAbortReason.CAMERA_TIMEOUT
    assert [frame.sequence for frame in outcome.produced] == [0, 1]
    assert [frame.sequence for frame in outcome.missing] == [2, 3]
    assert [frame.pattern_id for frame in outcome.missing] == [3, 4]
    assert "ordinal 2" in outcome.detail
    assert "pattern 3" in outcome.detail
    # The third pattern was written exactly once, and the panel was blanked.
    assert led.commands == ["00", "01", "11", "12", "13", "00"]
    assert led.commands.count("13") == 1
    assert camera.unstage_calls == 1
    assert stage.unstage_calls == 1
    assert led.unstage_calls == 1
    # The panel was blanked after the failed pattern, before anything unstaged.
    assert led.log[-2:] == ["led:00", "led:unstage"]
    assert _stop_statuses(documents) == ["abort"]
    assert len(_events(documents)) == 2


def test_an_uncertain_command_aborts_without_a_resend() -> None:
    camera = FakeCamera()
    stage = FakeStage()
    led = FakeIllumination([camera])
    led.uncertain_pattern = 2
    engine = RunEngine({})
    documents = _documents(engine)

    outcome = _run_plan(
        engine,
        detectors=[camera],
        stage=stage,
        illumination=led,
        run_id="run-20261103-c",
    )

    assert outcome.status is PlanStatus.ABORTED
    assert outcome.reason is PlanAbortReason.ILLUMINATION_UNCERTAIN
    assert "not resent" in outcome.detail
    # 12 is written once, and no further pattern is written.
    assert led.commands == ["00", "01", "11", "12", "00"]
    assert led.commands.count("12") == 1
    assert [frame.sequence for frame in outcome.produced] == [0]
    assert [frame.sequence for frame in outcome.missing] == [1, 2, 3]
    assert camera.unstage_calls == 1
    assert led.unstage_calls == 1
    assert _stop_statuses(documents) == ["abort"]


@pytest.mark.parametrize(
    ("interrupt", "exit_status"),
    [(RunEngine.stop, "success"), (RunEngine.abort, "abort")],
    ids=["stop", "abort"],
)
def test_a_stop_or_abort_still_blanks_and_unstages(
    interrupt: Callable[[RunEngine], Any], exit_status: str
) -> None:
    camera = FakeCamera()
    camera.drop_ordinals = {1, 2, 3}
    stage = FakeStage()
    led = FakeIllumination([camera])
    engine = RunEngine({})
    documents = _documents(engine)
    first_pattern = threading.Event()
    interrupted_too_early: list[bool] = []
    original_pattern = led.pattern

    async def observe_pattern(pattern_id: int) -> CommandReceipt:
        receipt = await original_pattern(pattern_id)
        first_pattern.set()
        return receipt

    led.pattern = observe_pattern  # type: ignore[method-assign]

    def plan() -> Any:
        yield from dpct_plan(
            [camera],
            stage,
            led,
            run_id="run-20261103-d",
            frame_timeout_s=30.0,
            frame_poll_s=0.01,
        )

    def interrupt_when_waiting() -> None:
        if first_pattern.wait(timeout=5.0):
            time.sleep(0.2)
            interrupt(engine)
        else:
            interrupted_too_early.append(True)

    interrupter = threading.Thread(target=interrupt_when_waiting)
    interrupter.start()
    try:
        with pytest.raises(RunEngineInterrupted):
            engine(plan())
    finally:
        interrupter.join(timeout=5.0)

    assert not interrupted_too_early
    assert not interrupter.is_alive()
    # The panel was blanked and every device was released.
    assert led.commands[-1] == "00"
    assert led.commands.count("00") >= 2
    assert camera.unstage_calls == 1
    assert stage.unstage_calls == 1
    assert led.unstage_calls == 1
    assert camera.staged is False
    assert led.staged is False
    assert _stop_statuses(documents) == [exit_status]


def test_a_second_detector_gets_its_own_frame_and_ordinal() -> None:
    dhm = FakeCamera("dhm")
    back = FakeCamera("back")
    led = FakeIllumination([dhm, back])
    engine = RunEngine({})
    documents = _documents(engine)

    outcome = _run_plan(
        engine,
        detectors=[dhm, back],
        stage=None,
        illumination=led,
        run_id="run-20261103-e",
    )

    assert outcome.status is PlanStatus.COMPLETE
    # One TTL per pattern still serves both cameras.
    assert led.commands == ["00", "01", "11", "12", "13", "14", "00"]
    assert dhm.counter is not None and dhm.counter.value == 4
    assert back.counter is not None and back.counter.value == 4
    assert dhm.prepare_values[0].number_of_events == 4
    assert back.prepare_values[0].number_of_events == 4

    events = _events(documents)
    assert len(events) == 8
    identities = [
        (
            event["data"]["dpct-detector-id"],
            event["data"]["dpct-sequence"],
            event["data"]["dpct-pattern-id"],
        )
        for event in events
    ]
    assert identities == [
        (detector, ordinal, ordinal + 1)
        for ordinal in range(4)
        for detector in ("dhm", "back")
    ]
    for event in events:
        assert event["data"]["dpct-command"] == f"{11 + event['data']['dpct-sequence']}"
        # Without a stage there is no position to claim.
        assert "dpct-position-quality" not in event["data"]


def test_the_sequence_is_editable_without_redefining_the_patterns() -> None:
    camera = FakeCamera()
    stage = FakeStage()
    led = FakeIllumination([camera])
    engine = RunEngine({})
    documents = _documents(engine)

    sequence = [PatternStep(2, "group2"), PatternStep(1, "group1")]
    outcome = _run_plan(
        engine,
        detectors=[camera],
        stage=stage,
        illumination=led,
        run_id="run-20261103-f",
        sequence=sequence,
    )

    assert outcome.status is PlanStatus.COMPLETE
    assert led.commands == ["00", "01", "12", "11", "00"]
    assert [event["data"]["dpct-pattern-id"] for event in _events(documents)] == [2, 1]


def test_an_empty_detector_list_is_refused() -> None:
    led = FakeIllumination()

    with pytest.raises(ValueError, match="at least one detector"):
        next(dpct_plan([], None, led, run_id="run"))


def test_a_detector_without_a_frame_source_is_named() -> None:
    led = FakeIllumination()

    with pytest.raises(ValueError, match="'dhm' has no frame source"):
        next(dpct_plan([_NoFrameSource()], None, led, run_id="run"))


def test_a_detector_that_cannot_be_armed_is_named() -> None:
    led = FakeIllumination()
    screen = _ScreenOnly()

    with pytest.raises(ValueError, match="cannot be armed"):
        next(dpct_plan([screen], None, led, run_id="run"))  # type: ignore[list-item]


def test_two_detectors_under_one_name_are_refused() -> None:
    led = FakeIllumination()

    with pytest.raises(ValueError, match="detector names must be unique"):
        next(dpct_plan([FakeCamera("dhm"), FakeCamera("dhm")], None, led, run_id="run"))


def test_an_illumination_capability_without_the_patterns_is_refused() -> None:
    led = FakeIllumination(
        capability=IlluminationCapability(pattern_ids=(1, 2), center_pattern_id=None)
    )

    with pytest.raises(ValueError, match=r"asks for \[3, 4\]"):
        next(dpct_plan([FakeCamera()], None, led, run_id="run"))


def test_a_pattern_outside_the_dpct_set_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        PatternStep(5, "centre")
    with pytest.raises(ValueError, match="at least one pattern"):
        next(
            dpct_plan(
                [FakeCamera()],
                None,
                FakeIllumination(),
                run_id="run",
                sequence=[],
            )
        )
    with pytest.raises(ValueError, match="request each pattern once"):
        next(
            dpct_plan(
                [FakeCamera()],
                None,
                FakeIllumination(),
                run_id="run",
                sequence=[PatternStep(1), PatternStep(1)],
            )
        )


def test_a_stage_without_a_position_surface_is_refused() -> None:
    class _MovingOnly:
        def move(self, position: Any) -> None:
            return None

    with pytest.raises(ValueError, match="position surface"):
        next(
            dpct_plan(
                [FakeCamera()],
                _MovingOnly(),  # type: ignore[arg-type]
                FakeIllumination(),
                run_id="run",
            )
        )


@pytest.mark.parametrize("bound", [0.0, -1.0, float("nan"), float("inf"), True, "1.0"])
def test_a_wait_bound_must_be_finite_and_positive(bound: Any) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        next(
            dpct_plan(
                [FakeCamera()],
                None,
                FakeIllumination(),
                run_id="run",
                frame_timeout_s=bound,
            )
        )


class _ScreenOnly:
    """A detector-shaped object with no preparation surface."""

    name = "screen"

    @property
    def frame_count(self) -> _FrameCounter:
        return _FrameCounter("screen-count")


class _NoFrameSource:
    """A detector that can be staged and armed but counts no frames."""

    name = "dhm"

    def stage(self) -> Status:
        return _CompletedStatus()

    def unstage(self) -> Status:
        return _CompletedStatus()

    def prepare(self, value: Any) -> Status:
        return _CompletedStatus()
