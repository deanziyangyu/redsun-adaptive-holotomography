"""Plan-level tests for the live-stream helpers.

A real :class:`bluesky.RunEngine` drives a fake flyer that writes stream
documents, so these tests assert the document order, the declared stream and
the cleanup the live presenter and the DPCT plan rely on. No hardware, no Qt.
"""

from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps
from bluesky.utils import RunEngineInterrupted
from event_model import ComposeStreamResource
from ophyd_async.core import TriggerInfo

from redsun_aht.plans import (
    collect_while_waiting_plan,
    live_stream,
    prepare_live_stream,
    teardown_live_stream,
)
from redsun_aht.protocols import LiveFlyer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from bluesky.protocols import Reading, Status, StreamAsset
    from bluesky.utils import Msg
    from event_model import DataKey, StreamRange

STREAM = "live_stream"

#: Number of frames a finite fake writes; the fake reports one collection per
#: event, and the stream datum's indices are what count the stored frames.
FRAMES = 5


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


class FakeFlyer:
    """A live flyer that writes a stream resource instead of real frames.

    With ``frames`` set, the flyer reports exactly that many written frames as
    soon as it is kicked off, which makes a finite capture deterministic. With
    ``frames=None`` it reports frames at ``rate_hz`` while writing, so a run
    kept open by an ``until`` plan has something to collect.
    """

    def __init__(
        self,
        *,
        frames: int | None = FRAMES,
        rate_hz: float = 200.0,
        name: str = "camera",
    ) -> None:
        self.name = name
        self.buffer: Any = object()
        self.frames = frames
        self.rate_hz = rate_hz
        self.prepare_calls = 0
        self.kickoff_calls = 0
        self.complete_calls = 0
        self.collect_calls = 0
        self.stage_calls = 0
        self.unstage_calls = 0
        self.stop_calls = 0
        self.staged = False
        self.trigger_info: Any = None
        self.completed_at: float | None = None
        self._writing = False
        self._written = 0
        self._started_at = 0.0
        self._last_emitted = 0
        self._resource_emitted = False
        self._bundle = ComposeStreamResource()(
            mimetype="application/x-zarr",
            uri=f"file:///fake/{name}.zarr",
            data_key=self.image_key,
            parameters={"chunk_shape": (1, 4, 4)},
            uid=None,
        )

    @property
    def image_key(self) -> str:
        """Return the stream data key of this flyer's image."""
        return f"{self.name}-image"

    def stage(self) -> Status:
        self.staged = True
        self.stage_calls += 1
        return _CompletedStatus()

    def unstage(self) -> Status:
        self.staged = False
        self.unstage_calls += 1
        return _CompletedStatus()

    def prepare(self, value: Any) -> Status:
        self.trigger_info = value
        self.prepare_calls += 1
        return _CompletedStatus()

    def kickoff(self) -> Status:
        self.kickoff_calls += 1
        self._writing = True
        self._started_at = time.monotonic()
        return _CompletedStatus()

    def complete(self) -> Status:
        self.complete_calls += 1
        if self._writing:
            self._written = self.get_index()
        self._writing = False
        self.completed_at = time.monotonic()
        return _CompletedStatus()

    def stop(self, success: bool = True) -> None:
        self.stop_calls += 1
        self._writing = False

    def get_index(self) -> int:
        if self.frames is not None:
            return self.frames if self.kickoff_calls else 0
        if not self._writing:
            return self._written
        return int((time.monotonic() - self._started_at) * self.rate_hz)

    def read(self) -> dict[str, Reading[Any]]:
        return {}

    def describe(self) -> dict[str, DataKey]:
        return {}

    def describe_collect(self) -> dict[str, DataKey]:
        return {
            self.image_key: {
                "source": f"fake://{self.name}",
                "shape": [1, 4, 4],
                "dtype": "array",
                "dtype_numpy": "<u2",
                "external": "STREAM:",
            }
        }

    async def collect_asset_docs(
        self, index: int | None = None
    ) -> AsyncIterator[StreamAsset]:
        self.collect_calls += 1
        if not self._resource_emitted:
            self._resource_emitted = True
            yield "stream_resource", self._bundle.stream_resource_doc
        stop = self.get_index() if index is None else index
        if stop > self._last_emitted:
            indices: StreamRange = {"start": self._last_emitted, "stop": stop}
            self._last_emitted = stop
            yield "stream_datum", self._bundle.compose_stream_datum(indices)


def _documents(engine: RunEngine) -> list[tuple[str, Any]]:
    documents: list[tuple[str, Any]] = []

    def record(name: str, doc: Any) -> None:
        documents.append((name, doc))

    engine.subscribe(record)  # type: ignore[no-untyped-call]
    return documents


def _run_keys(engine: RunEngine) -> list[tuple[str, Any]]:
    keys: list[tuple[str, Any]] = []

    def record(msg: Msg) -> None:
        keys.append((msg.command, msg.run))

    engine.msg_hook = record  # type: ignore[assignment]
    return keys


def test_live_stream_writes_a_nested_capture_run() -> None:
    flyer = FakeFlyer(frames=FRAMES)
    assert isinstance(flyer, LiveFlyer)
    engine = RunEngine({})
    documents = _documents(engine)
    keys = _run_keys(engine)
    returned: list[str] = []

    def plan() -> Any:
        outer = yield from bps.open_run()
        yield from bps.stage_all(flyer)
        yield from prepare_live_stream(
            [flyer], TriggerInfo(number_of_events=FRAMES), STREAM, declare=False
        )
        inner = yield from live_stream([flyer], STREAM, parent=outer)
        returned.append(inner)
        yield from bps.close_run()
        return outer, inner

    result = engine(plan())
    assert isinstance(result, tuple)
    outer_uid, inner_uid = result
    assert returned == [inner_uid]

    assert [name for name, _ in documents] == [
        "start",
        "start",
        "descriptor",
        "stream_resource",
        "stream_datum",
        "stop",
        "stop",
    ]
    outer_start, capture_start = documents[0][1], documents[1][1]
    assert outer_start["uid"] == outer_uid
    assert capture_start["uid"] == inner_uid
    assert capture_start["purpose"] == "capture"
    assert capture_start["parent"] == outer_uid
    assert capture_start["median_scan"] is None

    descriptor = documents[2][1]
    assert descriptor["name"] == STREAM
    assert descriptor["run_start"] == inner_uid
    data_keys = descriptor["data_keys"]
    assert set(data_keys) == {flyer.image_key}
    image_key = data_keys[flyer.image_key]
    assert image_key["external"] == "STREAM:"
    assert image_key["shape"] == [1, 4, 4]
    assert image_key["dtype"] == "array"

    resource = documents[3][1]
    assert resource["data_key"] == flyer.image_key
    assert resource["run_start"] == inner_uid
    assert resource["uri"] == f"file:///fake/{flyer.name}.zarr"

    datum = documents[4][1]
    assert datum["descriptor"] == descriptor["uid"]
    assert datum["indices"] == {"start": 0, "stop": FRAMES}
    # Bluesky numbers the stream's events from one, however the store indices
    # are numbered; a consumer pairing the two must not assume they are equal.
    assert datum["seq_nums"] == {"start": 1, "stop": FRAMES + 1}

    assert documents[5][1]["exit_status"] == "success"
    assert documents[6][1]["exit_status"] == "success"

    # The capture is keyed so it can nest beside the caller's default run.
    assert [key for command, key in keys if command == "open_run"] == [
        None,
        "capture",
    ]
    assert [key for command, key in keys if command == "close_run"] == [
        "capture",
        None,
    ]

    assert flyer.trigger_info.number_of_events == FRAMES
    assert flyer.prepare_calls == 1
    assert flyer.kickoff_calls == 1
    assert flyer.complete_calls == 1
    assert flyer.unstage_calls == 1
    assert flyer.collect_calls >= 1
    assert flyer.staged is False


def test_prepare_live_stream_prepares_then_declares() -> None:
    flyer = FakeFlyer(frames=2)
    engine = RunEngine({})
    documents = _documents(engine)

    def plan() -> Any:
        yield from bps.open_run()
        yield from bps.stage_all(flyer)
        yield from prepare_live_stream(
            [flyer], TriggerInfo(number_of_events=2), "preview"
        )
        yield from bps.close_run()

    engine(plan())

    assert [name for name, _ in documents] == ["start", "descriptor", "stop"]
    assert documents[1][1]["name"] == "preview"
    assert flyer.prepare_calls == 1
    assert flyer.kickoff_calls == 0
    assert flyer.trigger_info.number_of_events == 2


def test_until_ends_the_capture_after_the_wait_plan_returns() -> None:
    flyer = FakeFlyer(frames=None)
    engine = RunEngine({})
    documents = _documents(engine)
    until_finished_at: list[float] = []

    def wait_for_operator() -> Any:
        yield from bps.sleep(0.25)
        until_finished_at.append(time.monotonic())

    def plan() -> Any:
        yield from bps.stage_all(flyer)
        yield from prepare_live_stream(
            [flyer], TriggerInfo(number_of_events=0), STREAM, declare=False
        )
        return (
            yield from live_stream(
                [flyer], STREAM, until=wait_for_operator(), flush_period_s=0.05
            )
        )

    started = time.monotonic()
    engine(plan())
    elapsed = time.monotonic() - started

    assert until_finished_at
    assert flyer.completed_at is not None
    assert until_finished_at[0] <= flyer.completed_at
    # The capture ends when its wait plan returns; it is not a timer that runs
    # on after the operator action, and the flyer was told to write forever.
    assert elapsed < 3.0
    # Collected once while the wait plan slept, and once more at teardown.
    assert flyer.collect_calls >= 2
    assert flyer.complete_calls == 1
    assert flyer.unstage_calls == 1
    assert flyer.staged is False
    assert [doc["exit_status"] for name, doc in documents if name == "stop"] == [
        "success"
    ]
    # A finite stream still declares its stream before the first frame.
    names = [name for name, _ in documents]
    assert names[0] == "start"
    assert names[1] == "descriptor"
    assert "stream_datum" in names
    assert names[-1] == "stop"


def test_teardown_collects_and_unstages_after_a_failing_wait() -> None:
    flyer = FakeFlyer(frames=3)
    engine = RunEngine({})
    documents = _documents(engine)

    def failing_wait() -> Any:
        yield from bps.sleep(0.01)
        raise RuntimeError("the waiting plan failed")

    def plan() -> Any:
        yield from bps.stage_all(flyer)
        yield from prepare_live_stream(
            [flyer], TriggerInfo(number_of_events=3), STREAM, declare=False
        )
        return (yield from live_stream([flyer], STREAM, until=failing_wait()))

    with pytest.raises(RuntimeError, match="the waiting plan failed"):
        engine(plan())

    assert flyer.complete_calls == 1
    assert flyer.collect_calls >= 1
    assert flyer.unstage_calls == 1
    assert flyer.staged is False
    assert [doc["exit_status"] for name, doc in documents if name == "stop"] == ["fail"]


@pytest.mark.parametrize("failed_command", ["declare_stream", "kickoff"])
def test_capture_plan_unstages_and_closes_after_startup_failure(
    failed_command: str,
) -> None:
    """The plan itself cleans up before relying on RunEngine emergency cleanup."""
    flyer = FakeFlyer()
    plan = live_stream([flyer], STREAM)
    commands: list[str] = []

    with pytest.raises(RuntimeError, match="startup failed"):
        message = next(plan)
        while True:
            commands.append(message.command)
            if message.command == failed_command:
                message = plan.throw(RuntimeError("startup failed"))
            else:
                message = plan.send(None)

    assert commands[0] == "open_run"
    assert commands.index("unstage") > commands.index(failed_command)
    assert commands.index("close_run") > commands.index("unstage")
    assert "complete" not in commands


@pytest.mark.parametrize(
    ("interrupt", "exit_status"),
    [(RunEngine.stop, "success"), (RunEngine.abort, "abort")],
    ids=["stop", "abort"],
)
def test_teardown_collects_and_unstages_after_an_interrupt(
    interrupt: Callable[[RunEngine], Any], exit_status: str
) -> None:
    flyer = FakeFlyer(frames=None)
    engine = RunEngine({})
    documents = _documents(engine)
    started = threading.Event()
    interrupted_before_starting: list[bool] = []

    def wait_beside_the_operator() -> Any:
        started.set()
        while True:
            yield from bps.sleep(0.05)

    def plan() -> Any:
        yield from bps.stage_all(flyer)
        yield from prepare_live_stream(
            [flyer], TriggerInfo(number_of_events=0), STREAM, declare=False
        )
        return (
            yield from live_stream(
                [flyer], STREAM, until=wait_beside_the_operator(), flush_period_s=0.05
            )
        )

    def interrupt_when_running() -> None:
        if started.wait(timeout=5.0):
            time.sleep(0.1)
            interrupt(engine)
        else:
            interrupted_before_starting.append(True)

    interrupter = threading.Thread(target=interrupt_when_running)
    interrupter.start()
    try:
        with pytest.raises(RunEngineInterrupted):
            engine(plan())
    finally:
        interrupter.join(timeout=5.0)

    assert not interrupted_before_starting
    assert not interrupter.is_alive()
    assert flyer.complete_calls == 1
    assert flyer.collect_calls >= 1
    assert flyer.unstage_calls == 1
    assert flyer.staged is False
    assert [doc["exit_status"] for name, doc in documents if name == "stop"] == [
        exit_status
    ]


def test_callbacks_surround_completion_and_each_collection() -> None:
    flyer = FakeFlyer(frames=2)
    engine = RunEngine({})
    calls: list[tuple[str, str, bool | None]] = []

    def on_complete(detector: LiveFlyer, status: Status) -> Iterator[Any]:
        calls.append(("complete", detector.name, status.success))
        yield from ()

    def on_flush(detector: LiveFlyer) -> Iterator[Any]:
        calls.append(("flush", detector.name, None))
        yield from ()

    def plan() -> Any:
        yield from bps.stage_all(flyer)
        yield from prepare_live_stream(
            [flyer], TriggerInfo(number_of_events=2), STREAM, declare=False
        )
        yield from live_stream(
            [flyer], STREAM, on_complete=on_complete, on_flush=on_flush
        )

    engine(plan())

    assert calls == [
        ("complete", flyer.name, True),
        ("flush", flyer.name, None),
    ]


def _sleeper() -> Any:
    yield from bps.sleep(0.01)


@pytest.mark.parametrize(
    "factory",
    [
        lambda dets: prepare_live_stream(dets, TriggerInfo(number_of_events=1), STREAM),
        lambda dets: teardown_live_stream(dets, STREAM),
        lambda dets: collect_while_waiting_plan(
            _sleeper(), dets, STREAM, flush_period_s=0.01
        ),
        lambda dets: live_stream(dets, STREAM),
    ],
    ids=["prepare", "teardown", "collect-while-waiting", "live-stream"],
)
def test_an_empty_detector_list_is_refused(
    factory: Callable[[list[FakeFlyer]], Any],
) -> None:
    with pytest.raises(ValueError, match="at least one live detector is required"):
        next(factory([]))


def test_a_detector_without_a_buffer_is_named() -> None:
    flyer = FakeFlyer(frames=1)
    flyer.buffer = None

    with pytest.raises(ValueError, match="live detector 'camera' has no buffer signal"):
        next(prepare_live_stream([flyer], TriggerInfo(number_of_events=1), STREAM))


@pytest.mark.parametrize("flush_period_s", [0.0, -0.5, True, math.nan, math.inf])
def test_flush_period_must_be_finite_and_positive(flush_period_s: float) -> None:
    flyer = FakeFlyer(frames=1)

    with pytest.raises(ValueError, match="flush_period_s must be finite and positive"):
        next(teardown_live_stream([flyer], STREAM, flush_period_s=flush_period_s))


def test_empty_stream_name_and_parent_are_refused() -> None:
    flyer = FakeFlyer(frames=1)

    with pytest.raises(ValueError, match="stream_name must not be empty"):
        next(live_stream([flyer], ""))
    with pytest.raises(ValueError, match="parent must be a nonempty run UID"):
        next(live_stream([flyer], STREAM, parent="  "))
    with pytest.raises(ValueError, match="median_scan must be a nonempty run UID"):
        next(live_stream([flyer], STREAM, median_scan=""))
