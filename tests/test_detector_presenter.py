"""Tests for the detector presenter and its live-stream plan.

A real :class:`redsun.engine.RunEngine` drives the plan over a fake flyer whose
buffer is an ophyd-async signal, so the monitor produces real Event documents
and the presenter forwards the frames it is given. No hardware, no Qt, and no
session application is built here.

The monitor bookkeeping is the RunEngine's: ``bps.monitor`` subscribes the
current run to a signal, and the run's bundler clears those subscriptions when
it closes. These tests assert that both the engine's document subscription set
and each signal's listener count return to their starting size after a stop, so
a restart cannot leave a monitor from the previous run alive.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import bluesky.plan_stubs as bps
import numpy as np
import pytest
from bluesky.utils import RunEngineInterrupted
from event_model import ComposeStreamResource
from ophyd_async.core import (
    AsyncStatus,
    SignalR,
    StandardReadable,
    StandardReadableFormat,
    TriggerInfo,
    soft_signal_r_and_setter,
    soft_signal_rw,
)
from redsun import DevicesOf
from redsun.engine import Deferrals, RunEngine
from redsun.engine.actions import ActionState

from redsun_aht.contracts.devices import Roi
from redsun_aht.presenter.detector import (
    LIVE_STREAM,
    LIVE_VIEW_STREAM,
    STREAM,
    DescribesDetectors,
    DetectorDevice,
    DetectorPresenter,
)
from redsun_aht.protocols import LiveFlyer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from concurrent.futures import Future
    from pathlib import Path

    from bluesky.protocols import Reading, StreamAsset
    from event_model import StreamRange
    from event_model.documents import Event, EventDescriptor
    from ophyd_async.core import SignalRW

#: Shape of the fake frames, ``(height, width)``.
SHAPE = (4, 6)


class FakeFlyer(StandardReadable):
    """A live flyer whose buffer really publishes frames to its monitors.

    The settings are soft ophyd-async signals, so a setting the fake refuses
    fails in the same place a service refusal would: inside ``set``. The
    buffer's setter is what ``kickoff`` uses, so the Events the RunEngine
    composes are the ones a camera's service would have produced.
    """

    def __init__(
        self,
        name: str,
        *,
        frames: int = 3,
        value_offset: int = 0,
    ) -> None:
        with self.add_children_as_readables(StandardReadableFormat.CONFIG_SIGNAL):
            self.exposure = soft_signal_rw(float, initial_value=10.0)
            self.roi = soft_signal_rw(str, initial_value="0,0,6,4")
            self.pixel_dtype = soft_signal_rw(str, initial_value="uint8")
            self.Gain = soft_signal_rw(str, initial_value="0")
            self.sensor_size, _ = soft_signal_r_and_setter(
                np.ndarray, initial_value=np.array([6, 4])
            )
            self.last_setting_change, _ = soft_signal_r_and_setter(
                str, initial_value=""
            )
            self.setting_change_latency, _ = soft_signal_r_and_setter(
                str, initial_value=""
            )
        self.buffer, self._publish = soft_signal_r_and_setter(
            np.ndarray,
            initial_value=np.zeros(SHAPE, dtype=np.uint8),
            name=f"{name}-buffer",
        )
        self.properties: Mapping[str, SignalRW[str]] = {"Gain": self.Gain}
        self._bundle = ComposeStreamResource()(
            mimetype="application/x-zarr",
            uri=f"file:///fake/{name}.zarr",
            data_key=f"{name}-image",
            parameters={"chunk_shape": (1, *SHAPE)},
            uid=None,
        )
        self._resource_emitted = False
        self._last_emitted = 0
        super().__init__(name)
        self.frames = frames
        self.value_offset = value_offset
        self.prepare_calls = 0
        self.kickoff_calls = 0
        self.complete_calls = 0
        self.collect_calls = 0
        self.stage_calls = 0
        self.unstage_calls = 0
        self.stop_calls = 0
        self.staged = False
        self.trigger_info: TriggerInfo | None = None

    @AsyncStatus.wrap
    async def prepare(self, value: TriggerInfo) -> None:
        """Record the trigger information the plan asked for."""
        self.prepare_calls += 1
        self.trigger_info = value

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        """Publish the frames a live camera would publish."""
        self.kickoff_calls += 1
        for index in range(self.frames):
            self._publish(np.full(SHAPE, self.value_offset + index + 1, dtype=np.uint8))
            await asyncio.sleep(0.005)

    @AsyncStatus.wrap
    async def complete(self) -> None:
        """End the fake's writing."""
        self.complete_calls += 1

    def stop(self, success: bool = True) -> None:
        """Record an engine stop, as a camera device would."""
        self.stop_calls += 1

    def get_index(self) -> int:
        """Return the number of frames written so far."""
        return self.frames if self.kickoff_calls else 0

    async def describe_collect(self) -> dict[str, Any]:
        """Describe one collected stream, as ``declare_stream(collect=True)`` needs."""
        return {
            f"{self.name}-image": {
                "source": f"fake://{self.name}",
                "shape": [1, *SHAPE],
                "dtype": "array",
                "dtype_numpy": "|u1",
                "external": "STREAM:",
            }
        }

    async def collect_asset_docs(
        self, index: int | None = None
    ) -> AsyncIterator[StreamAsset]:
        """Collect one stream's resource and datum, as a stored write would."""
        self.collect_calls += 1
        if not self._resource_emitted:
            self._resource_emitted = True
            yield "stream_resource", self._bundle.stream_resource_doc
        stop = self.get_index() if index is None else index
        if stop > self._last_emitted:
            indices: StreamRange = {"start": self._last_emitted, "stop": stop}
            self._last_emitted = stop
            yield "stream_datum", self._bundle.compose_stream_datum(indices)

    def stage(self) -> AsyncStatus[None]:
        """Record the stage."""
        self.stage_calls += 1
        self.staged = True
        return super().stage()

    def unstage(self) -> AsyncStatus[None]:
        """Record the unstage."""
        self.unstage_calls += 1
        self.staged = False
        return super().unstage()


def _listener_count(signal: SignalR[Any]) -> int:
    """Return how many callbacks a signal's monitor cache holds."""
    return 0 if signal._cache is None else len(signal._cache._listeners)


def _monitor_counts(flyers: tuple[FakeFlyer, ...]) -> dict[str, int]:
    """Return the listener count of every signal the live plan monitors."""
    counts: dict[str, int] = {}
    for flyer in flyers:
        counts[f"{flyer.name}-buffer"] = _listener_count(flyer.buffer)
        counts[f"{flyer.name}-setting_changes"] = _listener_count(
            flyer.last_setting_change
        )
        counts[f"{flyer.name}-setting_latencies"] = _listener_count(
            flyer.setting_change_latency
        )
    return counts


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    """Fail unless *predicate* becomes true within *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached before the timeout")
        time.sleep(0.005)


def _start(engine: RunEngine, presenter: DetectorPresenter, *args: Any) -> Future[Any]:
    """Start the presenter's plan and wait until it offers the stream action."""
    offered = threading.Event()

    def note(name: str, state: str) -> None:
        if (name, state) == (STREAM.name, ActionState.OFFERED.value):
            offered.set()

    presenter.actions.sig_changed.connect(note)
    try:
        plan = presenter.plan_map()[LIVE_STREAM]["plan"]
        future: Future[Any] = engine(plan(*args))
        assert offered.wait(10), "the live stream never offered its action"
        return future
    finally:
        presenter.actions.sig_changed.disconnect(note)


def _stop(engine: RunEngine, future: Future[Any]) -> None:
    """Stop a running plan and consume the interruption it raises."""
    engine.stop()
    with pytest.raises(RunEngineInterrupted):
        future.result(timeout=10)


@pytest.fixture
def presenter() -> Iterator[DetectorPresenter]:
    """Return a presenter over one fake flyer, shut down afterwards."""
    flyer = FakeFlyer("cam", frames=3)
    controller = DetectorPresenter("det", detectors={"cam": flyer})
    try:
        yield controller
    finally:
        controller.shutdown()


def _value(frame: dict[str, Reading[Any]]) -> int:
    """Return the corner pixel of a forwarded buffer reading."""
    return int(frame["cam-buffer"]["value"][0, 0])


def test_live_run_forwards_frames_without_claiming_an_unpaired_roi() -> None:
    """Forward pixels with document time, without an unpaired ROI claim."""
    flyer = FakeFlyer("cam", frames=3)
    controller = DetectorPresenter("det", detectors={"cam": flyer})
    received: list[dict[str, Reading[Any]]] = []
    controller.sig_new_data.connect(received.append)
    documents: list[tuple[str, Any]] = []
    engine = RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))  # type: ignore[no-untyped-call]
    engine.subscribe(controller)  # type: ignore[no-untyped-call]
    try:
        future = _start(engine, controller, [flyer], 3, False)
        controller.actions.request(STREAM.name)
        # The monitor publishes the sensor's current frame when it opens too;
        # the frames this run produces are the non-zero ones.
        _wait_until(lambda: sum(1 for frame in received if _value(frame) != 0) >= 3)
        _stop(engine, future)

        assert [_value(frame) for frame in received if _value(frame) != 0] == [
            1,
            2,
            3,
        ]
        for frame in received:
            assert list(frame) == ["cam-buffer"]
            assert isinstance(frame["cam-buffer"]["timestamp"], float)
            assert "review-context" not in frame
            # The document time orders the frame; it is not exposure evidence.
            assert "exposure" not in frame["cam-buffer"]
        assert flyer.kickoff_calls >= 1
        assert [name for name, _ in documents].count("event") >= 3
        assert next(name for name, _ in documents) == "start"
        streams = {doc["name"] for kind, doc in documents if kind == "descriptor"}
        assert {
            f"cam_{LIVE_VIEW_STREAM}",
            "cam_setting_changes",
            "cam_setting_frame_latencies",
        } <= streams
    finally:
        controller.shutdown()


def test_preview_metadata_does_not_follow_a_newer_roi_revision() -> None:
    """A later ROI signal cannot be attached to pixels from an earlier frame."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    received: list[dict[str, Reading[Any]]] = []
    presenter.sig_new_data.connect(received.append)
    try:
        presenter.descriptor(
            cast(
                "EventDescriptor",
                {
                    "uid": "preview-desc",
                    "run_start": "run",
                    "data_keys": {"cam-buffer": {}},
                },
            )
        )
        old_pixels = np.zeros((4, 6), dtype=np.uint16)
        presenter._rois["cam"] = Roi(1, 1, 3, 2)
        presenter.event(
            cast(
                "Event",
                {
                    "uid": "preview-event",
                    "descriptor": "preview-desc",
                    "time": 1.0,
                    "data": {"cam-buffer": old_pixels},
                },
            )
        )

        assert len(received) == 1
        assert list(received[0]) == ["cam-buffer"]
        assert received[0]["cam-buffer"]["value"] is old_pixels
    finally:
        presenter.shutdown()


@pytest.mark.parametrize("selected", [1, 2], ids=["one-detector", "two-detectors"])
def test_one_or_two_detectors_can_be_selected(selected: int) -> None:
    """Monitor only the ticked detectors, and forward a frame from each."""
    flyers = (
        FakeFlyer("cam-a", frames=2, value_offset=0),
        FakeFlyer("cam-b", frames=2, value_offset=10),
    )
    chosen = flyers[:selected]
    presenter = DetectorPresenter(
        "det", detectors={flyer.name: flyer for flyer in flyers}
    )
    received: list[dict[str, Reading[Any]]] = []
    presenter.sig_new_data.connect(received.append)
    engine = RunEngine({})
    engine.subscribe(presenter)  # type: ignore[no-untyped-call]
    try:
        offered = threading.Event()
        presenter.actions.sig_changed.connect(
            lambda name, state: (
                offered.set()
                if (name, state) == (STREAM.name, ActionState.OFFERED.value)
                else None
            )
        )
        plan = presenter.plan_map()[LIVE_STREAM]["plan"]
        future: Future[Any] = engine(plan(chosen, 2, False))
        assert offered.wait(10)
        expected_monitors: dict[str, int] = {}
        for flyer in chosen:
            expected_monitors[f"{flyer.name}-buffer"] = 1
            expected_monitors[f"{flyer.name}-setting_changes"] = 1
            expected_monitors[f"{flyer.name}-setting_latencies"] = 1
        assert _monitor_counts(chosen) == expected_monitors

        def published(flyer: FakeFlyer) -> int:
            key = f"{flyer.name}-buffer"
            return sum(
                1
                for frame in received
                if key in frame and int(frame[key]["value"][0, 0]) != 0
            )

        presenter.actions.request(STREAM.name)
        _wait_until(lambda: all(published(flyer) >= 2 for flyer in chosen))
        _stop(engine, future)

        buffers = {
            key for frame in received for key in frame if key.endswith("-buffer")
        }
        assert buffers == {f"{flyer.name}-buffer" for flyer in chosen}
        for flyer in flyers:
            if flyer not in chosen:
                assert _monitor_counts((flyer,)) == {
                    f"{flyer.name}-buffer": 0,
                    f"{flyer.name}-setting_changes": 0,
                    f"{flyer.name}-setting_latencies": 0,
                }
    finally:
        presenter.shutdown()


def test_set_applies_a_setting_and_announces_the_readback() -> None:
    """Apply a writable setting and announce what the device read back."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    announced: list[tuple[str, str, Any]] = []
    presenter.sig_new_configuration.connect(lambda *args: announced.append(args))
    try:
        asyncio.run(presenter.set("cam-exposure", 5.0))

        assert announced == [("cam", "cam-exposure", 5.0)]
        assert asyncio.run(flyer.exposure.get_value()) == 5.0
    finally:
        presenter.shutdown()


def test_a_bare_setting_name_is_accepted_for_one_detector() -> None:
    """Accept a bare setting name when exactly one detector declares it."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    try:
        asyncio.run(presenter.set("exposure", 7.5))

        assert asyncio.run(flyer.exposure.get_value()) == 7.5
    finally:
        presenter.shutdown()


def test_the_view_signal_shape_reaches_the_same_setting() -> None:
    """Accept the view's ``(detector, property, value)`` emission unchanged."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    announced: list[tuple[str, str, Any]] = []
    presenter.sig_new_configuration.connect(lambda *args: announced.append(args))
    try:
        asyncio.run(presenter.set_property("cam", "exposure", 6.0))

        assert announced == [("cam", "cam-exposure", 6.0)]
        assert asyncio.run(flyer.exposure.get_value()) == 6.0
    finally:
        presenter.shutdown()


def test_a_refused_setting_leaves_the_value_unchanged() -> None:
    """Log a refused setting and announce nothing, so the view keeps its edit."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    announced: list[tuple[str, str, Any]] = []
    presenter.sig_new_configuration.connect(lambda *args: announced.append(args))
    try:
        asyncio.run(presenter.set("cam-exposure", "not a number"))

        assert announced == []
        assert asyncio.run(flyer.exposure.get_value()) == 10.0
    finally:
        presenter.shutdown()


def test_an_unknown_setting_is_ignored() -> None:
    """Ignore a setting neither the detector nor its properties declare."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    announced: list[tuple[str, str, Any]] = []
    presenter.sig_new_configuration.connect(lambda *args: announced.append(args))
    try:
        asyncio.run(presenter.set("cam-does-not-exist", 1))

        assert announced == []
    finally:
        presenter.shutdown()


def test_a_roi_change_during_a_plan_lands_between_two_messages() -> None:
    """Defer a ROI asked for during a plan until the current message finishes."""
    flyer = FakeFlyer("cam")
    engine = RunEngine({})
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    presenter.setup(Deferrals(engine))
    announced: list[tuple[str, str, Any]] = []
    presenter.sig_new_configuration.connect(lambda *args: announced.append(args))
    release = asyncio.Event()
    entered = threading.Event()

    async def hold() -> None:
        entered.set()
        await release.wait()

    def blocking_plan() -> Any:
        yield from bps.wait_for([hold])

    try:
        future = engine(blocking_plan())
        assert entered.wait(10)

        asyncio.run(presenter.set("cam-roi", "1,1,3,2"))
        # A ROI applied inside a point would put two frame shapes in one
        # stream, so the request waits for the message under way.
        assert asyncio.run(flyer.roi.get_value()) == "0,0,6,4"
        assert announced == []

        engine.loop.call_soon_threadsafe(release.set)
        future.result(timeout=10)
        _wait_until(lambda: bool(announced))

        assert asyncio.run(flyer.roi.get_value()) == "1,1,3,2"
        assert announced == [("cam", "cam-roi", "1,1,3,2")]
    finally:
        presenter.shutdown()


def test_stop_and_restart_leave_no_stale_subscriptions() -> None:
    """Stop and start again; the engine's subscriptions and monitors return home."""
    flyer = FakeFlyer("cam", frames=2)
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    received: list[dict[str, Reading[Any]]] = []
    presenter.sig_new_data.connect(received.append)
    engine = RunEngine({})
    engine.subscribe(presenter)  # type: ignore[no-untyped-call]

    def documents(name: str, doc: Any) -> None:
        del name, doc

    engine.subscribe(documents)  # type: ignore[no-untyped-call]
    subscriptions_before = len(engine.dispatcher._token_mapping)
    monitors_before = _monitor_counts((flyer,))
    assert set(monitors_before.values()) == {0}

    def run_and_stop() -> None:
        seen = len(received)
        future = _start(engine, presenter, [flyer], 2, False)
        presenter.actions.request(STREAM.name)
        _wait_until(lambda: len(received) >= seen + flyer.frames)
        assert set(_monitor_counts((flyer,)).values()) == {1}, "the run monitors"
        _stop(engine, future)
        assert _monitor_counts((flyer,)) == monitors_before
        assert len(engine._run_bundlers) == 0, "no run bundler survives the stop"
        assert len(engine.dispatcher._token_mapping) == subscriptions_before
        assert presenter._live_streams == {}, "the run's routing entries are gone"

    try:
        run_and_stop()
        # A restart reuses the same presenter and engine; nothing from the
        # previous run may still be subscribed.
        run_and_stop()
    finally:
        presenter.shutdown()


def test_a_finite_capture_uses_the_frame_count_as_trigger_info() -> None:
    """Ask the detectors to write exactly the requested number of frames."""
    flyer = FakeFlyer("cam", frames=1)
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    engine = RunEngine({})
    try:
        future = _start(engine, presenter, [flyer], 1, False)
        presenter.actions.request(STREAM.name)
        _wait_until(lambda: flyer.complete_calls >= 1)
        _stop(engine, future)

        assert flyer.prepare_calls >= 1
        assert flyer.trigger_info is not None
        assert flyer.trigger_info.number_of_events == 1
        assert flyer.complete_calls >= 1
        assert flyer.unstage_calls >= 1
        assert flyer.staged is False
    finally:
        presenter.shutdown()


def test_write_forever_ends_when_the_stream_is_released() -> None:
    """Keep the capture open until the stream action is released."""
    flyer = FakeFlyer("cam", frames=2)
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    received: list[dict[str, Reading[Any]]] = []
    presenter.sig_new_data.connect(received.append)
    engine = RunEngine({})
    engine.subscribe(presenter)  # type: ignore[no-untyped-call]
    try:
        future = _start(engine, presenter, [flyer], 2, True)
        presenter.actions.request(STREAM.name)
        _wait_until(lambda: sum(1 for frame in received if _value(frame) != 0) >= 2)
        assert flyer.trigger_info is not None
        assert flyer.trigger_info.number_of_events == 0, "unbounded capture"

        presenter.actions.request(STREAM.name, on=False)
        _wait_until(lambda: flyer.complete_calls >= 1)
        _stop(engine, future)

        assert flyer.complete_calls == 1
        assert flyer.unstage_calls >= 1
        assert flyer.staged is False
    finally:
        presenter.shutdown()


def test_the_camera_device_is_a_selectable_live_detector(tmp_path: Path) -> None:
    """An unconnected ``MMCamera`` satisfies the plan's detector protocol.

    This is the boundary between R03's device and R04's plan: the plan selects
    session devices by :class:`~redsun_aht.protocols.LiveFlyer`, and the
    presenter's settings map needs the device signals. Nothing is connected and
    no hardware is touched; only the device's declared surface is checked.
    """
    from redsun.path_provider import SessionPathProvider

    from redsun_aht.device.mmcore import MMCamera
    from redsun_aht.protocols import LiveFlyer

    camera = MMCamera(
        "R04-TEST-CAMERA:",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="detector"),
        name="cam",
    )

    # The device is a ``StandardDetector`` and the protocol composes several
    # Bluesky protocols, so mypy cannot express the instance check's narrowing;
    # the run-time check is what this test is about.
    surface: Any = camera
    assert isinstance(camera, LiveFlyer)
    assert isinstance(surface, DetectorDevice)
    assert camera.buffer.name == "cam-buffer"
    assert camera.settings_revision.name == "cam-settings_revision"


def test_the_session_can_resolve_the_presenter_annotations() -> None:
    """Resolve every annotation a Redsun session evaluates at run time.

    ``redsun.session`` reads a constructor with
    ``inspect.signature(..., eval_str=True)``, resolves ``setup`` the same way,
    and describes a plan-map entry by resolving the plan's annotations. A name
    imported only under ``TYPE_CHECKING`` makes the presenter undeclarable, so
    each surface is resolved here exactly as the session does it.
    """
    class_signature = inspect.signature(DetectorPresenter, eval_str=True)
    init = inspect.signature(DetectorPresenter.__init__, eval_str=True)
    setup = inspect.signature(DetectorPresenter.setup, eval_str=True)
    plan = inspect.signature(DetectorPresenter.live_stream, eval_str=True)

    detectors = init.parameters["detectors"].annotation
    assert get_origin(detectors) is Annotated
    assert get_args(detectors)[0] == Mapping[str, DetectorDevice]
    assert detectors == DevicesOf[DetectorDevice]
    assert "detectors" in class_signature.parameters

    assert setup.parameters["deferrals"].annotation == Deferrals | None
    assert plan.parameters["detectors"].annotation == Sequence[LiveFlyer]
    assert "stream" in plan.parameters

    for target in (
        DetectorPresenter.__init__,
        DetectorPresenter.setup,
        DetectorPresenter.live_stream,
        DetectorPresenter.plan_map,
        DetectorPresenter.detector_descriptors,
        DetectorPresenter.detector_readings,
        DetectorPresenter.descriptor,
        DetectorPresenter.event,
        DetectorPresenter.stop,
        DetectorPresenter.set,
        DetectorPresenter.set_property,
    ):
        get_type_hints(target, include_extras=True)


def test_the_presenter_describes_its_detectors() -> None:
    """Describe, read and lay out every detector, and expose the view protocol."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    try:
        assert isinstance(presenter, DescribesDetectors)
        assert "cam-exposure" in presenter.detector_readings()
        assert presenter.detector_layer_specs()["cam"]["shape"] == (4, 6)
        descriptors = presenter.detector_descriptors()
        assert descriptors["cam-sensor_size"]["source"].endswith(":readonly")
        assert not descriptors["cam-exposure"]["source"].endswith(":readonly")
        assert not descriptors["cam-roi"]["source"].endswith(":readonly")
        assert not descriptors["cam-Gain"]["source"].endswith(":readonly")
    finally:
        presenter.shutdown()


def test_an_event_from_an_unrouted_descriptor_is_ignored() -> None:
    """Ignore an Event whose descriptor carried no tracked buffer."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    received: list[Any] = []
    presenter.sig_new_data.connect(received.append)
    try:
        presenter.event(
            cast(
                "Event",
                {
                    "descriptor": "never-seen",
                    "time": 0.0,
                    "data": {"x": 1.0},
                },
            )
        )

        assert received == []
    finally:
        presenter.shutdown()


def test_a_descriptor_without_a_tracked_buffer_is_not_routed() -> None:
    """Remember only the descriptors the plan's buffer monitors produced."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    try:
        presenter.descriptor(
            cast(
                "EventDescriptor",
                {
                    "uid": "collect-desc",
                    "run_start": "run",
                    "data_keys": {f"{flyer.name}-image": {}},
                },
            )
        )

        assert presenter._live_streams == {}
    finally:
        presenter.shutdown()


def test_the_action_is_the_stream_toggle_and_the_plan_is_listed() -> None:
    """Expose the live stream plan and its start/stop action."""
    flyer = FakeFlyer("cam")
    presenter = DetectorPresenter("det", detectors={"cam": flyer})
    try:
        plans = presenter.plan_map()
        assert list(plans) == [LIVE_STREAM]
        assert plans[LIVE_STREAM]["plan"] == presenter.live_stream
        assert STREAM.name == "stream"
        assert STREAM.toggle_states == ("start", "stop")
        assert LIVE_VIEW_STREAM == "live_view"
    finally:
        presenter.shutdown()


def test_a_session_without_a_flyable_detector_names_the_role() -> None:
    """Refuse a run with no selectable detector, naming the role in the error."""
    presenter = DetectorPresenter("det", detectors={})
    engine = RunEngine({})
    try:
        plan = presenter.plan_map()[LIVE_STREAM]["plan"]
        with pytest.raises(ValueError, match="declare a camera device"):
            engine(plan()).result(timeout=10)
    finally:
        presenter.shutdown()
