"""Detector settings, live document routing and the live-stream plan.

Ported from ``redsun_searching_is_imaging/presenter/detector.py`` at donor
revision ``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed;
copyright the Redsun contributors. The plan's message order, the writable
setting map, the Event routing shape and the ``set`` semantics are the
donor's.

Changed for :mod:`redsun_aht`:

- The storage capture is this package's :func:`redsun_aht.plans.live.live_stream`
  and the prepare step is :func:`redsun_aht.plans.live.prepare_live_stream`; the
  searching bundle's nested-capture wrapper and its planner contract are gone.
  The outer run is closed in a ``finally`` here, where the donor left it to its
  session framework, so a plain engine teardown and a session teardown produce
  the same documents.
- ``plan_map`` returns the plan itself rather than a wrapped plan entry: the
  planner-contract decorator carried searching-bundle review context.
- The Event payload carries the detector's buffer reading, and nothing else. ROI
  is omitted because the buffer and ROI are not captured atomically. The
  searching bundle's ``review-context`` key is not carried over. A reading's
  timestamp is the Event document's time; no
  exposure is inferred from it, because a document time is when the document
  was emitted, not how long the sensor was exposed.
- ``set`` takes one canonical setting name, ``"<detector>-<setting>"`` (or the
  bare setting when a session has exactly one detector), applies it through the
  device's settings signal, reads it back and announces the applied value. An
  unknown or refused setting is logged and leaves the value unchanged, so the
  view keeps the pending edit instead of showing a value the camera refused.
- The ROI is followed from the device signal while the presenter lives and from
  configuration documents. It is not attached to preview frames because the
  buffer and ROI are not captured atomically.
- The presenter holds no monitor of its own. The live plan puts the buffer, the
  last setting change and the setting-change latency under ``bps.monitor``, so
  the RunEngine owns those subscriptions and tears them down with the run. Each
  buffer gets its own ``"<detector>_live_view"`` stream, because an Event stream
  carries one descriptor shape and two cameras have two data keys.
- ``set_live_monitor_signals`` and the donor's ``get_plan_block`` decoration are
  not carried over: the live monitor surface is the three signals above, and the
  plan map needs no block description.
"""

from __future__ import annotations

import contextlib

# These names are imported at run time on purpose: a Redsun session resolves
# the constructor and ``setup`` annotations with
# ``inspect.signature(..., eval_str=True)`` against this module's globals, and
# a plan map is described by resolving the plan's annotations the same way.
# Moving any of them under ``TYPE_CHECKING`` makes the presenter undeclarable.
from collections.abc import Callable, Mapping, Sequence  # noqa: TC003
from functools import partial
from typing import Any, Protocol, runtime_checkable

import bluesky.plan_stubs as bps
import numpy as np  # noqa: TC002 - DetectorDevice annotations resolve at runtime
from bluesky.protocols import Descriptor, Reading  # noqa: TC002
from bluesky.utils import MsgGenerator  # noqa: TC002
from event_model import DataKey, DocumentRouter
from event_model.documents import (  # noqa: TC002
    Event,
    EventDescriptor,
    RunStop,
)
from ophyd_async.core import SignalR, SignalRW, TriggerInfo
from psygnal import Signal
from redsun import DevicesOf, slot
from redsun.aio import run_coro
from redsun.engine import Deferrals  # noqa: TC002 - setup resolves at runtime
from redsun.engine.actions import ActionManager, PlanAction, continuous
from redsun.log import Loggable

from redsun_aht.contracts.devices import Roi
from redsun_aht.plans.live import live_stream as capture_live_stream
from redsun_aht.plans.live import prepare_live_stream
from redsun_aht.protocols import LiveFlyer

__all__ = [
    "LIVE_STREAM",
    "LIVE_VIEW_STREAM",
    "STREAM",
    "DescribesDetectors",
    "DetectorDevice",
    "DetectorPresenter",
]

#: The stream the nested capture declares and collects into.
LIVE_STREAM = "live_stream"

#: The suffix of the event stream a monitored detector buffer is emitted under.
#:
#: Each detector gets ``"<detector>_live_view"``: an Event stream carries one
#: descriptor shape, so two detectors cannot share one stream name without one
#: camera's data keys contradicting the other's.
LIVE_VIEW_STREAM = "live_view"

#: The action of :meth:`DetectorPresenter.live_stream` writing frames to disk.
STREAM = PlanAction(
    name="stream",
    description="Toggle data streaming to disk.",
    toggle_states=("start", "stop"),
)


@runtime_checkable
class DetectorDevice(LiveFlyer, Protocol):
    """A live flyer that also exposes the settings a detector view edits.

    The searching bundle called this surface ``DetectorProtocol``. It is the
    settings-and-capture surface layered on an ophyd-async ``StandardDetector``:
    the Bluesky methods keep their Bluesky meanings, and these attributes carry
    the signals the settings view and the live plan read.

    ``last_setting_change`` and ``setting_change_latency`` are optional on a
    device: a plan monitors them only when the device declares them.
    """

    exposure: SignalRW[float]
    """The exposure time, in milliseconds, as the camera service carries it."""

    roi: SignalRW[str]
    """The sensor region, ``"x,y,width,height"``, as the service carries it."""

    pixel_dtype: SignalRW[str]
    """The readout dtype the camera stores frames in."""

    sensor_size: SignalR[np.ndarray]
    """The full sensor as ``(width, height)``, whatever the ROI."""

    properties: Mapping[str, SignalRW[str]]
    """The camera properties the service exposes, keyed by property name."""

    async def read_configuration(self) -> dict[str, Reading[Any]]:
        """Return the configuration readings of the detector."""
        ...

    async def describe_configuration(self) -> dict[str, DataKey]:
        """Return the configuration descriptors of the detector."""
        ...


@runtime_checkable
class DescribesDetectors(Protocol):
    """The part of a presenter a detector view reads its panels from."""

    def detector_descriptors(self) -> dict[str, Descriptor]:
        """Return the configuration descriptors of every detector."""
        ...

    def detector_readings(self) -> dict[str, Reading[Any]]:
        """Return the configuration readings of every detector."""
        ...


def _writable_settings(detector: DetectorDevice) -> dict[str, SignalRW[Any]]:
    """Return a detector's writable settings, keyed as the view names them.

    A key is the signal's data key without the device-name prefix. The
    detector's exposure, ROI and pixel dtype come first, then every property
    the service exposes, because a Micro-Manager property is writable exactly
    when it arrives as a ``SignalRW``.
    """
    signals: list[SignalRW[Any]] = [
        detector.exposure,
        detector.roi,
        detector.pixel_dtype,
    ]
    signals.extend(detector.properties.values())
    return {signal.name.removeprefix(f"{detector.name}-"): signal for signal in signals}


class DetectorPresenter(DocumentRouter, Loggable):
    """Presenter for detector configuration and live data routing.

    It offers ``live_stream`` and holds as ``actions`` the
    :class:`~redsun.engine.actions.ActionManager` that plan waits on. Live
    frames arrive as Event documents, the plan having put each detector's
    buffer under ``bps.monitor``, so every displayed frame is part of the run
    and ordered against its other documents.

    Parameters
    ----------
    name
        The presenter's name, used in log records.
    detectors
        The session's detectors, keyed by the name the plan and the view use.
    timeout
        Timeout in seconds for async configuration calls; ``None`` means
        ``1.0``. Kept for the donor's surface; no call in this module waits on
        it yet.

    Notes
    -----
    The writable setting map is taken when the presenter is built, so a session
    connects its detectors before constructing their presenter. This is the
    donor's build order, and it is what makes a Micro-Manager camera's service
    properties visible to the settings tree and to ``set``.
    """

    sig_new_configuration = Signal(str, str, object)
    """Emitted after a detector setting is applied, with the detector name, the
    canonical key of the setting and the value the device read back."""

    sig_new_data = Signal(object)
    """Emitted for every live frame carried by an Event document, as a
    ``dict[str, Reading[Any]]`` containing the ``<detector>-buffer`` reading.
    ROI is omitted because it may belong to a different setting revision than
    the pixels. The timestamp is the Event document time, not exposure evidence."""

    def __init__(
        self,
        name: str,
        *,
        detectors: DevicesOf[DetectorDevice],
        timeout: float | None = 1.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.timeout = timeout or 1.0
        self.detectors: dict[str, DetectorDevice] = dict(detectors)
        #: buffer data keys this presenter forwards, by descriptor uid
        self._live_streams: dict[str, list[str]] = {}
        #: the run each remembered descriptor belongs to, so a stopped run's
        #: routing entries do not accumulate for the life of the presenter
        self._descriptor_runs: dict[str, str] = {}
        self._buffer_keys = {
            detector.buffer.name: name for name, detector in self.detectors.items()
        }
        self._writable_settings = {
            name: _writable_settings(detector)
            for name, detector in self.detectors.items()
        }
        #: the same settings by canonical key, to resolve ``set`` by one name
        self._settings_by_name: dict[str, tuple[str, str, SignalRW[Any]]] = {
            signal.name: (detector_name, key, signal)
            for detector_name, settings in self._writable_settings.items()
            for key, signal in settings.items()
        }
        #: each detector's ROI as last reported, kept by subscription so a
        #: frame is forwarded with the region it was taken with
        self._rois: dict[str, Roi] = {}
        self._roi_callbacks: dict[str, Callable[[dict[str, Reading[Any]]], None]] = {}
        self._deferrals: Deferrals | None = None
        self.actions = ActionManager()
        run_coro(self._subscribe_rois())

    def setup(self, deferrals: Deferrals | None = None) -> None:
        """Take the deferrals of the engine running the plans, if one shares them."""
        self._deferrals = deferrals

    async def _subscribe_rois(self) -> None:
        """Read every ROI once, then follow it, on the subscription's loop.

        The read places the first frame: a subscription reports its first value
        only when the transport gets to it. A device without a ROI signal, or
        one whose service has not published a region yet, is left without an
        entry rather than failing the presenter.
        """
        for name, detector in self.detectors.items():
            roi = getattr(detector, "roi", None)
            if roi is None:
                continue
            try:
                self._rois[name] = Roi.parse(await roi.get_value())
            except (TypeError, ValueError):
                continue
            callback = partial(self._store_roi, name)
            self._roi_callbacks[name] = callback
            roi.subscribe(callback)

    async def _unsubscribe_rois(self) -> None:
        """Stop following every detector's ROI."""
        for name, callback in list(self._roi_callbacks.items()):
            detector = self.detectors.get(name)
            roi = getattr(detector, "roi", None)
            if roi is not None:
                roi.clear_sub(callback)
        self._roi_callbacks.clear()

    def shutdown(self) -> None:
        """Stop following the ROIs, so no subscription outlives the presenter."""
        run_coro(self._unsubscribe_rois())

    def _store_roi(self, detector: str, reading: dict[str, Reading[Any]]) -> None:
        """Remember the ROI a device published, ignoring an empty placeholder."""
        text = next(iter(reading.values()))["value"]
        if isinstance(text, str) and text:
            try:
                self._rois[detector] = Roi.parse(text)
            except ValueError:
                self.logger.warning(f"Ignoring an unreadable ROI for {detector!r}")

    def descriptor(self, doc: EventDescriptor) -> None:
        """Remember which descriptors carry a tracked detector's buffer."""
        keys = [key for key in doc["data_keys"] if key in self._buffer_keys]
        if keys:
            self._live_streams[doc["uid"]] = keys
            self._descriptor_runs[doc["uid"]] = doc["run_start"]
        self._remember_configuration_rois(doc.get("configuration"))

    def _remember_configuration_rois(
        self, configuration: Mapping[str, Any] | None
    ) -> None:
        """Take the ROI a descriptor records, when it carries one."""
        if not configuration:
            return
        for block in configuration.values():
            data = block.get("data", {})
            for key, value in data.items():
                if not key.endswith("-roi") or not isinstance(value, str):
                    continue
                try:
                    self._rois[key.removesuffix("-roi")] = Roi.parse(value)
                except ValueError:
                    continue

    def event(self, doc: Event) -> Event:
        """Forward the raw frames of a live event to the viewer.

        The document's time is the reading timestamp. It is not copied into an
        exposure field: how long the sensor was exposed belongs to the device
        settings the run captured, not to when the document was emitted.
        """
        keys = self._live_streams.get(doc["descriptor"])
        if keys is None:
            return doc
        readings: dict[str, Reading[Any]] = {}
        for key in keys:
            if key not in doc["data"]:
                continue
            readings[key] = {"value": doc["data"][key], "timestamp": doc["time"]}
        if readings:
            self.sig_new_data.emit(readings)
        return doc

    def stop(self, doc: RunStop) -> None:
        """Forget the routing entries of a run that stopped."""
        run_start = doc["run_start"]
        for uid in [
            uid
            for uid, remembered in self._descriptor_runs.items()
            if remembered == run_start
        ]:
            self._live_streams.pop(uid, None)
            self._descriptor_runs.pop(uid, None)

    def plan_map(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the plan this presenter offers."""
        return {LIVE_STREAM: {"plan": self.live_stream}}

    @continuous
    def live_stream(
        self,
        detectors: Sequence[LiveFlyer] = (),
        frames: int = 10,
        write_forever: bool = False,
        /,
        stream: PlanAction = STREAM,
    ) -> MsgGenerator[None]:
        """Perform live data collection and optionally store data to disk.

        Every acquired frame is forwarded to ``sig_new_data`` through the
        monitor on the detector's buffer. The ``stream`` action writes the
        acquired frames to a store for ``frames`` frames, as a run nested in
        this one whose start document names this run as ``parent`` and whose
        run key is ``capture``. Live visualisation continues meanwhile.

        Parameters
        ----------
        detectors
            The detectors to collect from, ticked from the session's streaming
            devices. Leaving it empty streams every device the session declares
            that satisfies :class:`~redsun_aht.protocols.LiveFlyer`.
        frames
            The number of frames to stream to disk.
        write_forever
            Stream until the ``stream`` action is released, ignoring ``frames``.
        stream
            The action that starts and stops writing to disk.
        """
        if not isinstance(write_forever, bool):
            raise ValueError("write_forever must be boolean")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
            raise ValueError("frames must be a positive integer")
        selected: Sequence[LiveFlyer] = detectors or tuple(
            detector
            for detector in self.detectors.values()
            if isinstance(detector, LiveFlyer)
        )
        if not selected:
            raise ValueError(
                "live_stream needs a detector and this session declares none that can "
                "fly; declare a camera device (for example an MMCamera) in the session "
                "profile"
            )
        trigger_info = TriggerInfo(number_of_events=0 if write_forever else frames)

        parent = yield from bps.open_run()
        try:
            # live visualization travels as Event documents, so the viewer sees
            # frames through the same document sequence as everything else
            for detector in selected:
                yield from bps.monitor(
                    detector.buffer,
                    name=f"{detector.name}_{LIVE_VIEW_STREAM}",
                )
                setting_change = getattr(detector, "last_setting_change", None)
                if setting_change is not None:
                    # Keep camera-mode transitions in the run document stream
                    # so their timestamps can be compared with the frames.
                    yield from bps.monitor(
                        setting_change,
                        name=f"{detector.name}_setting_changes",
                    )
                setting_latency = getattr(detector, "setting_change_latency", None)
                if setting_latency is not None:
                    yield from bps.monitor(
                        setting_latency,
                        name=f"{detector.name}_setting_frame_latencies",
                    )
            while True:
                # preparing starts the live view and hands each detector the
                # store its next capture writes; the capture declares it
                yield from bps.stage_all(*selected)
                yield from prepare_live_stream(
                    selected, trigger_info, LIVE_STREAM, declare=False
                )
                action = yield from self.actions.wait(stream)
                try:
                    names = ", ".join(detector.name for detector in selected)
                    self.logger.info(f"live_stream: writing {names} to disk")
                    yield from capture_live_stream(
                        selected,
                        LIVE_STREAM,
                        parent=parent,
                        until=(
                            self.actions.wait_released(stream)
                            if write_forever
                            else None
                        ),
                    )
                    self.logger.info(f"live_stream: finished writing {names}")
                finally:
                    self.actions.done(action)
        finally:
            yield from bps.close_run()

    def detector_layer_specs(self) -> dict[str, dict[str, Any]]:
        """Return the layer spec of every detector.

        A layer is the size of the sensor, whatever the ROI: a cropped frame is
        drawn into the rectangle its ROI names.
        """
        specs: dict[str, dict[str, Any]] = {}
        for device in self.detectors.values():
            width, height = run_coro(device.sensor_size.get_value())
            dtype = run_coro(device.pixel_dtype.get_value())
            specs[device.name] = {"shape": (int(height), int(width)), "dtype": dtype}
        return specs

    def detector_readings(self) -> dict[str, Reading[Any]]:
        """Return the configuration readings of every detector."""
        result: dict[str, Reading[Any]] = {}
        for device in self.detectors.values():
            result.update(run_coro(device.read_configuration()))
        return result

    def detector_descriptors(self) -> dict[str, Descriptor]:
        """Return the configuration descriptors of every detector.

        A setting this presenter cannot write has ``:readonly`` appended to its
        source, which the settings tree shows as a label.
        """
        result: dict[str, Descriptor] = {}
        for name, device in self.detectors.items():
            writable = {
                f"{name}-{key}" for key in self._writable_settings.get(name, {})
            }
            for key, descriptor in run_coro(device.describe_configuration()).items():
                if key not in writable:
                    descriptor = {
                        **descriptor,
                        "source": f"{descriptor['source']}:readonly",
                    }
                result[key] = descriptor
        return result

    @slot
    async def set(self, name: str, value: Any) -> None:
        """Set a detector setting and announce the value the device read back.

        *name* is the canonical key, ``"<detector>-<setting>"``, as the view
        sees it in a descriptor. A bare setting name is accepted when exactly
        one detector declares it. An unknown pair is logged and ignored. A
        setting the device refuses is logged and announced not at all, so the
        value the view shows stays the value the camera holds.

        A ROI change is deferred to between two engine messages when the
        session has an engine, because a ROI applied inside a point would put
        frames of two shapes in one event stream.
        """
        resolved = self._resolve_setting(name)
        if resolved is None:
            self.logger.error(f"Unknown detector setting {name!r}")
            return
        detector, key, signal = resolved

        async def apply() -> None:
            try:
                await signal.set(value)
            except Exception as error:  # the view is told, the loop goes on
                self.logger.error(f"Failed to set {signal.name} to {value!r}: {error}")
                return
            new_reading = await signal.read()
            applied = new_reading[signal.name]["value"]
            if key == "roi" and isinstance(applied, str):
                with contextlib.suppress(ValueError):
                    self._rois[detector] = Roi.parse(applied)
            self.logger.info(f"{signal.name} set to {applied!r}")
            self.sig_new_configuration.emit(detector, signal.name, applied)

        if key == "roi" and self._deferrals is not None:
            self._deferrals.request(apply)
            return
        await apply()

    @slot
    async def set_property(self, detector: str, property: str, value: Any) -> None:
        """Set a setting as the detector view announces it.

        The view's ``sig_property_changed`` carries the detector and the bare
        property; this names the pair the way :meth:`set` and the descriptors
        do, so a session can connect the signal to this slot unchanged.
        """
        await self.set(f"{detector}-{property}", value)

    def _resolve_setting(self, name: str) -> tuple[str, str, SignalRW[Any]] | None:
        """Resolve a canonical or unambiguous bare setting name."""
        resolved = self._settings_by_name.get(name)
        if resolved is not None:
            return resolved
        matches = [
            entry for entry in self._settings_by_name.values() if entry[1] == name
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            self.logger.error(
                f"Ambiguous detector setting {name!r}; name it as "
                "'<detector>-<setting>'"
            )
        return None
