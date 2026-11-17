"""One Micro-Manager camera, served over PVAccess.

Ported, with adaptation, from
``redsun_searching_is_imaging/services/mmcore_camera.py`` at donor revision
``3a3bf06`` (``redsun-searching-is-imaging``). The process owns its own
``CMMCorePlus``, one camera per process, so a session can run several despite
Micro-Manager's one-camera limit.

Run as ``python -m redsun_aht.services.mmcore_camera --prefix AHT:CAM1:
--mm-config ix83.cfg --device Camera``, or, without a configuration,
``--adapter DemoCamera --device DCam``. The prefix and the name come from the
``redsun`` session that launches the service; ``--mm-path`` names the
Micro-Manager installation when it is not the one pymmcore finds.

Changed for this bundle:

- The service is served with ``redsun.services.serve_controller``, which
  Redsun 0.16 provides, so the donor's private ``_process`` module is not
  ported. Readiness and the stop request are the session's, as before.
- A Micro-Manager system configuration can be loaded (``--mm-config``) and the
  installation chosen (``--mm-path``); the camera is then selected by its
  device label instead of by an adapter device name.
- The shared-memory live ring and its sidecar are dropped. That ring fed the
  searching bundle's broom pipeline; this bundle previews through Bluesky
  Events on ``buffer`` and records through the capture.
- Captures use the storage package's ``RunFrameStore`` through the narrow
  ``FrameWriter`` protocol. Closing writes a verifiable manifest; callers can
  supply run identity later when assembling a receipt.
- Every property the camera lets one write is published by default, not none,
  so binning and the readout settings are editable without the launcher naming
  them; ``--properties`` still narrows the set.
- The camera's description is published as data rather than left implicit:
  ``settings_revision`` (rises once per accepted write and never for a refused
  one), ``exposure_range`` (the adapter's own limits in milliseconds, ``[0, 0]``
  when it declares none), ``pixel_dtypes`` and ``binning_factors``. The device
  turns those into a ``redsun_aht.contracts.CameraDescription``.
- ``roi`` carries ``redsun_aht.contracts.Roi`` text, so an invalid region is
  refused by that contract's own validation rather than by a private parser,
  and the refusal is still recorded as a setting change with ``outcome=error``.
- ``pymmcore_plus`` is imported where the core is built, not at module import,
  so the module can be imported on a host without the optional extra.
"""

from __future__ import annotations

import argparse
import asyncio
import enum
import json
import math
import re
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeVar, cast
from urllib.parse import urlparse
from urllib.request import url2pathname
from uuid import uuid4

import numpy as np
from fastcs.attributes import AttributeIO, AttributeIORef, AttrR, AttrRW, AttrW
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, Enum, Float, Int, String, Waveform
from fastcs.logging import logger
from fastcs.methods import scan
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from numpy.typing import NDArray
from redsun.services import configure_logging, identity, serve_controller

from redsun_aht.storage.bounded_frame_batch import BoundedFrameBatch
from redsun_aht.storage.writer import frame_store_factory, manifest_path

from ..contracts import Roi

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping

    from pymmcore_plus import CMMCorePlus

#: Micro-Manager's ``PixelType`` for each numpy dtype a camera reads out in.
#: Colour types are left out: a frame of one is not a 2D array.
PIXEL_TYPES: Final = {"uint8": "8bit", "uint16": "16bit", "uint32": "32bit"}

#: Seconds between two frames published on ``Buffer``, and therefore the cadence
#: ceiling a client can observe: a frame cannot be seen more often than the counter
#: is republished. Measured 2026-10-25 over 30 frames of a 10 ms exposure with an idle
#: PVCAM camera: 0.1 s gave 9.9 frames/s, 0.02 s gave 20.9 with client CPU rising from
#: 10.9 to 12.0 ms per frame, and 0.005 s gave 21.2 - saturated, because beyond that
#: the acquisition round trip dominates rather than the period. 0.02 is the knee. The
#: camera itself runs at 49 frames/s under Micro-Manager, so this is a service
#: property, not a hardware limit.
LIVE_PERIOD: Final = 0.02

#: Seconds the grabbing thread waits when the camera has no frame ready.
EMPTY_POLL: Final = 0.001

#: Seconds between two reads of a Micro-Manager property.
PROPERTY_POLL: Final = 1.0

#: Where the camera's own properties sit under its prefix.
PROPERTY_GROUP: Final = "properties"

#: What a PV name may not carry, replaced by an underscore.
NOT_IN_A_PV_NAME: Final = re.compile(r"[^A-Za-z0-9_]")

#: What ``State`` reads while the camera takes no frames.
IDLE: Final = "idle"

#: What ``State`` reads while the camera takes frames.
ACQUIRING: Final = "acquiring"

#: What ``State`` reads once grabbing stopped on an error.
FAULTED: Final = "faulted"

#: Schema of the record published for every setting write.
SETTING_CHANGE_SCHEMA: Final = "redsun-camera-setting-change-v1"

#: Schema of the batch published when a setting's next frame arrives.
SETTING_LATENCY_SCHEMA: Final = "redsun-camera-setting-frame-latency-v1"

#: Schema of the bounded per-frame timing sidecar of a capture.
CAMERA_FRAME_TIMING_SCHEMA: Final = "redsun-camera-frame-timing-v1"

#: Clock the service times its own observations with.
SERVICE_CLOCK_ID: Final = "python.time.monotonic_ns"

#: Clock the frame-arrival timing sidecar is expressed in.
CAMERA_FRAME_TIMING_CLOCK: Final = "python.time.monotonic_ns"

#: What the timing sidecar's timestamps mean, for a reader that must not over-read them.
CAMERA_FRAME_TIMING_SEMANTICS: Final = "service_arrival_after_core_frame_read"

#: The MMCore property whose limits are the camera's exposure range.
EXPOSURE_PROPERTY: Final = "Exposure"

#: The MMCore property that declares the binning factors a camera accepts.
BINNING_PROPERTY: Final = "Binning"

#: What ``ExposureRange`` publishes when the adapter reports no usable limits.
#: An empty interval is the service saying "I do not know", never an invented bound.
UNPUBLISHED_EXPOSURE_RANGE: Final = (0.0, 0.0)

T = TypeVar("T")


@dataclass
class CoreRef(AttributeIORef):  # type: ignore[misc]
    """Names the camera setting an attribute reads and writes."""

    setting: str = ""


def as_is(apply: Callable[[], None]) -> None:
    """Run *apply* as it is, for a camera nothing sequences."""
    apply()


def within(roi: Roi, outer: Roi) -> bool:
    """Whether *roi* falls inside *outer*, both on the sensor's own axes."""
    return (
        roi.x >= outer.x
        and roi.y >= outer.y
        and roi.x + roi.width <= outer.x + outer.width
        and roi.y + roi.height <= outer.y + outer.height
    )


def uncrop(core: CMMCorePlus) -> Roi:
    """Put the readout back over the whole sensor, and return what it covers.

    The first clear zeroes the offsets and leaves the size they allowed; the
    second takes the size zero offsets allow, which is the whole sensor.
    """
    core.clearROI()
    core.clearROI()
    return Roi(*core.getROI())


def _binning_xy(value: str) -> tuple[int, int]:
    """Parse the active camera binning as horizontal and vertical factors."""
    match = re.fullmatch(r"\s*(\d+)\s*(?:[xX]\s*(\d+)\s*)?", value)
    if match is None:
        raise ValueError(f"unsupported camera binning readback: {value!r}")
    bin_x = int(match.group(1))
    bin_y = int(match.group(2) or bin_x)
    if bin_x <= 0 or bin_y <= 0:
        raise ValueError(f"camera binning factors must be positive: {value!r}")
    return bin_x, bin_y


def exposure_limits_ms(core: CMMCorePlus, label: str) -> tuple[float, float] | None:
    """Return the exposure limits MMCore reports for the camera, in milliseconds.

    ``None`` when the adapter declares no ``Exposure`` property, answers with
    no limits, or answers with a range that is not finite, increasing and
    positive. The service then publishes the empty range: a client is told
    that the bound is unknown instead of being handed one this service made up.
    """
    if not core.hasProperty(label, EXPOSURE_PROPERTY):
        return None
    try:
        low = float(core.getPropertyLowerLimit(label, EXPOSURE_PROPERTY))
        high = float(core.getPropertyUpperLimit(label, EXPOSURE_PROPERTY))
    except Exception as error:
        logger.debug(f"Camera {label!r} reports no {EXPOSURE_PROPERTY} limits: {error}")
        return None
    if not math.isfinite(low) or not math.isfinite(high):
        return None
    if low >= high or high <= 0:
        return None
    return (low, high)


def declared_binning_factors(core: CMMCorePlus, label: str) -> tuple[int, ...]:
    """Return the binning factors the camera declares, ``(1,)`` when it declares none.

    The factors are the horizontal factor of each allowed ``Binning`` value, so
    ``("1x1", "2x2")`` and ``("1", "2")`` both read as ``(1, 2)``.
    """
    if not core.hasProperty(label, BINNING_PROPERTY):
        return (1,)
    factors: set[int] = set()
    for text in core.getAllowedPropertyValues(label, BINNING_PROPERTY):
        try:
            factors.add(_binning_xy(str(text))[0])
        except ValueError:
            continue
    return tuple(sorted(factors)) or (1,)


def apply_initial_properties(
    core: CMMCorePlus, label: str, properties: Mapping[str, str]
) -> None:
    """Apply and verify the explicitly configured camera property defaults.

    Raises
    ------
    ValueError
        If the camera has no such property, the property is read-only, or the
        value is not one the property allows.
    RuntimeError
        If the camera reads the property back as something else.
    """
    for name, value in properties.items():
        if not core.hasProperty(label, name):
            raise ValueError(f"camera {label!r} has no property {name!r}")
        if core.isPropertyReadOnly(label, name):
            raise ValueError(f"camera property {name!r} is read-only")
        allowed = tuple(core.getAllowedPropertyValues(label, name))
        if allowed and value not in allowed:
            raise ValueError(
                f"camera property {name!r} does not allow {value!r}; "
                f"choose from {allowed}"
            )

    for name, value in properties.items():
        previous = str(core.getProperty(label, name))
        core.setProperty(label, name, value)
        actual = str(core.getProperty(label, name))
        logger.info(
            f"Camera property default {name!r}: {previous!r} -> "
            f"{actual!r} (requested {value!r})"
        )
        if actual != value:
            raise RuntimeError(
                f"camera property {name!r} read back {actual!r}, "
                f"not requested default {value!r}"
            )


class WhenIdle(Protocol):
    """Runs a read only while the camera takes no sequence."""

    def __call__(self, read: Callable[[], T], /) -> T | None:
        """Return what *read* returns, or ``None`` while a sequence runs."""


def run_now[T](read: Callable[[], T]) -> T | None:
    """Run *read* at once, for a camera nothing sequences."""
    return read()


class CoreIO(AttributeIO[Any, CoreRef]):  # type: ignore[misc]
    """Reads and writes camera settings through Micro-Manager.

    *without_sequence* runs a callable with no sequence acquisition running,
    for the settings Micro-Manager refuses during one. *layout_changed* is
    handed a frame snapped after a setting that changes what frames look
    like. *capturing* says whether a capture is writing, during which
    the pixel dtype is refused: the store's dtype is fixed when it opens.

    ``fastcs`` ships no type information, so this class subclasses an ``Any``
    base and carries the ignore above; the same applies to every ``fastcs``
    subclass in this module.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        without_sequence: Callable[[Callable[[], None]], None] = as_is,
        layout_changed: Callable[[NDArray[Any]], Awaitable[None]] | None = None,
        capturing: Callable[[], bool] = lambda: False,
        when_idle: WhenIdle = run_now,
        record_change: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        defer_snap: bool = False,
        is_busy: Callable[[], bool] = lambda: False,
    ) -> None:
        super().__init__()
        self._core = core
        self._without_sequence = without_sequence
        self._layout_changed = layout_changed
        self._capturing = capturing
        self._when_idle = when_idle
        self._record_change = record_change
        self._defer_snap = defer_snap
        self._is_busy = is_busy

    async def send(self, attr: AttrW[Any, CoreRef], value: Any) -> None:
        """Apply *value* to the camera, and read back what it took.

        Raises
        ------
        ValueError
            For a pixel dtype the camera does not read out in.
        """
        setting = attr.io_ref.setting
        started_ns = time.perf_counter_ns()
        previous = await asyncio.to_thread(
            self._when_idle, partial(self._read_setting, setting)
        )
        outcome = "ok"
        failure: str | None = None
        actual: Any = ""
        try:
            if self._defer_snap and self._is_busy():
                raise RuntimeError("camera setting changes require idle capture")
            match setting:
                case "exposure":
                    await asyncio.to_thread(
                        self._without_sequence,
                        lambda: self._core.setExposure(float(value)),
                    )
                    actual = await asyncio.to_thread(self._core.getExposure)
                case "roi":
                    roi = Roi.parse(value)
                    await asyncio.to_thread(
                        self._without_sequence, lambda: self._set_roi(roi)
                    )
                    actual = Roi(*await asyncio.to_thread(self._core.getROI)).as_text()
                case "pixel_dtype":
                    await self._set_pixel_dtype(cast("enum.Enum", value).name)
                    actual = await asyncio.to_thread(self._when_idle, self._pixel_dtype)
                case _:
                    raise ValueError(f"no camera setting named {setting!r}")
            if isinstance(attr, AttrR):
                # Without this the readback carries the old value until the next
                # scan, and a client that reads straight after writing sees it.
                await self.update(attr)
        except BaseException as exc:
            outcome = (
                "outcome_unknown"
                if isinstance(exc, asyncio.CancelledError)
                else "error"
            )
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self._record_change is not None:
                try:
                    await self._record_change(
                        {
                            "schema": SETTING_CHANGE_SCHEMA,
                            "setting": setting,
                            "requested": str(value),
                            "previous": (None if previous is None else str(previous)),
                            "readback": None if actual is None else str(actual),
                            "outcome": outcome,
                            "error": failure,
                            "service_roundtrip_ms": round(
                                (time.perf_counter_ns() - started_ns) / 1_000_000,
                                4,
                            ),
                            "service_monotonic_ns": time.monotonic_ns(),
                            "scope": (
                                "service_roundtrip_ms covers write/readback only; "
                                "next-frame arrival is reported separately"
                            ),
                        }
                    )
                except Exception as exc:
                    logger.warning(f"Could not publish camera setting timing: {exc}")

    def _read_setting(self, setting: str) -> Any:
        """Read one current setting before changing it."""
        match setting:
            case "exposure":
                return self._core.getExposure()
            case "roi":
                return Roi(*self._core.getROI()).as_text()
            case "pixel_dtype":
                return self._pixel_dtype()
            case _:
                raise ValueError(f"no camera setting named {setting!r}")

    def _set_roi(self, roi: Roi) -> None:
        """Put the camera's readout over *roi*, from wherever it stands now.

        A ROI reaching outside what the camera reads now needs the readout
        uncropped first, since an adapter that caps the width at what is left
        of the sensor beyond the current offset refuses it while the camera is
        cropped. A ROI inside goes straight to the camera, and one the camera
        already reads goes nowhere: every write rebuilds the driver's buffers.
        """
        current = Roi(*self._core.getROI())
        if not within(roi, current):
            current = uncrop(self._core)
        if roi != current:
            self._core.setROI(roi.x, roi.y, roi.width, roi.height)

    async def update(self, attr: AttrR[Any, CoreRef]) -> None:
        """Read the camera's own value of the setting into *attr*."""
        match attr.io_ref.setting:
            case "exposure":
                await attr.update(await asyncio.to_thread(self._core.getExposure))
            case "roi":
                roi = await asyncio.to_thread(self._core.getROI)
                await attr.update(Roi(*roi).as_text())
            case "pixel_dtype":
                # reading a camera property ends the sequence on some adapters
                # and the attribute needs no
                # poll anyway: ``follow_layout`` retypes it from every frame
                dtype = await asyncio.to_thread(self._when_idle, self._pixel_dtype)
                members = cast("Enum[Any]", attr.datatype).enum_cls.__members__
                if dtype in members:
                    await attr.update(members[dtype])
            case _:
                raise ValueError(f"no camera setting named {attr.io_ref.setting!r}")

    def pixel_dtypes(self) -> dict[str, str]:
        """Return the dtypes the camera reads out in, each with its ``PixelType``."""
        label = str(self._core.getCameraDevice())
        if not self._core.hasProperty(label, "PixelType"):
            return {}
        allowed = set(self._core.getAllowedPropertyValues(label, "PixelType"))
        return {
            dtype: pixel for dtype, pixel in PIXEL_TYPES.items() if pixel in allowed
        }

    def _pixel_dtype(self) -> str | None:
        """Return the dtype of the camera's ``PixelType``, ``None`` if unmapped."""
        label = str(self._core.getCameraDevice())
        if not self._core.hasProperty(label, "PixelType"):
            return None
        pixel = str(self._core.getProperty(label, "PixelType"))
        return next((d for d, p in PIXEL_TYPES.items() if p == pixel), None)

    async def _set_pixel_dtype(self, dtype: str) -> None:
        if self._capturing():
            raise RuntimeError(
                "the pixel dtype cannot change while a capture writes; "
                "the store's dtype was fixed when the capture started"
            )
        supported = await asyncio.to_thread(self.pixel_dtypes)
        if dtype not in supported:
            raise ValueError(
                f"{dtype!r} is not a pixel dtype this camera reads out in; "
                f"one of {sorted(supported)}"
            )
        label = str(self._core.getCameraDevice())
        snapped: list[NDArray[Any]] = []

        def apply() -> None:
            self._core.setProperty(label, "PixelType", supported[dtype])
            if not self._defer_snap:
                snapped.append(self._core.snap())

        await asyncio.to_thread(self._without_sequence, apply)
        if self._layout_changed is not None and snapped:
            await self._layout_changed(snapped[0])


@dataclass
class PropertyRef(AttributeIORef):  # type: ignore[misc]
    """Names the Micro-Manager property an attribute reads and writes."""

    property: str = ""


class PropertyIO(AttributeIO[str, PropertyRef]):  # type: ignore[misc]
    """Reads and writes one Micro-Manager property of the camera.

    Every property travels as text, as Micro-Manager holds it; a client
    wanting a number parses it.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        without_sequence: Callable[[Callable[[], None]], None] = as_is,
        layout_changed: Callable[[NDArray[Any]], Awaitable[None]] | None = None,
        when_idle: WhenIdle = run_now,
        record_change: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        *,
        defer_snap: bool = False,
        is_busy: Callable[[], bool] = lambda: False,
    ) -> None:
        super().__init__()
        self._core = core
        self._label = label
        self._without_sequence = without_sequence
        self._layout_changed = layout_changed
        self._when_idle = when_idle
        self._record_change = record_change
        self._defer_snap = defer_snap
        self._is_busy = is_busy

    async def send(self, attr: AttrW[str, PropertyRef], value: str) -> None:
        """Write the property, and read back what the camera made of it.

        Normal mode pauses a sequence and snaps to discover a changed layout.
        Deferred mode requires an idle camera and does not snap; its next
        capture reads a fresh layout before starting acquisition.
        """
        started_ns = time.perf_counter_ns()
        previous = await asyncio.to_thread(
            self._when_idle,
            partial(self._core.getProperty, self._label, attr.io_ref.property),
        )
        read: list[str] = []
        snapped: list[NDArray[Any]] = []
        outcome = "ok"
        failure: str | None = None
        snap_ms: float | None = None
        layout_ms: float | None = None
        try:
            if self._defer_snap and self._is_busy():
                raise RuntimeError("camera property changes require idle capture")

            def apply() -> None:
                nonlocal snap_ms
                self._core.setProperty(self._label, attr.io_ref.property, value)
                read.append(
                    str(self._core.getProperty(self._label, attr.io_ref.property))
                )
                if not self._defer_snap:
                    snap_started_ns = time.perf_counter_ns()
                    snapped.append(self._core.snap())
                    snap_ms = (time.perf_counter_ns() - snap_started_ns) / 1_000_000

            await asyncio.to_thread(self._without_sequence, apply)
            if isinstance(attr, AttrR):
                await attr.update(read[0])
            if self._layout_changed is not None and snapped:
                layout_started_ns = time.perf_counter_ns()
                await self._layout_changed(snapped[0])
                layout_ms = (time.perf_counter_ns() - layout_started_ns) / 1_000_000
        except BaseException as exc:
            outcome = (
                "outcome_unknown"
                if isinstance(exc, asyncio.CancelledError)
                else "error"
            )
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self._record_change is not None:
                try:
                    await self._record_change(
                        {
                            "schema": SETTING_CHANGE_SCHEMA,
                            "setting": attr.io_ref.property,
                            "requested": str(value),
                            "previous": None if previous is None else str(previous),
                            "readback": read[0] if read else "",
                            "outcome": outcome,
                            "error": failure,
                            "service_roundtrip_ms": round(
                                (time.perf_counter_ns() - started_ns) / 1_000_000,
                                4,
                            ),
                            "snap_ms": None if snap_ms is None else round(snap_ms, 4),
                            "layout_update_ms": (
                                None if layout_ms is None else round(layout_ms, 4)
                            ),
                            "service_monotonic_ns": time.monotonic_ns(),
                            "scope": (
                                "service_roundtrip_ms covers write/readback/snap/"
                                "layout; next-frame arrival is reported separately"
                            ),
                        }
                    )
                except Exception as exc:
                    logger.warning(f"Could not publish camera setting timing: {exc}")

    async def update(self, attr: AttrR[str, PropertyRef]) -> None:
        """Read the camera's own value of the property, unless it is sequencing.

        Reading a property ends the sequence on some adapters, which then
        deliver no further frame. Nothing but
        ``send`` changes a property meanwhile, and ``send`` reads its value
        back while the sequence is paused, so the poll is skipped rather than
        deferred.
        """
        value = await asyncio.to_thread(
            self._when_idle,
            partial(self._core.getProperty, self._label, attr.io_ref.property),
        )
        if value is not None:
            await attr.update(str(value))


class FrameWriter(Protocol):
    """What a capture writes frames into, in arrival order.

    The storage package owns the recording format; this protocol is the whole
    of what the service needs from it; the storage package supplies the
    default ``RunFrameStore`` implementation. Where that writer supports a
    setting timeline, the service supplies the revision current at append time;
    it is not an atomic exposure-time configuration stamp.
    """

    @property
    def frames_written(self) -> int:
        """Number of frames successfully appended to the store."""
        ...

    def append(self, frame: NDArray[Any]) -> int:
        """Persist one YX frame and return its zero-based index."""
        ...

    def close(self) -> None:
        """Finish the store, writing anything still held."""
        ...


def local_store_path(uri: str) -> Path:
    """Return the filesystem path *uri* names, refusing a non-local store."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    if "://" in uri:
        raise ValueError("a capture store must be a local file path")
    return Path(uri)


type FrameStoreFactory = Callable[[str, str, NDArray[Any]], FrameWriter]

#: The store a capture opens. Tests and the storage package replace this
#: binding; the controller only ever calls it.
FrameStore: Final[FrameStoreFactory] = frame_store_factory


class MMCameraController(Controller):  # type: ignore[misc]
    """A camera's settings, its latest frame, and the capture it writes."""

    exposure = AttrRW(
        Float(prec=6), io_ref=CoreRef(setting="exposure", update_period=1.0)
    )
    # text, "x,y,width,height": a client can put a string over PVAccess, and
    # not the array a waveform is served as
    roi = AttrRW(String(), io_ref=CoreRef(setting="roi", update_period=1.0))
    sensor_size = AttrR(Waveform(np.int32, shape=(2,)))
    sensor_size_binning = AttrR(String())
    #: The exposure interval the adapter declares, in milliseconds; ``[0, 0]``
    #: when it declares none, which a client must read as "unknown", not as a
    #: bound.
    exposure_range = AttrR(Waveform(np.float64, shape=(2,)))
    #: The readout dtypes the camera offers, comma separated, mirroring the
    #: choices ``pixel_dtype`` is an enum over.
    pixel_dtypes = AttrR(String())
    #: The binning factors the camera declares, comma separated.
    binning_factors = AttrR(String())
    #: Rises by one for every setting write the camera accepted and none for a
    #: write it refused, so a frame can name the configuration it was taken
    #: under.
    settings_revision = AttrR(Int())
    acquire = AttrRW(Bool())
    snap_request = AttrW(Bool(), description="Request one idle software snap")
    frame_count = AttrR(Int())
    capture = AttrRW(Bool())
    file_path = AttrRW(String())
    data_key = AttrRW(String())
    num_capture = AttrRW(Int())
    record_frame_timing = AttrRW(Bool())
    captured = AttrR(Int())
    buffered_frames = AttrR(
        Int(), description="Frames accepted into the most recent RAM batch"
    )
    state = AttrR(String())
    sequence_started = AttrR(Bool())
    last_error = AttrR(String())
    last_setting_change = AttrR(String())
    setting_change_latency = AttrR(String())

    def __init__(
        self,
        core: CMMCorePlus,
        label: str,
        data_key: str,
        properties: Iterable[str] | None = None,
        *,
        defer_initial_frame: bool = False,
        buffered_capture: bool = False,
        finite_capture: bool = False,
        buffer_limit_mib: int = 256,
        allow_unbounded_capture: bool = False,
    ) -> None:
        self._property_io = PropertyIO(
            core,
            label,
            self._without_sequence,
            self.publish_layout,
            self._when_idle,
            self._record_setting_change,
            defer_snap=defer_initial_frame,
            is_busy=lambda: (
                self._store is not None
                or self._grabbing.is_set()
                or self._snap_in_progress
            ),
        )
        self._core_io = CoreIO(
            core,
            self._without_sequence,
            self.publish_layout,
            lambda: self._store is not None,
            self._when_idle,
            self._record_setting_change,
            defer_snap=defer_initial_frame,
            is_busy=lambda: (
                self._store is not None
                or self._grabbing.is_set()
                or self._snap_in_progress
            ),
        )
        super().__init__(ios=[self._core_io, self._property_io])
        self._core = core
        self._control_lock = asyncio.Lock()
        self._publish_frame_lock = asyncio.Lock()
        self._snap_in_progress = False
        self._defer_initial_frame = defer_initial_frame
        if buffered_capture and not defer_initial_frame:
            raise ValueError("buffered capture requires deferred-frame mode")
        if finite_capture and not buffered_capture:
            raise ValueError("finite capture requires buffered capture")
        if type(buffer_limit_mib) is not int or not 1 <= buffer_limit_mib <= 4096:
            raise ValueError("buffer limit must be 1..4096 MiB")
        self._buffer_limit_bytes = buffer_limit_mib * 1024 * 1024
        self._buffered_capture = buffered_capture
        self._allow_unbounded_capture = allow_unbounded_capture
        self._finite_capture = finite_capture
        self._sequence_is_finite = False
        self._batch: BoundedFrameBatch | None = None
        self._accepted = 0
        self._label = label
        self._default_data_key = data_key
        self._properties = None if properties is None else set(properties)
        self._sensor_size_native_xy: tuple[int, int] | None = None
        self._latest: NDArray[Any] | None = None
        self._grabbed = 0
        self._published = 0
        self._written = 0
        self._capture_full = False
        self._store: FrameWriter | None = None
        self._store_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._grabbing = threading.Event()
        self._stopped = threading.Event()
        self._stopped.set()
        self._grab_task: asyncio.Task[None] | None = None
        self._sequencing = False
        self._sequence_started = threading.Event()
        # a layout change that arrived mid-sequence defers its geometry
        # readbacks until the camera stops, see publish_layout
        self._geometry_refresh_pending = False
        self._camera_lock = threading.Lock()
        # the grabbing thread takes the camera lock again right after it
        # releases it, and a lock is not fair, so a write waiting for it can
        # wait for every frame that follows: whoever wants the camera takes
        # this lock first, and the grabbing thread takes and releases it
        # between frames
        self._camera_request_lock = threading.Lock()
        self._error: str | None = None
        self._timing_records: list[dict[str, Any]] = []
        self._timing_limit = 100_000
        self._capture_record_timing = False
        self._capture_uri: str | None = None
        self._capture_data_key: str | None = None
        self._setting_change_lock = threading.Lock()
        self._pending_setting_changes: list[dict[str, Any]] = []
        self._setting_latency_updates: list[dict[str, Any]] = []
        self._setting_change_publish_lock = asyncio.Lock()
        self._latest_setting_change: dict[str, Any] | None = None

        self.acquire.add_on_update_callback(self._on_acquire)
        self.snap_request.set_on_put_callback(self._on_snap_request)
        self.capture.add_on_update_callback(self._on_capture)
        self.file_path.add_on_update_callback(self._on_file_path)

    async def _record_setting_change(self, record: dict[str, Any]) -> None:
        """Publish a setting transition and queue its next-frame measurement."""
        item = {
            **record,
            "setting_change_id": uuid4().hex,
            "service_clock_id": SERVICE_CLOCK_ID,
            "frame_latency_status": (
                "pending" if record.get("outcome") == "ok" else "not_applicable"
            ),
            "next_frame_service_monotonic_ns": None,
            "readback_to_next_frame_arrival_ms": None,
            "next_frame_latency_scope": "readback_to_next_camera_frame_arrival",
        }
        if item["frame_latency_status"] == "pending":
            with self._setting_change_lock:
                self._pending_setting_changes.append(item)
        async with self._setting_change_publish_lock:
            revision = self.settings_revision.get()
            if item["outcome"] == "ok":
                # only an applied change moves the revision on, so a frame
                # labelled with it names a configuration the camera took
                revision += 1
                await self.settings_revision.update(revision)
            item["settings_revision"] = revision
            self._latest_setting_change = item
            await self.last_setting_change.update(json.dumps(item, sort_keys=True))

    async def initialise(self) -> None:
        """Declare the frame buffer, whose shape and dtype the camera decides.

        The sensor size is read here too, before any ROI crops the camera,
        since a ROI is expressed against it.
        """
        self._loop = asyncio.get_running_loop()
        frame = await asyncio.to_thread(
            self._layout_template if self._defer_initial_frame else self.grab_once
        )
        if frame is None:
            raise RuntimeError("the camera gave no frame to size the buffer from")
        self.buffer = AttrR(Waveform(frame.dtype, shape=frame.shape))
        # one choice per dtype the camera reads out in, so a client offers
        # exactly those; a camera without a PixelType offers the one it has
        supported = await asyncio.to_thread(self._core_io.pixel_dtypes)
        names = list(supported) or [frame.dtype.name]
        self.pixel_dtype = AttrRW(
            # the members are worth their names, not their ordinals, so a
            # client that puts the dtype as text is understood as well as one
            # that puts the member the transport resolved
            Enum(enum.Enum("PixelDtype", {name: name for name in names})),
            io_ref=CoreRef(setting="pixel_dtype", update_period=1.0),
        )
        await self.follow_layout(frame, force=True)
        sensor_width, sensor_height = await asyncio.to_thread(
            lambda: (self._core.getImageWidth(), self._core.getImageHeight())
        )
        await self.sensor_size.update(
            np.array(
                [sensor_width, sensor_height],
                dtype=np.int32,
            )
        )
        binning = "unavailable"
        if await asyncio.to_thread(self._core.hasProperty, self._label, "Binning"):
            binning = str(
                await asyncio.to_thread(self._core.getProperty, self._label, "Binning")
            )
            bin_x, bin_y = _binning_xy(binning)
        else:
            bin_x, bin_y = (1, 1)
        self._sensor_size_native_xy = (
            int(sensor_width) * bin_x,
            int(sensor_height) * bin_y,
        )
        await self.sensor_size_binning.update(binning)
        limits = await asyncio.to_thread(exposure_limits_ms, self._core, self._label)
        await self.exposure_range.update(
            np.array(
                UNPUBLISHED_EXPOSURE_RANGE if limits is None else limits,
                dtype=np.float64,
            )
        )
        await self.pixel_dtypes.update(",".join(names))
        await self.binning_factors.update(
            ",".join(
                str(factor)
                for factor in await asyncio.to_thread(
                    declared_binning_factors, self._core, self._label
                )
            )
        )
        await self.data_key.update(self._default_data_key)
        self.add_sub_controller(PROPERTY_GROUP, await self._build_properties())

    def _layout_template(self) -> NDArray[Any]:
        """Build a layout-only template without taking or counting a frame."""

        def read() -> NDArray[Any]:
            dtype = self._core_io._pixel_dtype()
            label = str(self._core.getCameraDevice())
            if dtype is None and self._core.hasProperty(label, "Pixel Format"):
                pixel_format = str(self._core.getProperty(label, "Pixel Format"))
                dtype = {"Mono8": "uint8", "Mono16": "uint16"}.get(pixel_format)
            width, height = self._core.getImageWidth(), self._core.getImageHeight()
            if dtype not in PIXEL_TYPES or width <= 0 or height <= 0:
                raise ValueError("camera does not report a supported monochrome layout")
            return np.empty((height, width), dtype=dtype)

        frame = self._when_idle(read)
        if frame is None:
            raise RuntimeError("cannot read camera layout during a sequence")
        return frame

    async def _build_properties(self) -> Controller:
        """Publish the camera's writable properties, all of them or the ones chosen.

        The attributes are built here rather than declared on the class, since
        a camera's properties are known only once it is loaded. A chosen name
        the camera does not have, or cannot write, is logged and skipped.
        """
        names: list[str] = list(
            await asyncio.to_thread(self._core.getDevicePropertyNames, self._label)
        )
        read_only = await asyncio.to_thread(
            lambda: {
                name: self._core.isPropertyReadOnly(self._label, name) for name in names
            }
        )
        if self._properties is not None:
            for name in self._properties:
                if name not in names or read_only[name]:
                    logger.warning(
                        f"Property {name!r} is not one the camera lets one write"
                    )
            names = [name for name in names if name in self._properties]
        properties = Controller(ios=[self._property_io])
        for name in names:
            # PixelType is pixel_dtype, in numpy's names
            if read_only[name] or name == "PixelType":
                continue
            # a PV name takes no spaces or brackets, and Micro-Manager's do
            # ("Photon Conversion Factor"): the attribute is named for the PV,
            # the reference keeps the name the camera answers to
            attribute = NOT_IN_A_PV_NAME.sub("_", name)
            if attribute in properties.attributes:
                logger.warning(f"Property {name!r} clashes with {attribute!r}, skipped")
                continue
            setattr(
                properties,
                attribute,
                AttrRW(
                    String(),
                    io_ref=PropertyRef(property=name, update_period=PROPERTY_POLL),
                ),
            )
        return properties

    async def disconnect(self) -> None:
        """Stop grabbing and finish a capture left open."""
        async with self._control_lock:
            await self._stop_grabbing()
            await asyncio.to_thread(self._close_store)

    @scan(LIVE_PERIOD)  # type: ignore[untyped-decorator]
    async def publish_frame(self) -> None:
        """Publish the latest frame and what the capture has written.

        ``FrameCount`` counts frames the camera took, and a tick with no new
        frame publishes nothing, so a client can wait on it for a frame newer
        than its last move.
        """
        async with self._publish_frame_lock:
            await self._publish_state()
            buffered = self._accepted if self._batch is not None else 0
            if self.buffered_frames.get() != buffered:
                await self.buffered_frames.update(buffered)
            grabbed = self._grabbed
            frame = self._latest
            if frame is None or grabbed == self._published:
                return
            if tuple(frame.shape) != self.buffer.datatype.shape:
                await self.publish_layout(frame)
            else:
                await self.follow_layout(frame)
                await self.buffer.update(frame)
            await self.frame_count.update(grabbed)
            self._published = grabbed
            await self._publish_setting_latency_updates()
            if self.captured.get() != self._written:
                await self.captured.update(self._written)

    async def _publish_setting_latency_updates(self) -> None:
        """Publish frame-arrival records and enrich the latest setting record."""
        with self._setting_change_lock:
            updates, self._setting_latency_updates = (
                self._setting_latency_updates,
                [],
            )
        if not updates:
            return
        await self.setting_change_latency.update(
            json.dumps(
                {
                    "schema": SETTING_LATENCY_SCHEMA,
                    "batch_id": uuid4().hex,
                    "service_clock_id": SERVICE_CLOCK_ID,
                    "measurements": updates,
                },
                sort_keys=True,
            )
        )
        async with self._setting_change_publish_lock:
            current = self._latest_setting_change
            if current is None:
                return
            latest_id = current.get("setting_change_id")
            update = next(
                (
                    item
                    for item in reversed(updates)
                    if item["setting_change_id"] == latest_id
                ),
                None,
            )
            if update is None:
                return
            current = {**current, **update}
            self._latest_setting_change = current
            await self.last_setting_change.update(json.dumps(current, sort_keys=True))

    async def follow_layout(self, frame: NDArray[Any], *, force: bool = False) -> None:
        """Retype ``buffer`` and ``pixel_dtype`` to *frame* when its dtype is new.

        The declared dtype would otherwise cast every frame to the first
        frame's. A dtype ``pixel_dtype`` has no choice for leaves it as is.
        """
        if force or frame.dtype != self.buffer.datatype.array_dtype:
            self.buffer.update_datatype(
                Waveform(frame.dtype, shape=self.buffer.datatype.shape)
            )
            members = cast("Enum[Any]", self.pixel_dtype.datatype).enum_cls.__members__
            if frame.dtype.name in members:
                await self.pixel_dtype.update(members[frame.dtype.name])

    async def publish_layout(self, frame: NDArray[Any]) -> None:
        """Publish a changed image layout and its fresh camera geometry readbacks."""
        await self.follow_layout(frame)
        if tuple(frame.shape) != self.buffer.datatype.shape:
            self.buffer.update_datatype(Waveform(frame.dtype, shape=tuple(frame.shape)))
        await self.buffer.update(frame)
        if self._sequencing:
            # Some adapters end a sequence when a property is read while it
            # runs, so the geometry readbacks wait for the camera to stop.
            self._geometry_refresh_pending = True
        else:
            await self._publish_geometry()

    async def _publish_geometry(self) -> None:
        """Publish the camera's ROI and post-binning sensor size."""
        roi = await asyncio.to_thread(self._core.getROI)
        await self.roi.update(Roi(*roi).as_text())
        if await asyncio.to_thread(self._core.hasProperty, self._label, "Binning"):
            binning = str(
                await asyncio.to_thread(self._core.getProperty, self._label, "Binning")
            )
            bin_x, bin_y = _binning_xy(binning)
            if self._sensor_size_native_xy is not None:
                sensor_width, sensor_height = self._sensor_size_native_xy
                if sensor_width % bin_x or sensor_height % bin_y:
                    raise RuntimeError(
                        "camera sensor dimensions are not divisible by active binning"
                    )
                await self.sensor_size.update(
                    np.array(
                        [sensor_width // bin_x, sensor_height // bin_y],
                        dtype=np.int32,
                    )
                )
            await self.sensor_size_binning.update(binning)

    async def reconnect(self) -> None:
        """Forget the last error and grab again if the camera should be."""
        self._error = None
        await super().reconnect()
        if self.acquire.get():
            await self._start_grabbing()

    async def _publish_state(self) -> None:
        """Report what the camera is doing and what stopped it, if anything."""
        sequence_started = self._sequence_started.is_set()
        if self.sequence_started.get() != sequence_started:
            await self.sequence_started.update(sequence_started)
        error = self._error
        if error is not None and self.last_error.get() != error:
            await self.last_error.update(error)
        if error is not None:
            state = FAULTED
        elif self._grabbing.is_set():
            state = ACQUIRING
        else:
            state = IDLE
        if self.state.get() != state:
            await self.state.update(state)

    async def _on_acquire(self, acquiring: bool) -> None:
        """Start or stop the thread grabbing frames."""
        if acquiring and self._snap_in_progress:
            raise RuntimeError("software snap in progress")
        async with self._control_lock:
            if acquiring:
                logger.info("Acquisition started")
                await self._start_grabbing()
            else:
                await self._stop_grabbing()
                await self.publish_frame()
                logger.info("Acquisition stopped")

    async def _on_snap_request(self, attr: AttrW[Any, Any], requested: bool) -> None:
        """Expose one frame only while the service is idle, with no replay."""
        del attr
        if requested is not True:
            raise ValueError("snap request must be true")
        if not self._defer_initial_frame:
            raise RuntimeError("software snap requires deferred-frame mode")
        if self._control_lock.locked():
            raise RuntimeError("camera control is busy")
        async with self._control_lock:
            if self._grabbing.is_set() or self._grab_task is not None:
                raise RuntimeError("software snap requires idle camera")
            self._snap_in_progress = True
            task = asyncio.create_task(asyncio.to_thread(self.grab_once))
            try:
                try:
                    frame = await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
                if frame is None:
                    raise RuntimeError("software snap returned no frame")
                await self.publish_frame()
            finally:
                self._snap_in_progress = False

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block until the grabbing thread stops, and return whether it did.

        The thread stops when ``Acquire`` is cleared and when the camera
        fails, so a fault is recorded by the time this returns ``True``.
        """
        return self._stopped.wait(timeout)

    async def _start_grabbing(self) -> None:
        """Put the grabbing thread to work, unless it already is."""
        if self._grabbing.is_set():
            return
        # a thread that stopped on a fault has set _stopped, but its task is
        # done only once the loop has run the completion callback; a restart
        # asked for in between must not be dropped for a thread that is gone
        if self._grab_task is not None:
            await self._grab_task
        if self._store is None:
            # A previous bounded capture must not affect a later live-only
            # sequence; its extra-frame checks belong to the old sequence.
            self._capture_full = False
        # a fault is what stopped the last thread; starting another is the
        # request to try again, and its state reads so at once
        self._error = None
        self._grabbing.set()
        self._stopped.clear()
        self._grab_task = asyncio.create_task(asyncio.to_thread(self._grab_loop))

    async def _on_file_path(self, path: str) -> None:
        """Reset ``Captured``: a client names the store before it starts a capture.

        A client waits for ``Captured`` to grow past what it read while
        preparing; left at the last capture's total, the next would be waited
        on for twice its frames.
        """
        if self._store is not None:
            logger.warning(
                "Capture file path changed while a capture is open; "
                "the open capture remains pinned to its original location"
            )
            return
        await asyncio.to_thread(self._reset_written)
        await self.captured.update(0)
        await self.buffered_frames.update(0)

    async def _on_capture(self, capturing: bool) -> None:
        if capturing and self._snap_in_progress:
            raise RuntimeError("software snap in progress")
        async with self._control_lock:
            await self._on_capture_locked(capturing)

    async def _on_capture_locked(self, capturing: bool) -> None:
        """Open the store frames are written to, or finish the one open.

        Closing publishes the count too, since an unbounded capture ends here.
        """
        if not capturing:
            try:
                await asyncio.to_thread(self._close_store)
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
                await self._publish_state()
                raise
            await self.captured.update(self._written)
            logger.info(f"Capture closed: {self._written} frames written")
            return
        if self._store is not None:
            return
        wanted = self.num_capture.get()
        if wanted < 0:
            raise ValueError("capture frame count cannot be negative")
        if self._buffered_capture and not (
            1 <= wanted <= 2048 or (wanted == 0 and self._allow_unbounded_capture)
        ):
            raise ValueError("buffered capture requires 1..2048 frames")
        record_timing = self.record_frame_timing.get()
        if record_timing and not 0 < wanted <= self._timing_limit:
            raise ValueError(
                "frame timing requires a bounded capture between 1 and "
                f"{self._timing_limit} frames"
            )
        frame = (
            await asyncio.to_thread(self._layout_template)
            if self._defer_initial_frame
            else self._latest
        )
        if frame is None:
            raise RuntimeError("the camera has taken no frame to size the store from")
        if self._buffered_capture and frame.nbytes * wanted > self._buffer_limit_bytes:
            limit_mib = self._buffer_limit_bytes // (1024 * 1024)
            raise ValueError(f"buffered capture exceeds {limit_mib} MiB")
        self._written = 0
        self._accepted = 0
        await self.buffered_frames.update(0)
        self._batch = (
            BoundedFrameBatch(max_frames=wanted, max_bytes=self._buffer_limit_bytes)
            if self._buffered_capture and wanted
            else None
        )
        self._capture_full = False
        self._capture_wanted = wanted
        self._timing_records = []
        self._capture_record_timing = record_timing
        self._capture_uri = self.file_path.get()
        self._capture_data_key = self.data_key.get()
        try:
            self._store = FrameStore(self._capture_uri, self._capture_data_key, frame)
            set_planned = getattr(self._store, "set_planned_frames", None)
            if wanted and callable(set_planned):
                set_planned(wanted)
        except BaseException:
            if self._store is not None:
                with suppress(BaseException):
                    self._store.close()
                self._store = None
            self._batch = None
            self._capture_uri = None
            self._capture_data_key = None
            raise
        logger.info(
            f"Capture opened: {self._capture_data_key!r} in {self._capture_uri}"
        )

    async def _end_capture(self) -> None:
        """Finish a capture that has written every frame it was asked for."""
        try:
            await asyncio.to_thread(self._close_store)
            logger.info(f"Capture complete: {self._written} frames written")
        except Exception as error:
            self._error = f"{type(error).__name__}: {error}"
            logger.exception("Capture flush failed; stored outcome may be partial")
            await self._publish_state()
        finally:
            await self.captured.update(self._written)
            await self.capture.update(False)

    async def _stop_grabbing(self) -> None:
        """Ask the grabbing thread to end, and wait for it."""
        self._grabbing.clear()
        if self._grab_task is not None:
            await self._grab_task
            self._grab_task = None
        if self._geometry_refresh_pending:
            # the layout change that arrived mid-sequence refreshes now that
            # no property read can disturb a running adapter
            self._geometry_refresh_pending = False
            await self._publish_geometry()

    def _reset_written(self) -> None:
        with self._store_lock:
            self._written = 0
            self._accepted = 0

    def _close_store(self) -> None:
        """Finish the capture's store, if one is open.

        Held under the lock the grabbing thread appends beneath, so no store
        is closed between its check and its append. Closing writes what the
        store still holds, so it runs off the event loop.

        A store that recorded no frame holds only the placeholder a capture
        opens before its first frame arrives, so it is removed instead of left
        behind to look like a run.
        """
        with self._store_lock:
            if self._store is not None:
                store = self._store
                try:
                    if self._batch is not None:
                        try:
                            self._batch.flush(store)
                        except BaseException:
                            with suppress(BaseException):
                                store.close()
                            raise
                    store.close()
                    self._written = store.frames_written
                finally:
                    self._store = None
                    self._batch = None
                written = self._written
                if written == 0:
                    self._discard_empty_store()
                else:
                    self._write_timing_sidecar()
                self._capture_uri = None
                self._capture_data_key = None

    def _discard_empty_store(self) -> None:
        """Remove the store of a capture that recorded no frame.

        The store is created when the capture opens so frames can stream into
        it, before the first frame exists; a capture stopped before that frame
        leaves only the placeholder. A store with any chunk directory is never
        touched, so a recorded run cannot be removed here.
        """
        uri = self._capture_uri or self.file_path.get()
        path = local_store_path(uri)
        key = self._capture_data_key or self.data_key.get()
        array_path = path / key
        try:
            # Only remove this plane's empty array and manifest. A store
            # directory may also contain another plane's recorded frames.
            if array_path.is_dir() and not any(array_path.rglob("c")):
                import shutil

                shutil.rmtree(array_path)
            manifest = manifest_path(str(path), key)
            if manifest.exists():
                manifest.unlink()
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
            logger.info(f"Removed empty plane store: {array_path}")
        except OSError as exc:  # pragma: no cover - a racing filesystem
            logger.warning(f"could not remove empty store {path}: {exc}")

    def _write_timing_sidecar(self) -> None:
        """Atomically persist the bounded service-arrival timing series."""
        if not self._capture_record_timing:
            return
        uri = self._capture_uri or self.file_path.get()
        path = local_store_path(uri)
        data_key = self._capture_data_key or self.data_key.get()
        sidecar = path / f"{data_key}.timing.json"
        temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
        payload = {
            "schema": CAMERA_FRAME_TIMING_SCHEMA,
            "timestamp_semantics": CAMERA_FRAME_TIMING_SEMANTICS,
            "clock": CAMERA_FRAME_TIMING_CLOCK,
            "high_resolution_clock": "python.time.perf_counter_ns",
            "frame_metadata_source": (
                "per-record; null when the camera is not sequencing"
            ),
            "frame_metadata_semantics": self._frame_metadata_semantics(),
            "store_uri": str(path),
            "data_key": data_key,
            "written_count": self._written,
            "records": self._timing_records,
            "camera_frame_counter": None,
        }
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(sidecar)

    def _frame_metadata_semantics(self) -> str:
        return (
            "raw per-frame MMCore tags; timestamps are not assumed to be "
            "exposure intervals or mapped to the host monotonic clock"
        )

    def _frame_metadata_fields(self, metadata: dict[str, str] | None) -> dict[str, Any]:
        return {
            "camera_frame_counter": None,
            "mmcore_metadata_source": (
                "CMMCorePlus.popNextImageAndMD" if metadata is not None else None
            ),
            "mmcore_metadata": metadata,
        }

    def take_frame(self) -> NDArray[Any] | None:
        """Return the camera's next frame, or ``None`` while it has none.

        With a sequence running the frame comes from Micro-Manager's circular
        buffer; without one, each call exposes the camera. Taking it holds the
        camera lock: a driver reached from two threads at once, one exposing
        and one changing what frames look like, can end the process.
        """
        with self._camera_request_lock:
            pass
        with self._camera_lock:
            if not self._sequencing:
                return self._core.snap()
            if self._core.getRemainingImageCount() < 1:
                return None
            return self._core.popNextImage()

    def take_frame_with_metadata(
        self,
    ) -> tuple[NDArray[Any] | None, dict[str, str] | None]:
        """Pop one sequence frame with its unmodified MMCore tags.

        The metadata is retained as evidence only. Adapter timestamps can use
        a different epoch and event (for example, buffer insertion) from the
        host monotonic arrival stamp, so this method does not infer exposure
        start/end or convert the clocks.
        """
        with self._camera_request_lock:
            pass
        with self._camera_lock:
            if not self._sequencing:
                return self._core.snap(), None
            if self._core.getRemainingImageCount() < 1:
                return None, None
            frame, metadata = self._core.popNextImageAndMD()
            return frame, {str(key): str(value) for key, value in metadata.items()}

    def grab_once(self) -> NDArray[Any] | None:
        """Take one frame, and write it if a capture wants it.

        Runs in the grabbing thread and only appends to the store; on the last
        frame of a bounded capture it hands the closing to the event loop
        through ``_end_capture``.
        """
        record_timing = self._capture_record_timing
        if record_timing:
            frame, frame_metadata = self.take_frame_with_metadata()
        else:
            frame = self.take_frame()
            frame_metadata = None
        if frame is None:
            return None
        arrival_ns = time.monotonic_ns()
        arrival_perf_ns = time.perf_counter_ns()
        with self._setting_change_lock:
            ready = [
                item
                for item in self._pending_setting_changes
                if int(item["service_monotonic_ns"]) <= arrival_ns
            ]
            self._pending_setting_changes = [
                item
                for item in self._pending_setting_changes
                if int(item["service_monotonic_ns"]) > arrival_ns
            ]
            self._setting_latency_updates.extend(
                {
                    "setting_change_id": item["setting_change_id"],
                    "frame_latency_status": "measured",
                    "next_frame_id": self._grabbed + 1,
                    "next_frame_id_semantics": "camera_service_grab_ordinal",
                    "next_frame_service_monotonic_ns": arrival_ns,
                    "readback_to_next_frame_arrival_ms": round(
                        (arrival_ns - int(item["service_monotonic_ns"])) / 1_000_000,
                        4,
                    ),
                }
                for item in ready
            )
        self._latest = frame
        self._grabbed += 1

        with self._store_lock:
            store = self._store
            if store is None or self._capture_full:
                return frame
            if record_timing and len(self._timing_records) >= self._timing_limit:
                raise RuntimeError("frame timing record capacity exceeded")
            # This is the service's revision at append time. The adapter does
            # not provide an atomic exposure/configuration stamp for the frame.
            set_revision = getattr(store, "set_setting_revision", None)
            if callable(set_revision):
                set_revision(self.settings_revision.get())
            if self._batch is not None:
                self._batch.append(frame)
            else:
                store.append(frame)
            if record_timing:
                self._timing_records.append(
                    {
                        "written_index": self._accepted,
                        "frame_id": self._grabbed,
                        "service_monotonic_ns": arrival_ns,
                        "service_perf_counter_ns": arrival_perf_ns,
                        **self._frame_metadata_fields(frame_metadata),
                    }
                )
            self._accepted += 1
            if self._batch is None:
                self._written += 1
            wanted = self._capture_wanted
            if wanted and self._accepted >= wanted:
                self._capture_full = True
                self._request_capture_end()
        return frame

    def _request_capture_end(self) -> None:
        """Ask the event loop to end the capture this thread has filled."""
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._end_capture(), self._loop)

    def _grab_loop(self) -> None:
        """Take frames until asked to stop, or until the camera fails."""
        try:
            with self._camera_lock:
                self._start_sequence()
            while self._grabbing.is_set():
                if self._sequence_is_finite:
                    with self._camera_lock:
                        if self._core.isBufferOverflowed():
                            raise RuntimeError("camera buffer overflow")
                if self._sequence_is_finite and self._capture_full:
                    time.sleep(EMPTY_POLL)
                    continue
                if self.grab_once() is None:
                    if self._sequence_is_finite and not self._capture_full:
                        with self._camera_lock:
                            if not self._core.isSequenceRunning():
                                raise RuntimeError(
                                    "camera sequence ended before all frames"
                                )
                    time.sleep(EMPTY_POLL)
        except Exception as error:
            self._error = f"{type(error).__name__}: {error}"
            self._grabbing.clear()
            # a client waits for Capture to fall to know its capture is over,
            # and no frame will arrive to end it now
            self._request_capture_end()
        finally:
            try:
                with self._camera_lock:
                    self._stop_sequence()
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
                raise
            finally:
                self._stopped.set()

    def _when_idle(self, read: Callable[[], T]) -> T | None:
        """Return what *read* returns, or ``None`` while a sequence runs.

        The check and the read hold the camera lock, so no sequence can start
        between them: on some adapters a read during a sequence ends it.
        """
        with self._camera_request_lock, self._camera_lock:
            return None if self._sequencing else read()

    def _without_sequence(self, apply: Callable[[], None]) -> None:
        """Run *apply* with no sequence running, and resume one that was.

        The camera lock is held throughout, so the grabbing thread waits for
        the frame it is taking to end and takes no other until *apply* is
        done.
        """
        with self._camera_request_lock, self._camera_lock:
            was_sequencing = self._sequencing
            if was_sequencing:
                self._stop_sequence()
            try:
                apply()
            finally:
                if was_sequencing:
                    self._start_sequence()

    def _start_sequence(self) -> None:
        """Ask the camera for a continuous sequence, and note whether it took.

        An adapter that refuses one leaves ``take_frame`` exposing per frame.
        """
        # the flag is set before the sequence starts and cleared after it
        # stops: the grabbing thread exposes only while the flag is clear,
        # and Micro-Manager refuses an exposure during a sequence
        self._sequencing = True
        try:
            self._sequence_is_finite = (
                self._finite_capture
                and self._store is not None
                and self._capture_wanted > 0
            )
            if self._sequence_is_finite:
                if not 1 <= self._capture_wanted <= 2048:
                    raise RuntimeError(
                        "finite sequence requires an open bounded capture"
                    )
                if self._core.isSequenceRunning():
                    raise RuntimeError("camera sequence already running")
                self._core.clearCircularBuffer()
                self._core.startSequenceAcquisition(self._capture_wanted, 0, True)
            else:
                self._core.startContinuousSequenceAcquisition(0)
            self._sequence_started.set()
        except Exception as error:
            self._sequencing = False
            self._sequence_is_finite = False
            if self._defer_initial_frame:
                raise RuntimeError(
                    "camera refused required sequence acquisition"
                ) from error
            logger.warning(f"Camera refused a sequence, exposing per frame: {error}")

    def _stop_sequence(self) -> None:
        """End the sequence, if the camera is running one."""
        self._sequence_started.clear()
        if not self._sequencing:
            return
        try:
            if not self._sequence_is_finite:
                self._core.stopSequenceAcquisition()
                if (
                    self._buffered_capture
                    and self._capture_full
                    and (
                        self._core.isBufferOverflowed()
                        or self._core.getRemainingImageCount()
                    )
                ):
                    raise RuntimeError("camera sequence overflow or extra frames")
                return
            if self._capture_full:
                deadline = time.monotonic() + 1.0
                while self._core.isSequenceRunning() and time.monotonic() < deadline:
                    time.sleep(0.001)
            if self._core.isSequenceRunning():
                self._core.stopSequenceAcquisition()
                if self._capture_full:
                    raise TimeoutError("finite camera sequence did not stop")
            if self._capture_full and (
                self._core.isBufferOverflowed() or self._core.getRemainingImageCount()
            ):
                raise RuntimeError("camera sequence overflow or extra frames")
        finally:
            self._sequencing = False
            self._sequence_is_finite = False


def build_controller(
    adapter: str | None,
    device: str | None,
    label: str,
    data_key: str,
    properties: Iterable[str] | None = None,
    initial_properties: Mapping[str, str] | None = None,
    *,
    mm_config: str | None = None,
    mm_path: str | None = None,
    defer_initial_frame: bool = False,
    buffered_capture: bool = False,
    finite_capture: bool = False,
    buffer_limit_mib: int = 256,
) -> MMCameraController:
    """Load the camera, apply declared properties, and preserve exposure and ROI.

    *mm_config* loads a Micro-Manager system configuration instead of one
    adapter device; *device* then names the camera's device label in it, and
    only without it does *device* name the adapter's device while *label*
    names the MMCore device. *mm_path* is the Micro-Manager installation to
    load adapters from.

    Raises
    ------
    ValueError
        If neither a configuration nor an adapter device is given, or no
        camera can be selected from the configuration.
    """
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus(mm_path=mm_path) if mm_path else CMMCorePlus()
    if mm_config is not None:
        core.loadSystemConfiguration(mm_config)
        if device:
            core.setCameraDevice(device)
            label = device
        else:
            label = str(core.getCameraDevice())
        if not label:
            raise ValueError(
                f"no camera is selected in {mm_config!r}; give --device to select one"
            )
    else:
        if adapter is None or device is None:
            raise ValueError(
                "an adapter and device are required unless a Micro-Manager "
                "configuration is loaded"
            )
        core.loadDevice(label, adapter, device)
        core.initializeDevice(label)
        core.setCameraDevice(label)
    if initial_properties:
        apply_initial_properties(core, label, initial_properties)
    return MMCameraController(
        core,
        label,
        data_key,
        properties,
        defer_initial_frame=defer_initial_frame,
        buffered_capture=buffered_capture,
        finite_capture=finite_capture,
        buffer_limit_mib=buffer_limit_mib,
    )


def initial_properties_from_arguments(
    assignments: Iterable[str], parser: argparse.ArgumentParser
) -> dict[str, str]:
    """Read ``NAME=VALUE`` property defaults, refusing a malformed or repeated one."""
    initial_properties: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not name.strip():
            parser.error(
                f"invalid --initial-property {assignment!r}; expected NAME=VALUE"
            )
        name = name.strip()
        if name in initial_properties:
            parser.error(f"duplicate --initial-property for {name!r}")
        initial_properties[name] = value
    return initial_properties


def run(serving: Coroutine[Any, Any, None]) -> int:
    """Run *serving* until the session or Ctrl+C stops it; return the exit code.

    Ctrl+C, the way a service run on its own is stopped, counts as a clean
    stop: `asyncio.run` has cancelled and awaited *serving* before it raises.
    """
    with suppress(KeyboardInterrupt):
        asyncio.run(serving)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the service, taking its identity from the session that launched it."""
    service = identity()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default=service.prefix if service else "",
        help="PV prefix, the session's unless given",
    )
    parser.add_argument(
        "--name",
        default=service.name if service else "camera",
        help="name of this service, the session's unless given",
    )
    parser.add_argument(
        "--mm-config",
        help="Micro-Manager system configuration to load, instead of an adapter device",
    )
    parser.add_argument(
        "--mm-path",
        help="Micro-Manager installation to load device adapters from",
    )
    parser.add_argument(
        "--adapter",
        help="Micro-Manager adapter, when no configuration is loaded",
    )
    parser.add_argument(
        "--device",
        help=(
            "camera to serve: the adapter's device name without a configuration, "
            "or the device label within one"
        ),
    )
    parser.add_argument(
        "--properties",
        default=None,
        help=(
            "comma-separated camera properties to publish; every property the "
            "camera lets one write without it"
        ),
    )
    parser.add_argument(
        "--initial-property",
        nargs="+",
        default=[],
        metavar="NAME=VALUE",
        help="camera property defaults to apply after initialization",
    )
    parser.add_argument(
        "--defer-initial-frame",
        action="store_true",
        help="initialize monochrome layout without snapping a first frame",
    )
    parser.add_argument(
        "--buffered-capture",
        action="store_true",
        help="buffer a finite capture in RAM before writing (requires deferred mode)",
    )
    parser.add_argument(
        "--finite-capture",
        action="store_true",
        help="use finite internal sequences for prepared buffered captures",
    )
    parser.add_argument(
        "--buffer-limit-mib",
        type=int,
        default=256,
        help="maximum retained pixel bytes for a buffered capture (default: 256 MiB)",
    )
    options = parser.parse_args(argv)

    if not 1 <= options.buffer_limit_mib <= 4096:
        parser.error("--buffer-limit-mib must be 1..4096")
    if options.finite_capture and not options.buffered_capture:
        parser.error("--finite-capture requires --buffered-capture")
    if options.buffered_capture and not options.defer_initial_frame:
        parser.error("--buffered-capture requires --defer-initial-frame")

    initial_properties = initial_properties_from_arguments(
        options.initial_property, parser
    )
    properties: list[str] | None = None
    if options.properties is not None:
        properties = [name for name in options.properties.split(",") if name]
    configure_logging()
    controller = build_controller(
        options.adapter,
        options.device,
        str(options.name),
        str(options.name),
        properties,
        initial_properties,
        mm_config=options.mm_config,
        mm_path=options.mm_path,
        defer_initial_frame=options.defer_initial_frame,
        buffered_capture=options.buffered_capture,
        finite_capture=options.finite_capture,
        buffer_limit_mib=options.buffer_limit_mib,
    )
    return run(
        serve_controller(controller, EpicsPVATransport(), options.prefix or None)
    )


if __name__ == "__main__":
    sys.exit(main())
