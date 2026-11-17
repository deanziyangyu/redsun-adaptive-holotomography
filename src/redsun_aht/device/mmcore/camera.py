"""A Micro-Manager camera, reached through the PVA service that owns it.

Ported from ``redsun_searching_is_imaging/device/mmcore/_camera.py`` at donor
revision ``3a3bf06`` (``redsun-searching-is-imaging``).

Changed:

- The module is ``device/mmcore/camera.py``, not ``_camera.py``, so a session
  imports the device from a public module.
- The shared-memory live ring is gone with the service's publishing of it; the
  searching bundle's broom consumed that ring and this bundle previews through
  Bluesky Events instead.
- The dark-frame retry is described as the PVCAM/shutter behaviour it is, and
  stays off unless a session opts in: a camera whose shutter reports nothing
  may repeat one unlit frame, a camera without that shutter must not pay a
  second exposure for it.
- The device also implements the plan-facing surface of
  ``redsun_aht.contracts.devices.Camera``: ``describe_settings``,
  ``apply_settings``, ``capture_frame`` and ``stop``. They are an adapter over
  the service signals, not a second device: ``trigger``/``read``/``describe``
  keep their Bluesky meanings and are untouched, and only ``apply_settings``
  converts seconds to the milliseconds the service takes.
- The soft-signal based ``FakeDetector`` doubles that the searching bundle's
  plan tests shared are not part of this port.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from ophyd_async.core import (
    AsyncStatus,
    DetectorAcquireLogic,
    DetectorDataLogic,
    DetectorTriggerLogic,
    SignalR,
    SignalRW,
    SignalW,
    StandardDetector,
    StreamResourceDataProvider,
    StreamResourceInfo,
    TriggerInfo,
    callback_on_mock_put,
    set_mock_value,
    wait_for_value,
)
from ophyd_async.fastcs.core import fastcs_connector
from redsun.log import Loggable
from redsun.writers import ZARR

from ...contracts import (
    CameraDescription,
    CameraReadout,
    CameraSettings,
    Roi,
    SettingRefusal,
    SettingRefused,
    TriggerMode,
)
from ._containers import ReadableDeviceMap  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bluesky.protocols import Reading
    from event_model import DataKey
    from ophyd_async.core import PathProvider, StreamableDataProvider

#: Seconds ``trigger`` waits for a frame taken after it was called.
DEFAULT_TIMEOUT: Final = 5.0

logger = logging.getLogger(__name__)

#: Default spread, in counts, below which a frame is treated as carrying no light.
#:
#: The test is structural rather than absolute, so it does not depend on a camera's
#: offset: an unlit PVCAM frame here reads min 86-90, median 100, p99 104 (a spread of
#: about 18 counts), while a lit one reads min ~90, median 197, p99 505-757 (a spread of
#: 400 or more). 40 counts sits between them with room on both sides. A uniformly lit
#: blank field also fails the test and costs one extra exposure, which is the price of
#: not guessing an absolute floor per camera. Measured on the IX83 PVCAM camera on
#: 2026-10-30; it is not a property of Micro-Manager or of ophyd-async, which is why
#: the retry that uses it is off unless a session asks for it.
DEFAULT_DARK_FRAME_STRUCTURE_DN: Final = 40.0


def frame_has_signal(frame: Any, structure_dn: float) -> bool:
    """Return whether *frame* carries more than *structure_dn* counts of spread.

    Measured on the IX83 PVCAM camera on 2026-10-30: an unlit frame spreads about 18
    counts around its offset, a frame lit through the VMM-D1 at 2 mW spreads 400 or
    more.
    """
    values = np.asarray(frame)
    if values.size == 0:
        return False
    spread = float(np.percentile(values, 99) - values.min())
    return spread > float(structure_dn)


async def trigger_until_lit(
    trigger_once: Any,
    read_frame: Any,
    *,
    retries: int,
    structure_dn: float,
    describe: str = "camera",
) -> int:
    """Take a frame, retaking it while it carries no light; return the exposures taken.

    A mechanical shutter that is commanded open and does not deliver light for one frame
    is a transient, not a measurement: measured 2026-10-30 through the VMM-D1, one or
    two frames in ten sat at the exact closed-shutter floor while the same stage
    position was lit on the next run, and the driver has no readback that could tell a
    lost command from a blade that did not travel. Retaking the frame inside the trigger
    keeps one capture one event, so a plan's frame-to-position association is untouched.

    It can only ever *replace* an unlit frame: a frame that carries light is returned as
    it is, and the last attempt is returned whether or not it carries light.
    """
    attempts = max(0, int(retries)) + 1
    for attempt in range(1, attempts + 1):
        await trigger_once()
        if attempt == attempts:
            return attempt
        frame = await read_frame()
        if frame_has_signal(frame, structure_dn):
            return attempt
        logger.warning(
            f"{describe} frame {attempt} carried no light "
            f"(spread <= {structure_dn:g} counts); retaking it"
        )
    return attempts


#: Seconds between two reads of the frame counter in ``trigger``. The service publishes
#: the counter every ``LIVE_PERIOD`` (0.02 s), so polling faster than half that only
#: adds traffic: a frame becomes visible at the next publish, whichever pole runs
#: faster.
FRAME_POLL: Final = 0.01

#: Milliseconds per second, the units a camera takes its exposure in.
MILLISECONDS = 1000.0

#: What the service reports as its state once grabbing stopped on an error.
FAULTED: Final = "faulted"


@dataclass
class ServiceTriggerLogic(DetectorTriggerLogic):
    """Trigger logic telling the service how many frames the next capture writes."""

    camera: MMCamera

    def config_sigs(self) -> set[SignalR[Any]]:
        """Return the settings that describe how the frames were taken."""
        return {
            self.camera.exposure,
            self.camera.roi,
            self.camera.pixel_dtype,
            self.camera.sensor_size,
            self.camera.sensor_size_binning,
            self.camera.record_frame_timing,
        }

    async def prepare_internal(
        self, num: int, livetime: float, deadtime: float
    ) -> None:
        """Set the exposure and the number of frames to write, 0 for unbounded."""
        if livetime:
            await self.camera.exposure.set(livetime * MILLISECONDS)
        await self.camera.num_capture.set(num)
        if num == 0:
            # Per-frame timing sidecars are bounded; a previous stack may
            # have left timing enabled before this indefinite live capture.
            await self.camera.record_frame_timing.set(False)

    async def default_trigger_info(self) -> TriggerInfo:
        """Return the unbounded capture a plan without `prepare` gets."""
        return TriggerInfo(number_of_events=0)


@dataclass
class ServiceAcquireLogic(DetectorAcquireLogic):
    """Acquire logic starting and stopping the camera and the capture it writes.

    The camera takes frames from ``stage`` to ``unstage`` whether or not
    anything is written: a viewer watches the buffer, and a read after a move
    wants the frame taken since.
    """

    camera: MMCamera

    async def ensure_ready(self) -> None:
        """Have the camera take frames."""
        await self.camera.acquire.set(True)

    async def start_acquiring(self) -> None:
        """Start writing frames."""
        await self.camera.capture.set(True)

    async def wait_for_idle(self) -> None:
        """Wait for a bounded capture's last frame, or end an unbounded capture.

        Ending an unbounded capture here rather than at unstage keeps the
        count the documents report equal to the frames on disk.
        """
        if await self.camera.num_capture.get_value():
            capture_ended = asyncio.ensure_future(
                wait_for_value(self.camera.capture, False, timeout=None)
            )
            fault_raised = asyncio.ensure_future(
                wait_for_value(self.camera.state, FAULTED, timeout=None)
            )
            try:
                await asyncio.wait(
                    [capture_ended, fault_raised], return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                capture_ended.cancel()
                fault_raised.cancel()
                # a cancelled wait closes its subscription only once it runs
                # again; a read made before then waits on the closing
                # subscription's first value, which never comes
                await asyncio.gather(
                    capture_ended, fault_raised, return_exceptions=True
                )
            await self.camera.faulted()
        else:
            await self.camera.capture.set(False)

    async def ensure_stopped(self) -> None:
        """End the capture and stop the camera."""
        await self.camera.capture.set(False)
        await self.camera.acquire.set(False)


@dataclass
class ServiceDataLogic(DetectorDataLogic):
    """Data logic naming the store the service writes and reporting what it wrote."""

    camera: MMCamera
    path_provider: PathProvider

    def get_hinted_fields(self, datakey_name: str) -> Sequence[str]:
        """Return the stream a viewer plots by default."""
        return [datakey_name]

    async def prepare_unbounded(self, datakey_name: str) -> StreamableDataProvider:
        """Hand the service a store to write, and describe what lands in it."""
        info = self.path_provider(datakey_name)
        directory = Path(info.directory_path)
        directory.mkdir(parents=True, exist_ok=True)
        store = directory / f"{info.filename}.zarr"

        await asyncio.gather(
            self.camera.file_path.set(str(store)),
            self.camera.data_key.set(datakey_name),
        )
        # the service starts its count over when it is handed a store, and
        # what is counted from here is what this capture writes
        await wait_for_value(self.camera.captured, 0, timeout=DEFAULT_TIMEOUT)
        shape, dtype = await frame_shape_and_dtype(self.camera)

        return StreamResourceDataProvider(
            uri=f"{info.directory_uri}{info.filename}.zarr",
            resources=[
                StreamResourceInfo(
                    data_key=datakey_name,
                    shape=shape,
                    chunk_shape=(1, *shape),
                    dtype_numpy=np.dtype(dtype).str,
                    parameters={"dataset": datakey_name},
                )
            ],
            mimetype=ZARR,
            collections_written_signal=self.camera.captured,
        )


async def frame_shape_and_dtype(camera: MMCamera) -> tuple[tuple[int, int], str]:
    """Return the ``(height, width)`` and dtype of the frames a camera sends."""
    text, dtype = await asyncio.gather(
        camera.roi.get_value(), camera.pixel_dtype.get_value()
    )
    roi = Roi.parse(text)
    return (roi.height, roi.width), dtype


class MMCamera(StandardDetector, Loggable):
    """A Micro-Manager camera, reached through the service that owns it.

    The service publishes PVI, so each signal below is built from its
    annotation and paired by name; the adapter and the device belong to the
    service's declaration.

    Parameters
    ----------
    prefix :
        PV prefix of the service, ending in ``:``; a device declared with
        ``service=`` receives it from that service.
    path_provider :
        Where a capture writes; the session passes its own.
    dark_frame_retries :
        Extra exposures ``trigger`` may spend on a frame that carries no
        light. Zero, the default, takes exactly one exposure per trigger. A
        session whose PVCAM camera sits behind a mechanical shutter that
        reports nothing opts in; nothing in Micro-Manager or ophyd-async
        requires it.
    dark_frame_structure_dn :
        Counts of spread below which a frame counts as carrying no light.
        Unused unless ``dark_frame_retries`` is non-zero.
    """

    # the filler reads these annotations at runtime to build the signals, so
    # the types they name are imported at runtime too. The settings a reading
    # is described by are registered by the trigger logic, since a detector is
    # not a StandardReadable and cannot carry the annotation
    exposure: SignalRW[float]
    roi: SignalRW[str]
    pixel_dtype: SignalRW[str]
    sensor_size: SignalR[np.ndarray]
    sensor_size_binning: SignalR[str]
    #: ``[low, high]`` exposure limits in milliseconds, or ``[0, 0]`` when the
    #: service's adapter declares none.
    exposure_range: SignalR[np.ndarray]
    #: The readout dtypes the camera offers, comma separated.
    pixel_dtypes: SignalR[str]
    #: The binning factors the camera declares, comma separated.
    binning_factors: SignalR[str]
    #: The revision of the settings the service has applied.
    settings_revision: SignalR[int]
    buffer: SignalR[np.ndarray]
    acquire: SignalRW[bool]
    snap_request: SignalW[bool]
    frame_count: SignalR[int]
    capture: SignalRW[bool]
    file_path: SignalRW[str]
    data_key: SignalRW[str]
    num_capture: SignalRW[int]
    record_frame_timing: SignalRW[bool]
    captured: SignalR[int]
    state: SignalR[str]
    sequence_started: SignalR[bool]
    last_error: SignalR[str]
    last_setting_change: SignalR[str]
    setting_change_latency: SignalR[str]
    properties: ReadableDeviceMap[SignalRW[str]]

    # The planner may read this live buffer after a completed stage move.
    # The trigger waits for a new free-running frame; it does not request an
    # exposure or prove that the exposure began after that move.
    planner_snap_readable = True

    def __init__(
        self,
        prefix: str,
        *,
        path_provider: PathProvider,
        name: str = "",
        dark_frame_retries: int = 0,
        dark_frame_structure_dn: float = DEFAULT_DARK_FRAME_STRUCTURE_DN,
    ) -> None:
        super().__init__(name=name, connector=fastcs_connector(prefix, self))
        if (
            isinstance(dark_frame_retries, bool)
            or not isinstance(dark_frame_retries, int)
            or dark_frame_retries < 0
        ):
            raise ValueError("dark_frame_retries must be a non-negative integer")
        if (
            isinstance(dark_frame_structure_dn, bool)
            or not isinstance(dark_frame_structure_dn, (int, float))
            or not np.isfinite(dark_frame_structure_dn)
            or dark_frame_structure_dn < 0
        ):
            raise ValueError("dark_frame_structure_dn must be finite and non-negative")
        self._dark_frame_retries = dark_frame_retries
        self._dark_frame_structure_dn = float(dark_frame_structure_dn)
        self.path_provider = path_provider
        self._acquire = ServiceAcquireLogic(self)
        self.add_detector_logics(
            ServiceTriggerLogic(self),
            self._acquire,
            ServiceDataLogic(self, path_provider),
        )

    async def connect(self, *args, **kwargs) -> None:
        await super().connect(*args, **kwargs)
        if not kwargs.get("mock", args[0] if args else False):
            return

        shape = (4, 4)
        dtype = np.dtype("uint16")
        set_mock_value(self.sensor_size, np.asarray(shape[::-1], dtype=np.int32))
        set_mock_value(self.pixel_dtype, dtype.name)
        set_mock_value(self.pixel_dtypes, dtype.name)
        set_mock_value(self.roi, Roi(0, 0, shape[1], shape[0]).as_text())
        set_mock_value(self.buffer, np.zeros(shape, dtype=dtype))

        async def snap(requested: bool) -> None:
            if requested:
                count = await self.frame_count.get_value() + 1
                set_mock_value(self.buffer, np.full(shape, count % 65536, dtype=dtype))
                set_mock_value(self.frame_count, count)

        callback_on_mock_put(self.snap_request, snap)

    def dpct_store_uri(self, run_id: str) -> str:
        """Reserve a separate local pixel store for one software DPCT run."""
        info = self.path_provider(f"dpct-{run_id}-{self.name}")
        directory = Path(info.directory_path)
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory / f"{info.filename}.zarr")

    async def read_configuration(self) -> dict[str, Reading[Any]]:
        """Return the settings by name, then every property the camera can write."""
        settings, properties, timing = await asyncio.gather(
            super().read_configuration(),
            self.properties.read(),
            self.last_setting_change.read(),
        )
        # the trigger logic hands its settings over as a set, in no fixed order
        return {**dict(sorted(settings.items())), **properties, **timing}

    async def describe_configuration(self) -> dict[str, DataKey]:
        """Describe the settings by name, then every property the camera can write."""
        settings, properties, timing = await asyncio.gather(
            super().describe_configuration(),
            self.properties.describe(),
            self.last_setting_change.describe(),
        )
        # the trigger logic hands its settings over as a set, in no fixed order
        return {**dict(sorted(settings.items())), **properties, **timing}

    async def faulted(self) -> None:
        """Raise with the camera's own error if it stopped on a fault.

        A fault ends the grabbing thread, so waiting on a frame would only
        time out; this names the cause instead.
        """
        if await self.state.get_value() == FAULTED:
            raise RuntimeError(
                f"{self.name} stopped: {await self.last_error.get_value()}"
            )

    @AsyncStatus.wrap
    async def trigger(self) -> None:  # type: ignore[override]
        """Wait for a frame taken after this call, retaking one that carries no light.

        The camera runs continuously, so the frame already published may
        predate the move a plan just made.

        With ``dark_frame_retries`` set, an unlit frame is retaken here rather
        than kept. Doing it inside the trigger is what keeps one capture one
        event, so a plan's frame-to-position association and its frame count
        are unchanged; the frame a reader sees afterwards is the last one taken.
        A session whose gate is a mechanical shutter that reports nothing opts
        into this, because a shutter commanded open that delivered no light for
        one frame is a transient. With the default of zero, one trigger takes
        exactly one frame.

        The counter is polled, not watched. A monitor opened on it while a
        plan runs comes back from the service as ``Monitor Create implied
        error`` often enough to abort a scan, and every later read through
        that subscription times out. This is a workaround: watching the
        counter is the right shape once a monitor holds.
        """

        async def take_one() -> None:
            await self._wait_for_new_frame()

        async def read_frame() -> Any:
            return await self.buffer.get_value()

        await trigger_until_lit(
            take_one,
            read_frame,
            retries=self._dark_frame_retries,
            structure_dn=self._dark_frame_structure_dn,
            describe=self.name,
        )

    async def _wait_for_new_frame(self) -> None:
        """Block until the service publishes a frame newer than the one just seen."""
        seen = await self.frame_count.get_value(cached=False)
        await self.faulted()
        deadline = time.monotonic() + DEFAULT_TIMEOUT
        while await self.frame_count.get_value(cached=False) <= seen:
            if time.monotonic() > deadline:
                await self.faulted()
                raise TimeoutError(
                    f"{self.name} took no frame in {DEFAULT_TIMEOUT} s, "
                    f"its frame count still at {seen}"
                )
            await asyncio.sleep(FRAME_POLL)

    async def describe_settings(self) -> CameraDescription:
        """Return what this camera declares it can do, and its current revision.

        The description is read from the signals the service publishes, so it
        describes the service's view of the adapter, not this device's own
        configuration.

        Raises
        ------
        SettingRefused
            With ``EXPOSURE_OUT_OF_RANGE`` when the service publishes the empty
            exposure range, because the adapter names no usable limits, or when
            the limits it does name cannot be a
            :class:`~redsun_aht.contracts.CameraDescription` range, which
            requires a finite, increasing, non-negative interval. A refusal is
            deliberate: a description that invented a bound would let a plan
            request an exposure no camera checked.
        """
        width, height = (int(size) for size in await self.sensor_size.get_value())
        limits = await self.exposure_range.get_value()
        low = float(limits[0])
        high = float(limits[1])
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
            raise SettingRefused(
                SettingRefusal.EXPOSURE_OUT_OF_RANGE,
                {
                    "detector_id": self._detector_id,
                    "reason": (
                        "the service publishes no exposure range; its adapter "
                        "declares no usable limits, or none that are finite, "
                        "increasing and positive"
                    ),
                    "reported_ms": [low, high],
                },
            )
        return CameraDescription(
            detector_id=self._detector_id,
            shape=(height, width),
            pixel_dtypes=await self._declared_pixel_dtypes(),
            exposure_range_s=(low / MILLISECONDS, high / MILLISECONDS),
            # the service always serves a ROI: whether an adapter honours one
            # is a property of that adapter, not of the control surface
            roi_supported=True,
            binning=await self._declared_binning(),
            # the service takes frames from a continuous sequence; a client can
            # ask for the next one, and the illumination TTL is not read here,
            # so EXTERNAL is not declared until commissioning reads the
            # adapter's trigger property
            trigger_modes=(TriggerMode.FREE_RUN, TriggerMode.SOFTWARE),
            revision=int(await self.settings_revision.get_value()),
        )

    async def apply_settings(self, settings: CameraSettings) -> int:
        """Apply *settings* through the service, and return the new revision.

        Only the conversion to the service's units happens here: exposure
        arrives in seconds and is written in milliseconds, which is what
        Micro-Manager holds. Binning is written in the text form the camera
        already reports, so ``"1x1"`` stays ``"2x2"`` and ``"1"`` stays ``"2"``.

        Raises
        ------
        SettingRefused
            Before any write, when the description does not declare the
            requested settings, and when a binning factor the description
            declares has no writable service property behind it.
        """
        description = await self.describe_settings()
        settings.validate_for(description)
        binning = self.properties.get("Binning")
        if binning is not None:
            await binning.set(await self._binning_text(settings.binning))
        elif settings.binning != 1:
            raise SettingRefused(
                SettingRefusal.BINNING_UNSUPPORTED,
                {
                    "detector_id": description.detector_id,
                    "binning": settings.binning,
                    "reason": "the service publishes no writable Binning property",
                },
            )
        await self.roi.set(settings.roi.as_text())
        await self.exposure.set(float(settings.exposure_s) * MILLISECONDS)
        await self.pixel_dtype.set(settings.pixel_dtype)
        return int(await self.settings_revision.get_value())

    async def capture_frame(self) -> CameraReadout:
        """Return one frame with the settings revision it was taken under.

        The camera free-runs, so this waits for the next frame rather than
        asking for one; ``trigger`` remains the Bluesky shape of the same wait.
        """
        await self._wait_for_new_frame()
        frame = await self.buffer.get_value()
        if frame is None:
            raise RuntimeError(f"{self.name} published no frame to read")
        # The service keeps publishing into its live buffer while a plan is
        # appending this readout. Give the plan an owned snapshot so a later
        # buffer update cannot change pixels after their sequence was assigned.
        data = np.array(frame, copy=True)
        return CameraReadout(
            data=data,
            sequence=int(await self.frame_count.get_value(cached=False)),
            setting_revision=int(await self.settings_revision.get_value()),
            exposure_s=float(await self.exposure.get_value()) / MILLISECONDS,
            captured_monotonic_ns=time.monotonic_ns(),
        )

    async def stop(self, success: bool = True) -> None:
        """End the capture, stop the camera, and say whether the stop was asked for.

        ``StandardDetector`` has no ``stop`` of its own in ophyd-async 0.21, so
        this is what makes the camera a Bluesky ``Stoppable`` and therefore a
        :class:`redsun_aht.protocols.LiveFlyer`. *success* is the Bluesky flag
        for "the stop was intended", not an outcome: this device has no move
        status to cancel, so both values do the same thing.
        """
        del success
        await self._acquire.ensure_stopped()

    async def _declared_pixel_dtypes(self) -> tuple[str, ...]:
        """Return the readout dtypes the service publishes, or the current one."""
        text = await self.pixel_dtypes.get_value()
        names = tuple(name.strip() for name in text.split(",") if name.strip())
        if names:
            return names
        return (str(await self.pixel_dtype.get_value()),)

    async def _declared_binning(self) -> tuple[int, ...]:
        """Return the binning factors the service publishes, ``(1,)`` when none."""
        text = await self.binning_factors.get_value()
        factors = tuple(int(part) for part in text.split(",") if part.strip().isdigit())
        return factors or (1,)

    @property
    def _detector_id(self) -> str:
        """Return the identity frames are recorded under.

        The service publishes no separate detector identity, so the device's
        own name is what a frame is labelled with, which is also the name the
        stream's data keys carry.
        """
        return self.name

    async def _binning_text(self, factor: int) -> str:
        """Return *factor* in the text form the camera reports its binning in.

        A camera that reads back ``"1x1"`` takes ``"2x2"``, and one that reads
        back ``"1"`` takes ``"2"``. The service reads every property write back
        and records the outcome, so a form the adapter does not take surfaces as
        a failed setting change rather than as a silent no-op.
        """
        current = str(await self.sensor_size_binning.get_value())
        return f"{factor}x{factor}" if "x" in current.lower() else str(factor)


__all__ = [
    "DEFAULT_DARK_FRAME_STRUCTURE_DN",
    "DEFAULT_TIMEOUT",
    "FRAME_POLL",
    "MILLISECONDS",
    "MMCamera",
    "frame_has_signal",
    "frame_shape_and_dtype",
    "trigger_until_lit",
]
