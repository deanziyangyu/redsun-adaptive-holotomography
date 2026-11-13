"""Four-pattern DPCT snapshots from a free-running camera and MM ASCII panel.

Each pattern is set once, then each detector returns a fresh frame from its
software-visible buffer. The plan owns the pixel store, so an Event is emitted
only after that frame has been appended. This mode makes no camera TTL claim.
"""

from __future__ import annotations

import logging
import math
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast

from bluesky import plan_stubs as bps
from bluesky.utils import RunEngineControlException

from redsun_aht.contracts import (
    CameraDescription,
    CameraReadout,
    FrameIdentity,
    Illumination,
    IlluminationAction,
    PartialRun,
    PlanAbortReason,
    PlanOutcome,
    PlanStatus,
    RunIdentity,
    TriggerMode,
)
from redsun_aht.storage.writer import (
    FrameStamp,
    RunFrameStore,
    StoreRefused,
    local_store_path,
)

from .dpct import (
    DEVICE_TIMEOUT_S,
    DPCT_PATTERN_SEQUENCE,
    FRAME_TIMEOUT_S,
    STREAM_NAME,
    DpctError,
    IlluminationUncertain,
    MeasuredStage,
    PatternStep,
    _blank_and_open,
    _bounded_call,
    _confirmed,
    _emit_frame,
    _finalize,
    _FrameSlot,
    _receipt,
    _stageable,
    _start_document,
    _validate_bound,
    _validate_illumination,
    _validate_sequence,
    _validate_stage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bluesky.protocols import Status
    from bluesky.utils import MsgGenerator

    from .dpct import DpctDetector

logger = logging.getLogger(__name__)


class _CameraFrameTimeout(TimeoutError):
    """A free-running camera did not publish a frame before its bound."""


class SoftwareDpctDetector(Protocol):
    """A staged camera that can return a fresh YX frame and name its store."""

    name: str

    def stage(self) -> Status:
        """Start the camera's free-running acquisition."""
        ...

    def unstage(self) -> Status:
        """Stop acquisition and release the camera."""
        ...

    async def describe_settings(self) -> CameraDescription:
        """Describe the camera trigger modes and detector identity."""
        ...

    async def capture_frame(self) -> CameraReadout:
        """Return a new frame after the most recent pattern selection."""
        ...

    def dpct_store_uri(self, run_id: str) -> str:
        """Name the local store for this run and detector."""
        ...


def software_dpct_plan(
    detectors: Sequence[SoftwareDpctDetector],
    stage: MeasuredStage | None,
    illumination: Illumination,
    *,
    run_id: str,
    sequence: Sequence[PatternStep] = DPCT_PATTERN_SEQUENCE,
    settle_s: float = 0.02,
    frame_timeout_s: float = FRAME_TIMEOUT_S,
    device_timeout_s: float = DEVICE_TIMEOUT_S,
    stream_name: str = STREAM_NAME,
) -> MsgGenerator[PlanOutcome]:
    """Record one frame per pattern and detector without a hardware trigger."""
    cameras = tuple(detectors)
    if not cameras:
        raise ValueError("a DPCT run needs at least one detector")
    names = tuple(camera.name for camera in cameras)
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("DPCT detector names must be non-empty and unique")
    steps = _validate_sequence(sequence)
    capability = _validate_illumination(illumination, steps)
    axes = _validate_stage(stage)
    _validate_bound(device_timeout_s, "device_timeout_s")
    _validate_bound(frame_timeout_s, "frame_timeout_s")
    if (
        not isinstance(settle_s, (int, float))
        or not math.isfinite(settle_s)
        or settle_s < 0
    ):
        raise ValueError("settle_s must be finite and non-negative")
    run = RunIdentity(
        run_id=run_id, plan_name="dpct", purpose="four-pattern DPCT", scan_id=0
    )
    locations: dict[str, str] = {}
    for camera in cameras:
        description = yield from _bounded_call(
            camera.describe_settings,
            timeout_s=device_timeout_s,
            what=f"detector {camera.name!r} description",
        )
        if (
            not isinstance(description, CameraDescription)
            or description.detector_id != camera.name
            or TriggerMode.FREE_RUN not in description.trigger_modes
        ):
            raise ValueError(
                f"detector {camera.name!r} cannot supply free-running frames"
            )
        if not callable(getattr(camera, "capture_frame", None)) or not callable(
            getattr(camera, "dpct_store_uri", None)
        ):
            raise ValueError(
                f"detector {camera.name!r} has no software frame store path"
            )
        uri = camera.dpct_store_uri(run_id)
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError(f"detector {camera.name!r} gave no store URI")
        store_path = local_store_path(uri).resolve()
        prior_paths = {
            local_store_path(value).resolve() for value in locations.values()
        }
        if store_path in prior_paths:
            raise ValueError(
                f"detector {camera.name!r} shares a pixel store location with "
                "another detector"
            )
        locations[camera.name] = uri

    requested = tuple(
        FrameIdentity(
            run=run,
            detector_id=name,
            sequence=ordinal,
            plane_ordinal=0,
            pattern_id=step.pattern_id,
            scan_index=0,
        )
        for ordinal, step in enumerate(steps)
        for name in names
    )
    produced: list[FrameIdentity] = []
    slots = _FrameSlot(axes)
    stores: dict[str, RunFrameStore] = {}
    failure: list[tuple[PlanAbortReason, str]] = []
    metadata = _start_document(run, names, steps, capability, axes)
    metadata["frame_trigger"] = "software-fresh-frame-after-pattern"
    metadata["dpct_store_uris"] = locations
    metadata["dpct_producer_sequence_available"] = False
    metadata["dpct_setting_revision_available"] = False

    yield from bps.open_run(md=metadata)
    staged: list[Any] = []
    try:
        for device in _stageable(
            cast("Sequence[DpctDetector]", cameras), stage, illumination
        ):
            yield from bps.stage(device)
            staged.append(device)
        yield from _blank_and_open(illumination, device_timeout_s)
        for ordinal, step in enumerate(steps):
            result = yield from _bounded_call(
                partial(illumination.pattern, step.pattern_id),
                timeout_s=device_timeout_s,
                what=f"pattern {step.pattern_id}",
            )
            command = _confirmed(
                _receipt(
                    result,
                    action=IlluminationAction.PATTERN,
                    pattern_id=step.pattern_id,
                ),
                what=f"pattern {step.pattern_id}",
            )
            if settle_s:
                yield from bps.sleep(settle_s)
            position = None
            if stage is not None:
                position = yield from _bounded_call(
                    stage.read_position,
                    timeout_s=device_timeout_s,
                    what="reading stage position",
                )
            for camera in cameras:
                try:
                    frame = yield from _bounded_call(
                        camera.capture_frame,
                        timeout_s=frame_timeout_s,
                        what=f"detector {camera.name!r} fresh frame",
                    )
                except DpctError as error:
                    if "did not answer" in str(error):
                        raise _CameraFrameTimeout(str(error)) from error
                    raise
                if not isinstance(frame, CameraReadout):
                    raise ValueError(
                        f"detector {camera.name!r} returned no pixel frame"
                    )
                store = stores.get(camera.name)
                if store is None:
                    store = RunFrameStore(
                        locations[camera.name],
                        camera.name,
                        frame.data,
                        run=run,
                        detector_id=camera.name,
                        planned_frames=len(steps),
                        # PVA exposes the revision separately from the buffer;
                        # do not attach that read to these pixels.
                        setting_revision=None,
                    )
                    stores[camera.name] = store
                store.append(
                    frame.data,
                    stamp=FrameStamp(
                        pattern_id=step.pattern_id,
                        # The service publishes buffer and counter separately,
                        # so this readout's count cannot be proven to identify
                        # these pixels under continuous acquisition.
                        producer_sequence=None,
                        setting_revision=None,
                    ),
                )
                identity = FrameIdentity(
                    run=run,
                    detector_id=camera.name,
                    sequence=ordinal,
                    plane_ordinal=0,
                    pattern_id=step.pattern_id,
                    scan_index=0,
                    producer_sequence=None,
                )
                yield from _emit_frame(stream_name, slots, identity, command, position)
                produced.append(identity)
    except RunEngineControlException:
        raise
    except StoreRefused as error:
        failure.append(
            (PlanAbortReason.STORAGE_REFUSED, f"{type(error).__name__}: {error}")
        )
    except IlluminationUncertain as error:
        failure.append(
            (PlanAbortReason.ILLUMINATION_UNCERTAIN, f"{type(error).__name__}: {error}")
        )
    except Exception as error:
        reason = (
            PlanAbortReason.CAMERA_TIMEOUT
            if isinstance(error, TimeoutError)
            else PlanAbortReason.DEVICE_FAULT
        )
        failure.append((reason, f"{type(error).__name__}: {error}"))
    finally:
        for store in stores.values():
            try:
                store.close()
            except Exception:
                logger.exception("could not close DPCT store")
                failure.append(
                    (PlanAbortReason.DEVICE_FAULT, "could not close DPCT store")
                )
        yield from _finalize(illumination, staged, device_timeout_s)

    if failure:
        yield from bps.close_run(exit_status="abort", reason=failure[0][1])
        return PlanOutcome(
            run=run,
            status=PlanStatus.ABORTED,
            reason=failure[0][0],
            partial=PartialRun(requested=requested, produced=tuple(produced)),
            detail=failure[0][1],
            finished_monotonic_ns=time.monotonic_ns(),
        )
    yield from bps.close_run()
    return PlanOutcome(
        run=run,
        status=PlanStatus.COMPLETE,
        detail=f"{len(steps)} patterns, {len(produced)} stored frames",
        finished_monotonic_ns=time.monotonic_ns(),
    )
