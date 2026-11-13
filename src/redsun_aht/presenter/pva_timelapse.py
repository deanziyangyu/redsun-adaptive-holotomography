"""Explicitly bound plan presenter for service-owned time-lapse rehearsals."""

from __future__ import annotations

import logging
from collections.abc import Sequence  # noqa: TC003 - resolved by Redsun
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp

from redsun_aht.device.pva_automatic_stack import acquire_pva_automatic_stack
from redsun_aht.device.pva_gated_shots import acquire_pva_gated_shots
from redsun_aht.device.unified_mcu import AngularStack
from redsun_aht.plans.confirmation_ids import new_confirmation_id
from redsun_aht.plans.dpct import _bounded_call
from redsun_aht.plans.pva_timelapse import pva_timelapse_plan
from redsun_aht.plans.timelapse import _PointRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from redsun.registry import PlanEntry


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PvaTimeLapseBindings:
    """Session-supplied service owners and instrument-specific plan operations."""

    owner: Any
    camera: Any
    stack_for_block: Callable[[int], AngularStack]
    store_for_plane: Callable[[int, int], str]
    move_position: Callable[[Any], Generator]
    configure_channel: Callable[[Any], Generator]
    open_gate: Callable[[], Generator]
    close_gate: Callable[[], Generator]


class PvaTimeLapsePresenter:
    """Offer the service-owned plan only when explicit bindings are supplied.

    The managed entry acquires the MCU reservation. Neither entry acquires a
    camera lease or establishes the asserted camera configuration.
    """

    def __init__(self, name: str, *, bindings: PvaTimeLapseBindings | None) -> None:
        self.name = name
        self.bindings = bindings

    def plan_map(self) -> Mapping[str, PlanEntry]:
        """Return the plan entry only for a configured presenter."""
        return (
            {
                "pva_timelapse": {"plan": self.timelapse},
                "pva_managed_timelapse": {"plan": self.managed_timelapse},
                "pva_automatic_stack": {"plan": self.automatic_stack},
                "pva_gated_stack": {"plan": self.gated_stack},
            }
            if self.bindings
            else {}
        )

    def gated_stack(
        self,
        channel: Any,
        patterns: Sequence[int],
        *,
        first: int,
        step: int,
        planes: int,
        lower: int,
        upper: int,
        resources_exclusive: bool,
        external_trigger_configured: bool,
        illumination_ready: bool,
        bind_stage: bool = True,
        timeout_s: float = 60,
    ) -> Generator:
        """Run host-paced c1-gated shots into one continuously armed store."""
        binding = self.bindings
        if binding is None:
            raise RuntimeError("time-lapse bindings are not configured")
        if any(
            flag is not True
            for flag in (
                resources_exclusive,
                external_trigger_configured,
                illumination_ready,
            )
        ):
            raise ValueError("exclusive configured resources required")
        AngularStack(1, first, step, planes, lower, upper)
        patterns = tuple(patterns)
        if not 1 <= len(patterns) <= 64 or any(
            type(pattern) is not int or not 1 <= pattern <= 5 for pattern in patterns
        ):
            raise ValueError("invalid patterns")
        if type(bind_stage) is not bool or not 0 < timeout_s <= 300:
            raise ValueError("invalid stage binding or timeout")
        owner = binding.owner
        yield from bps.abs_set(owner.auto_pulses, True, wait=True)
        yield from _bounded_call(owner.reserve, timeout_s=5, what="reserve MCU")
        failed = False
        try:
            ids = yield from _bounded_call(
                lambda: owner.allocate_run_ids(1),
                timeout_s=5,
                what="allocate MCU run ID",
            )
            definition = AngularStack(ids[0], first, step, planes, lower, upper)
            store_uri = binding.store_for_plane(0, 0)
            yield from binding.configure_channel(channel)
            yield from binding.open_gate()

            @bpp.run_decorator(
                md={
                    "plan_name": "pva_gated_stack",
                    "store_uri": store_uri,
                    "patterns": patterns,
                    "planes": planes,
                    "stage_bound": bind_stage,
                    "association_quality": "count_matched_host_order_inferred",
                }
            )
            def run() -> Generator:
                result = yield from _bounded_call(
                    lambda: acquire_pva_gated_shots(
                        owner,
                        binding.camera,
                        definition,
                        patterns,
                        store_uri,
                        resources_exclusive=True,
                        external_trigger_configured=True,
                        automatic_pulses_enabled=True,
                        illumination_ready=True,
                        bind_stage=bind_stage,
                        timeout_s=timeout_s,
                    ),
                    timeout_s=timeout_s + 15,
                    what="gated stack",
                )
                record = _PointRecord()
                record.values = {
                    "attempted_pulses": result.attempted_pulses,
                    "stored_frames": result.evidence.stored_frames,
                    "camera_frame_count_delta": result.camera_count_delta,
                }
                yield from bps.create("gated_stack")
                yield from bps.read(record)
                yield from bps.save()
                for index, (plane, target, pattern) in enumerate(
                    result.requested_frames
                ):
                    record.values = {
                        "shot_index": index,
                        "plane_index": plane,
                        "target_tenths": target,
                        "pattern": pattern,
                    }
                    yield from bps.create("gated_frame")
                    yield from bps.read(record)
                    yield from bps.save()
                return result

            return (yield from run())
        except BaseException:
            failed = True
            raise
        finally:
            if owner.reservation_held:
                cleanup_error = None
                try:
                    yield from binding.close_gate()
                except Exception as error:
                    cleanup_error = error
                    logger.exception("Gated stack gate cleanup failed")
                try:
                    yield from _bounded_call(
                        lambda: owner.set_reserved_illumination("00"),
                        timeout_s=5,
                        what="blank LED",
                    )
                except Exception as error:
                    cleanup_error = cleanup_error or error
                    logger.exception("Gated stack LED blank failed")
                try:
                    yield from _bounded_call(
                        owner.release_reservation, timeout_s=5, what="release MCU"
                    )
                except Exception as error:
                    cleanup_error = cleanup_error or error
                    logger.exception("Gated stack reservation release failed")
                if cleanup_error is not None and not failed:
                    raise cleanup_error

    def automatic_stack(
        self,
        channel: Any,
        shots_per_plane: int,
        pattern_mask: int,
        interval_ms: int,
        *,
        first: int,
        step: int,
        planes: int,
        lower: int,
        upper: int,
        resources_exclusive: bool,
        external_trigger_configured: bool,
        continuous_capture_configured: bool,
        illumination_ready: bool,
        trigger_mask: int = 1,
        timeout_s: float = 60,
    ) -> Generator:
        """Acquire one MCU-triggered stack into a single continuous camera store."""
        binding = self.bindings
        if binding is None:
            raise RuntimeError("time-lapse bindings are not configured")
        if any(
            value is not True
            for value in (
                resources_exclusive,
                external_trigger_configured,
                continuous_capture_configured,
                illumination_ready,
            )
        ):
            raise ValueError("exclusive configured resources required")
        AngularStack(1, first, step, planes, lower, upper, trigger_mask)
        if type(shots_per_plane) is not int or not 1 <= shots_per_plane <= 64:
            raise ValueError("invalid shots per plane")
        if type(pattern_mask) is not int or not 1 <= pattern_mask <= 31:
            raise ValueError("invalid pattern mask")
        if type(interval_ms) is not int or not 1 <= interval_ms < 30000:
            raise ValueError("invalid interval")
        if not 0 < timeout_s <= 300:
            raise ValueError("invalid timeout")
        owner = binding.owner
        yield from _bounded_call(owner.reserve, timeout_s=5, what="reserve MCU")
        failed = False
        try:
            ids = yield from _bounded_call(
                lambda: owner.allocate_run_ids(1),
                timeout_s=5,
                what="allocate MCU run ID",
            )
            definition = AngularStack(
                ids[0], first, step, planes, lower, upper, trigger_mask
            )
            store_uri = binding.store_for_plane(0, 0)
            yield from binding.configure_channel(channel)
            yield from binding.open_gate()

            @bpp.run_decorator(
                md={
                    "plan_name": "pva_automatic_stack",
                    "store_uri": store_uri,
                    "first_tenths_degree": first,
                    "step_tenths_degree": step,
                    "planes": planes,
                    "shots_per_plane": shots_per_plane,
                    "pattern_mask": pattern_mask,
                    "interval_ms": interval_ms,
                    "association_quality": "count_matched_host_order_inferred",
                }
            )
            def run() -> Generator:
                result = yield from _bounded_call(
                    lambda: acquire_pva_automatic_stack(
                        owner,
                        binding.camera,
                        definition,
                        shots_per_plane,
                        pattern_mask,
                        interval_ms,
                        store_uri,
                        resources_exclusive=True,
                        external_trigger_configured=True,
                        continuous_capture_configured=True,
                        illumination_ready=True,
                        timeout_s=timeout_s,
                    ),
                    timeout_s=timeout_s + 15,
                    what="automatic stack",
                )
                record = _PointRecord()
                record.values = {
                    "attempted_pulses": result.attempted_pulses,
                    "stored_frames": result.evidence.stored_frames,
                    "camera_frame_count_delta": result.camera_count_delta,
                }
                yield from bps.create("automatic_stack")
                yield from bps.read(record)
                yield from bps.save()
                return result

            return (yield from run())
        except BaseException:
            failed = True
            raise
        finally:
            if owner.reservation_held:
                cleanup_error = None
                try:
                    yield from binding.close_gate()
                except Exception as error:
                    cleanup_error = error
                    logger.exception("Automatic stack gate cleanup failed")
                try:
                    yield from _bounded_call(
                        lambda: owner.set_reserved_illumination("00"),
                        timeout_s=5,
                        what="blank LED",
                    )
                except Exception as error:
                    cleanup_error = cleanup_error or error
                    logger.exception("Automatic stack LED blank failed")
                try:
                    yield from _bounded_call(
                        owner.release_reservation, timeout_s=5, what="release MCU"
                    )
                except Exception as error:
                    cleanup_error = cleanup_error or error
                    logger.exception("Automatic stack reservation release failed")
                if cleanup_error is not None and not failed:
                    raise cleanup_error
            elif not failed:
                raise RuntimeError("MCU reservation uncertain; reconciliation required")

    def managed_timelapse(
        self,
        positions: Sequence[Any],
        timepoints: int,
        channels: Sequence[Any],
        channel_patterns: Sequence[int],
        channel_exposures_ms: Sequence[float],
        frames_per_plane: int,
        *,
        first: int,
        step: int,
        planes: int,
        lower: int,
        upper: int,
        resources_exclusive: bool,
        internal_trigger_configured: bool,
        illumination_ready: bool,
        software_trigger_configured: bool = False,
        confirmation_id: str | None = None,
        **options: Any,
    ) -> Generator:
        """Reserve the MCU and allocate IDs for one bounded time-lapse run.

        Camera exclusivity and trigger configuration remain caller assertions.
        An uncertain reservation is intentionally left for reconciliation.
        """
        binding = self.bindings
        if binding is None:
            raise RuntimeError("time-lapse bindings are not configured")
        positions, channels = tuple(positions), tuple(channels)
        if (
            not positions
            or not channels
            or type(timepoints) is not int
            or timepoints < 1
        ):
            raise ValueError(
                "nonempty positions and channels and positive timepoints required"
            )
        count = len(positions) * len(channels) * timepoints
        if count > 10_000:
            raise ValueError("too many blocks")
        AngularStack(1, first, step, planes, lower, upper)
        if (
            resources_exclusive is not True
            or illumination_ready is not True
            or type(internal_trigger_configured) is not bool
            or type(software_trigger_configured) is not bool
            or internal_trigger_configured == software_trigger_configured
        ):
            raise ValueError("exclusive configured resources required")
        if confirmation_id is None:
            confirmation_id = new_confirmation_id("pva_managed_timelapse")
        owner = binding.owner
        yield from _bounded_call(owner.reserve, timeout_s=5, what="reserve MCU")
        failed = False
        try:
            ids = yield from _bounded_call(
                lambda: owner.allocate_run_ids(count),
                timeout_s=5,
                what="allocate MCU run IDs",
            )
            yield from pva_timelapse_plan(
                owner,
                binding.camera,
                positions,
                timepoints,
                channels,
                channel_patterns,
                channel_exposures_ms,
                frames_per_plane,
                stack_for_block=lambda index: AngularStack(
                    ids[index], first, step, planes, lower, upper
                ),
                store_for_plane=binding.store_for_plane,
                move_position=binding.move_position,
                configure_channel=binding.configure_channel,
                open_gate=binding.open_gate,
                close_gate=binding.close_gate,
                resources_exclusive=True,
                internal_trigger_configured=internal_trigger_configured,
                software_trigger_configured=software_trigger_configured,
                illumination_ready=True,
                confirmation_id=confirmation_id,
                **options,
            )
        except BaseException:
            failed = True
            raise
        finally:
            if owner.reservation_held:
                try:
                    yield from _bounded_call(
                        owner.release_reservation, timeout_s=5, what="release MCU"
                    )
                except Exception:
                    if not failed:
                        raise
                    logger.exception(
                        "MCU reservation release failed after acquisition error"
                    )
            elif not failed:
                raise RuntimeError("MCU reservation uncertain; reconciliation required")
            else:
                logger.warning("MCU reservation uncertain; reconciliation required")

    def timelapse(
        self,
        positions: Sequence[Any],
        timepoints: int,
        channels: Sequence[Any],
        channel_patterns: Sequence[int],
        channel_exposures_ms: Sequence[float],
        frames_per_plane: int,
        *,
        resources_exclusive: bool,
        internal_trigger_configured: bool,
        illumination_ready: bool,
        software_trigger_configured: bool = False,
        **options: Any,
    ) -> Generator:
        """Run with session bindings and caller-asserted device readiness."""
        binding = self.bindings
        if binding is None:
            raise RuntimeError("time-lapse bindings are not configured")
        yield from pva_timelapse_plan(
            binding.owner,
            binding.camera,
            positions,
            timepoints,
            channels,
            channel_patterns,
            channel_exposures_ms,
            frames_per_plane,
            stack_for_block=binding.stack_for_block,
            store_for_plane=binding.store_for_plane,
            move_position=binding.move_position,
            configure_channel=binding.configure_channel,
            open_gate=binding.open_gate,
            close_gate=binding.close_gate,
            resources_exclusive=resources_exclusive,
            internal_trigger_configured=internal_trigger_configured,
            software_trigger_configured=software_trigger_configured,
            illumination_ready=illumination_ready,
            **options,
        )
