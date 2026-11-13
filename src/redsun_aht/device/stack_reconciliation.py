"""Reconcile independently collected camera counters with MCU stack reports.

Order-inferred associations require a caller-established camera baseline and
exclusive, bounded capture. They do not prove electrical or exposure timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redsun_aht.device.unified_mcu import AngularStack, StackProgress


@dataclass(frozen=True, slots=True)
class PlaneFrame:
    """One order-inferred association, never a measured trigger-to-frame link."""

    plane: int
    target_tenths: int
    camera_sequence: int


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """Counts and associations; a rejected run has no inferred associations."""

    run_id: int
    attempts: tuple[int, ...]
    camera_sequences: tuple[int, ...]
    status: str
    associations: tuple[PlaneFrame, ...]
    quality: str = "unassociated"


def reconcile_stack(
    stack: AngularStack,
    progress: StackProgress,
    camera_sequences: tuple[int, ...],
    *,
    camera_baseline: int,
    capture_exclusive: bool,
) -> Reconciliation:
    """Associate only a complete matching report and contiguous camera stream.

    The baseline is the last camera sequence before capture was armed. The
    caller must establish that no other source can generate frames. Counter
    rollover, reset, loss and unknown baselines are deliberately not inferred.
    """
    if type(camera_baseline) is not int or camera_baseline < 0:
        raise ValueError("nonnegative camera baseline required")
    if type(capture_exclusive) is not bool:
        raise ValueError("capture exclusivity must be explicit")
    if not isinstance(camera_sequences, tuple) or any(
        type(seq) is not int or seq < 0 for seq in camera_sequences
    ):
        raise ValueError("invalid camera sequence numbers")
    expected = tuple(stack.first + stack.step * i for i in range(stack.count))
    if (
        type(progress.run_id) is not int
        or progress.run_id != stack.run_id
        or type(progress.complete) is not bool
        or progress.quality != "controller_reported"
        or any(type(target) is not int for target in progress.targets_tenths)
        or progress.targets_tenths != expected[: len(progress.targets_tenths)]
        or len(progress.targets_tenths) > stack.count
    ):
        raise ValueError("progress does not match requested stack")
    status = "complete_order_inferred"
    if not capture_exclusive:
        status = "capture_not_exclusive"
    elif not progress.complete or len(progress.targets_tenths) != stack.count:
        status = "stack_incomplete"
    elif camera_sequences != tuple(
        range(camera_baseline + 1, camera_baseline + 1 + stack.count)
    ):
        status = "camera_count_or_sequence_mismatch"
    associations: tuple[PlaneFrame, ...] = ()
    if status == "complete_order_inferred":
        associations = tuple(
            PlaneFrame(i, target, sequence)
            for i, (target, sequence) in enumerate(
                zip(progress.targets_tenths, camera_sequences, strict=True)
            )
        )
    return Reconciliation(
        stack.run_id,
        progress.targets_tenths,
        camera_sequences,
        status,
        associations,
        "order_inferred" if associations else "unassociated",
    )
