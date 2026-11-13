"""Camera/MCU evidence reconciliation without physical hardware."""

import pytest

from redsun_aht.device.stack_reconciliation import reconcile_stack
from redsun_aht.device.unified_mcu import AngularStack, StackProgress


def test_matching_streams_are_order_inferred_only():
    stack = AngularStack(7, 0, 18, 3, -180, 180)
    report = StackProgress(7, (0, 18, 36), complete=True)
    result = reconcile_stack(
        stack, report, (101, 102, 103), camera_baseline=100, capture_exclusive=True
    )
    assert result.status == "complete_order_inferred"
    assert result.quality == "order_inferred"
    assert [
        (p.plane, p.target_tenths, p.camera_sequence) for p in result.associations
    ] == [(0, 0, 101), (1, 18, 102), (2, 36, 103)]


def test_no_association_for_missing_extra_duplicate_or_partial_evidence():
    stack = AngularStack(7, 0, 18, 3, -180, 180)
    report = StackProgress(7, (0, 18, 36), complete=True)
    for frames in ((101, 102), (101, 102, 103, 104), (101, 101, 103), (101, 103, 104)):
        result = reconcile_stack(
            stack, report, frames, camera_baseline=100, capture_exclusive=True
        )
        assert result.status == "camera_count_or_sequence_mismatch"
        assert not result.associations
    partial = reconcile_stack(
        stack,
        StackProgress(7, (0, 18)),
        (101, 102),
        camera_baseline=100,
        capture_exclusive=True,
    )
    assert partial.status == "stack_incomplete" and not partial.associations
    unknown = reconcile_stack(
        stack, report, (101, 102, 103), camera_baseline=100, capture_exclusive=False
    )
    assert unknown.status == "capture_not_exclusive" and not unknown.associations
    with pytest.raises(ValueError, match="requested stack"):
        reconcile_stack(
            stack,
            StackProgress(8, (0,)),
            (),
            camera_baseline=100,
            capture_exclusive=True,
        )
