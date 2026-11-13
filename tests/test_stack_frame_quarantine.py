"""Bounded unreconciled-frame quarantine tests."""

import numpy as np
import pytest

from redsun_aht.device.stack_frame_quarantine import FrameQuarantine
from redsun_aht.device.unified_mcu import AngularStack, StackProgress


def test_overflow_rejects_association_and_preserves_owned_pixels():
    buffer = FrameQuarantine(max_frames=1, max_bytes=128)
    pixels = np.ones((2, 2), dtype=np.uint8)
    buffer.append(1, pixels)
    pixels[:] = 9
    buffer.append(2, pixels)
    result = buffer.finish(
        AngularStack(7, 0, 1, 2, -10, 10),
        StackProgress(7, (0, 1), complete=True),
        camera_baseline=0,
        capture_exclusive=True,
    )
    assert result.status == "buffer_overflow" and not result.associations
    assert len(result.frames) == 1 and int(result.frames[0].pixels[0, 0]) == 1
    assert not result.frames[0].pixels.flags.writeable
    with pytest.raises(RuntimeError, match="finalized"):
        buffer.append(3, pixels)


def test_byte_limit_latches_even_with_correct_count():
    buffer = FrameQuarantine(max_frames=3, max_bytes=3)
    buffer.append(1, np.zeros((2, 2), dtype=np.uint8))
    result = buffer.finish(
        AngularStack(7, 0, 1, 1, -10, 10),
        StackProgress(7, (0,), complete=True),
        camera_baseline=0,
        capture_exclusive=True,
    )
    assert result.status == "buffer_overflow" and not result.frames
