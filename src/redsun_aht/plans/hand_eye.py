"""Axis definitions and single-axis sampling for 3D hand-eye acquisition."""

import math
from collections.abc import Iterator, Mapping


HAND_EYE_AXES = {
    "arh": ("x", "y", "z", "roll", "pitch", "yaw"),
    "holospim": ("x", "y", "z", "roll"),
}


def signed_offsets(
    instrument: str, steps: Mapping[str, float],
) -> Iterator[tuple[str | None, float]]:
    """Yield the anchor and signed offsets in instrument order.

    Linear offsets use micrometres and rotary offsets use degrees. A caller
    resolves each offset through its commissioned hardware axis mapping.
    """
    axes = HAND_EYE_AXES[instrument]
    if not steps or steps.keys() - set(axes):
        raise ValueError(f"Select supported {instrument} calibration axes: {', '.join(axes)}")
    for step in steps.values():
        if not math.isfinite(step) or step <= 0:
            raise ValueError("Calibration steps must be positive and finite")
    yield None, 0.0
    for axis in axes:
        if axis in steps:
            yield axis, -steps[axis]
            yield axis, steps[axis]


def calibration_waypoints(
    instrument: str, steps: Mapping[str, float],
) -> tuple[tuple[float, ...], ...]:
    """Return relative poses in instrument axis order, with an anchor after each move."""
    axes = HAND_EYE_AXES[instrument]
    anchor = (0.0,) * len(axes)
    path = [anchor]
    for axis, offset in signed_offsets(instrument, steps):
        if axis is not None:
            path.append(tuple(offset if key == axis else 0.0 for key in axes))
            path.append(anchor)
    return tuple(path)
