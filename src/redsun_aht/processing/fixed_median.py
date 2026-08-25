"""Explicit fixed-background median construction for finite DPCT stacks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class FixedMedian:
    """One immutable SYX background formed over all acquired Z positions."""

    values_syx: npt.NDArray[np.float32]
    source_checksum: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        """Validate the four-shot result and detach mutable metadata."""
        values = np.asarray(self.values_syx, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 4:
            raise ValueError("fixed median requires a four-shot SYX array")
        if not np.all(np.isfinite(values)):
            raise ValueError("fixed median contains non-finite values")
        if len(self.source_checksum) != 64:
            raise ValueError("fixed median source checksum must be SHA-256")
        object.__setattr__(self, "values_syx", values.copy())
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def fixed_median_from_czyx(
    acquired_czyx: npt.ArrayLike,
    *,
    source_checksum: str,
) -> FixedMedian:
    """Median every one of four fixed illumination shots over acquired Z."""
    acquired = np.asarray(acquired_czyx)
    if acquired.ndim != 4 or acquired.shape[0] != 4 or acquired.shape[1] <= 0:
        raise ValueError("fixed median requires acquired CZYX data with four shots")
    values = np.median(acquired, axis=1).astype(np.float32)
    checksum = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    return FixedMedian(
        values,
        source_checksum,
        {
            "kind": "fixed-median-over-acquired-z",
            "input_axes": ("c", "z", "y", "x"),
            "output_axes": ("s", "y", "x"),
            "shot_count": int(acquired.shape[0]),
            "z_count": int(acquired.shape[1]),
            "median_checksum": checksum,
        },
    )


def fixed_median_from_czyx_store(
    acquired_czyx: Any,
    *,
    source_checksum: str,
) -> FixedMedian:
    """Compute the same median one shot at a time from a CZYX store.

    This avoids materializing a whole physical 4xZ stack solely to make the
    fixed background, while retaining the exact per-shot median.
    """
    shape = tuple(int(value) for value in acquired_czyx.shape)
    if len(shape) != 4 or shape[0] != 4 or shape[1] <= 0:
        raise ValueError("fixed median requires acquired CZYX data with four shots")
    values = np.empty((shape[0], shape[2], shape[3]), dtype=np.float32)
    for shot_index in range(shape[0]):
        values[shot_index] = np.median(
            np.asarray(acquired_czyx[shot_index]), axis=0
        ).astype(np.float32)
    checksum = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    return FixedMedian(
        values,
        source_checksum,
        {
            "kind": "fixed-median-over-acquired-z",
            "input_axes": ("c", "z", "y", "x"),
            "output_axes": ("s", "y", "x"),
            "shot_count": shape[0],
            "z_count": shape[1],
            "median_checksum": checksum,
            "input_read_mode": "per-shot-store",
        },
    )


__all__ = [
    "FixedMedian",
    "fixed_median_from_czyx",
    "fixed_median_from_czyx_store",
]
