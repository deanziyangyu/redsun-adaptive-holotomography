"""Traceable host-side illumination maps for the active 91-LED DPCT panel."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

# These four ordered groups are transcribed from
# ``04_embedded_controller_src/led_triple_mode_ttl/led_triple_mode_ttl.ino``.
# The donor's repeated addresses are deliberately retained here for a
# reviewable source correspondence; a RAW_U8 map has one intensity per physical
# address, so :func:`panel91_dpct_quadrant_patterns` collapses repeats.
PANEL91_DPCT_DONOR_GROUPS: tuple[tuple[int, ...], ...] = (
    (
        0,
        3,
        4,
        10,
        11,
        12,
        13,
        24,
        25,
        26,
        27,
        28,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        0,
        1,
        2,
        7,
        8,
        9,
        10,
        19,
        20,
        21,
        22,
        23,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
    ),
    (
        0,
        1,
        2,
        7,
        8,
        9,
        10,
        19,
        20,
        21,
        22,
        23,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        0,
        1,
        6,
        7,
        16,
        17,
        18,
        19,
        33,
        34,
        35,
        36,
        37,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ),
    (
        0,
        1,
        6,
        7,
        16,
        17,
        18,
        19,
        33,
        34,
        35,
        36,
        37,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        0,
        4,
        5,
        13,
        14,
        15,
        16,
        28,
        29,
        30,
        31,
        32,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
    ),
    (
        0,
        4,
        5,
        13,
        14,
        15,
        16,
        28,
        29,
        30,
        31,
        32,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        0,
        3,
        4,
        10,
        11,
        12,
        13,
        24,
        25,
        26,
        27,
        28,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
    ),
)


@dataclass(frozen=True, slots=True)
class Panel91DpctPattern:
    """One native-v1 RAW_U8 pattern derived from a donor quadrant group."""

    pattern_id: int
    quadrant_index: int
    addresses: tuple[int, ...]
    values: bytes


def panel91_dpct_quadrant_patterns(
    *,
    pattern_ids: tuple[int, int, int, int] = (101, 102, 103, 104),
    intensity: int = 255,
) -> Mapping[int, Panel91DpctPattern]:
    """Build four 91-byte DPCT maps without inferring optical azimuths.

    Quadrant order is exactly the donor group order. Mapping that order to
    physical azimuths remains a calibration responsibility and is intentionally
    not asserted by this panel-address profile.
    """
    if len(set(pattern_ids)) != len(pattern_ids):
        raise ValueError("DPCT quadrant pattern IDs must be unique")
    if any(not 0 <= pattern_id <= 0xFFFF for pattern_id in pattern_ids):
        raise ValueError("DPCT quadrant pattern IDs must fit unsigned 16-bit integers")
    if not 1 <= intensity <= 255:
        raise ValueError("DPCT quadrant intensity must be in the range 1..255")
    patterns: dict[int, Panel91DpctPattern] = {}
    for quadrant_index, (pattern_id, donor_group) in enumerate(
        zip(pattern_ids, PANEL91_DPCT_DONOR_GROUPS, strict=True)
    ):
        addresses = tuple(dict.fromkeys(donor_group))
        if any(not 0 <= address < 91 for address in addresses):
            raise AssertionError("91-LED DPCT donor group contains an invalid address")
        values = bytearray(91)
        for address in addresses:
            values[address] = intensity
        patterns[pattern_id] = Panel91DpctPattern(
            pattern_id=pattern_id,
            quadrant_index=quadrant_index,
            addresses=addresses,
            values=bytes(values),
        )
    if set().union(*(set(pattern.addresses) for pattern in patterns.values())) != set(
        range(91)
    ):
        raise AssertionError("91-LED DPCT donor quadrants do not cover the panel")
    return MappingProxyType(patterns)


__all__ = [
    "PANEL91_DPCT_DONOR_GROUPS",
    "Panel91DpctPattern",
    "panel91_dpct_quadrant_patterns",
]
