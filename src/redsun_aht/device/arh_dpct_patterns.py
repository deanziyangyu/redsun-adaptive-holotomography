"""Native ARH Due masks for the dynamic DPCT demonstration."""

# These masks follow grp1–grp4 in the Due firmware, including center index 60.
GROUPS = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 40, 41, 42, 43, 44, 45, 46, 52, 53, 54, 55, 56, 60),
    (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 28, 29, 30, 31, 32, 33, 34,
     35, 36, 43, 44, 45, 46, 47, 48, 49, 54, 55, 56, 57, 58, 60),
    (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 32, 33, 34, 35, 36, 37, 38,
     39, 24, 46, 47, 48, 49, 50, 51, 40, 56, 57, 58, 59, 52, 60),
    (0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 36, 37,
     38, 39, 40, 41, 42, 43, 49, 50, 51, 52, 53, 54, 58, 59, 60),
)
RINGS = (range(60, 61), range(54, 60), range(42, 54), range(24, 42), range(24))
LEVELS = (32, 48, 80, 130, 205)


def demo_phases() -> tuple[tuple[bytes, ...], ...]:
    """Return the semicircle pair, graded groups 1–4 and solid groups 1–4."""
    radial = {index: level for ring, level in zip(RINGS, LEVELS, strict=True) for index in ring}
    graded = tuple(bytes(radial[i] if i in group else 0 for i in range(61)) for group in GROUPS)
    solid = tuple(bytes(LEVELS[-1] if i in group else 0 for i in range(61)) for group in GROUPS)
    return (graded[0], solid[0]), graded, solid
