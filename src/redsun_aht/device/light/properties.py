"""Generic Micro-Manager property introspection and typed value handling.

Shared classification rules for discovered device properties so every AHT
Micro-Manager adapter exposes the same enum/bool/numeric/string/read-only
semantics without bespoke per-property classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

type MMPropertyValue = str | int | float | bool

_BOOLEAN_ALLOWED_VALUES = ("0", "1")


class MMPropertyKind(StrEnum):
    """Typed classification of one discovered Micro-Manager property."""

    ENUMERABLE = "enumerable"
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class MMProperty:
    """One classified Micro-Manager device property snapshot."""

    name: str
    value: str
    read_only: bool
    kind: MMPropertyKind
    allowed_values: tuple[str, ...]
    limits: tuple[float, float] | None

    def typed_value(self) -> str | float | bool:
        """Return the readback coerced to the classified Python type."""
        if self.kind is MMPropertyKind.BOOLEAN:
            return self.value == "1"
        if self.kind is MMPropertyKind.NUMERIC:
            return float(self.value)
        return self.value


def describe_mm_property(
    name: str,
    *,
    value: str,
    read_only: bool,
    allowed_values: tuple[str, ...],
    limits: tuple[float, float] | None,
) -> MMProperty:
    """Classify one property snapshot with deterministic precedence."""
    normalized_allowed = tuple(allowed_values)
    if normalized_allowed == _BOOLEAN_ALLOWED_VALUES:
        kind = MMPropertyKind.BOOLEAN
    elif normalized_allowed:
        kind = MMPropertyKind.ENUMERABLE
    elif limits is not None or _parses_as_float(value):
        kind = MMPropertyKind.NUMERIC
    else:
        kind = MMPropertyKind.TEXT
    return MMProperty(
        name=name,
        value=value,
        read_only=read_only,
        kind=kind,
        allowed_values=normalized_allowed,
        limits=limits,
    )


def format_mm_property_value(prop: MMProperty, value: MMPropertyValue) -> str:
    """Validate one typed write against the property and render its wire form.

    Raises ``ValueError`` for writes that violate the property's classified
    type, allowed values, or inclusive numeric limits.
    """
    if prop.kind is MMPropertyKind.BOOLEAN:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int) and value in (0, 1):
            return str(value)
        if isinstance(value, str) and value in _BOOLEAN_ALLOWED_VALUES:
            return value
        raise ValueError(
            f"property {prop.name!r} accepts boolean or '0'/'1' values, got {value!r}"
        )
    if prop.kind is MMPropertyKind.ENUMERABLE:
        rendered = str(value)
        if rendered not in prop.allowed_values:
            raise ValueError(
                f"property {prop.name!r} accepts {list(prop.allowed_values)}, "
                f"got {rendered!r}"
            )
        return rendered
    if prop.kind is MMPropertyKind.NUMERIC:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"property {prop.name!r} requires a numeric value, got {value!r}"
            ) from None
        if limits := prop.limits:
            lower, upper = limits
            if not lower <= numeric <= upper:
                raise ValueError(
                    f"property {prop.name!r} requires a value within "
                    f"[{lower}, {upper}], got {numeric!r}"
                )
        return repr(numeric)
    return str(value)


def _parses_as_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


__all__ = [
    "MMProperty",
    "MMPropertyKind",
    "MMPropertyValue",
    "describe_mm_property",
    "format_mm_property_value",
]
