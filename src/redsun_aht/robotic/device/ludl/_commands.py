"""Typed scalar-PV command payloads for the native Ludl service."""

from __future__ import annotations

import json
import math
from typing import Any

TRAVERSE_COMMAND_SCHEMA = "redsun.ludl-stage-traverse/v1"
RELATIVE_MOVE_SCHEMA = "redsun.ludl-stage-relative-move/v1"


def encode_traverse_command(target_xy_um: tuple[float, float]) -> str:
    """Encode one finite absolute XY endpoint into a PVA-writable string."""
    if not isinstance(target_xy_um, (tuple, list)) or len(target_xy_um) != 2:
        raise ValueError("traverse target must be a finite XY pair")
    values: list[float] = []
    for value in target_xy_um:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("traverse target must be a finite XY pair")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("traverse target must be a finite XY pair")
        values.append(number)
    return json.dumps(
        {
            "schema": TRAVERSE_COMMAND_SCHEMA,
            "target_x_um": values[0],
            "target_y_um": values[1],
        },
        separators=(",", ":"),
    )


def decode_traverse_command(payload: str) -> tuple[float, float]:
    """Validate one versioned JSON command received over a scalar PVA field."""
    try:
        value: Any = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("traverse command must be valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "target_x_um",
        "target_y_um",
    }:
        raise ValueError("traverse command must contain schema and finite XY targets")
    if value["schema"] != TRAVERSE_COMMAND_SCHEMA:
        raise ValueError("traverse command schema is unsupported")
    coordinates: list[float] = []
    for key in ("target_x_um", "target_y_um"):
        coordinate = value[key]
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(
                "traverse command must contain schema and finite XY targets"
            )
        number = float(coordinate)
        if not math.isfinite(number):
            raise ValueError(
                "traverse command must contain schema and finite XY targets"
            )
        coordinates.append(number)
    return coordinates[0], coordinates[1]


def encode_relative_move_command(delta_xy_um: tuple[float, float]) -> str:
    """Encode one finite XY displacement in physical micrometres."""
    values = _validate_xy(delta_xy_um, "relative displacement")
    return json.dumps(
        {
            "schema": RELATIVE_MOVE_SCHEMA,
            "delta_x_um": values[0],
            "delta_y_um": values[1],
        },
        separators=(",", ":"),
    )


def decode_relative_move_command(payload: str) -> tuple[float, float]:
    """Validate one versioned relative XY command received over PVA."""
    try:
        value: Any = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("relative move command must be valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "delta_x_um",
        "delta_y_um",
    }:
        raise ValueError("relative move command must contain schema and XY deltas")
    if value["schema"] != RELATIVE_MOVE_SCHEMA:
        raise ValueError("relative move schema is unsupported")
    return _validate_xy(
        (value["delta_x_um"], value["delta_y_um"]), "relative displacement"
    )


def _validate_xy(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} must contain two finite coordinates")
    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(f"{label} must contain two finite coordinates")
        number = float(coordinate)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain two finite coordinates")
        coordinates.append(number)
    return coordinates[0], coordinates[1]


__all__ = [
    "RELATIVE_MOVE_SCHEMA",
    "TRAVERSE_COMMAND_SCHEMA",
    "decode_relative_move_command",
    "decode_traverse_command",
    "encode_relative_move_command",
    "encode_traverse_command",
]
