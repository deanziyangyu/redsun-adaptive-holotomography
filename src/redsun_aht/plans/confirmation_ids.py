"""Human-readable plan confirmation identifiers, compatible with the IX83 format."""

from __future__ import annotations

import re
from datetime import datetime


def new_confirmation_id(plan_name: str, *, now: datetime | None = None) -> str:
    """Format local time and plan name; this label is not a unique run UID."""
    if not isinstance(plan_name, str):
        raise ValueError("plan name must be a string")
    token = re.sub(r"[^A-Za-z0-9]+", "_", plan_name).strip("_").upper() or "PLAN"
    return f"{(now or datetime.now()).strftime('%Y%m%dT%H%M')}_{token}"
