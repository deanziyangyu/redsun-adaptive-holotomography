"""Compose a completed reviewed revisit with an independent time-lapse run.

The design preserves the IX83 reviewed-revisit/time-lapse chain's use of the
actual returned target order. The caller owns the revisit, stage, illumination,
time-lapse storage and device lifecycle; this module opens no devices itself.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bluesky.utils import RunEngineControlException

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RevisitedTarget:
    """A target actually returned by the revisit, in reported XY micrometres."""

    target_id: str
    x_um: float
    y_um: float

    def __post_init__(self) -> None:
        """Require a named target with finite coordinates."""
        if (
            not self.target_id.strip()
            or not math.isfinite(self.x_um)
            or not math.isfinite(self.y_um)
        ):
            raise ValueError("target requires an ID and finite XY")


def reviewed_revisit_timelapse_chain(
    *,
    read_anchor: Callable[[], Generator],
    revisit: Callable[[], Generator],
    run_timelapse: Callable[[Sequence[RevisitedTarget]], Generator],
    return_to_anchor: Callable[[tuple[float, float]], Generator],
) -> Generator:
    """Use the returned revisit order for a separate time-lapse run.

    An unreadable anchor leaves no return target. If the time-lapse succeeds,
    a failed anchor return is logged without invalidating its acquired run,
    matching the donor chain. No return is inferred after failed acquisition.
    """
    anchor = None
    try:
        value = yield from read_anchor()
        if value is not None:
            x, y = value
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("non-finite anchor")
            anchor = (float(x), float(y))
    except RunEngineControlException:
        raise
    except Exception:
        logger.warning(
            "Could not read XY anchor; no return will be attempted", exc_info=True
        )

    targets = tuple((yield from revisit()))
    if not targets or any(
        not isinstance(target, RevisitedTarget) for target in targets
    ):
        raise ValueError("revisit must return at least one valid target")
    yield from run_timelapse(targets)
    if anchor is not None:
        try:
            yield from return_to_anchor(anchor)
        except RunEngineControlException:
            raise
        except Exception:
            logger.warning(
                "Time-lapse completed but XY anchor return failed", exc_info=True
            )
