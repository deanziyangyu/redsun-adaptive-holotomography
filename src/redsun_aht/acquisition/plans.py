"""RedSun-executed Bluesky plans for AHT acquisition."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp

from redsun_aht.streams import PRIMARY_STREAM

if TYPE_CHECKING:
    from collections.abc import Generator

    from bluesky.utils import Msg

    from redsun_aht.acquisition.dpct import DpctRecipe, DpctRunResult
    from redsun_aht.device.dpct import (
        DpctDetectorGroupDevice,
        DpctPatternDevice,
        DpctStageDevice,
    )


def dpct_plan(
    stage: DpctStageDevice,
    illumination: DpctPatternDevice,
    detectors: DpctDetectorGroupDevice,
    recipe: DpctRecipe,
) -> Generator[Msg, object, DpctRunResult]:
    """Acquire one pose/pattern product through RedSun-compatible devices."""

    def inner() -> Generator[Msg, object, DpctRunResult]:
        for scan_point in recipe.scan_points:
            yield from bps.abs_set(stage, scan_point, wait=True)
            pose = stage.pose
            if pose is None:  # pragma: no cover - completed set invariant
                raise RuntimeError("DPCT stage set completed without a pose")
            for pattern_id in recipe.pattern_ids:
                yield from bps.abs_set(illumination, pattern_id, wait=True)
                detectors.set_context(scan_point.index, pattern_id, pose)
                yield from bps.trigger(detectors, wait=True)
                yield from bps.create(name=PRIMARY_STREAM)
                yield from bps.read(stage)
                yield from bps.read(illumination)
                yield from bps.read(detectors)
                yield from bps.save()
        yield from bps.wait_for([detectors.finalize])
        result = detectors.result
        if result is None:  # pragma: no cover - finalize invariant
            raise RuntimeError("DPCT detector group did not finalize a result")
        return result

    staged = bpp.stage_wrapper(  # type: ignore[no-untyped-call]
        inner(), [stage, illumination, detectors]
    )
    yield from bps.open_run(
        md={
            "purpose": "dpct",
            "aht_run_id": recipe.run_id,
            "aht_pattern_ids": list(recipe.pattern_ids),
            "aht_scan_count": len(recipe.scan_points),
        },
    )
    try:
        result = yield from staged
    except BaseException as error:
        yield from bps.close_run(exit_status="fail", reason=str(error))
        raise
    else:
        yield from bps.close_run()
        return cast("DpctRunResult", result)


__all__ = ["dpct_plan"]
