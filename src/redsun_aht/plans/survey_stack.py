"""Finite camera snaps at verified survey Z positions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from redsun.engine.plan_stubs import lock_wrapper

from redsun_aht.device.pva_capture_evidence import (
    PvaCaptureEvidence,
    require_idle_camera,
)
from redsun_aht.plans.dpct import _bounded_call
from redsun_aht.plans.timelapse import _PointRecord

if TYPE_CHECKING:
    from bluesky.utils import MsgGenerator

    from redsun_aht.device.esp300_stage import Esp300Stage
    from redsun_aht.device.mmcore.camera import MMCamera


def z_targets(origin: float, step_um: float, planes: int) -> tuple[float, ...]:
    """Validate a relative, monotone stack before issuing any move."""
    if type(planes) is not int or not 1 <= planes <= 32:
        raise ValueError("planes must be 1..32")
    if not math.isfinite(origin) or not math.isfinite(step_um):
        raise ValueError("origin and step must be finite")
    if planes > 1 and (step_um == 0 or abs(step_um) * (planes - 1) > 1000):
        raise ValueError("stack must have a nonzero step and span at most 1000 um")
    targets = tuple(origin + index * step_um for index in range(planes))
    if not all(math.isfinite(target) for target in targets):
        raise ValueError("stack target is not finite")
    if len(set(targets)) != planes:
        raise ValueError("stack step is too small to produce distinct Z targets")
    return targets


def survey_z_stack(
    camera: MMCamera,
    survey: Esp300Stage,
    step_um: float,
    planes: int,
    frames_per_plane: int,
) -> MsgGenerator[None]:
    """Capture one store and return to the origin after a verified stack."""
    z_targets(0.0, step_um, planes)
    if type(frames_per_plane) is not int or not 1 <= frames_per_plane <= 32:
        raise ValueError("frames per plane must be 1..32")
    info = camera.path_provider(camera.name)
    uri = str(Path(info.directory_path) / f"{info.filename}.zarr")
    capture = PvaCaptureEvidence(camera)

    @bpp.run_decorator(
        md={
            "plan_name": "holospim_survey_z_stack",
            "camera": camera.name,
            "store_uri": uri,
            "step_um": step_um,
            "planes": planes,
            "frames_per_plane": frames_per_plane,
            "association": "host_order_inferred",
        }
    )
    def run() -> MsgGenerator[None]:
        yield from bps.clear_checkpoint()
        yield from _bounded_call(
            lambda: require_idle_camera(camera, internal_trigger=True),
            timeout_s=5,
            what="read camera trigger mode",
        )
        if set(survey.axes) != set("xyz"):
            raise RuntimeError("survey XYZ bounds are required")

        def read_position():
            reading = yield from _bounded_call(
                survey.read_position, timeout_s=5, what="read survey position"
            )
            values = {axis.name: axis.value for axis in reading.axes}
            if set(values) != set("xyz") or not all(
                map(math.isfinite, values.values())
            ):
                raise RuntimeError("survey XYZ readback is unavailable")
            return values

        initial = yield from read_position()
        targets = z_targets(initial["z"], step_um, planes)
        z = survey.axes["z"]
        if any(not z.lower <= target <= z.upper for target in targets):
            raise ValueError("survey stack exceeds Z travel bounds")

        def check(values, target):
            expected = {**initial, "z": target}
            if any(
                abs(values[k] - expected[k]) > 2 * survey.axes[k].precision
                for k in "xyz"
            ):
                raise RuntimeError("survey position differs from the stack waypoint")

        try:
            yield from _bounded_call(
                lambda: capture.prepare(
                    uri,
                    planes * frames_per_plane,
                    internal_trigger_configured=True,
                    software_snap=True,
                ),
                timeout_s=10,
                what="prepare survey stack store",
            )
            point = _PointRecord()
            for plane, target in enumerate(targets):
                if plane:
                    yield from bps.abs_set(survey.axis["z"], target, wait=True)
                check((yield from read_position()), target)
                for frame in range(frames_per_plane):
                    count = yield from _bounded_call(
                        capture.snap, timeout_s=10, what="snap survey stack frame"
                    )
                    after = yield from read_position()
                    check(after, target)
                    point.values = {
                        "plane": plane,
                        "frame": frame,
                        "camera_count": count,
                        "target_z_um": target,
                        "readback_z_um": after["z"],
                    }
                    yield from bps.create("survey_z_frame")
                    yield from bps.read(point)
                    yield from bps.save()
            evidence = yield from _bounded_call(
                capture.collect,
                timeout_s=15,
                what="collect survey stack",
            )
            if evidence.stored_frames != planes * frames_per_plane:
                raise RuntimeError("survey stack store is incomplete")
            if planes > 1:
                yield from bps.abs_set(survey.axis["z"], initial["z"], wait=True)
                check((yield from read_position()), initial["z"])
        finally:
            yield from _bounded_call(
                capture.close, timeout_s=10, what="close survey stack"
            )

    yield from lock_wrapper(run(), camera, survey)
