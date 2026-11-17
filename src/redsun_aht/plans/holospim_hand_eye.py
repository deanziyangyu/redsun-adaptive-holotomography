"""Capture a reviewed HoloSPIM calibration path and record joint observations."""

from __future__ import annotations

import hashlib
import json
import math
import time
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
from redsun_aht.plans.hand_eye import HAND_EYE_AXES, calibration_waypoints

if TYPE_CHECKING:
    from bluesky.utils import MsgGenerator

    from redsun_aht.device.esp300_stage import Esp300Stage
    from redsun_aht.device.mmcore.camera import MMCamera
    from redsun_aht.device.pva_rotation import PvaRotationMotor


def _read_record(filename: str) -> tuple[dict, bytes]:
    with Path(filename).open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError("hand-eye record exceeds 16 KiB")
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise ValueError("hand-eye record must be a JSON object")
    return record, raw


def capture_reviewed_path(
    camera: MMCamera,
    survey: Esp300Stage,
    rotation: PvaRotationMotor,
    preview_path: str,
    review_path: str,
    roll_limits: tuple[float, float] | None,
) -> MsgGenerator[None]:
    """Execute the signed path after verifying its review and current anchor."""
    path, path_bytes = _read_record(preview_path)
    review, review_bytes = _read_record(review_path)
    path_hash = hashlib.sha256(path_bytes).hexdigest()
    axes = HAND_EYE_AXES["holospim"]
    parked = path.get("parked_x28_mm")
    if (
        path.get("schema") != "redsun.hand_eye.path.v1"
        or path.get("instrument") != "holospim"
        or path.get("detector") != camera.name
        or path.get("waypoint_axes") != list(axes)
        or path.get("waypoint_units") != ["um", "um", "um", "deg"]
        or review.get("schema") != "redsun.hand_eye.sampled_collision.v1"
        or review.get("instrument") != "holospim"
        or review.get("path_sha256") != path_hash
        or type(parked) not in (float, int)
        or not math.isfinite(parked)
        or review.get("focus_z_mm") != parked
        or review.get("checked_paths") != ["nominal"]
        or type(review.get("sample_count")) is not int
        or review["sample_count"] < 1
        or "first_collision_sample" not in review
        or review["first_collision_sample"] is not None
        or not review.get("mesh_sha256")
        or not review.get("profile_sha256")
    ):
        raise ValueError(
            "a clear simulator review of the exact HoloSPIM preview is required"
        )
    rows = path["relative_waypoints"]
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 17
        or any(
            not isinstance(row, list)
            or len(row) != 4
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in row
            )
            for row in rows
        )
    ):
        raise ValueError("invalid XYZ/roll waypoints")
    steps = {axis: max(abs(row[i]) for row in rows) for i, axis in enumerate(axes)}
    selected = {axis: step for axis, step in steps.items() if step}
    if (
        [list(row) for row in calibration_waypoints("holospim", selected)] != rows
        or path.get("calibration_axes") != list(selected)
        or any(steps[axis] > 1000 for axis in "xyz")
    ):
        raise ValueError("expected the bounded signed calibration path")
    captures = [0, *range(1, len(rows), 2)]
    if (
        path.get("capture_waypoint_indices") != captures
        or review.get("capture_waypoint_indices") != captures
    ):
        raise ValueError("reviewed capture waypoint indices do not match the path")
    origin = (*path["origin_um"], path["origin_roll_deg"])
    if len(origin) != 4 or not all(math.isfinite(value) for value in origin):
        raise ValueError("invalid calibration origin")
    axis = rotation.rotation
    targets = [
        tuple(zero + offset for zero, offset in zip(origin, row, strict=True))
        for row in rows
    ]
    roll_resolution = abs(axis.to_degrees(1))
    if steps["roll"] and (
        roll_limits is None
        or path["roll_limits_deg"] != list(roll_limits)
        or any(
            not roll_limits[0] <= target[3] <= roll_limits[1]
            or abs(axis.to_degrees(axis.to_steps(target[3])) - target[3]) > 1e-8
            for target in targets
        )
    ):
        raise ValueError(
            "reviewed roll path differs from the configured bounds or step grid"
        )

    async def position():
        reading = await survey.read_position()
        by_axis = {row.name: row.value for row in reading.axes}
        values = (*(by_axis[name] for name in "xyz"), await axis.read_position())
        roll_quality = "simulated" if axis.simulated else "commanded"
        if (
            reading.quality != path["survey_anchor_quality"]
            or roll_quality != path["roll_anchor_quality"]
            or not all(math.isfinite(value) for value in values)
        ):
            raise RuntimeError("live readback quality differs from the preview")
        return values

    watched = (camera, survey, rotation)
    info = camera.path_provider(camera.name)
    store = Path(info.directory_path) / f"{info.filename}.zarr"
    capture = PvaCaptureEvidence(camera)

    @bpp.run_decorator(
        md={
            "plan_name": "holospim_hand_eye_capture",
            "store_uri": str(store),
            "preview_path_sha256": path_hash,
            "simulator_review_sha256": hashlib.sha256(review_bytes).hexdigest(),
            "instrument": "holospim",
            "association": "host_order_inferred",
        }
    )
    def run() -> MsgGenerator[None]:
        yield from bps.clear_checkpoint()
        yield from _bounded_call(
            lambda: require_idle_camera(camera, internal_trigger=True),
            timeout_s=5,
            what="verify camera ready for internal snaps",
        )
        limits = {name: [spec.lower, spec.upper] for name, spec in survey.axes.items()}
        if limits != path["travel_limits_um"] or set(limits) != set("xyz"):
            raise ValueError("live survey limits differ from the preview")
        for target in targets:
            for name, value in zip("xyz", target[:3], strict=True):
                if not limits[name][0] <= value <= limits[name][1]:
                    raise ValueError(
                        f"reviewed position exceeds live {name.upper()} limits"
                    )
        tolerance = (
            *(2 * survey.axes[name].precision for name in "xyz"),
            roll_resolution / 2,
        )

        def read_position():
            return (
                yield from _bounded_call(
                    position, timeout_s=5, what="read calibration pose"
                )
            )

        def check(actual, expected):
            if any(
                abs(a - b) > limit
                for a, b, limit in zip(actual, expected, tolerance, strict=True)
            ):
                raise RuntimeError(
                    f"calibration pose differs from reviewed path: {actual}"
                )

        check((yield from read_position()), origin)
        annotations = []
        complete = False
        previous = targets[0]
        try:
            yield from _bounded_call(
                lambda: capture.prepare(
                    str(store),
                    len(captures),
                    internal_trigger_configured=True,
                    software_snap=True,
                ),
                timeout_s=10,
                what="prepare calibration camera",
            )
            for index, target in enumerate(targets):
                for joint, before, after in zip(axes, previous, target, strict=True):
                    if before == after:
                        continue
                    device = axis if joint == "roll" else survey.axis[joint]
                    yield from bps.abs_set(device, after, wait=True)
                previous = target
                before = yield from read_position()
                check(before, target)
                if index in captures:
                    started = time.monotonic_ns()
                    camera_count = yield from _bounded_call(
                        capture.snap, timeout_s=10, what="snap calibration frame"
                    )
                    finished = time.monotonic_ns()
                    after = yield from read_position()
                    check(after, target)
                    check(after, before)
                    annotations.append(
                        {
                            "stored_index": len(annotations),
                            "waypoint_index": index,
                            "target": target,
                            "before": before,
                            "after": after,
                            "camera_count": camera_count,
                            "host_snap_start_ns": started,
                            "host_snap_end_ns": finished,
                        }
                    )
            evidence = yield from _bounded_call(
                capture.collect, timeout_s=15, what="collect calibration frames"
            )
            if evidence.stored_frames != len(captures):
                raise RuntimeError("calibration frame store is incomplete")
            complete = True
        finally:
            try:
                yield from _bounded_call(
                    capture.close, timeout_s=10, what="close calibration capture"
                )
            except BaseException:
                complete = False
                raise
            finally:
                if store.is_dir():
                    (store / "hand_eye_preview.json").write_bytes(path_bytes)
                    (store / "simulator_review.json").write_bytes(review_bytes)
                    output = (
                        "hand_eye_path.json" if complete else "hand_eye_partial.json"
                    )
                    with (store / output).open("x", encoding="utf-8") as stream:
                        json.dump(
                            {**path, "observations": annotations}, stream, indent=2
                        )
                        stream.write("\n")

    yield from lock_wrapper(run(), *watched)
