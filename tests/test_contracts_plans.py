"""Plan contracts: selection, grids, and frame-exact partial reporting."""

from __future__ import annotations

import json
from typing import Any

import pytest

from redsun_aht.contracts.devices import (
    CENTER_PATTERN_ID,
    DPCT_PATTERN_IDS,
    AxisSpec,
    IlluminationCapability,
)
from redsun_aht.contracts.frames import FrameIdentity, RunIdentity
from redsun_aht.contracts.plans import (
    AxisGrid,
    DeviceRole,
    PartialRun,
    PlanAbortReason,
    PlanOutcome,
    PlanRequest,
    PlanStatus,
    ScanPoint,
)

RUN = RunIdentity(
    run_id="run-20261103-a",
    plan_name="dpct",
    purpose="four-pattern",
    scan_id=7,
)
Z = AxisSpec("z", "um", -50.0, 50.0, 0.1)
Y = AxisSpec("y", "um", 0.0, 100.0, 0.5)
X = AxisSpec("x", "um", 0.0, 100.0, 0.5)


def _devices(
    **roles: tuple[str, ...],
) -> dict[DeviceRole, tuple[str, ...]]:
    values: dict[DeviceRole, tuple[str, ...]] = {
        DeviceRole.CAMERA: ("dhm",),
        DeviceRole.STAGE: ("stage",),
        DeviceRole.ILLUMINATION: ("led",),
    }
    for name, names in roles.items():
        values[DeviceRole(name)] = names
    return values


def _request(**overrides: Any) -> PlanRequest:
    values: dict[str, Any] = {
        "run": RUN,
        "devices": _devices(),
        "detector_ids": ("dhm",),
        "grid": (),
        "pattern_ids": (1, 2, 3, 4),
        "snake": False,
    }
    values.update(overrides)
    return PlanRequest(**values)


def _frame(
    *,
    sequence: int,
    pattern_id: int = 1,
    plane_ordinal: int = 0,
    scan_index: int = 0,
    producer_sequence: int | None = None,
) -> FrameIdentity:
    return FrameIdentity(
        run=RUN,
        detector_id="dhm",
        sequence=sequence,
        plane_ordinal=plane_ordinal,
        pattern_id=pattern_id,
        scan_index=scan_index,
        producer_sequence=producer_sequence,
    )


def _dpct_frames() -> tuple[FrameIdentity, ...]:
    return tuple(
        _frame(sequence=index, pattern_id=index + 1, plane_ordinal=index)
        for index in range(4)
    )


def test_device_roles_are_the_three_selected_roles() -> None:
    assert {member.value for member in DeviceRole} == {
        "camera",
        "stage",
        "illumination",
    }


def test_plan_request_requires_a_camera_and_its_detectors() -> None:
    with pytest.raises(ValueError, match="at least one camera"):
        _request(devices={DeviceRole.STAGE: ("stage",)})
    with pytest.raises(ValueError, match="not a selected camera"):
        _request(detector_ids=("other",))
    with pytest.raises(ValueError, match="detector identity"):
        _request(detector_ids=())
    with pytest.raises(ValueError, match="unique"):
        _request(detector_ids=("dhm", "dhm"))


def test_plan_request_requires_the_role_a_requested_capability_needs() -> None:
    with pytest.raises(ValueError, match="must select a stage"):
        _request(
            grid=(AxisGrid(Z, (0.0,)),),
            devices={DeviceRole.CAMERA: ("dhm",)},
        )
    with pytest.raises(ValueError, match="needs an illumination device"):
        _request(
            devices={
                DeviceRole.CAMERA: ("dhm",),
                DeviceRole.STAGE: ("stage",),
            }
        )


def test_plan_request_refuses_a_pattern_outside_the_dpct_frame_set() -> None:
    with pytest.raises(ValueError, match="outside"):
        _request(pattern_ids=(1, 5))
    with pytest.raises(ValueError, match="outside"):
        _request(pattern_ids=(0,))
    with pytest.raises(ValueError, match="unique"):
        _request(pattern_ids=(1, 1))


def test_the_dpct_plan_does_not_request_the_centre_pattern() -> None:
    # the centre is a real illumination pattern on the 91-LED panel, but it is
    # not part of the four-frame DPCT set
    with pytest.raises(ValueError, match="outside"):
        _request(pattern_ids=(1, 2, 3, 4, CENTER_PATTERN_ID))
    assert CENTER_PATTERN_ID in IlluminationCapability().pattern_ids
    assert CENTER_PATTERN_ID not in DPCT_PATTERN_IDS


def test_axis_grid_refuses_an_untraversable_line() -> None:
    with pytest.raises(ValueError, match="at least one position"):
        AxisGrid(Z, ())
    with pytest.raises(ValueError, match="distinct"):
        AxisGrid(Z, (0.0, 0.0))
    with pytest.raises(ValueError, match="one direction"):
        AxisGrid(Z, (0.0, 1.0, -1.0))
    with pytest.raises(ValueError, match="outside"):
        AxisGrid(Z, (0.0, 60.0))
    assert AxisGrid(Z, (2.0, 1.0, 0.0)).direction == -1
    assert AxisGrid(Z, (0.0, 1.0, 2.0)).direction == 1


def test_plan_request_grid_axes_are_unique() -> None:
    with pytest.raises(ValueError, match="grid axes must be unique"):
        _request(
            grid=(
                AxisGrid(Z, (0.0,)),
                AxisGrid(AxisSpec("z", "um", 0, 10, 0.1), (1.0,)),
            )
        )


def test_scan_points_visit_the_outer_axis_slowest() -> None:
    request = _request(grid=(AxisGrid(Y, (0.0, 10.0)), AxisGrid(X, (0.0, 5.0, 10.0))))
    points = request.scan_points()
    assert [point.index for point in points] == [0, 1, 2, 3, 4, 5]
    assert [point.as_mapping()["y"] for point in points] == [
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
    ]
    assert [point.as_mapping()["x"] for point in points] == [
        0.0,
        5.0,
        10.0,
        0.0,
        5.0,
        10.0,
    ]
    assert request.axes == (Y, X)


def test_snaking_reverses_alternate_lines() -> None:
    straight = _request(grid=(AxisGrid(Y, (0.0, 10.0)), AxisGrid(X, (0.0, 5.0, 10.0))))
    snaked = _request(
        grid=(AxisGrid(Y, (0.0, 10.0)), AxisGrid(X, (0.0, 5.0, 10.0))),
        snake=True,
    )
    assert [point.as_mapping()["x"] for point in straight.scan_points()] == [
        0.0,
        5.0,
        10.0,
        0.0,
        5.0,
        10.0,
    ]
    assert [point.as_mapping()["x"] for point in snaked.scan_points()] == [
        0.0,
        5.0,
        10.0,
        10.0,
        5.0,
        0.0,
    ]
    assert [point.as_mapping()["y"] for point in snaked.scan_points()] == [
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
    ]


def test_a_single_position_request_has_one_scan_point() -> None:
    points = _request().scan_points()
    assert points == (ScanPoint(index=0, positions=()),)
    assert points[0].as_mapping() == {}


def test_plan_request_freezes_the_selected_devices() -> None:
    devices = _devices()
    request = _request(devices=devices)
    devices[DeviceRole.CAMERA] = ("other",)
    assert request.cameras == ("dhm",)
    assert request.device_names(DeviceRole.STAGE) == ("stage",)
    assert request.device_names(DeviceRole("stage")) == ("stage",)
    with pytest.raises(TypeError):
        request.devices[DeviceRole.CAMERA] = ("other",)  # type: ignore[index]


def test_partial_run_names_the_missing_frames() -> None:
    requested = _dpct_frames()
    produced = (
        _frame(sequence=0, pattern_id=1, plane_ordinal=0, producer_sequence=10),
        _frame(sequence=1, pattern_id=2, plane_ordinal=1, producer_sequence=11),
        _frame(sequence=2, pattern_id=3, plane_ordinal=2, producer_sequence=12),
    )
    partial = PartialRun(requested=requested, produced=produced)
    assert [frame.sequence for frame in partial.missing] == [3]
    assert partial.missing[0].pattern_id == 4
    assert partial.is_complete is False
    assert partial.requested_count == 4
    assert partial.produced_count == 3
    payload = partial.as_payload()
    assert [item["pattern_id"] for item in payload["missing"]] == [4]


def test_partial_run_accounts_for_every_requested_frame() -> None:
    requested = _dpct_frames()
    produced = tuple(
        _frame(sequence=index, pattern_id=index + 1, plane_ordinal=index)
        for index in range(4)
    )
    assert PartialRun(requested=requested, produced=produced).is_complete is True


def test_partial_run_refuses_a_frame_outside_the_request() -> None:
    with pytest.raises(ValueError, match="outside the request"):
        PartialRun(requested=_dpct_frames(), produced=(_frame(sequence=9),))
    with pytest.raises(ValueError, match="unique"):
        PartialRun(
            requested=_dpct_frames(),
            produced=(_frame(sequence=0), _frame(sequence=0)),
        )
    with pytest.raises(ValueError, match="requested frames"):
        PartialRun(requested=(), produced=())


def test_partial_outcome_names_frames_rather_than_a_boolean() -> None:
    requested = _dpct_frames()
    partial = PartialRun(requested=requested, produced=requested[:3])
    outcome = PlanOutcome(run=RUN, status=PlanStatus.PARTIAL, partial=partial)
    assert outcome.status is PlanStatus.PARTIAL
    assert outcome.is_complete is False
    assert isinstance(outcome.missing, tuple)
    assert {frame.pattern_id for frame in outcome.missing} == {4}
    assert {frame.pattern_id for frame in outcome.produced} == {1, 2, 3}
    assert outcome.as_payload()["status"] == "partial"
    assert outcome.as_payload()["partial"] is not None


def test_complete_and_aborted_outcomes_are_explicit() -> None:
    complete = PlanOutcome(run=RUN, status=PlanStatus.COMPLETE)
    assert complete.is_complete is True
    assert complete.missing == ()
    with pytest.raises(ValueError, match="no partial report"):
        PlanOutcome(
            run=RUN,
            status=PlanStatus.COMPLETE,
            partial=PartialRun(requested=_dpct_frames()),
        )
    with pytest.raises(ValueError, match="no abort reason"):
        PlanOutcome(
            run=RUN,
            status=PlanStatus.COMPLETE,
            reason=PlanAbortReason.DEVICE_FAULT,
        )
    aborted = PlanOutcome(
        run=RUN,
        status=PlanStatus.ABORTED,
        reason=PlanAbortReason.ILLUMINATION_UNCERTAIN,
        partial=PartialRun(requested=_dpct_frames(), produced=_dpct_frames()[:1]),
        detail="pattern 12 timed out; the run was not resent",
    )
    assert aborted.reason is PlanAbortReason.ILLUMINATION_UNCERTAIN
    assert [frame.sequence for frame in aborted.missing] == [1, 2, 3]
    with pytest.raises(ValueError, match="must name its reason"):
        PlanOutcome(run=RUN, status=PlanStatus.ABORTED)


def test_a_partial_outcome_without_a_missing_frame_is_refused() -> None:
    requested = _dpct_frames()
    with pytest.raises(ValueError, match="at least one missing frame"):
        PlanOutcome(
            run=RUN,
            status=PlanStatus.PARTIAL,
            partial=PartialRun(requested=requested, produced=requested),
        )
    with pytest.raises(ValueError, match="needs its partial report"):
        PlanOutcome(run=RUN, status=PlanStatus.PARTIAL)


def test_plan_outcome_payload_is_json_compatible() -> None:
    outcome = PlanOutcome(
        run=RUN,
        status=PlanStatus.PARTIAL,
        partial=PartialRun(requested=_dpct_frames(), produced=_dpct_frames()[:2]),
        finished_monotonic_ns=5_000,
    )
    document = json.loads(json.dumps(outcome.as_payload(), allow_nan=False))
    assert document["run"]["run_id"] == RUN.run_id
    assert document["status"] == "partial"
    assert len(document["partial"]["missing"]) == 2
