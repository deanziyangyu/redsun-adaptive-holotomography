"""Reservation grouping with explicit uncertain-outcome handling."""

import pytest
from test_motion_coordinator import setup

from redsun_aht.device.coordinated_reservation import (
    CoordinatedReservation,
    GroupState,
    ReservationResource,
)


class Service:
    def __init__(self, name, calls, *, fail_acquire=False, fail_release=False):
        self.name, self.calls = name, calls
        self.fail_acquire, self.fail_release = fail_acquire, fail_release

    async def acquire(self):
        self.calls.append("acquire " + self.name)
        if self.fail_acquire:
            raise TimeoutError("ack lost")

    async def release(self):
        self.calls.append("release " + self.name)
        if self.fail_release:
            raise TimeoutError("ack lost")

    def binding(self):
        return ReservationResource(self.name, self.acquire, self.release)


async def test_group_acquires_in_order_and_releases_in_reverse():
    coord, _, _ = setup()
    calls = []
    a, b = Service("survey", calls), Service("mcu", calls)
    group = CoordinatedReservation(coord, [a.binding(), b.binding()])
    token = await group.acquire()
    assert group.state is GroupState.HELD and group.acquired == ["survey", "mcu"]
    with pytest.raises(RuntimeError, match="reservation"):
        await coord.jog_by({"survey_x": 1})
    await coord.move({"survey_x": 1}, token=token)
    await group.release()
    assert calls == ["acquire survey", "acquire mcu", "release mcu", "release survey"]
    assert group.state is GroupState.RELEASED and not group.acquired


async def test_failed_acquire_retains_prior_and_local_ownership():
    coord, _, _ = setup()
    calls = []
    a, b = Service("survey", calls), Service("mcu", calls, fail_acquire=True)
    group = CoordinatedReservation(coord, [a.binding(), b.binding()])
    with pytest.raises(TimeoutError, match="ack lost"):
        await group.acquire()
    assert group.state is GroupState.UNCERTAIN
    assert group.acquired == ["survey"] and group.attempting == "mcu"
    assert calls == ["acquire survey", "acquire mcu"]
    with pytest.raises(RuntimeError, match="reserved"):
        coord.reserve()
    with pytest.raises(RuntimeError, match="not certainly held"):
        await group.release()


async def test_failed_release_preserves_remaining_ownership():
    coord, _, _ = setup()
    calls = []
    a, b = Service("survey", calls, fail_release=True), Service("mcu", calls)
    group = CoordinatedReservation(coord, [a.binding(), b.binding()])
    await group.acquire()
    with pytest.raises(TimeoutError, match="ack lost"):
        await group.release()
    assert group.state is GroupState.UNCERTAIN
    assert group.acquired == ["survey"] and group.attempting == "survey"
    with pytest.raises(RuntimeError, match="reserved"):
        coord.reserve()
