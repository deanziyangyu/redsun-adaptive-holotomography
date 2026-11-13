"""Explicit-binding presenter contract."""

import pytest
from bluesky import RunEngine

from redsun_aht.presenter.pva_timelapse import PvaTimeLapsePresenter


def test_unconfigured_presenter_offers_no_plan():
    presenter = PvaTimeLapsePresenter("timelapse", bindings=None)
    assert presenter.plan_map() == {}
    with pytest.raises(RuntimeError, match="not configured"):
        RunEngine({})(
            presenter.timelapse(
                ["xy"],
                1,
                ["red"],
                [1],
                [1.0],
                4,
                resources_exclusive=True,
                internal_trigger_configured=True,
                illumination_ready=True,
            )
        )


def test_managed_presenter_releases_reservation_after_failure(monkeypatch):
    from bluesky import plan_stubs as bps

    from redsun_aht.presenter.pva_timelapse import PvaTimeLapseBindings

    class Owner:
        reservation_held = False
        releases = 0

        async def reserve(self):
            self.reservation_held = True

        async def allocate_run_ids(self, count):
            return tuple(range(9, 9 + count))

        async def release_reservation(self):
            self.reservation_held = False
            self.releases += 1

    def noop(*args):
        yield from bps.null()

    def fail(*args, **kwargs):
        yield from bps.null()
        raise RuntimeError("capture failed")

    monkeypatch.setattr("redsun_aht.presenter.pva_timelapse.pva_timelapse_plan", fail)
    owner = Owner()
    bindings = PvaTimeLapseBindings(
        owner, object(), lambda i: None, lambda b, p: "x", noop, noop, noop, noop
    )
    presenter = PvaTimeLapsePresenter("test", bindings=bindings)
    with pytest.raises(RuntimeError, match="capture failed"):
        RunEngine({})(
            presenter.managed_timelapse(
                ["xy"],
                1,
                ["red"],
                [1],
                [1.0],
                4,
                first=0,
                step=18,
                planes=2,
                lower=-180,
                upper=180,
                resources_exclusive=True,
                internal_trigger_configured=True,
                illumination_ready=True,
            )
        )
    assert owner.releases == 1 and not owner.reservation_held


def test_ix83_style_confirmation_label_is_not_a_run_uid():
    from datetime import datetime

    from redsun_aht.plans.confirmation_ids import new_confirmation_id

    assert (
        new_confirmation_id("my-plan name", now=datetime(2026, 11, 8, 14, 7))
        == "20261108T1407_MY_PLAN_NAME"
    )
    assert (
        new_confirmation_id("", now=datetime(2026, 11, 8, 14, 7))
        == "20261108T1407_PLAN"
    )


def test_automatic_stack_plan_emits_counts_and_cleans_up(monkeypatch):
    from types import SimpleNamespace

    from bluesky import plan_stubs as bps

    from redsun_aht.presenter.pva_timelapse import PvaTimeLapseBindings

    class Owner:
        reservation_held = False
        blanked = False

        async def reserve(self):
            self.reservation_held = True

        async def allocate_run_ids(self, count):
            return (42,)

        async def set_reserved_illumination(self, command):
            self.blanked = command == "00"

        async def release_reservation(self):
            self.reservation_held = False

    async def capture(owner, camera, definition, *args, **kwargs):
        assert definition.run_id == 42
        return SimpleNamespace(
            attempted_pulses=48,
            camera_count_delta=48,
            evidence=SimpleNamespace(stored_frames=48),
        )

    def noop(*args):
        yield from bps.null()

    monkeypatch.setattr(
        "redsun_aht.presenter.pva_timelapse.acquire_pva_automatic_stack", capture
    )
    owner = Owner()
    binding = PvaTimeLapseBindings(
        owner,
        object(),
        lambda i: None,
        lambda b, p: "stack.zarr",
        noop,
        noop,
        noop,
        noop,
    )
    presenter = PvaTimeLapsePresenter("test", bindings=binding)
    engine = RunEngine({})
    docs = []
    engine.subscribe(lambda name, doc: docs.append((name, doc)))
    engine(
        presenter.automatic_stack(
            "blue",
            16,
            1,
            50,
            first=0,
            step=18,
            planes=3,
            lower=-180,
            upper=180,
            resources_exclusive=True,
            external_trigger_configured=True,
            continuous_capture_configured=True,
            illumination_ready=True,
        )
    )
    event = next(doc for name, doc in docs if name == "event")
    assert event["data"]["stored_frames"] == 48
    assert owner.blanked and not owner.reservation_held
