"""Headless software-step plan tests with fake stage and camera."""

import hashlib

import numpy as np
import pytest
import zarr
from bluesky import RunEngine

from redsun_aht.device.software_stepped_plan import software_stepped_plan
from redsun_aht.storage import read_manifest
from redsun_aht.storage.writer import StoreRefused


class Stage:
    def __init__(self):
        self.moves = []

    def validate_position_um(self, value):
        if value not in (0, 5, 10):
            raise ValueError("bad target")

    def set_position_um(self, value):
        self.moves.append(value)

    def get_position_um(self):
        return self.moves[-1]


class Camera:
    def __init__(self, *, fail_at=None):
        self.count = 0
        self.fail_at = fail_at
        self.buffer = self

    async def get_value(self, *, cached=False):
        return np.zeros((2, 2), dtype=np.uint8)

    async def snap(self):
        self.count += 1
        if self.count == self.fail_at:
            raise RuntimeError("camera failed")
        return self.count, np.full((2, 2), self.count, dtype=np.uint8)


def test_plan_emits_each_host_ordered_plane():
    stage = Stage()
    camera = Camera()
    events = []
    engine = RunEngine({}, context_managers=[])
    engine.subscribe(
        lambda name, doc: events.append(doc["data"]) if name == "event" else None
    )
    engine(
        software_stepped_plan(
            stage, camera, [0, 5, 10], reservation_held=True, camera_exclusive=True
        )
    )
    assert stage.moves == [0, 5, 10]
    assert [event["plane"] for event in events] == [0, 1, 2]
    assert [event["camera_frame_count"] for event in events] == [1, 2, 3]
    assert [event["controller_position_um"] for event in events] == [0, 5, 10]


def test_prevalidation_and_failure_do_not_queue_later_moves():
    stage = Stage()
    camera = Camera()
    engine = RunEngine({}, context_managers=[])
    with pytest.raises(ValueError, match="bad target"):
        engine(
            software_stepped_plan(
                stage, camera, [0, 12], reservation_held=True, camera_exclusive=True
            )
        )
    assert not stage.moves
    with pytest.raises(RuntimeError, match="camera failed"):
        engine(
            software_stepped_plan(
                stage,
                Camera(fail_at=2),
                [0, 5, 10],
                reservation_held=True,
                camera_exclusive=True,
            )
        )
    assert stage.moves == [0, 5]


def test_storage_full_and_partial_are_distinguishable(tmp_path):
    engine = RunEngine({}, context_managers=[])
    full = tmp_path / "full.zarr"
    engine(
        software_stepped_plan(
            Stage(),
            Camera(),
            [0, 5, 10],
            reservation_held=True,
            camera_exclusive=True,
            store_uri=str(full),
        )
    )
    manifest = read_manifest(str(full), data_key="software_snap")
    assert manifest.frame_count == manifest.planned_frames == 3
    assert manifest.is_complete
    images = zarr.open_array(str(full / "software_snap"), mode="r")
    assert all(
        hashlib.sha256(images[i].tobytes(order="C")).hexdigest() == entry.checksum
        for i, entry in enumerate(manifest.frames)
    )
    partial = tmp_path / "partial.zarr"
    with pytest.raises(RuntimeError, match="camera failed"):
        engine(
            software_stepped_plan(
                Stage(),
                Camera(fail_at=2),
                [0, 5, 10],
                reservation_held=True,
                camera_exclusive=True,
                store_uri=str(partial),
            )
        )
    manifest = read_manifest(str(partial), data_key="software_snap")
    assert manifest.frame_count == 1 and manifest.planned_frames == 3
    assert not manifest.is_complete
    stage = Stage()
    with pytest.raises(StoreRefused):
        engine(
            software_stepped_plan(
                stage,
                Camera(),
                [0, 5, 10],
                reservation_held=True,
                camera_exclusive=True,
                store_uri=str(full),
            )
        )
    assert not stage.moves
