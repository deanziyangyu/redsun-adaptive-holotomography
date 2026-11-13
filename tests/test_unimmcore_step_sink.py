"""Ordered UniMMCore software-frame sink tests with synthetic arrays."""

import hashlib

import numpy as np
import pytest
import zarr

from redsun_aht.storage import read_manifest
from redsun_aht.storage.unimmcore_step_sink import UniMMCoreStepSink


def test_ordered_sink_writes_verified_frames(tmp_path):
    uri = tmp_path / "full.zarr"
    template = np.zeros((2, 3), dtype=np.uint16)
    sink = UniMMCoreStepSink(str(uri), template, count=3)
    for i in range(3):
        assert sink.append(np.full((2, 3), i, dtype=np.uint16), {"z": i}) == i
    sink.close()
    manifest = read_manifest(str(uri), data_key="software_snap")
    assert manifest.is_complete and manifest.frame_count == manifest.planned_frames == 3
    pixels = zarr.open_array(str(uri / "software_snap"), mode="r")
    assert all(
        hashlib.sha256(pixels[i].tobytes(order="C")).hexdigest() == row.checksum
        for i, row in enumerate(manifest.frames)
    )
    with pytest.raises(RuntimeError, match="already closed"):
        sink.close()


def test_invalid_index_latches_and_preserves_partial(tmp_path):
    uri = tmp_path / "partial.zarr"
    image = np.zeros((2, 3), dtype=np.uint16)
    sink = UniMMCoreStepSink(str(uri), image, count=3)
    sink.append(image, {"z": 0})
    with pytest.raises(ValueError, match="out of order"):
        sink.append(image, {"z": 2})
    with pytest.raises(RuntimeError, match="uncertain"):
        sink.append(image, {"z": 1})
    with pytest.raises(RuntimeError, match="partial"):
        sink.close()
    manifest = read_manifest(str(uri), data_key="software_snap")
    assert (
        manifest.frame_count == 1
        and manifest.planned_frames == 3
        and not manifest.is_complete
    )


def test_writer_failure_latches_without_retry(tmp_path, monkeypatch):
    uri = tmp_path / "failed.zarr"
    image = np.zeros((2, 3), dtype=np.uint16)
    sink = UniMMCoreStepSink(str(uri), image, count=3)
    original = sink.store.append
    calls = []

    def fail_second(frame):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("injected write failure")
        return original(frame)

    monkeypatch.setattr(sink.store, "append", fail_second)
    sink.append(image, {"z": 0})
    with pytest.raises(OSError, match="injected"):
        sink.append(image, {"z": 1})
    with pytest.raises(RuntimeError, match="uncertain"):
        sink.append(image, {"z": 1})
    with pytest.raises(RuntimeError, match="partial"):
        sink.close()
    assert len(calls) == 2
    assert read_manifest(str(uri), data_key="software_snap").frame_count == 1
