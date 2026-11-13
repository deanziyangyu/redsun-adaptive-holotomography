"""Finite frame-buffer ownership, overflow and failed-flush checks."""

import numpy as np
import pytest

from redsun_aht.storage.bounded_frame_batch import BoundedFrameBatch


class Sink:
    def __init__(self, fail_at=None):
        self.frames = []
        self.fail_at = fail_at

    def append(self, frame):
        if len(self.frames) == self.fail_at:
            raise OSError("write uncertain")
        self.frames.append(frame.copy())
        return len(self.frames) - 1


def test_owned_frames_flush_once_in_order():
    batch = BoundedFrameBatch(max_frames=2, max_bytes=8)
    frame = np.ones((2, 2), dtype=np.uint8)
    batch.append(frame)
    frame[:] = 2
    batch.append(frame)
    frame[:] = 9
    sink = Sink()
    batch.flush(sink)
    assert [int(image[0, 0]) for image in sink.frames] == [1, 2]
    assert batch.persisted == 2 and not batch.overflowed
    with pytest.raises(RuntimeError, match="flush"):
        batch.flush(sink)


def test_overflow_preserves_prefix_and_is_latched():
    batch = BoundedFrameBatch(max_frames=2, max_bytes=5)
    frame = np.ones((2, 2), dtype=np.uint8)
    batch.append(frame)
    with pytest.raises(BufferError):
        batch.append(frame)
    assert batch.overflowed and batch.buffered == 1
    with pytest.raises(BufferError):
        batch.append(frame)
    sink = Sink()
    batch.flush(sink)
    assert batch.persisted == 1 and batch.overflowed


def test_failed_flush_preserves_prefix_but_disallows_retry():
    batch = BoundedFrameBatch(max_frames=2, max_bytes=8)
    batch.append(np.ones((2, 2), dtype=np.uint8))
    batch.append(np.ones((2, 2), dtype=np.uint8))
    with pytest.raises(OSError, match="uncertain"):
        batch.flush(Sink(fail_at=1))
    assert batch.persisted == 1 and batch.buffered == 2
    with pytest.raises(RuntimeError, match="flush"):
        batch.flush(Sink())


def test_bulk_flush_preserves_manifest_and_frame_hashes(tmp_path):
    from redsun_aht.storage import read_manifest
    from redsun_aht.storage.writer import RunFrameStore, frame_checksum

    values = [np.full((2, 2), i, dtype=np.uint16) for i in range(4)]
    store = RunFrameStore(str(tmp_path / "batch"), "cam", values[0])
    batch = BoundedFrameBatch(max_frames=4, max_bytes=32)
    for value in values:
        batch.append(value)
    batch.flush(store)
    store.close()
    manifest = read_manifest(str(tmp_path / "batch"), data_key="cam")
    assert batch.persisted == manifest.frame_count == 4
    assert [x.checksum for x in manifest.frames] == [frame_checksum(x) for x in values]
    assert [x.index for x in manifest.frames] == [0, 1, 2, 3]


def test_bulk_partial_write_is_uncertain_and_has_no_manifest(tmp_path):
    from redsun_aht.storage.writer import RunFrameStore, manifest_path

    frame = np.ones((2, 2), dtype=np.uint16)
    uri = str(tmp_path / "partial")
    store = RunFrameStore(uri, "cam", frame)
    original = store._array

    class PartialWrite:
        def __getattr__(self, name):
            return getattr(original, name)

        def append(self, data, *, axis):
            original.append(data[:1], axis=axis)
            raise OSError("partial batch")

    store._array = PartialWrite()
    batch = BoundedFrameBatch(max_frames=2, max_bytes=16)
    batch.append(frame)
    batch.append(frame)
    with pytest.raises(OSError, match="partial batch"):
        batch.flush(store)
    assert batch.persisted == 0
    with pytest.raises(RuntimeError, match="uncertain"):
        store.close()
    with pytest.raises(RuntimeError, match="uncertain"):
        store.append(frame)
    assert not manifest_path(uri, "cam").exists()


def test_timed_store_uses_sequential_fallback(tmp_path):
    from redsun_aht.storage import read_manifest
    from redsun_aht.storage.writer import RunFrameStore

    frame = np.ones((2, 2), dtype=np.uint16)
    uri = str(tmp_path / "timed")
    store = RunFrameStore(uri, "cam", frame, record_timing=True)
    batch = BoundedFrameBatch(max_frames=2, max_bytes=16)
    batch.append(frame)
    batch.append(frame)
    batch.flush(store)
    store.close()
    assert batch.persisted == read_manifest(uri, data_key="cam").frame_count == 2
