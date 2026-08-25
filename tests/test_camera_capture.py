from __future__ import annotations

from pathlib import Path

import numpy as np

from redsun_aht.configurations import run_processing_simulation
from redsun_aht.processing import observations_from_offline_bundle
from redsun_aht.storage import CameraCaptureZarrReplay, CameraCaptureZarrStore


def test_camera_capture_store_round_trips_one_czyx_frame(tmp_path: Path) -> None:
    root = Path(tmp_path) / "capture"
    source = np.arange(20, dtype=np.uint16).reshape(4, 5)
    manifest = CameraCaptureZarrStore(root).write(
        run_id="capture-run",
        service_id="fluorescence",
        frame_id="frame-1",
        sequence=7,
        timestamp_ns=123,
        array=source,
    )

    replay = CameraCaptureZarrReplay(root)
    assert replay.verify() == manifest
    np.testing.assert_array_equal(replay.read(), source[np.newaxis, np.newaxis])
    observation = observations_from_offline_bundle(root)[0]
    assert observation.detector_id == "fluorescence"
    assert observation.channel_id == "single-camera-frame"
    assert observation.array.shape == (1, 1, 4, 5)


def test_camera_capture_replays_through_reference_processors(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    CameraCaptureZarrStore(root).write(
        run_id="capture-run",
        service_id="dhm",
        frame_id="frame-1",
        sequence=0,
        timestamp_ns=123,
        array=np.arange(64, dtype=np.uint16).reshape(8, 8),
    )

    result = run_processing_simulation(root, tmp_path / "products")

    assert set(result.results) == {"mean-projection", "quality-metrics"}
    assert not result.failures
