from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from redsun_aht.calibration import (
    ArtifactIntegrityError,
    CalibrationArtifactStore,
    apply_background,
    temporal_median,
)
from redsun_aht.domain import CalibrationKind


def test_median_artifact_reproduces_corrected_output(tmp_path: Path) -> None:
    frames = [
        np.array([[1, 0], [3, 4]], dtype=np.uint16),
        np.array([[3, 0], [5, 6]], dtype=np.uint16),
        np.array([[2, 0], [4, 5]], dtype=np.uint16),
    ]
    median = temporal_median(frames)
    store = CalibrationArtifactStore(tmp_path)
    artifact = store.create(
        kind=CalibrationKind.MEDIAN,
        array=median,
        validity_scope={"detector_id": "dhm", "channel_id": "dhm"},
        source_run_id="run-calibration",
        settings={"exposure_s": 0.01},
        created_ns=1,
    )
    median[:] = 99

    replayed, persisted = store.read(artifact.artifact_id)
    raw = np.array([[4, 9], [8, 10]], dtype=np.uint16)
    corrected = apply_background(raw, persisted)

    assert replayed == artifact
    np.testing.assert_array_equal(persisted, [[2, 0], [4, 5]])
    np.testing.assert_allclose(corrected, [[2, 1], [2, 2]])


def test_registration_transform_is_locked_as_artifact(tmp_path: Path) -> None:
    transform = np.eye(3, dtype=np.float64)
    store = CalibrationArtifactStore(tmp_path)
    artifact = store.create(
        kind=CalibrationKind.REGISTRATION,
        array=transform,
        validity_scope={
            "source_detector": "fluorescence",
            "target_detector": "dhm",
            "confidence": 0.99,
            "residual_px": 0.2,
        },
        source_run_id="registration-run",
        settings={"model": "affine"},
        created_ns=2,
    )
    transform[0, 0] = 7

    _, persisted = store.read(artifact.artifact_id)
    np.testing.assert_array_equal(persisted, np.eye(3))


def test_artifact_integrity_and_recipe_validation(tmp_path: Path) -> None:
    store = CalibrationArtifactStore(tmp_path)
    artifact = store.create(
        kind=CalibrationKind.DARK,
        array=np.ones((2, 2), dtype=np.uint8),
        validity_scope={},
        source_run_id="run",
        settings={},
        created_ns=3,
    )
    array_path = Path(artifact.payload.uri.removeprefix("file:///"))
    np.save(array_path, np.zeros((2, 2), dtype=np.uint8), allow_pickle=False)

    with pytest.raises(ArtifactIntegrityError):
        store.read(artifact.artifact_id)
    with pytest.raises(ValueError, match="at least one"):
        temporal_median([])
    with pytest.raises(ValueError, match="shapes differ"):
        apply_background(np.ones((1, 1)), np.ones((2, 2)))
