from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import numpy as np
import pytest

from redsun_aht.acquisition import (
    PANEL91_DPCT_DONOR_GROUPS,
    panel91_dpct_quadrant_patterns,
)
from redsun_aht.configurations import (
    build_panel91_dpct_simulation,
    run_processing_simulation,
)
from redsun_aht.processing import (
    OfflineTiledJobRequest,
    QuantitativeTileConfig,
    TilingConfig,
    fixed_median_from_czyx,
    fixed_median_from_czyx_store,
)
from redsun_aht.storage import DpctZarrReplay


def test_panel91_dpct_profile_matches_the_restored_four_group_donor() -> None:
    patterns = panel91_dpct_quadrant_patterns()

    assert [len(group) for group in PANEL91_DPCT_DONOR_GROUPS] == [54, 54, 54, 54]
    assert [pattern.quadrant_index for pattern in patterns.values()] == [0, 1, 2, 3]
    assert [len(pattern.addresses) for pattern in patterns.values()] == [51, 48, 51, 48]
    assert {
        address for pattern in patterns.values() for address in pattern.addresses
    } == set(range(91))
    assert [
        hashlib.sha256(pattern.values).hexdigest() for pattern in patterns.values()
    ] == [
        "ff2972cd2bbc8a21aab3fa0d8fe210a2a899e6f91c1d0a2476cc65abc3e53676",
        "54407df8d5c478ccc521ed1f8b7de05323af5e3c6adcda6ab4a51f853a338d9b",
        "12ade182cbd9af7c2bc2ce1569d0c9604bd7eef499b1b42c09d96e4b4dced425",
        "03bbf990405f1360fe3362122cd936146d1204a041ce1deede67e27ea366c041",
    ]


def test_panel91_dpct_profile_applies_intensity_to_only_its_group_members() -> None:
    patterns = panel91_dpct_quadrant_patterns(pattern_ids=(2, 3, 5, 7), intensity=37)

    assert tuple(patterns) == (2, 3, 5, 7)
    for pattern in patterns.values():
        assert {index for index, value in enumerate(pattern.values) if value} == set(
            pattern.addresses
        )
        assert set(pattern.values) == {0, 37}


def test_fixed_median_is_one_value_per_shot_over_all_acquired_z() -> None:
    acquired = np.arange(4 * 5 * 2 * 3, dtype=np.uint16).reshape(4, 5, 2, 3)

    median = fixed_median_from_czyx(acquired, source_checksum="a" * 64)

    np.testing.assert_array_equal(median.values_syx, np.median(acquired, axis=1))
    assert median.provenance["shot_count"] == 4
    assert median.provenance["z_count"] == 5


def test_store_fixed_median_matches_the_in_memory_result() -> None:
    acquired = np.arange(4 * 5 * 2 * 3, dtype=np.uint16).reshape(4, 5, 2, 3)

    median = fixed_median_from_czyx_store(acquired, source_checksum="a" * 64)

    np.testing.assert_array_equal(median.values_syx, np.median(acquired, axis=1))
    assert median.provenance["input_read_mode"] == "per-shot-store"


@pytest.mark.parametrize(
    ("pattern_ids", "intensity", "message"),
    [
        ((1, 1, 2, 3), 255, "must be unique"),
        ((1, 2, 3, 0x10000), 255, "unsigned 16-bit"),
        ((1, 2, 3, 4), 0, "range 1..255"),
        ((1, 2, 3, 4), 256, "range 1..255"),
    ],
)
def test_panel91_dpct_profile_rejects_ambiguous_pattern_requests(
    pattern_ids: tuple[int, int, int, int], intensity: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        panel91_dpct_quadrant_patterns(
            pattern_ids=pattern_ids,
            intensity=intensity,
        )


def test_panel91_dpct_four_pattern_simulation_runs_the_reference_reconstruction(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "panel91-input"
    simulation = build_panel91_dpct_simulation(output_root=input_root, intensity=52)

    result = asyncio.run(simulation.run())

    assert [shot.pattern_id for shot in result.shots] == [
        101,
        102,
        103,
        104,
        101,
        102,
        103,
        104,
    ]
    replay = DpctZarrReplay(input_root).verify()
    assert replay.pattern_ids == (101, 102, 103, 104)
    assert replay.frame_commit_count == 16
    processing = run_processing_simulation(
        input_root,
        tmp_path / "panel91-products",
        tiled=OfflineTiledJobRequest(
            detector_id="dhm",
            optical=QuantitativeTileConfig(
                model="dpct",
                wavelength_um=0.625,
                numerical_aperture=0.8,
                pixel_size_um=0.108333,
                pixel_size_z_um=1.0,
                source_azimuth_deg=(0, 90, 180, 270),
            ),
            tiling=TilingConfig(
                explicit_shape_yx=(6, 6),
                explicit_overlap_yx=(2, 2),
                min_shape_yx=(2, 2),
                max_shape_yx=(8, 8),
                alignment_px=2,
            ),
            fixed_median_from_sample=True,
        ),
    )

    assert not processing.failures
    assert set(processing.results) == {
        "mean-projection",
        "quality-metrics",
        "tiled-dpct-numpy",
    }
