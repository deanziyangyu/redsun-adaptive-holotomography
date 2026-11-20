"""Exercise the public headless DPCT TIFF commands."""

import json

import numpy as np
import tifffile
import yaml

from redsun_aht.__main__ import main


def test_legacy_dpct_tiff_with_roi_and_recorded_zero_policy(tmp_path):
    sample = np.full((3, 4, 12, 12), 120, dtype=np.uint16)
    sample[1, 2, 4, 4] = 0
    reference = np.full((4, 12, 12), 100, dtype=np.uint16)
    sample_path = tmp_path / "sample.tif"
    reference_path = tmp_path / "reference.tif"
    tifffile.imwrite(sample_path, sample, imagej=True, metadata={"axes": "ZCYX"})
    tifffile.imwrite(
        reference_path,
        reference,
        compression="deflate",
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        yaml.safe_dump(
            {
                "optical_setup": {
                    "wavelength_um": 0.465,
                    "numerical_aperture": 0.95,
                    "refractive_index_medium": 1.35,
                    "pixel_size_detector_um": 6.5,
                    "pixel_size_z_um": 1,
                    "mag": 40,
                },
                "solvers": {
                    "dpct": {
                        "src_azimuth_agl_deg": [270, 0, 90, 180],
                        "background_removal": True,
                        "background_removal_params": {"inv_bg_divide": True},
                        "tiling": {"mode": "off"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    migrated = tmp_path / "migrated.yaml"
    assert (
        main(
            [
                "migrate-recon-config",
                "--input",
                str(legacy),
                "--output",
                str(migrated),
            ]
        )
        == 0
    )
    output = tmp_path / "result.tif"
    argv = [
        "reconstruct-tiff",
        "--sample",
        str(sample_path),
        "--reference",
        str(reference_path),
        "--config",
        str(migrated),
        "--output",
        str(output),
        "--roi",
        "2,2,8,8",
    ]
    assert main(argv) == 1
    assert not output.exists()
    assert main([*argv, "--zero-policy", "neutral"]) == 0
    with tifffile.TiffFile(output) as product:
        assert product.series[0].axes == "ZYX"
        assert product.series[0].shape == (3, 8, 8)
        assert product.series[0].dtype == np.uint16
    document = json.loads(output.with_suffix(".provenance.json").read_text("utf-8"))
    assert document["neutral_replacements"] == 1
    assert document["config"]["input"]["normalization"] == "reference_over_sample"
    assert main([*argv, "--zero-policy", "neutral"]) == 1


def test_dpct_tv_medians_repeated_reference_and_writes_float_ri(tmp_path):
    rng = np.random.default_rng(11)
    sample = tmp_path / "sample.tif"
    reference = tmp_path / "reference.tif"
    config = tmp_path / "dpct.yaml"
    output = tmp_path / "ri.tif"
    data = rng.integers(102, 121, size=(3, 4, 12, 12), dtype=np.uint16)
    repeats = np.broadcast_to(
        np.array([96, 101, 102, 350], dtype=np.uint16)[:, None, None, None],
        (4, 4, 12, 12),
    ).copy()
    tifffile.imwrite(
        sample, data, ome=True, metadata={"axes": "ZCYX"}, photometric="minisblack"
    )
    tifffile.imwrite(
        reference,
        repeats,
        ome=True,
        metadata={"axes": "TCYX"},
        photometric="minisblack",
    )
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "optics": {
                    "wavelength_um": 0.465,
                    "pixel_size_um": 0.18333333,
                    "refractive_index_medium": 1.33,
                    "numerical_aperture": 0.95,
                },
                "input": {
                    "z_spacing_um_override": 0.25,
                    "normalization": "reference_over_sample",
                    "allow_reference_broadcast": True,
                },
                "dpct": {
                    "source_azimuth_deg": [180, 90, 0, 270],
                    "dpct_tv_iterations": 2,
                    "tiling": {"mode": "off"},
                },
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "reconstruct-tiff",
                "--solver",
                "dpct",
                "--method",
                "tv",
                "--sample",
                str(sample),
                "--reference",
                str(reference),
                "--config",
                str(config),
                "--output",
                str(output),
                "--output-dtype",
                "float32",
            ]
        )
        == 0
    )
    ri = tifffile.imread(output)
    assert ri.shape == (3, 12, 12) and ri.dtype == np.float32
    assert np.isfinite(ri).all() and np.ptp(ri) > 1e-5
    receipt = json.loads(
        output.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    assert receipt["optics"]["dpct_method"] == "tv"
    assert receipt["reference_median_repeats"] == 4
