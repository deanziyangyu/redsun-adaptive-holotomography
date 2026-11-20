"""Test the public offline IDT path with repeated-channel background TIFFs."""

import json

import numpy as np
import tifffile
import yaml

from redsun_aht.__main__ import main


def test_linear_idt_uses_median_per_led_and_preserves_xy_axes(tmp_path):
    sample_path = tmp_path / "sample.tif"
    repeats_path = tmp_path / "background.tif"
    config_path = tmp_path / "idt.yaml"
    led_path = tmp_path / "led.npz"
    output = tmp_path / "ri.tif"
    median_path = tmp_path / "median.tif"
    rng = np.random.default_rng(22)
    sample = rng.integers(101, 121, size=(4, 12, 16), dtype=np.uint16)
    offsets = np.array([0, 8, 9, 80], dtype=np.uint16)[:, None, None, None]
    background = np.array([93, 94, 95, 96], dtype=np.uint16)[None, :, None, None]
    repeats = np.broadcast_to(offsets + background, (4, 4, 12, 16)).copy()
    channel_names = ["40", "41", "42", "43"]
    tifffile.imwrite(
        sample_path,
        sample,
        imagej=True,
        metadata={"axes": "CYX", "Labels": channel_names},
        photometric="minisblack",
    )
    tifffile.imwrite(
        repeats_path,
        repeats,
        ome=True,
        metadata={"axes": "TCYX", "Channel": {"Name": channel_names}},
        photometric="minisblack",
    )
    directions = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    np.savez(
        led_path,
        illumination_fxy=directions,
        firmware_led_index_zero_based=[43, 42, 41, 40],
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "optics": {
                    "wavelength_um": 0.465,
                    "pixel_size_um": 0.1833333333,
                    "pixel_size_z_um": 0.25,
                    "refractive_index_medium": 1.35,
                    "numerical_aperture": 0.95,
                },
                "idt": {
                    "illumination_file": led_path.name,
                    "depth_range_um": [-0.25, 0.25],
                },
            }
        ),
        encoding="utf-8",
    )
    args = [
        "reconstruct-tiff",
        "--solver",
        "idt",
        "--sample",
        str(sample_path),
        "--reference",
        str(repeats_path),
        "--config",
        str(config_path),
        "--output",
        str(output),
        "--median-output",
        str(median_path),
    ]
    assert main(args) == 1
    assert not output.exists()
    np.savez(
        led_path,
        illumination_fxy=directions,
        firmware_led_index_zero_based=[40, 41, 42, 43],
    )
    assert main(args) == 0
    with tifffile.TiffFile(output) as result:
        assert result.series[0].shape == (3, 12, 16)
        assert result.series[0].axes == "ZYX"
        assert result.imagej_metadata["spacing"] == 0.25
        assert result.series[0].dtype == np.float32
        assert np.isfinite(result.asarray()).all()
        assert np.ptp(result.asarray()) > 1e-5
    median = tifffile.imread(median_path)
    assert median.dtype == np.float32
    assert np.array_equal(median, np.median(repeats, axis=0))
    receipt = json.loads(
        output.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    assert receipt["reference_median_repeats"] == 4
    assert (
        receipt["sample_channel_labels"]
        == receipt["reference_channel_labels"]
        == channel_names
    )
    assert receipt["defocus_um"] == [-0.25, 0.0, 0.25]
    assert main(args) == 1
