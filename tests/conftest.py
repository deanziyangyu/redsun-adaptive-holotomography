"""Shared pytest configuration and semantic test-suite groups for AHT."""

from __future__ import annotations

from pathlib import Path

import pytest

_TEST_GROUPS: dict[str, frozenset[str]] = {
    "core": frozenset(
        {
            "test_buffers.py",
            "test_calibration.py",
            "test_documents.py",
            "test_domain.py",
            "test_journal.py",
            "test_manifest.py",
            "test_manifest_integrity.py",
            "test_registration.py",
            "test_streams_and_providers.py",
            "test_transport.py",
        }
    ),
    "service": frozenset(
        {
            "test_camera_acquisition_gui.py",
            "test_camera_capture.py",
            "test_camera_ioc.py",
            "test_composite_stage.py",
            "test_detector_services.py",
            "test_dpct_acquisition.py",
            "test_dpct_illumination.py",
            "test_esp300_stage.py",
            "test_mcu_commands.py",
            "test_mcu_human.py",
            "test_mcu_protocol.py",
            "test_mcu_serial.py",
            "test_mmcore_camera.py",
            "test_mmcore_light.py",
            "test_motion_scans.py",
        }
    ),
    "processing": frozenset(
        {
            "test_gpu_processing_hardware.py",
            "test_gpu_resources.py",
            "test_gpu_solver.py",
            "test_gpu_stdio.py",
            "test_msbp_matlab.py",
            "test_multilayer_gpu_hardware.py",
            "test_multilayer_iterative.py",
            "test_multilayer_processing_job.py",
            "test_multilayer.py",
            "test_multislice_schema.py",
            "test_processing.py",
            "test_processing_flyer.py",
            "test_quantitative_gpu_hardware.py",
            "test_quantitative_tiling.py",
            "test_tiled_processing_job.py",
            "test_tiled_reconstruction.py",
        }
    ),
    "integration": frozenset(
        {
            "test_cli.py",
            "test_configurations.py",
            "test_dpct_zarr.py",
            "test_redsun_composition.py",
            "test_redsun_native_architecture.py",
            "test_tiled_catalog.py",
        }
    ),
    "gui": frozenset(
        {
            "test_camera_acquisition_gui.py",
            "test_multishot_gui.py",
            "test_offline_processing_gui.py",
        }
    ),
    "redsun": frozenset(
        {
            "test_camera_acquisition_gui.py",
            "test_configurations.py",
            "test_dpct_acquisition.py",
            "test_multishot_gui.py",
            "test_offline_processing_gui.py",
            "test_processing_flyer.py",
            "test_redsun_composition.py",
            "test_redsun_native_architecture.py",
        }
    ),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply stable suite markers without forcing tests into one directory."""
    for item in items:
        filename = Path(item.path).name
        for group, filenames in _TEST_GROUPS.items():
            if filename in filenames:
                item.add_marker(group)
