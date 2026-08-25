from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from redsun_aht.configurations import (
    build_dpct_simulation,
    build_processing_gui_presenter,
    run_processing_gui,
)
from redsun_aht.presenter import ProcessingViewResult
from redsun_aht.processing import (
    GpuAdmissionError,
    GpuDeviceInfo,
    MultiLayerReconstructionConfig,
    MultiLayerSolveConfig,
    OfflineMultiLayerJobRequest,
    OfflineProcessingRequest,
    OfflineTiledJobRequest,
    ProcessingBatchResult,
    QuantitativeTileConfig,
    TilingConfig,
    resolve_offline_processing,
)
from redsun_aht.processing import offline as offline_module

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _bundle(tmp_path: Path) -> Path:
    input_root = tmp_path / "input bundle"
    asyncio.run(build_dpct_simulation("gui-run", output_root=input_root).run())
    return input_root


def _tiled_request() -> OfflineTiledJobRequest:
    return OfflineTiledJobRequest(
        "dhm",
        QuantitativeTileConfig(
            model="dpct",
            wavelength_um=0.625,
            numerical_aperture=0.8,
            pixel_size_um=0.108333,
            pixel_size_z_um=1.0,
        ),
        TilingConfig(
            explicit_shape_yx=(6, 6),
            explicit_overlap_yx=(2, 2),
            min_shape_yx=(2, 2),
            max_shape_yx=(8, 8),
            alignment_px=2,
        ),
    )


def _multilayer_request() -> OfflineMultiLayerJobRequest:
    return OfflineMultiLayerJobRequest(
        "dhm",
        MultiLayerReconstructionConfig(
            model="multislice",
            depth_layers=3,
            voxel_size_zyx_um=(0.7, 0.12, 0.11),
            wavelength_um=0.625,
            numerical_aperture=0.8,
            refractive_index_medium=1.33,
            illumination_na=0.45,
            solve=MultiLayerSolveConfig(max_iterations=1, step_size=2e-5),
        ),
    )


def test_offline_processing_plan_builds_typed_jobs_before_quoted_commands(
    tmp_path: Path,
) -> None:
    input_root = _bundle(tmp_path)
    output_root = tmp_path / "output products"
    plan = resolve_offline_processing(OfflineProcessingRequest(input_root, output_root))

    assert plan.source_run_id == "gui-run"
    assert set(plan.jobs) == {"mean-projection", "quality-metrics"}
    assert all(job.latency_class == "offline" for job in plan.jobs.values())
    assert plan.command_argv() == (
        "aht",
        "process-headless",
        "--input",
        str(input_root.resolve()),
        "--output",
        str(output_root.resolve()),
    )
    assert f"'{input_root.resolve()}'" in plan.render_command(platform="posix")
    assert f'"{input_root.resolve()}"' in plan.render_command(platform="windows")

    gpu_plan = resolve_offline_processing(
        OfflineProcessingRequest(input_root, output_root, gpu_id=1)
    )
    assert "gpu-mean-projection" in gpu_plan.jobs
    assert gpu_plan.command_argv()[-4:] == (
        "--gpu-id",
        "1",
        "--gpu-reservation-mib",
        "256",
    )

    tiled_gpu_plan = resolve_offline_processing(
        OfflineProcessingRequest(
            input_root,
            output_root,
            tiled=replace(_tiled_request(), backend="cupy", gpu_id=1),
        )
    )
    assert "tiled-dpct-cupy" in tiled_gpu_plan.jobs
    tiled_gpu_command = tiled_gpu_plan.command_argv()
    assert tiled_gpu_command[tiled_gpu_command.index("--tiled-backend") + 1] == "cupy"
    assert tiled_gpu_command[tiled_gpu_command.index("--tiled-gpu-id") + 1] == "1"

    multilayer_plan = resolve_offline_processing(
        OfflineProcessingRequest(
            input_root,
            output_root,
            multilayer=_multilayer_request(),
        )
    )
    assert "multilayer-multislice-numpy" in multilayer_plan.jobs
    multilayer_command = multilayer_plan.command_argv()
    assert (
        multilayer_command[multilayer_command.index("--multilayer-model") + 1]
        == "multislice"
    )
    assert "--multilayer-step-size" in multilayer_command


@pytest.mark.parametrize(
    ("input_name", "output_name", "gpu_id", "reservation", "message"),
    [
        ("missing", "output", None, 1024 * 1024, "not a directory"),
        ("input", "input/products", None, 1024 * 1024, "immutable input"),
        ("input", "output", -1, 1024 * 1024, "cannot be negative"),
        ("input", "output", None, 0, "must be positive"),
        ("input", "output", None, 1024 * 1024 + 1, "whole mebibytes"),
    ],
)
def test_offline_processing_request_rejects_invalid_paths_and_resources(
    tmp_path: Path,
    input_name: str,
    output_name: str,
    gpu_id: int | None,
    reservation: int,
    message: str,
) -> None:
    if input_name == "input":
        (tmp_path / input_name).mkdir()
    with pytest.raises(ValueError, match=message):
        OfflineProcessingRequest(
            tmp_path / input_name,
            tmp_path / output_name,
            gpu_id=gpu_id,
            gpu_reservation_bytes=reservation,
        )


def test_offline_tiled_request_rejects_invalid_gpu_selection() -> None:
    base = _tiled_request()
    with pytest.raises(ValueError, match="requires an explicit GPU"):
        replace(base, backend="cupy")
    with pytest.raises(ValueError, match="cannot select a GPU"):
        replace(base, gpu_id=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(base, backend="cupy", gpu_id=-1)
    with pytest.raises(ValueError, match="must be positive"):
        replace(base, gpu_reservation_bytes=0)
    with pytest.raises(ValueError, match="whole mebibytes"):
        replace(base, gpu_reservation_bytes=1024 * 1024 + 1)


def test_offline_multilayer_request_rejects_invalid_selection() -> None:
    base = _multilayer_request()
    with pytest.raises(ValueError, match="requires an explicit GPU"):
        replace(base, backend="cupy")
    with pytest.raises(ValueError, match="cannot select a GPU"):
        replace(base, gpu_id=0)
    with pytest.raises(ValueError, match="requires background normalization"):
        replace(base, background_detector_id="fluorescence")
    background = replace(base.reconstruction, normalization="background")
    with pytest.raises(ValueError, match="requires a detector"):
        replace(base, reconstruction=background)


def test_processing_presenter_executes_and_verifies_products(tmp_path: Path) -> None:
    input_root = _bundle(tmp_path)
    presenter = build_processing_gui_presenter()
    plan = presenter.resolve(str(input_root), str(tmp_path / "products"))

    command = presenter.preview_command(plan, platform="windows")
    result = presenter.execute(plan)

    assert "process-headless" in command
    assert presenter.last_plan == plan
    assert presenter.last_result == result
    assert result.source_run_id == "gui-run"
    assert set(result.products) == {"mean-projection", "quality-metrics"}
    assert result.products["mean-projection"].ndim == 3
    assert not result.failures


def test_offline_plan_runs_tiled_job_with_its_selected_observation(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "four-shot-input"
    asyncio.run(
        build_dpct_simulation(
            "integrated-tiled-run",
            output_root=input_root,
            pattern_ids=(10, 11, 12, 13),
        ).run()
    )
    presenter = build_processing_gui_presenter()
    plan = presenter.resolve(
        str(input_root),
        str(tmp_path / "products"),
        tiled=_tiled_request(),
    )

    assert set(plan.jobs) == {
        "mean-projection",
        "quality-metrics",
        "tiled-dpct-numpy",
    }
    assert len(plan.job_observations["mean-projection"]) == 2
    assert tuple(
        item.detector_id for item in plan.job_observations["tiled-dpct-numpy"]
    ) == ("dhm",)
    command = plan.command_argv()
    assert command[command.index("--tiled-model") + 1] == "dpct"
    shape_index = command.index("--tile-shape-yx")
    assert command[shape_index + 1 : shape_index + 3] == ("6", "6")

    result = presenter.execute(plan)

    assert set(result.products) == {
        "mean-projection",
        "quality-metrics",
        "tiled-dpct-numpy",
    }
    assert result.products["tiled-dpct-numpy"].shape == (2, 8, 8)
    assert not result.failures


def test_offline_presenter_runs_multilayer_job_with_selected_observation(
    tmp_path: Path,
) -> None:
    input_root = _bundle(tmp_path)
    presenter = build_processing_gui_presenter()
    plan = presenter.resolve(
        str(input_root),
        str(tmp_path / "multilayer products"),
        multilayer=_multilayer_request(),
    )

    assert set(plan.jobs) == {
        "mean-projection",
        "quality-metrics",
        "multilayer-multislice-numpy",
    }
    assert tuple(
        item.detector_id
        for item in plan.job_observations["multilayer-multislice-numpy"]
    ) == ("dhm",)

    result = presenter.execute(plan)

    assert result.products["multilayer-multislice-numpy"].shape == (3, 8, 8)
    assert not result.failures


def test_offline_plan_admits_independent_summary_and_tiled_gpu_leases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = resolve_offline_processing(
        OfflineProcessingRequest(
            _bundle(tmp_path),
            tmp_path / "products",
            gpu_id=0,
            tiled=replace(_tiled_request(), backend="cupy", gpu_id=1),
        )
    )
    devices = (
        GpuDeviceInfo(0, "gpu-0", 8 * 1024**3, 8 * 1024**3, "test"),
        GpuDeviceInfo(1, "gpu-1", 8 * 1024**3, 8 * 1024**3, "test"),
    )
    captured: dict[str, Any] = {}
    expected = ProcessingBatchResult({}, {}, {})

    class Supervisor:
        def __init__(self, solvers: object, **kwargs: object) -> None:
            captured.update(solvers=solvers, **kwargs)

        def start(self) -> None:
            captured["started"] = True

        def run_batch(self, requests: object) -> Any:
            captured["requests"] = requests
            return expected

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(offline_module, "discover_nvidia_gpus", lambda: devices)
    monkeypatch.setattr(offline_module, "ProcessingSupervisor", Supervisor)

    result = offline_module.run_offline_processing_plan(plan)

    assert result is expected
    assert captured["started"] and captured["closed"]
    leases = captured["gpu_leases"]
    assert set(leases) == {"gpu-mean-projection", "tiled-dpct-cupy"}
    assert leases["gpu-mean-projection"].device_id == 0
    assert leases["tiled-dpct-cupy"].device_id == 1
    cache_directories = captured["gpu_cache_directories"]
    assert set(cache_directories) == set(leases)

    conflicting = resolve_offline_processing(
        OfflineProcessingRequest(
            plan.request.input_root,
            tmp_path / "conflicting-products",
            gpu_id=0,
            tiled=replace(_tiled_request(), backend="cupy", gpu_id=0),
        )
    )
    with pytest.raises(GpuAdmissionError, match="no GPU can satisfy"):
        offline_module.run_offline_processing_plan(conflicting)


def test_processing_gui_composition_resolves_preloads_and_delegates_lazily(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    received: dict[str, object] = {}
    module = ModuleType("redsun_aht.view.processing")

    def launch_processing_gui(
        presenter: object,
        *,
        input_root: str | None,
        output_root: str | None,
        run_event_loop: bool,
    ) -> tuple[str, object]:
        received.update(
            presenter=presenter,
            input_root=input_root,
            output_root=output_root,
            run_event_loop=run_event_loop,
        )
        return "viewer", presenter

    module.launch_processing_gui = launch_processing_gui  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redsun_aht.view.processing", module)

    viewer, presenter = run_processing_gui(
        input_root=input_root,
        output_root=output_root,
        run_event_loop=False,
    )

    assert viewer == "viewer"
    assert presenter is received["presenter"]
    assert received["input_root"] == str(input_root.resolve())
    assert received["output_root"] == str(output_root.resolve())
    assert received["run_event_loop"] is False


class _Viewer:
    def __init__(self) -> None:
        self.layers: list[tuple[np.ndarray[Any, Any], str]] = []

    def add_image(self, array: np.ndarray[Any, Any], *, name: str) -> None:
        self.layers.append((array, name))


def test_qt_processing_widget_validates_copies_and_presents_without_hardware(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from qtpy import QtWidgets

    from redsun_aht.view.processing import ProcessingWidget

    application_instance = QtWidgets.QApplication.instance()
    application = (
        application_instance
        if isinstance(application_instance, QtWidgets.QApplication)
        else QtWidgets.QApplication([])
    )
    input_root = _bundle(tmp_path)
    viewer = _Viewer()
    widget = ProcessingWidget(build_processing_gui_presenter(), viewer=viewer)
    widget.input_edit.setText(str(input_root))
    widget.output_edit.setText(str(tmp_path / "gui products"))

    widget.preview_command()
    assert "process-headless" in widget.command_edit.toPlainText()
    assert widget.status_label.text().startswith("Validated 2 jobs")
    widget.copy_command()
    clipboard = application.clipboard()
    assert clipboard is not None
    assert "process-headless" in clipboard.text()

    widget.tiled_group.setChecked(True)
    widget.tile_max_shape.setText("8, 8")
    widget.tile_min_shape.setText("2, 2")
    widget.tile_alignment.setValue(2)
    widget.tile_shape.setText("6, 6")
    widget.tile_overlap.setText("2, 2")
    widget.preview_command()
    assert widget.status_label.text().startswith("Validated 3 jobs")
    assert "--tiled-model dpct" in widget.command_edit.toPlainText()
    widget.fixed_median.setChecked(True)
    widget.preview_command()
    assert "--fixed-median-from-sample" in widget.command_edit.toPlainText()
    assert not widget.background_detector.isEnabled()
    widget.tiled_gpu_check.setChecked(True)
    widget.tiled_gpu_id.setValue(1)
    widget.preview_command()
    assert "--tiled-backend cupy" in widget.command_edit.toPlainText()
    assert "--tiled-gpu-id 1" in widget.command_edit.toPlainText()

    widget.multilayer_group.setChecked(True)
    widget.multilayer_model.setCurrentText("multislice")
    widget.multilayer_depth.setValue(3)
    widget.multilayer_voxel_size.setText("0.7, 0.12, 0.11")
    widget.multilayer_iterations.setValue(1)
    widget.multilayer_step.setValue(0.00002)
    widget.preview_command()
    assert widget.status_label.text().startswith("Validated 4 jobs")
    assert "--multilayer-model multislice" in widget.command_edit.toPlainText()
    assert "--multilayer-step-size 2e-05" in widget.command_edit.toPlainText()
    widget.multilayer_gpu_check.setChecked(True)
    widget.multilayer_gpu_id.setValue(2)
    widget.preview_command()
    assert "--multilayer-backend cupy" in widget.command_edit.toPlainText()
    assert "--multilayer-gpu-id 2" in widget.command_edit.toPlainText()

    selections = iter((str(input_root), str(tmp_path / "browsed products")))
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *args: next(selections),
    )
    widget._browse(widget.input_edit, True)
    widget._browse(widget.output_edit, False)
    assert widget.input_edit.text() == str(input_root)
    assert widget.output_edit.text().endswith("browsed products")

    shown = ProcessingViewResult(
        "gui-run",
        {"mean-projection": np.ones((2, 3, 4), dtype=np.float32)},
        {"mean-projection": "file:///product.ome.zarr#0"},
        {},
    )
    widget._on_succeeded(shown)
    assert viewer.layers[0][1] == "gui-run:mean-projection"
    assert widget.status_label.text() == "Completed 1 products; 0 failures"

    widget.input_edit.clear()
    widget.preview_command()
    assert widget.status_label.text().startswith("Invalid:")
    assert not widget.command_edit.toPlainText()

    widget._on_failed("worker stopped")
    assert widget.status_label.text() == "Failed: worker stopped"
    assert widget.run_button.isEnabled()
    widget.close()
