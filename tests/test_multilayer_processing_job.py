from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr

from redsun_aht.configurations import build_dpct_simulation
from redsun_aht.domain.models import thaw_json
from redsun_aht.processing import (
    MultiLayerProcessingRequest,
    MultiLayerReconstructionConfig,
    MultiLayerSolveConfig,
    resolve_multilayer_processing,
    run_multilayer_processing_plan,
)
from redsun_aht.processing.multilayer_kernel import MultiLayerKernel
from redsun_aht.processing.solvers import ProcessingCancelled, build_solver


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "multilayer-input"
    asyncio.run(
        build_dpct_simulation(
            "multilayer-run",
            output_root=root,
            pattern_ids=(10, 11, 12, 13),
        ).run()
    )
    return root


def _reconstruction(**updates: Any) -> MultiLayerReconstructionConfig:
    base: dict[str, Any] = {
        "model": "multislice",
        "depth_layers": 3,
        "voxel_size_zyx_um": (0.7, 0.12, 0.11),
        "wavelength_um": 0.625,
        "numerical_aperture": 0.8,
        "refractive_index_medium": 1.33,
        "illumination_na": 0.45,
        "source_z_index": 0,
        "padding_yx": (1, 2),
        "defocus_um": 0.3,
        "solve": MultiLayerSolveConfig(max_iterations=1, step_size=2e-5),
    }
    base.update(updates)
    return MultiLayerReconstructionConfig(**base)


def test_multilayer_job_runs_in_spawned_worker_with_verified_provenance(
    tmp_path: Path,
) -> None:
    request = MultiLayerProcessingRequest(
        _bundle(tmp_path),
        tmp_path / "products",
        "dhm",
        _reconstruction(),
    )
    plan = resolve_multilayer_processing(request)
    kernel = build_solver(plan.job.solver_id)

    assert plan.source_run_id == "multilayer-run"
    assert plan.job.solver_id == "multilayer-multislice-numpy"
    assert plan.job.latency_class == "offline"
    assert plan.job.resource_request["working_set_estimate_bytes"] == (
        plan.working_set_estimate_bytes
    )
    assert kernel.capabilities.live_supported is False
    kernel.close()

    batch = run_multilayer_processing_plan(plan)

    assert not batch.failures
    result = batch.results[plan.job.solver_id]
    assert result.output.shape == (3, 8, 8)
    assert result.metrics["model"] == "multislice"
    assert result.metrics["backend"] == "numpy"
    assert result.metrics["iterations"] == 1
    assert isinstance(result.metrics["provenance_sha256"], str)
    group = zarr.open_group(
        Path(plan.job.output_destination) / "product.ome.zarr", mode="r"
    )
    aht = cast("dict[str, object]", group.attrs["aht"])
    provenance = cast("dict[str, object]", aht["provenance"])
    donor = cast("dict[str, object]", provenance["donor"])
    source = cast("dict[str, object]", provenance["source"])
    assert donor["commit"] == "e8dd96bfa2d54784e00209769d594f305863b01a"
    assert source["sample_detector_id"] == "dhm"
    assert provenance["scientific_status"] == "donor-characterized-cpu-reference"
    assert len(cast("list[float]", provenance["loss_history"])) == 1

    replayed = run_multilayer_processing_plan(plan)
    assert replayed.results[plan.job.solver_id].cached


def test_multilayer_background_selection_and_direct_cancellation(
    tmp_path: Path,
) -> None:
    request = MultiLayerProcessingRequest(
        _bundle(tmp_path),
        tmp_path / "products",
        "dhm",
        _reconstruction(normalization="background"),
        background_detector_id="fluorescence",
        background_z_index=1,
    )
    plan = resolve_multilayer_processing(request)
    kernel = build_solver(plan.job.solver_id)
    assert tuple(item.detector_id for item in plan.observations) == (
        "dhm",
        "fluorescence",
    )
    with pytest.raises(ProcessingCancelled, match="cancelled"):
        kernel.run(plan.job, plan.observations, lambda: True)
    kernel.close()


def test_multilayer_resolution_rejects_memory_and_selection_errors(
    tmp_path: Path,
) -> None:
    input_root = _bundle(tmp_path)
    base = MultiLayerProcessingRequest(
        input_root,
        tmp_path / "products",
        "dhm",
        _reconstruction(),
    )
    with pytest.raises(ValueError, match="inside the input"):
        replace(base, output_root=input_root / "products")
    with pytest.raises(ValueError, match="not a directory"):
        replace(base, input_root=tmp_path / "missing")
    with pytest.raises(ValueError, match="identity"):
        replace(base, detector_id="")
    with pytest.raises(ValueError, match="absent"):
        resolve_multilayer_processing(replace(base, detector_id="missing"))
    with pytest.raises(ValueError, match="out of range"):
        resolve_multilayer_processing(
            replace(
                base,
                reconstruction=replace(base.reconstruction, source_z_index=2),
            )
        )
    with pytest.raises(MemoryError, match="exceeds budget"):
        resolve_multilayer_processing(
            replace(
                base,
                reconstruction=replace(
                    base.reconstruction,
                    depth_layers=5000,
                    memory_budget_bytes=1024 * 1024,
                ),
            )
        )
    with pytest.raises(ValueError, match="requires a detector"):
        replace(
            base,
            reconstruction=replace(base.reconstruction, normalization="background"),
        )
    with pytest.raises(ValueError, match="requires background normalization"):
        replace(base, background_detector_id="fluorescence")
    with pytest.raises(ValueError, match="must differ"):
        replace(
            base,
            reconstruction=replace(base.reconstruction, normalization="background"),
            background_detector_id="dhm",
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(
            base,
            reconstruction=replace(base.reconstruction, normalization="background"),
            background_detector_id="fluorescence",
            background_z_index=-1,
        )
    with pytest.raises(ValueError, match="backend"):
        replace(base, backend="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires an explicit GPU"):
        replace(base, backend="cupy")
    with pytest.raises(ValueError, match="cannot select a GPU"):
        replace(base, gpu_id=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(base, backend="cupy", gpu_id=-1)
    with pytest.raises(ValueError, match="whole mebibytes"):
        replace(base, gpu_reservation_bytes=1024 * 1024 + 1)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model": "other"}, "model"),
        ({"depth_layers": 0}, "depth"),
        ({"source_z_index": -1}, "source Z"),
        ({"illumination_na": 2.0}, "illumination NA"),
        ({"normalization": "other"}, "normalization"),
        ({"memory_budget_bytes": 0}, "memory budget"),
        ({"memory_budget_bytes": 1024 * 1024 + 1}, "memory budget"),
    ],
)
def test_reconstruction_config_rejects_invalid_values(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _reconstruction(**updates)


def test_multilayer_kernel_rejects_changed_resource_preflight(tmp_path: Path) -> None:
    plan = resolve_multilayer_processing(
        MultiLayerProcessingRequest(
            _bundle(tmp_path),
            tmp_path / "products",
            "dhm",
            _reconstruction(),
        )
    )
    kernel = build_solver(plan.job.solver_id)
    tampered = replace(
        plan.job,
        resource_request={
            **plan.job.resource_request,
            "working_set_estimate_bytes": 1,
        },
    )
    with pytest.raises(MemoryError, match="preflight"):
        kernel.run(tampered, plan.observations, lambda: False)
    kernel.close()


def _tamper_job(plan: Any, mutate: Any) -> Any:
    configuration = thaw_json(plan.job.configuration)
    assert isinstance(configuration, dict)
    mutate(configuration)
    return replace(plan.job, configuration=configuration)


def test_multilayer_kernel_rejects_changed_configuration_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = resolve_multilayer_processing(
        MultiLayerProcessingRequest(
            _bundle(tmp_path),
            tmp_path / "products",
            "dhm",
            _reconstruction(),
        )
    )
    kernel = cast("MultiLayerKernel", build_solver(plan.job.solver_id))
    with pytest.raises(ValueError, match="sample and optional"):
        kernel.run(plan.job, (), lambda: False)

    def change_model(configuration: dict[str, Any]) -> None:
        cast("dict[str, Any]", configuration["reconstruction"])["model"] = "multi_born"

    with pytest.raises(ValueError, match="does not match worker"):
        kernel.run(_tamper_job(plan, change_model), plan.observations, lambda: False)

    def change_source_z(configuration: dict[str, Any]) -> None:
        cast("dict[str, Any]", configuration["reconstruction"])["source_z_index"] = 99

    with pytest.raises(ValueError, match="source Z index"):
        kernel.run(_tamper_job(plan, change_source_z), plan.observations, lambda: False)

    def change_normalization(configuration: dict[str, Any]) -> None:
        cast("dict[str, Any]", configuration["reconstruction"])["normalization"] = (
            "other"
        )

    with pytest.raises(ValueError, match="normalization is unsupported"):
        kernel.run(
            _tamper_job(plan, change_normalization),
            plan.observations,
            lambda: False,
        )

    def require_background(configuration: dict[str, Any]) -> None:
        cast("dict[str, Any]", configuration["reconstruction"])["normalization"] = (
            "background"
        )
        configuration["background_z_index"] = 0

    with pytest.raises(ValueError, match="requires an observation"):
        kernel.run(
            _tamper_job(plan, require_background),
            plan.observations,
            lambda: False,
        )
    with pytest.raises(ValueError, match="unused background"):
        kernel.run(
            plan.job,
            (plan.observations[0], plan.observations[0]),
            lambda: False,
        )

    def change_frequencies(configuration: dict[str, Any]) -> None:
        configuration["illumination_fxy"] = [[0.0, 0.0]]

    with pytest.raises(ValueError, match="geometry changed"):
        kernel.run(
            _tamper_job(plan, change_frequencies),
            plan.observations,
            lambda: False,
        )

    monkeypatch.setattr(
        "redsun_aht.processing.multilayer_kernel.read_observation_array",
        lambda _observation: np.zeros((4, 2, 8, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="zero mean"):
        kernel.run(plan.job, plan.observations, lambda: False)
    monkeypatch.undo()
    kernel.close()


def test_multilayer_kernel_constructor_rejects_inconsistent_placement() -> None:
    with pytest.raises(ValueError, match="requires a GPU"):
        MultiLayerKernel("multislice", backend="cupy")
    with pytest.raises(ValueError, match="cannot receive a GPU"):
        MultiLayerKernel("multislice", gpu_id=0)
