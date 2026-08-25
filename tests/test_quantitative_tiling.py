from __future__ import annotations

import hashlib
import importlib
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from redsun_aht.processing import (
    CupyQuantitativeRuntime,
    CupyQuantitativeTileSolver,
    NumpyQuantitativeTileSolver,
    QuantitativeTileConfig,
    TiledVolumeReconstructor,
    TilingConfig,
)
from redsun_aht.processing.solvers import build_solver


class _Context:
    def __enter__(self) -> _Context:
        return self

    def __exit__(self, *_: object) -> None:
        pass


class _Stream(_Context):
    def __init__(self, *, non_blocking: bool) -> None:
        assert non_blocking

    def synchronize(self) -> None:
        pass


class _MemoryPool:
    def total_bytes(self) -> int:
        return 4096

    def used_bytes(self) -> int:
        return 2048

    def free_all_blocks(self) -> None:
        pass


class _FakeCupy:
    def __init__(self) -> None:
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: 2,
                runtimeGetVersion=lambda: 12020,
                driverGetVersion=lambda: 12040,
            ),
            Device=lambda _gpu_id: _Context(),
            Stream=_Stream,
        )
        self._pool = _MemoryPool()

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)

    def get_default_memory_pool(self) -> _MemoryPool:
        return self._pool

    @staticmethod
    def asnumpy(value: Any, *, stream: object | None = None) -> np.ndarray[Any, Any]:
        del stream
        return np.asarray(value)


def _sample() -> np.ndarray[Any, Any]:
    generator = np.random.default_rng(4)
    return (1000 + generator.normal(0, 20, (3, 4, 16, 16))).astype(np.float32)


def _dpct_config() -> QuantitativeTileConfig:
    return QuantitativeTileConfig(
        model="dpct",
        wavelength_um=0.625,
        numerical_aperture=0.8,
        pixel_size_um=0.108333,
        pixel_size_z_um=1.0,
        source_azimuth_deg=(90, 180, 270, 0),
    )


def _qobt_config() -> QuantitativeTileConfig:
    return QuantitativeTileConfig(
        model="qobt",
        wavelength_um=0.660,
        numerical_aperture=0.8,
        pixel_size_um=0.108333,
        pixel_size_z_um=1.0,
        refractive_index_medium=1.33,
        source_azimuth_deg=(45, 315, 225, 135),
        source_elevation_deg=50,
        source_sigma=0.5,
        background_division="sample_over_background",
    )


@pytest.mark.parametrize(
    ("config", "expected_hash", "minimum", "maximum", "mean"),
    [
        (
            _dpct_config(),
            "c9db373eee7a92ccb9f1b10607b303dd2b774da34ba90af562fa50e49c144139",
            0.9960982203483582,
            1.0041929483413696,
            0.9999992847442627,
        ),
        (
            _qobt_config(),
            "f7aa921e7a0f37928ff896a292370fdf89c146b3fb27b926754d7256f75731d7",
            1.3298841714859009,
            1.3300893306732178,
            1.3300000429153442,
        ),
    ],
)
def test_numpy_quantitative_adapters_match_frozen_donor_tikhonov_outputs(
    config: QuantitativeTileConfig,
    expected_hash: str,
    minimum: float,
    maximum: float,
    mean: float,
) -> None:
    solver = NumpyQuantitativeTileSolver(config)
    solver.prepare((3, 16, 16))

    result = solver.reconstruct_tile(_sample(), None)

    assert result.shape == (3, 16, 16)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
    assert hashlib.sha256(result.tobytes(order="C")).hexdigest() == expected_hash
    assert float(np.min(result)) == minimum
    assert float(np.max(result)) == maximum
    assert float(np.mean(result)) == mean
    assert solver.gpu_telemetry == {
        "backend": "numpy",
        "device_id": None,
        "allocated_memory_bytes": 0,
        "observed_memory_bytes": 0,
    }


@pytest.mark.parametrize("config", [_dpct_config(), _qobt_config()])
def test_cupy_adapter_matches_numpy_through_worker_owned_runtime_contract(
    config: QuantitativeTileConfig,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCupy()
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake)
    sample = _sample()
    cpu = NumpyQuantitativeTileSolver(config)
    cpu.prepare((3, 16, 16))
    expected = cpu.reconstruct_tile(sample, None)
    cpu.close()
    runtime = CupyQuantitativeRuntime(
        1, cache_directory=str((tmp_path / "cupy-cache").resolve())
    )
    gpu = CupyQuantitativeTileSolver(config, runtime)
    gpu.prepare((3, 16, 16))

    actual = gpu.reconstruct_tile(sample, None)

    np.testing.assert_array_equal(actual, expected)
    assert gpu.gpu_telemetry["device_id"] == 1
    assert gpu.gpu_telemetry["observed_memory_bytes"] == 2048
    gpu.close()
    gpu.close()
    runtime.close()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        gpu.prepare((3, 16, 16))


def test_cupy_quantitative_runtime_and_allowlist_reject_invalid_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CupyQuantitativeRuntime(-1)
    fake = _FakeCupy()
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake)
    with pytest.raises(ValueError, match="must be absolute"):
        CupyQuantitativeRuntime(0, cache_directory="relative")
    with pytest.raises(ValueError, match="does not exist"):
        CupyQuantitativeRuntime(2)
    with pytest.raises(ValueError, match="explicit GPU ID"):
        build_solver("tiled-dpct-cupy")
    kernel = build_solver("tiled-qobt-cupy", gpu_id=1)
    assert kernel.capabilities.requires_gpu
    assert kernel.capabilities.gpu_runtime == "cupy-cuda12x"
    assert kernel.gpu_id == 1
    kernel.close()

    def missing(_name: str) -> Any:
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="install the gpu-cupy extra"):
        CupyQuantitativeRuntime(0)


def test_quantitative_background_conventions_match_explicitly_transformed_input() -> (
    None
):
    sample = _sample()
    background = np.full(sample.shape[1:], 900, dtype=np.float32)
    background_over = NumpyQuantitativeTileSolver(_dpct_config())
    background_over.prepare((3, 16, 16))
    transformed = NumpyQuantitativeTileSolver(_dpct_config())
    transformed.prepare((3, 16, 16))

    with_background = background_over.reconstruct_tile(sample, background)
    direct = transformed.reconstruct_tile(background[None, ...] / sample, None)

    np.testing.assert_array_equal(with_background, direct)

    sample_over_config = replace(
        _qobt_config(), background_division="sample_over_background"
    )
    sample_over = NumpyQuantitativeTileSolver(sample_over_config)
    sample_over.prepare((3, 16, 16))
    transformed_qobt = NumpyQuantitativeTileSolver(sample_over_config)
    transformed_qobt.prepare((3, 16, 16))
    np.testing.assert_array_equal(
        sample_over.reconstruct_tile(sample, background),
        transformed_qobt.reconstruct_tile(sample / background[None, ...], None),
    )


def test_real_dpct_adapter_runs_through_overlapped_tiler() -> None:
    generator = np.random.default_rng(8)
    sample = (1000 + generator.normal(0, 20, (3, 4, 24, 24))).astype(np.float32)
    solver = NumpyQuantitativeTileSolver(_dpct_config())
    reconstructor = TiledVolumeReconstructor(
        solver,
        TilingConfig(
            explicit_shape_yx=(16, 16),
            explicit_overlap_yx=(8, 8),
            min_shape_yx=(8, 8),
            max_shape_yx=(32, 32),
            alignment_px=8,
        ),
    )

    result = reconstructor.run(sample)

    assert result.volume_zyx.shape == (3, 24, 24)
    assert np.all(np.isfinite(result.volume_zyx))
    assert len(result.plan.placements) == 4
    assert result.as_provenance()["gpu"]["backend"] == "numpy"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model": "bad"}, "model"),
        ({"wavelength_um": 0}, "must be positive"),
        ({"numerical_aperture": 1.5}, "cannot exceed"),
        ({"source_azimuth_deg": (0, 90, 180)}, "four finite"),
        ({"source_elevation_deg": float("nan")}, "elevation"),
        ({"background_division": "bad"}, "division convention"),
        ({"axial_window": "blackman"}, "axial window"),
    ],
)
def test_quantitative_config_rejects_invalid_scientific_inputs(
    changes: dict[str, Any], message: str
) -> None:
    values: dict[str, Any] = {
        "model": "dpct",
        "wavelength_um": 0.625,
        "numerical_aperture": 0.8,
        "pixel_size_um": 0.1,
        "pixel_size_z_um": 1.0,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        QuantitativeTileConfig(**values)


def test_quantitative_solver_rejects_invalid_lifecycle_shapes_and_division() -> None:
    solver = NumpyQuantitativeTileSolver(_dpct_config())
    with pytest.raises(ValueError, match="positive ZYX"):
        solver.prepare((0, 16, 16))
    solver.prepare((3, 16, 16))
    with pytest.raises(ValueError, match="ZSYX"):
        solver.reconstruct_tile(np.ones((4, 16, 16), dtype=np.float32), None)
    with pytest.raises(ValueError, match="wrong shot count"):
        solver.reconstruct_tile(np.ones((3, 3, 16, 16), dtype=np.float32), None)
    with pytest.raises(ValueError, match="prepared shape"):
        solver.reconstruct_tile(np.ones((2, 4, 16, 16), dtype=np.float32), None)
    with pytest.raises(ValueError, match="matching SYX"):
        solver.reconstruct_tile(
            np.ones((3, 4, 16, 16), dtype=np.float32),
            np.ones((4, 8, 8), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="zero pixels"):
        solver.reconstruct_tile(
            np.zeros((3, 4, 16, 16), dtype=np.float32),
            np.ones((4, 16, 16), dtype=np.float32),
        )
    solver.close()
    solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.prepare((3, 16, 16))


def test_qobt_rejects_source_angles_without_two_opposed_pairs() -> None:
    solver = NumpyQuantitativeTileSolver(
        replace(_qobt_config(), source_azimuth_deg=(0, 10, 20, 30))
    )
    with pytest.raises(ValueError, match="opposed pairs"):
        solver.prepare((3, 16, 16))
