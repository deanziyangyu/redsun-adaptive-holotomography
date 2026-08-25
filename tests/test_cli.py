from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from redsun_aht.__main__ import main
from redsun_aht.configurations import build_dpct_simulation
from redsun_aht.processing import OfflineMultiLayerJobRequest, OfflineTiledJobRequest


def test_cli_lists_profiles(capsys: object) -> None:
    assert main(["profiles"]) == 0


def test_cli_runs_hardware_free_simulation(tmp_path: Path, capsys: object) -> None:
    journal = tmp_path / "cli-events.jsonl"

    assert main(["simulate", "--journal", str(journal)]) == 0
    assert journal.read_text(encoding="utf-8").count("\n") == 2
    document_journal = journal.with_name("cli-events.documents.jsonl")
    assert document_journal.read_text(encoding="utf-8").count("\n") == 4
    assert journal.with_name("cli-events.manifest.json").is_file()


def test_cli_runs_dual_detector_simulation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["simulate-detectors"]) == 0
    captured = capsys.readouterr()
    assert "fluorescence#0" in captured.out
    assert "dhm#0" in captured.out


def test_cli_runs_dpct_simulation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate-dpct"]) == 0
    assert "completed 4 DPCT shots from 2 detectors" in capsys.readouterr().out


def test_cli_persists_dpct_simulation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "dpct-output"

    assert main(["simulate-dpct", "--output", str(output)]) == 0

    captured = capsys.readouterr().out
    assert "completed 4 DPCT shots from 2 detectors" in captured
    assert output.resolve().as_uri() in captured
    assert (output / "acquisition-manifest.json").is_file()


def test_cli_creates_four_pattern_quantitative_reference_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "four-pattern-output"

    assert (
        main(
            [
                "simulate-dpct",
                "--output",
                str(output),
                "--pattern-id",
                "10",
                "--pattern-id",
                "11",
                "--pattern-id",
                "12",
                "--pattern-id",
                "13",
            ]
        )
        == 0
    )

    assert "completed 8 DPCT shots" in capsys.readouterr().out


def test_cli_replays_two_processing_flyers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = tmp_path / "flyer-input"
    output_root = tmp_path / "flyer-output"
    asyncio.run(build_dpct_simulation("cli-flyer", output_root=input_root).run())

    assert (
        main(
            [
                "process-flyer-replay",
                "--input",
                str(input_root),
                "--output",
                str(output_root),
            ]
        )
        == 0
    )

    captured = capsys.readouterr().out
    assert "mean-projection: computed" in captured
    assert "quality-metrics: computed" in captured
    assert "journaled" in captured
    assert (output_root / "processing-documents.jsonl").is_file()


def test_cli_registers_completed_flyer_replay_with_tiled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = SimpleNamespace(uri="file:///product.ome.zarr#0")
    result = SimpleNamespace(cached=False, output=output)
    replay = SimpleNamespace(
        results={"mean-projection": result},
        failures={},
        documents=(object(),),
        journal_path=Path("processing-documents.jsonl"),
    )
    client = object()

    async def register(
        selected_client: object,
        journal_path: Path,
        results: Any,
    ) -> SimpleNamespace:
        assert selected_client is client
        assert journal_path == replay.journal_path
        assert results is replay.results
        return SimpleNamespace(
            catalog_run_uid="catalog-uid",
            asset_paths={"raw": "raw", "mean-projection": "mean"},
        )

    monkeypatch.setattr(
        "redsun_aht.__main__.run_processing_flyer_replay",
        lambda _input, _output: replay,
    )
    monkeypatch.setattr("redsun_aht.__main__.connect_tiled", lambda _uri: client)
    monkeypatch.setattr("redsun_aht.__main__.register_processing_run", register)

    assert (
        main(
            [
                "process-flyer-replay",
                "--input",
                "input",
                "--output",
                "output",
                "--tiled-uri",
                "http://localhost:8000",
            ]
        )
        == 0
    )
    assert "registered Tiled run catalog-uid with 2 assets" in capsys.readouterr().out


def test_cli_launches_hardware_free_processing_gui(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    received: dict[str, Path | None] = {}

    def launch(*, input_root: Path | None, output_root: Path | None) -> None:
        received["input"] = input_root
        received["output"] = output_root

    monkeypatch.setattr("redsun_aht.__main__.run_processing_gui", launch)

    assert (
        main(
            [
                "process-gui",
                "--input",
                str(input_root),
                "--output",
                str(output_root),
            ]
        )
        == 0
    )
    assert received == {"input": input_root, "output": output_root}


def test_cli_builds_typed_tiled_quantitative_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    received: dict[str, object] = {}

    def run(input_path: Path, output_path: Path, **kwargs: object) -> SimpleNamespace:
        received.update(input=input_path, output=output_path, **kwargs)
        return SimpleNamespace(results={}, failures={})

    monkeypatch.setattr("redsun_aht.__main__.run_processing_simulation", run)

    assert (
        main(
            [
                "process-headless",
                "--input",
                str(input_root),
                "--output",
                str(tmp_path / "output"),
                "--tiled-model",
                "qobt",
                "--tiled-detector",
                "dhm",
                "--tiled-backend",
                "cupy",
                "--tiled-gpu-id",
                "1",
                "--tiled-gpu-reservation-mib",
                "384",
                "--wavelength-um",
                "0.660",
                "--numerical-aperture",
                "0.8",
                "--pixel-size-um",
                "0.108333",
                "--pixel-size-z-um",
                "1.0",
                "--source-azimuth-deg",
                "45",
                "315",
                "225",
                "135",
                "--tile-shape-yx",
                "512",
                "512",
                "--tile-overlap-yx",
                "96",
                "96",
                "--fixed-median-from-sample",
            ]
        )
        == 0
    )
    tiled = cast("OfflineTiledJobRequest", received["tiled"])
    assert tiled.optical.model == "qobt"
    assert tiled.tiling.explicit_shape_yx == (512, 512)
    assert tiled.backend == "cupy"
    assert tiled.gpu_id == 1
    assert tiled.gpu_reservation_bytes == 384 * 1024 * 1024
    assert tiled.fixed_median_from_sample


def test_cli_rejects_incomplete_tiled_request(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()

    assert (
        main(
            [
                "process-headless",
                "--input",
                str(input_root),
                "--output",
                str(tmp_path / "output"),
                "--tiled-model",
                "dpct",
            ]
        )
        == 2
    )
    assert "--tiled-detector" in capsys.readouterr().err


def test_cli_builds_typed_multilayer_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    received: dict[str, object] = {}

    def run(input_path: Path, output_path: Path, **kwargs: object) -> SimpleNamespace:
        received.update(input=input_path, output=output_path, **kwargs)
        return SimpleNamespace(results={}, failures={})

    monkeypatch.setattr("redsun_aht.__main__.run_processing_simulation", run)
    assert (
        main(
            [
                "process-headless",
                "--input",
                str(input_root),
                "--output",
                str(tmp_path / "output"),
                "--multilayer-model",
                "multislice",
                "--multilayer-detector",
                "dhm",
                "--multilayer-backend",
                "cupy",
                "--multilayer-gpu-id",
                "1",
                "--multilayer-gpu-reservation-mib",
                "768",
                "--multilayer-depth-layers",
                "32",
                "--multilayer-voxel-size-zyx-um",
                "0.25",
                "0.108333",
                "0.108333",
                "--multilayer-wavelength-um",
                "0.625",
                "--multilayer-na",
                "0.8",
                "--multilayer-illumination-na",
                "0.45",
                "--multilayer-max-iterations",
                "3",
                "--multilayer-step-size",
                "0.0002",
                "--multilayer-tv-weight",
                "0.00001",
            ]
        )
        == 0
    )
    request = cast("OfflineMultiLayerJobRequest", received["multilayer"])
    assert request.reconstruction.model == "multislice"
    assert request.reconstruction.depth_layers == 32
    assert request.reconstruction.solve.max_iterations == 3
    assert request.reconstruction.solve.tv_weight == 1e-5
    assert request.backend == "cupy"
    assert request.gpu_id == 1
    assert request.gpu_reservation_bytes == 768 * 1024 * 1024


def test_cli_rejects_incomplete_multilayer_request(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    assert (
        main(
            [
                "process-headless",
                "--input",
                str(input_root),
                "--output",
                str(tmp_path / "output"),
                "--multilayer-model",
                "multi_born",
            ]
        )
        == 2
    )
    assert "--multilayer-detector" in capsys.readouterr().err


def test_cli_runs_multilayer_worker_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = tmp_path / "multilayer-input"
    output_root = tmp_path / "multilayer-output"
    asyncio.run(
        build_dpct_simulation(
            "cli-multilayer",
            output_root=input_root,
            pattern_ids=(10, 11, 12, 13),
        ).run()
    )

    assert (
        main(
            [
                "process-headless",
                "--input",
                str(input_root),
                "--output",
                str(output_root),
                "--multilayer-model",
                "multislice",
                "--multilayer-detector",
                "dhm",
                "--multilayer-depth-layers",
                "3",
                "--multilayer-voxel-size-zyx-um",
                "0.7",
                "0.12",
                "0.11",
                "--multilayer-wavelength-um",
                "0.625",
                "--multilayer-na",
                "0.8",
                "--multilayer-illumination-na",
                "0.45",
                "--multilayer-max-iterations",
                "1",
                "--multilayer-step-size",
                "0.00002",
            ]
        )
        == 0
    )
    captured = capsys.readouterr().out
    assert "multilayer-multislice-numpy: computed" in captured
    assert (
        output_root / "multilayer-multislice-numpy" / "result-manifest.json"
    ).is_file()
