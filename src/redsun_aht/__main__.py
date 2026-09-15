"""Command-line entry point for inspectable AHT workflows."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from redsun_aht.catalog import connect_tiled, register_processing_run
from redsun_aht.configurations import (
    PROFILES,
    build_simulated_mcu_test_session,
    profile_metadata,
    run_camera_acquisition_gui,
    run_dpct_simulation,
    run_dual_detector_simulation,
    run_processing_flyer_replay,
    run_processing_gui,
    run_processing_simulation,
    run_simulation,
)
from redsun_aht.device.mcu import McuCommandFailure
from redsun_aht.domain import MultiSliceAcquisitionMode
from redsun_aht.mcu import HUMAN_MCU_COMMAND_HELP, HumanMcuCommandError
from redsun_aht.processing import (
    MultiLayerReconstructionConfig,
    MultiLayerSolveConfig,
    OfflineMultiLayerJobRequest,
    OfflineTiledJobRequest,
    QuantitativeTileConfig,
    TilingConfig,
)
from redsun_aht.ri_analysis import (
    analyze_ri_tiff,
    export_analysis_bundle,
    load_config,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aht")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("profiles", help="list declared composition profiles")
    simulate = subparsers.add_parser(
        "simulate", help="run and replay a hardware-free lifecycle smoke test"
    )
    simulate.add_argument(
        "--journal",
        type=Path,
        default=Path("run-events.jsonl"),
        help="append-only event-journal path",
    )
    subparsers.add_parser(
        "simulate-detectors",
        help="acquire from two isolated hardware-free detector services",
    )
    simulate_dpct = subparsers.add_parser(
        "simulate-dpct",
        help="run a hardware-free stage/pattern/dual-detector DPCT smoke test",
    )
    simulate_dpct.add_argument(
        "--output",
        type=Path,
        help="fresh run-bundle directory for OME-Zarr persistence and replay",
    )
    simulate_dpct.add_argument(
        "--pattern-id",
        dest="pattern_ids",
        action="append",
        type=int,
        help="illumination pattern ID; repeat one to four times (default: 10, 11)",
    )
    process_headless = subparsers.add_parser(
        "process-headless",
        help="run detached offline solvers over a verified DPCT bundle",
    )
    process_headless.add_argument("--input", type=Path, required=True)
    process_headless.add_argument("--output", type=Path, required=True)
    process_headless.add_argument(
        "--gpu-id",
        type=int,
        help="use the CuPy mean-projection worker on this explicit NVIDIA GPU",
    )
    process_gui = subparsers.add_parser(
        "process-gui",
        help="launch the hardware-free Napari processing application",
    )
    process_gui.add_argument("--input", type=Path)
    process_gui.add_argument("--output", type=Path)
    camera_gui = subparsers.add_parser(
        "camera-gui", help="launch a one-frame GUI client for an existing camera IOC"
    )
    camera_gui.add_argument("--prefix", required=True)
    process_headless.add_argument(
        "--gpu-reservation-mib",
        type=int,
        default=256,
        help="exclusive VRAM reservation used for GPU admission (default: 256)",
    )
    _add_tiled_processing_arguments(process_headless)
    _add_multilayer_processing_arguments(process_headless)
    analyze_ri = subparsers.add_parser(
        "analyze-ri",
        help="analyze one donor-format uint16 ImageJ RI TIFF without hardware",
    )
    analyze_ri.add_argument("--input", type=Path, required=True)
    analyze_ri.add_argument("--output", type=Path, required=True)
    analyze_ri.add_argument(
        "--config",
        type=Path,
        help="optional strict JSON RI-analysis configuration",
    )
    export_ri = subparsers.add_parser(
        "export-ri-analysis",
        help="export verified RI-analysis TIFF and CSV inspection products",
    )
    export_ri.add_argument("--input", type=Path, required=True)
    export_ri.add_argument("--output", type=Path, required=True)
    export_ri.add_argument(
        "--format",
        dest="formats",
        choices=("tiff", "csv"),
        action="append",
        help="export format; repeat to select both (default: tiff and csv)",
    )
    process_flyer = subparsers.add_parser(
        "process-flyer-replay",
        help="replay a verified run through two Bluesky processing flyers",
    )
    process_flyer.add_argument("--input", type=Path, required=True)
    process_flyer.add_argument("--output", type=Path, required=True)
    process_flyer.add_argument(
        "--tiled-uri",
        help="register completed documents and Zarr assets with this Tiled server",
    )
    mcu_test = subparsers.add_parser(
        "mcu-test",
        help="execute human-readable MCU v1 commands against the simulator",
    )
    command_source = mcu_test.add_mutually_exclusive_group()
    command_source.add_argument(
        "--command",
        dest="mcu_commands",
        action="append",
        help="execute one command; repeat to run an ordered command list",
    )
    command_source.add_argument(
        "--script",
        type=Path,
        help="read newline-separated commands from this UTF-8 file",
    )
    mcu_test.add_argument("--led-count", type=int, default=8)
    mcu_test.add_argument("--panel-revision", type=int, default=1)
    mcu_test.add_argument(
        "--show-command-help",
        action="store_true",
        help="print the readable command grammar and exit",
    )
    return parser


def _add_tiled_processing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the explicit offline quantitative and XY-tiling controls."""
    parser.add_argument(
        "--tiled-model",
        choices=("dpct", "qobt"),
        help="append one offline tiled quantitative job",
    )
    parser.add_argument(
        "--tiled-backend",
        choices=("numpy", "cupy"),
        default="numpy",
        help="quantitative execution backend (default: numpy)",
    )
    parser.add_argument("--tiled-detector")
    parser.add_argument("--tiled-gpu-id", type=int)
    parser.add_argument(
        "--tiled-gpu-reservation-mib",
        type=int,
        default=256,
        help="exclusive tiled-worker VRAM reservation (default: 256)",
    )
    parser.add_argument("--background-detector")
    parser.add_argument("--background-z-index", type=int)
    parser.add_argument(
        "--fixed-median-from-sample",
        action="store_true",
        help="form one fixed four-shot background median over all acquired Z",
    )
    parser.add_argument("--wavelength-um", type=float)
    parser.add_argument("--numerical-aperture", type=float)
    parser.add_argument("--pixel-size-um", type=float)
    parser.add_argument("--pixel-size-z-um", type=float)
    parser.add_argument("--refractive-index-medium", type=float, default=1.0)
    parser.add_argument("--refractive-index-sample", type=float, default=1.4)
    parser.add_argument(
        "--source-azimuth-deg",
        type=float,
        nargs=4,
        metavar=("A0", "A1", "A2", "A3"),
        default=(0.0, 90.0, 180.0, 270.0),
    )
    parser.add_argument("--source-elevation-deg", type=float, default=45.0)
    parser.add_argument("--source-sigma", type=float, default=0.5)
    parser.add_argument(
        "--background-division",
        choices=("background_over_sample", "sample_over_background"),
        default="background_over_sample",
    )
    parser.add_argument("--regularization-real", type=float, default=5e-5)
    parser.add_argument("--regularization-imaginary", type=float, default=5e-5)
    parser.add_argument("--regularization-scalar", type=float, default=1e-2)
    parser.add_argument(
        "--axial-window",
        choices=("hamming", "rectangular", "none"),
        default="hamming",
    )
    parser.add_argument("--tile-mode", choices=("auto", "off"), default="auto")
    parser.add_argument("--tile-voxel-budget", type=int, default=512 * 512 * 100)
    parser.add_argument(
        "--tile-max-shape-yx", type=int, nargs=2, metavar=("Y", "X"), default=(768, 768)
    )
    parser.add_argument(
        "--tile-min-shape-yx", type=int, nargs=2, metavar=("Y", "X"), default=(256, 256)
    )
    parser.add_argument("--tile-alignment-px", type=int, default=32)
    parser.add_argument("--tile-overlap-fraction", type=float, default=0.20)
    parser.add_argument(
        "--tile-pad-mode", choices=("reflect", "edge"), default="reflect"
    )
    parser.add_argument("--tile-shape-yx", type=int, nargs=2, metavar=("Y", "X"))
    parser.add_argument("--tile-overlap-yx", type=int, nargs=2, metavar=("Y", "X"))


def _add_multilayer_processing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add explicit offline nonlinear reconstruction controls."""
    parser.add_argument(
        "--multilayer-model",
        choices=("multi_born", "multislice"),
        help="append one offline nonlinear multi-layer job",
    )
    parser.add_argument(
        "--multilayer-backend", choices=("numpy", "cupy"), default="numpy"
    )
    parser.add_argument("--multilayer-detector")
    parser.add_argument("--multilayer-background-detector")
    parser.add_argument("--multilayer-background-z-index", type=int)
    parser.add_argument("--multilayer-gpu-id", type=int)
    parser.add_argument("--multilayer-gpu-reservation-mib", type=int, default=512)
    parser.add_argument("--multilayer-depth-layers", type=int)
    parser.add_argument(
        "--multilayer-voxel-size-zyx-um",
        type=float,
        nargs=3,
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument("--multilayer-wavelength-um", type=float)
    parser.add_argument("--multilayer-na", type=float)
    parser.add_argument("--multilayer-medium-index", type=float, default=1.33)
    parser.add_argument("--multilayer-illumination-na", type=float)
    parser.add_argument(
        "--multilayer-acquisition-mode",
        choices=tuple(mode.value for mode in MultiSliceAcquisitionMode),
        default=MultiSliceAcquisitionMode.AHT_REAL_AMPLITUDE_FOCAL_STACK.value,
        help=(
            "measurement contract: AHT real-amplitude focal stack or "
            "interferometric complex field"
        ),
    )
    parser.add_argument("--multilayer-source-z-index", type=int, default=0)
    parser.add_argument(
        "--multilayer-padding-yx",
        type=int,
        nargs=2,
        metavar=("Y", "X"),
        default=(0, 0),
    )
    parser.add_argument("--multilayer-defocus-um", type=float, default=0.0)
    parser.add_argument(
        "--multilayer-focus-offset-slices", type=float, nargs="+", default=(0.0,)
    )
    parser.add_argument(
        "--multilayer-illumination-fxy",
        type=float,
        nargs=2,
        action="append",
        metavar=("FX", "FY"),
    )
    parser.add_argument("--multilayer-skip-shots", type=int, nargs="*", default=())
    parser.add_argument(
        "--multilayer-normalization",
        choices=("none", "shot_mean", "background"),
        default="shot_mean",
    )
    parser.add_argument("--multilayer-memory-budget-mib", type=int, default=512)
    parser.add_argument("--multilayer-max-iterations", type=int, default=5)
    parser.add_argument("--multilayer-step-size", type=float)
    parser.add_argument(
        "--multilayer-optimizer",
        choices=("fista", "gradient_descent"),
        default="fista",
    )
    parser.add_argument(
        "--no-multilayer-restart",
        dest="multilayer_restart",
        action="store_false",
        default=True,
    )
    parser.add_argument("--multilayer-random-order", action="store_true")
    parser.add_argument("--multilayer-seed", type=int, default=0)
    parser.add_argument(
        "--multilayer-measurement-domain",
        choices=("intensity", "amplitude", "field"),
        default="intensity",
    )
    parser.add_argument("--multilayer-l2-weight", type=float, default=0.0)
    parser.add_argument("--multilayer-tv-weight", type=float, default=0.0)
    parser.add_argument("--multilayer-tv-iterations", type=int, default=15)
    parser.add_argument("--multilayer-early-stopping-relative", type=float)
    parser.add_argument("--multilayer-subtract-first-slice-mean", action="store_true")
    parser.add_argument(
        "--no-multilayer-enforce-physical-sign",
        dest="multilayer_enforce_physical_sign",
        action="store_false",
        default=True,
    )
    parser.add_argument("--multilayer-recover-pupil", action="store_true")
    parser.add_argument("--multilayer-pupil-step-size", type=float, default=0.0)
    parser.add_argument(
        "--multilayer-pupil-update-method",
        choices=("gradient", "gauss_newton"),
        default="gradient",
    )


def _tiled_request_from_args(args: argparse.Namespace) -> OfflineTiledJobRequest | None:
    """Build validated typed quantitative intent from parsed CLI values."""
    if args.tiled_model is None:
        return None
    required = {
        "--tiled-detector": args.tiled_detector,
        "--wavelength-um": args.wavelength_um,
        "--numerical-aperture": args.numerical_aperture,
        "--pixel-size-um": args.pixel_size_um,
        "--pixel-size-z-um": args.pixel_size_z_um,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("tiled processing requires " + ", ".join(missing))
    optical = QuantitativeTileConfig(
        model=args.tiled_model,
        wavelength_um=args.wavelength_um,
        numerical_aperture=args.numerical_aperture,
        pixel_size_um=args.pixel_size_um,
        pixel_size_z_um=args.pixel_size_z_um,
        refractive_index_medium=args.refractive_index_medium,
        refractive_index_sample=args.refractive_index_sample,
        source_azimuth_deg=tuple(args.source_azimuth_deg),
        source_elevation_deg=args.source_elevation_deg,
        source_sigma=args.source_sigma,
        background_division=args.background_division,
        regularization_real=args.regularization_real,
        regularization_imaginary=args.regularization_imaginary,
        regularization_scalar=args.regularization_scalar,
        axial_window=args.axial_window,
    )
    tiling = TilingConfig(
        mode=args.tile_mode,
        voxel_budget=args.tile_voxel_budget,
        max_shape_yx=tuple(args.tile_max_shape_yx),
        min_shape_yx=tuple(args.tile_min_shape_yx),
        alignment_px=args.tile_alignment_px,
        overlap_fraction=args.tile_overlap_fraction,
        pad_mode=args.tile_pad_mode,
        explicit_shape_yx=(
            None if args.tile_shape_yx is None else tuple(args.tile_shape_yx)
        ),
        explicit_overlap_yx=(
            None if args.tile_overlap_yx is None else tuple(args.tile_overlap_yx)
        ),
    )
    return OfflineTiledJobRequest(
        args.tiled_detector,
        optical,
        tiling,
        background_detector_id=args.background_detector,
        background_z_index=args.background_z_index,
        fixed_median_from_sample=args.fixed_median_from_sample,
        backend=args.tiled_backend,
        gpu_id=args.tiled_gpu_id,
        gpu_reservation_bytes=args.tiled_gpu_reservation_mib * 1024 * 1024,
    )


def _multilayer_request_from_args(
    args: argparse.Namespace,
) -> OfflineMultiLayerJobRequest | None:
    """Build validated nonlinear reconstruction intent from CLI values."""
    if args.multilayer_model is None:
        return None
    required = {
        "--multilayer-detector": args.multilayer_detector,
        "--multilayer-depth-layers": args.multilayer_depth_layers,
        "--multilayer-voxel-size-zyx-um": args.multilayer_voxel_size_zyx_um,
        "--multilayer-wavelength-um": args.multilayer_wavelength_um,
        "--multilayer-na": args.multilayer_na,
        "--multilayer-illumination-na": args.multilayer_illumination_na,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("multi-layer processing requires " + ", ".join(missing))
    solve = MultiLayerSolveConfig(
        max_iterations=args.multilayer_max_iterations,
        step_size=args.multilayer_step_size,
        optimizer=args.multilayer_optimizer,
        restart_on_loss_increase=args.multilayer_restart,
        random_order=args.multilayer_random_order,
        seed=args.multilayer_seed,
        measurement_domain=args.multilayer_measurement_domain,
        l2_weight=args.multilayer_l2_weight,
        tv_weight=args.multilayer_tv_weight,
        tv_iterations=args.multilayer_tv_iterations,
        enforce_physical_sign=args.multilayer_enforce_physical_sign,
        recover_pupil=args.multilayer_recover_pupil,
        pupil_step_size=args.multilayer_pupil_step_size,
        pupil_update_method=args.multilayer_pupil_update_method,
        early_stopping_relative=args.multilayer_early_stopping_relative,
        subtract_first_slice_mean=args.multilayer_subtract_first_slice_mean,
    )
    reconstruction = MultiLayerReconstructionConfig(
        model=args.multilayer_model,
        depth_layers=args.multilayer_depth_layers,
        voxel_size_zyx_um=tuple(args.multilayer_voxel_size_zyx_um),
        wavelength_um=args.multilayer_wavelength_um,
        numerical_aperture=args.multilayer_na,
        refractive_index_medium=args.multilayer_medium_index,
        illumination_na=args.multilayer_illumination_na,
        acquisition_mode=MultiSliceAcquisitionMode(args.multilayer_acquisition_mode),
        source_z_index=args.multilayer_source_z_index,
        padding_yx=tuple(args.multilayer_padding_yx),
        defocus_um=args.multilayer_defocus_um,
        focus_offsets_slices=tuple(args.multilayer_focus_offset_slices),
        illumination_fxy=(
            None
            if args.multilayer_illumination_fxy is None
            else tuple(tuple(pair) for pair in args.multilayer_illumination_fxy)
        ),
        skip_shots=tuple(args.multilayer_skip_shots),
        normalization=args.multilayer_normalization,
        solve=solve,
        memory_budget_bytes=args.multilayer_memory_budget_mib * 1024 * 1024,
    )
    return OfflineMultiLayerJobRequest(
        args.multilayer_detector,
        reconstruction,
        background_detector_id=args.multilayer_background_detector,
        background_z_index=args.multilayer_background_z_index,
        backend=args.multilayer_backend,
        gpu_id=args.multilayer_gpu_id,
        gpu_reservation_bytes=args.multilayer_gpu_reservation_mib * 1024 * 1024,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AHT command line."""
    args = _build_parser().parse_args(argv)
    if args.command == "profiles":
        for profile in PROFILES.values():
            metadata = profile_metadata(profile)
            print(f"{metadata['profile']}: {metadata['description']}")
        return 0
    if args.command == "simulate":
        snapshot = run_simulation(args.journal)
        print(f"run {snapshot.run_id} replayed as {snapshot.state.value}")
        return 0
    if args.command == "simulate-detectors":
        frames = run_dual_detector_simulation()
        print(
            "acquired "
            + ", ".join(
                f"{detector_id}#{frame.sequence}"
                for detector_id, frame in frames.items()
            )
        )
        return 0
    if args.command == "simulate-dpct":
        dpct_result = run_dpct_simulation(
            output_root=args.output,
            pattern_ids=(10, 11)
            if args.pattern_ids is None
            else tuple(args.pattern_ids),
        )
        print(
            f"completed {len(dpct_result.shots)} DPCT shots from "
            f"{len(dpct_result.detector_ids)} detectors"
        )
        if dpct_result.storage_uri is not None:
            print(f"stored verified run bundle at {dpct_result.storage_uri}")
        return 0
    if args.command == "process-headless":
        try:
            tiled = _tiled_request_from_args(args)
            multilayer = _multilayer_request_from_args(args)
            processing = run_processing_simulation(
                args.input,
                args.output,
                gpu_id=args.gpu_id,
                gpu_reservation_bytes=args.gpu_reservation_mib * 1024 * 1024,
                tiled=tiled,
                multilayer=multilayer,
            )
        except ValueError as error:
            print(f"ERROR configuration: {error}", file=sys.stderr)
            return 2
        for solver_id, result in processing.results.items():
            state = "cached" if result.cached else "computed"
            print(f"{solver_id}: {state} {result.output.uri}")
        for solver_id, failure in processing.failures.items():
            print(f"{solver_id}: failed {failure}")
        return 0 if not processing.failures else 1
    if args.command == "analyze-ri":
        try:
            ri_result = analyze_ri_tiff(
                args.input, args.output, load_config(args.config)
            )
        except (OSError, ValueError) as error:
            print(f"ERROR RI analysis: {error}", file=sys.stderr)
            return 2
        print(
            f"RI analysis: {ri_result.label_count} labels, "
            f"{ri_result.grid_cell_count} grid cells at {ri_result.bundle_path}"
        )
        return 0
    if args.command == "export-ri-analysis":
        try:
            outputs = export_analysis_bundle(
                args.input,
                args.output,
                formats=tuple(args.formats or ("tiff", "csv")),
            )
        except (OSError, ValueError) as error:
            print(f"ERROR RI analysis export: {error}", file=sys.stderr)
            return 2
        print(f"exported {len(outputs)} RI analysis inspection files to {args.output}")
        return 0
    if args.command == "process-gui":
        run_processing_gui(input_root=args.input, output_root=args.output)
        return 0
    if args.command == "camera-gui":
        run_camera_acquisition_gui(args.prefix)
        return 0
    if args.command == "process-flyer-replay":
        replay = run_processing_flyer_replay(args.input, args.output)
        for solver_id, result in replay.results.items():
            state = "cached" if result.cached else "computed"
            print(f"{solver_id}: {state} {result.output.uri}")
        print(f"journaled {len(replay.documents)} Bluesky documents")
        for solver_id, failure in replay.failures.items():
            print(f"{solver_id}: failed {failure}")
        if args.tiled_uri is not None and not replay.failures:
            registration = asyncio.run(
                register_processing_run(
                    connect_tiled(args.tiled_uri),
                    replay.journal_path,
                    replay.results,
                )
            )
            print(
                f"registered Tiled run {registration.catalog_run_uid} with "
                f"{len(registration.asset_paths)} assets"
            )
        return 0 if not replay.failures else 1
    if args.command == "mcu-test":
        if args.show_command_help:
            print(HUMAN_MCU_COMMAND_HELP)
            return 0
        try:
            session = build_simulated_mcu_test_session(
                led_count=args.led_count,
                panel_revision=args.panel_revision,
            )
        except ValueError as error:
            print(f"ERROR configuration: {error}", file=sys.stderr)
            return 2
        if args.mcu_commands is not None:
            lines = args.mcu_commands
        elif args.script is not None:
            lines = args.script.read_text(encoding="utf-8").splitlines()
        else:
            lines = sys.stdin
        for line_number, line in enumerate(lines, start=1):
            try:
                command_result = session.console.execute_line(line)
            except (HumanMcuCommandError, McuCommandFailure) as error:
                print(f"ERROR line {line_number}: {error}", file=sys.stderr)
                return 1
            if command_result is not None:
                print(command_result)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
