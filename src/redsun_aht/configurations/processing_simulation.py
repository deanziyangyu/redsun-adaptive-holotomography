"""Hardware-free two-solver offline replay composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from redsun_aht.processing import (
    OfflineMultiLayerJobRequest,
    OfflineProcessingRequest,
    OfflineTiledJobRequest,
    ProcessingBatchResult,
    resolve_offline_processing,
    run_offline_processing_plan,
)

if TYPE_CHECKING:
    from pathlib import Path


def run_processing_simulation(
    input_root: Path,
    output_root: Path,
    *,
    gpu_id: int | None = None,
    gpu_reservation_bytes: int = 256 * 1024 * 1024,
    tiled: OfflineTiledJobRequest | None = None,
    multilayer: OfflineMultiLayerJobRequest | None = None,
) -> ProcessingBatchResult:
    """Verify a DPCT or single-camera bundle and run two reference pipelines."""
    plan = resolve_offline_processing(
        OfflineProcessingRequest(
            input_root,
            output_root,
            gpu_id=gpu_id,
            gpu_reservation_bytes=gpu_reservation_bytes,
            tiled=tiled,
            multilayer=multilayer,
        )
    )
    return run_offline_processing_plan(plan)


__all__ = ["run_offline_processing_plan", "run_processing_simulation"]
