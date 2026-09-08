"""Hardware-free two-solver offline replay composition."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from redsun.containers import AppContainer, declare_presenter

from redsun_aht.presenter import OfflineProcessingPresenter
from redsun_aht.processing import (
    OfflineMultiLayerJobRequest,
    OfflineProcessingRequest,
    OfflineTiledJobRequest,
    ProcessingBatchResult,
    resolve_offline_processing,
)

from .profiles import profile_path

if TYPE_CHECKING:
    from pathlib import Path


def build_processing_container() -> AppContainer:
    """Build no hardware while giving RedSun the offline app lifecycle."""

    class AHTOfflineProcessingContainer(
        AppContainer, config=profile_path("process-headless")
    ):
        processing = declare_presenter(OfflineProcessingPresenter)

    return AHTOfflineProcessingContainer(
        session="AHT offline processing", frontend="headless"
    )


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
    container = build_processing_container().build()
    try:
        presenter = cast(
            "OfflineProcessingPresenter", container.presenters["processing"]
        )
        return presenter.execute_batch(plan)
    finally:
        container.shutdown()


__all__ = ["build_processing_container", "run_processing_simulation"]
