"""Hardware-free processing presenter shared by CLI previews and Qt views."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from redsun_aht.processing import (
    OfflineMultiLayerJobRequest,
    OfflineProcessingPlan,
    OfflineProcessingRequest,
    OfflineTiledJobRequest,
    read_processing_result_array,
    resolve_offline_processing,
    run_offline_processing_plan,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from redsun_aht.processing import ProcessingBatchResult


@dataclass(frozen=True, slots=True)
class ProcessingViewResult:
    """Lightweight outcome and verified arrays ready for presentation."""

    source_run_id: str
    products: Mapping[str, npt.NDArray[Any]]
    output_uris: Mapping[str, str]
    failures: Mapping[str, str]

    def __post_init__(self) -> None:
        """Detach GUI-facing mappings from mutable execution state."""
        object.__setattr__(self, "products", MappingProxyType(dict(self.products)))
        object.__setattr__(
            self, "output_uris", MappingProxyType(dict(self.output_uris))
        )
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))


@dataclass(slots=True)
class OfflineProcessingPresenter:
    """Resolve, preview, execute, and verify processing without hardware."""

    executable: str = "aht"
    last_plan: OfflineProcessingPlan | None = field(default=None, init=False)
    last_result: ProcessingViewResult | None = field(default=None, init=False)

    def resolve(
        self,
        input_text: str,
        output_text: str,
        *,
        gpu_id: int | None = None,
        gpu_reservation_mib: int = 256,
        tiled: OfflineTiledJobRequest | None = None,
        multilayer: OfflineMultiLayerJobRequest | None = None,
    ) -> OfflineProcessingPlan:
        """Validate text fields and build typed jobs before any CLI rendering."""
        if not input_text.strip() or not output_text.strip():
            raise ValueError("processing input and output paths are required")
        plan = resolve_offline_processing(
            OfflineProcessingRequest(
                Path(input_text),
                Path(output_text),
                gpu_id=gpu_id,
                gpu_reservation_bytes=gpu_reservation_mib * 1024 * 1024,
                tiled=tiled,
                multilayer=multilayer,
            )
        )
        self.last_plan = plan
        return plan

    def preview_command(
        self,
        plan: OfflineProcessingPlan,
        *,
        platform: Literal["windows", "posix"] | None = None,
    ) -> str:
        """Render a copyable command only after typed-job resolution."""
        return plan.render_command(
            executable=self.executable,
            platform=platform,
        )

    def execute(self, plan: OfflineProcessingPlan) -> ProcessingViewResult:
        """Run the resolved plan and verify every published product for display."""
        batch = run_offline_processing_plan(plan)
        result = self._to_view_result(plan, batch)
        self.last_plan = plan
        self.last_result = result
        return result

    @staticmethod
    def _to_view_result(
        plan: OfflineProcessingPlan,
        batch: ProcessingBatchResult,
    ) -> ProcessingViewResult:
        products = {
            solver_id: read_processing_result_array(result)
            for solver_id, result in batch.results.items()
        }
        return ProcessingViewResult(
            plan.source_run_id,
            products,
            {
                solver_id: result.output.uri
                for solver_id, result in batch.results.items()
            },
            batch.failures,
        )


__all__ = ["OfflineProcessingPresenter", "ProcessingViewResult"]
