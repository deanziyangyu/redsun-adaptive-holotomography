"""Hardware-free processing presenter shared by CLI previews and Qt views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import dependency_injector.providers as dip
from redsun.presenter import Presenter

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
    from ophyd_async.core import Device
    from redsun.virtual import ProviderKey, VirtualContainer

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


class OfflineProcessingPresenter(Presenter):
    """Resolve, preview, execute, and verify processing without hardware."""

    def __init__(
        self,
        name: str = "offline-processing",
        devices: Mapping[str, Device] | None = None,
        /,
        *,
        executable: str = "aht",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, {} if devices is None else devices, **kwargs)
        self.executable = executable
        self.last_plan: OfflineProcessingPlan | None = None
        self.last_result: ProcessingViewResult | None = None

    def register_providers(self, container: VirtualContainer) -> None:
        """Publish this hardware-free processing boundary to RedSun views."""
        container.provide(OFFLINE_PROCESSING_PRESENTER, self)

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
        batch = self.execute_batch(plan)
        result = self._to_view_result(plan, batch)
        self.last_result = result
        return result

    def execute_batch(self, plan: OfflineProcessingPlan) -> ProcessingBatchResult:
        """Execute a detached batch for a headless RedSun application."""
        batch = run_offline_processing_plan(plan)
        self.last_plan = plan
        return batch

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


OFFLINE_PROCESSING_PRESENTER: ProviderKey[OfflineProcessingPresenter] = dip.Dependency(
    instance_of=OfflineProcessingPresenter
)

__all__ = [
    "OFFLINE_PROCESSING_PRESENTER",
    "OfflineProcessingPresenter",
    "ProcessingViewResult",
]
