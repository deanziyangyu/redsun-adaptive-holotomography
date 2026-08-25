"""Hardware-free presenters used to verify RedSun composition contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from redsun.presenter import Presenter
from redsun.storage import PATH_PROVIDER, BaseStorage
from redsun.virtual import Signal

from redsun_aht.providers import RUN_STORAGE
from redsun_aht.storage import InMemoryStorageIO

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ophyd_async.core import Device
    from redsun.virtual import VirtualContainer


class SimulationLifecyclePresenter(Presenter):
    """Own the simulated run lifecycle and its in-memory storage instance."""

    sig_run_started = Signal(str)
    sig_run_finished = Signal()

    def __init__(
        self,
        name: str,
        devices: Mapping[str, Device],
        /,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, devices, **kwargs)
        self._io = InMemoryStorageIO()
        self._storage: BaseStorage | None = None

    @property
    def io(self) -> InMemoryStorageIO:
        """In-memory backend retained for inspection and replay tests."""
        return self._io

    @property
    def storage(self) -> BaseStorage:
        """Profile-owned storage, available after container injection."""
        if self._storage is None:
            raise RuntimeError("storage is available only after container build")
        return self._storage

    def inject_dependencies(self, container: VirtualContainer) -> None:
        """Resolve RedSun's path owner and publish one `BaseStorage`."""
        path_provider = container.require(PATH_PROVIDER)
        self._storage = BaseStorage(io=self._io, path_provider=path_provider)
        container.provide(RUN_STORAGE, self._storage)

    def announce_started(self, plan_name: str) -> None:
        """Publish the plan identity before a simulated run begins."""
        self.sig_run_started.emit(plan_name)

    def announce_finished(self) -> None:
        """Publish run completion so path ownership resets."""
        self.sig_run_finished.emit()


__all__ = ["SimulationLifecyclePresenter"]
