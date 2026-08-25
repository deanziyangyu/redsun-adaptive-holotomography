"""Inspectable hardware-free RedSun 0.11 container composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from redsun.containers import AppContainer, declare_presenter
from redsun.presenter.builtins import StoragePresenter

from redsun_aht.presenter import SimulationLifecyclePresenter

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Protocol, Self

    from ophyd_async.core import Device
    from redsun.presenter import PPresenter
    from redsun.view import PView
    from redsun.virtual import VirtualContainer

    class RedSunSimulationContainer(Protocol):
        """Typed public surface of the locally declared RedSun container."""

        storage_ctrl: StoragePresenter
        lifecycle: SimulationLifecyclePresenter

        @property
        def is_built(self) -> bool: ...

        @property
        def devices(self) -> Mapping[str, Device]: ...

        @property
        def presenters(self) -> Mapping[str, PPresenter]: ...

        @property
        def views(self) -> Mapping[str, PView]: ...

        @property
        def virtual_container(self) -> VirtualContainer: ...

        def build(self) -> Self: ...

        def shutdown(self) -> None: ...


def wire_storage_lifecycle(
    container: AppContainer,
    lifecycle: SimulationLifecyclePresenter,
    storage: StoragePresenter,
) -> None:
    """Route typed lifecycle ports to RedSun's path owner."""
    container.connect(lifecycle.sig_run_started, storage.set_plan)
    container.connect(lifecycle.sig_run_finished, storage.reset_plan)


def build_redsun_simulation_container(base_dir: Path) -> RedSunSimulationContainer:
    """Return an unbuilt, GUI-free RedSun container with no devices."""
    resolved_base_dir = base_dir.resolve()

    class AHTSimulationContainer(AppContainer):
        storage_ctrl = declare_presenter(
            StoragePresenter, base_dir=str(resolved_base_dir)
        )
        lifecycle = declare_presenter(SimulationLifecyclePresenter)

        def wire(self) -> None:
            wire_storage_lifecycle(self, self.lifecycle, self.storage_ctrl)

    return AHTSimulationContainer(session="aht-simulation")


def run_redsun_simulation_container(base_dir: Path) -> RedSunSimulationContainer:
    """Build the hardware-free RedSun container without entering a GUI loop."""
    return build_redsun_simulation_container(base_dir).build()


__all__ = [
    "build_redsun_simulation_container",
    "run_redsun_simulation_container",
    "wire_storage_lifecycle",
]
