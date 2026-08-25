from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from redsun.presenter.builtins import StoragePresenter
from redsun.storage import PATH_PROVIDER, BaseStorage, SessionPathProvider, StreamSpec

from redsun_aht.configurations import (
    build_redsun_simulation_container,
    run_redsun_simulation_container,
)
from redsun_aht.presenter import SimulationLifecyclePresenter
from redsun_aht.providers import RUN_STORAGE
from redsun_aht.storage import InMemoryStorageIO


def test_redsun_container_builds_typed_storage_wiring(tmp_path: Path) -> None:
    container = build_redsun_simulation_container(tmp_path)
    try:
        container.build()

        assert set(container.presenters) == {"storage_ctrl", "lifecycle"}
        assert container.devices == {}
        assert container.views == {}
        assert isinstance(container.storage_ctrl, StoragePresenter)
        assert isinstance(container.lifecycle, SimulationLifecyclePresenter)
        assert container.virtual_container.require(PATH_PROVIDER) is (
            container.storage_ctrl.path_provider
        )
        storage = container.virtual_container.require(RUN_STORAGE)
        assert isinstance(storage, BaseStorage)
        assert storage is container.lifecycle.storage
        assert len(container.virtual_container.connections) == 2
        assert not container.virtual_container.unconnected

        container.lifecycle.announce_started("phase1")
        assert container.storage_ctrl.path_provider().filename.startswith("phase1_")
        container.lifecycle.announce_finished()
        assert container.storage_ctrl.path_provider().filename.startswith("unknown_")
    finally:
        container.shutdown()


def test_simulation_presenter_requires_container_injection() -> None:
    presenter = SimulationLifecyclePresenter("lifecycle", {})

    with pytest.raises(RuntimeError, match="only after container build"):
        _ = presenter.storage


def test_run_factory_builds_without_gui_or_devices(tmp_path: Path) -> None:
    container = run_redsun_simulation_container(tmp_path)
    try:
        assert container.is_built
        assert container.devices == {}
    finally:
        container.shutdown()


def test_application_owned_memory_storage_uses_public_contract(tmp_path: Path) -> None:
    async def exercise_storage() -> None:
        io = InMemoryStorageIO()
        path_provider = SessionPathProvider(
            base_dir=tmp_path.resolve(),
            session="storage-contract",
        )
        storage = BaseStorage(io=io, path_provider=path_provider)
        spec = StreamSpec("frame", (2, 2), "uint16", 1)
        storage.register(spec)
        assert storage.uri_for("frame").endswith("#frame")
        resource_info = storage.resource_info_for(spec)
        assert resource_info.data_key == "frame"
        assert resource_info.shape == (2, 2)

        source = np.arange(4, dtype=np.uint16).reshape(2, 2)
        await storage.sink("frame").put(source)
        await storage.close()
        source[:] = 0

        assert len(io.stores) == 1
        store = io.stores[0]
        np.testing.assert_array_equal(
            store.arrays["frame"],
            [np.arange(4, dtype=np.uint16).reshape(2, 2)],
        )
        assert store.calls == [
            ("write", "frame"),
            ("release", "frame"),
            ("close", ""),
        ]
        assert storage.mimetype == "application/x-redsun-aht-memory"

    asyncio.run(exercise_storage())
