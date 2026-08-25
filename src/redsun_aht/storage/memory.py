"""Inspectable in-memory storage implementing RedSun's public protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from ophyd_async.core import StreamResourceInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    from ophyd_async.core import PathInfo
    from redsun.storage import OpenStore, StreamSpec


class InMemoryOpenStore:
    """Retain copied frames and lifecycle calls for deterministic tests."""

    def __init__(self, specs: Mapping[str, StreamSpec]) -> None:
        self.arrays: dict[str, list[npt.NDArray[Any]]] = {
            data_key: [] for data_key in specs
        }
        self.calls: list[tuple[str, str]] = []

    async def write(self, data_key: str, frame: npt.NDArray[Any]) -> None:
        """Copy a frame so later producer mutations cannot alter storage."""
        self.arrays[data_key].append(np.array(frame, copy=True))
        self.calls.append(("write", data_key))

    async def release(self, data_key: str) -> None:
        """Record stream release."""
        self.calls.append(("release", data_key))

    async def close(self) -> None:
        """Record store closure."""
        self.calls.append(("close", ""))


class InMemoryStorageIO:
    """Create inspectable stores through RedSun's public ``StorageIO`` API."""

    mimetype: ClassVar[str] = "application/x-redsun-aht-memory"
    extension: ClassVar[str] = ""

    def __init__(self) -> None:
        self.stores: list[InMemoryOpenStore] = []

    async def open(
        self,
        path: PathInfo,
        specs: Mapping[str, StreamSpec],
    ) -> OpenStore:
        """Open and retain a store for later inspection."""
        del path
        store = InMemoryOpenStore(specs)
        self.stores.append(store)
        return store

    def uri(self, path: PathInfo, data_key: str) -> str:
        """Return a stable URI for a stream in the simulated store."""
        directory = path.directory_path.as_posix()
        return f"memory://{directory}/{path.filename}#{data_key}"

    def resource_info(self, spec: StreamSpec) -> StreamResourceInfo:
        """Describe one frame stream for event-model resource documents."""
        return StreamResourceInfo(
            data_key=spec.data_key,
            shape=spec.shape,
            chunk_shape=(1, *spec.shape),
            dtype_numpy=np.dtype(spec.dtype).str,
            parameters={},
        )


__all__ = ["InMemoryOpenStore", "InMemoryStorageIO"]
