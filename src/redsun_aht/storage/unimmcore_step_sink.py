"""Finite Z-only UniMMCore frame sink using the shared create-only writer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from redsun_aht.storage.writer import RunFrameStore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


class UniMMCoreStepSink:
    """Persist ordered MDA frames with host-provided, unmeasured indices."""

    def __init__(self, uri: str, template: NDArray[Any], *, count: int) -> None:
        if type(count) is not int or not 1 <= count <= 32:
            raise ValueError("frame count must be 1..32")
        self.store = RunFrameStore(uri, "software_snap", template, planned_frames=count)
        self.expected = count
        self._failure: Exception | None = None
        self._closed = False

    def append(self, image: NDArray[Any], index: Mapping[str, int]) -> int:
        """Append once in Z order; a failed callback latches the sink."""
        if self._closed or self._failure is not None:
            raise RuntimeError("sink closed or uncertain")
        try:
            expected_index = self.store.frames_written
            if (
                set(index) != {"z"}
                or type(index["z"]) is not int
                or index["z"] != expected_index
            ):
                raise ValueError(
                    "MDA frame index is missing, duplicated or out of order"
                )
            return self.store.append(image)
        except Exception as error:
            self._failure = error
            raise

    def close(self) -> None:
        """Close once and refuse success for a failed or incomplete capture."""
        if self._closed:
            raise RuntimeError("sink already closed")
        self._closed = True
        self.store.close()
        if self._failure is not None:
            raise RuntimeError(
                "MDA sink failed; inspect partial store"
            ) from self._failure
        if self.store.frames_written != self.expected:
            raise RuntimeError("MDA sink incomplete; inspect partial store")
