"""In-memory finite frame batch; buffered pixels are not yet durable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FrameSink(Protocol):
    """Minimal sequential writer used during a single flush."""

    def append(self, frame: NDArray[Any]) -> int:
        """Persist one frame and return its index."""
        ...


class BoundedFrameBatch:
    """Own at most a fixed number of same-layout frames and flush only once."""

    def __init__(self, *, max_frames: int, max_bytes: int) -> None:
        if type(max_frames) is not int or not 1 <= max_frames <= 2048:
            raise ValueError("frame limit must be 1..2048")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("byte limit must be positive")
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self._frames: list[NDArray[Any]] = []
        self._bytes = 0
        self._layout: tuple[tuple[int, ...], np.dtype[Any]] | None = None
        self._overflow = False
        self._flush_attempted = False
        self._persisted = 0

    @property
    def buffered(self) -> int:
        """Number of owned frames retained in memory."""
        return len(self._frames)

    @property
    def overflowed(self) -> bool:
        """Whether capacity was exceeded and at least one frame was refused."""
        return self._overflow

    @property
    def persisted(self) -> int:
        """Frames acknowledged by the writer during the one flush attempt."""
        return self._persisted

    def append(self, frame: NDArray[Any]) -> None:
        """Copy one frame, latching overflow instead of dropping silently."""
        if self._flush_attempted:
            raise RuntimeError("batch already flushed or flush attempted")
        image = np.asarray(frame)
        if image.ndim != 2 or image.size == 0 or image.dtype.kind not in "uif":
            raise ValueError("numeric 2D frame required")
        layout = (image.shape, image.dtype)
        if self._layout is not None and layout != self._layout:
            raise ValueError("frame layout changed")
        if (
            self._overflow
            or len(self._frames) >= self.max_frames
            or self._bytes + image.nbytes > self.max_bytes
        ):
            self._overflow = True
            raise BufferError("frame batch capacity exceeded")
        owned = image.copy()
        owned.setflags(write=False)
        self._frames.append(owned)
        self._bytes += owned.nbytes
        self._layout = layout

    def flush(self, writer: FrameSink) -> None:
        """Write the retained prefix once; a failed write must not be retried."""
        if self._flush_attempted:
            raise RuntimeError("batch already flushed or flush attempted")
        self._flush_attempted = True
        bulk = getattr(writer, "append_batch", None)
        if (
            callable(bulk)
            and self._frames
            and getattr(writer, "supports_bulk_append", True)
        ):
            first = bulk(self._frames)
            if first != 0:
                raise RuntimeError("writer index mismatch; outcome uncertain")
            self._persisted = len(self._frames)
            return
        for index, frame in enumerate(self._frames):
            actual = writer.append(frame)
            if actual != index:
                raise RuntimeError("writer index mismatch; outcome uncertain")
            self._persisted += 1
