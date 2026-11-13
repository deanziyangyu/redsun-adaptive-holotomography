"""Napari image-layer adapter for the GUI-free live preview.

Ported from ``redsun_searching_is_imaging/view/preview.py`` at donor revision
``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed; copyright
the Redsun contributors. The controller contract - one named layer, a repeated
sequence is a no-op, ``close`` removes the layer - is the donor's.

Changed for :mod:`redsun_aht`:

- The image record is a small value type, :class:`PreviewFrame`, and the
  controller accepts anything with ``image`` and ``sequence`` through
  :class:`PreviewImage`. The searching bundle's ``acquisition.live.PreviewImage``
  and its shared-memory ring are not part of this package, so no import of a
  capture module is needed here.
- :class:`LatestPreview` is the one-slot, thread-safe source the donor kept in
  ``acquisition.live.PreviewBuffer``. Keeping the newest frame only is what
  bounds a slow viewer's memory; a dropped preview never drops a stored frame,
  because the store is fed by the Event documents, not by this layer.
- The module is Qt-free and imports no napari: a caller passes its viewer and
  refreshes the layer on its own timer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "LatestPreview",
    "PreviewFrame",
    "PreviewImage",
    "PreviewLayer",
    "PreviewLayerController",
    "PreviewLayers",
    "PreviewViewer",
]


@runtime_checkable
class PreviewImage(Protocol):
    """The part of one preview frame this controller uses."""

    @property
    def image(self) -> NDArray[Any]:
        """The two-dimensional frame to draw."""
        ...

    @property
    def sequence(self) -> int:
        """The source's sequence for the frame; a repeat is the same sequence."""
        ...


@dataclass(frozen=True, slots=True)
class PreviewFrame:
    """One immutable preview frame, with the sequence it arrived under."""

    image: NDArray[Any]
    sequence: int

    def __post_init__(self) -> None:
        """Refuse a frame whose sequence cannot order it."""
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("a preview frame sequence must be a non-negative integer")


@runtime_checkable
class PreviewLayer(Protocol):
    """The part of a napari image layer this controller uses."""

    data: NDArray[Any]

    def refresh(self) -> None:
        """Redraw the layer."""
        ...


@runtime_checkable
class PreviewLayers(Protocol):
    """The part of a napari layer list this controller uses."""

    def __contains__(self, name: str) -> bool:
        """Return whether *name* is present."""
        ...

    def __getitem__(self, name: str) -> PreviewLayer:
        """Return the layer named *name*."""
        ...

    def remove(self, name: str) -> None:
        """Remove the layer named *name*."""
        ...


@runtime_checkable
class PreviewViewer(Protocol):
    """The part of a napari viewer this controller uses."""

    @property
    def layers(self) -> PreviewLayers:
        """The viewer's layer list."""
        ...

    def add_image(self, data: NDArray[Any], *, name: str) -> PreviewLayer:
        """Add an image layer."""
        ...


class LatestPreview:
    """A one-slot, thread-safe holder for the newest preview frame.

    A producer offers frames from whatever thread delivers them; a viewer
    samples on its own cadence. Only the newest frame is kept, so a viewer that
    falls behind loses previews, never stored frames.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: PreviewFrame | None = None
        self._sequence = -1

    @property
    def sequence(self) -> int | None:
        """Return the newest sequence, or ``None`` before the first frame."""
        with self._lock:
            return None if self._frame is None else self._frame.sequence

    def offer(self, image: NDArray[Any], *, sequence: int | None = None) -> int:
        """Replace the held frame and return the sequence it was given."""
        with self._lock:
            next_sequence = self._sequence + 1 if sequence is None else sequence
            if type(next_sequence) is not int or next_sequence < 0:
                raise ValueError("a preview sequence must be a non-negative integer")
            self._frame = PreviewFrame(image=image, sequence=next_sequence)
            self._sequence = next_sequence
            return next_sequence

    def snapshot(self) -> PreviewFrame | None:
        """Return the newest frame, or ``None`` when nothing was offered."""
        with self._lock:
            return self._frame


class PreviewLayerController:
    """Draw the newest preview image into one napari image layer.

    The controller is GUI-free and Qt-free: a view calls :meth:`refresh` on its
    own timer, and a second call with the same ``sequence`` is a no-op. It never
    reads the ring or the buffer itself, so the viewer decides the refresh rate.
    """

    def __init__(self, viewer: PreviewViewer, *, name: str = "live-preview") -> None:
        if not name:
            raise ValueError("name must be nonempty")
        self._viewer = viewer
        self._name = name
        self._sequence: int | None = None

    @property
    def name(self) -> str:
        """Layer name this controller owns."""
        return self._name

    def refresh(self, image: PreviewImage | None = None) -> bool:
        """Draw *image*; return whether the layer changed."""
        if image is None or image.sequence == self._sequence:
            return False
        layers = self._viewer.layers
        if self._name in layers:
            layer = layers[self._name]
            layer.data = image.image
            layer.refresh()
        else:
            self._viewer.add_image(image.image, name=self._name)
        self._sequence = image.sequence
        return True

    def close(self) -> None:
        """Remove the layer if it is still present."""
        if self._name in self._viewer.layers:
            self._viewer.layers.remove(self._name)
        self._sequence = None
