"""Tests for the Qt-free napari preview layer adapter.

No napari viewer is constructed: the controller talks to the three-method
protocol the donor defined, and a small in-memory double records what a real
layer list would have seen. The latest-frame buffer is checked for its
one-frame bound, which is what keeps a slow preview from growing without limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from redsun_aht.view.preview import (
    LatestPreview,
    PreviewFrame,
    PreviewImage,
    PreviewLayerController,
    PreviewViewer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _Layer:
    """Record what a napari image layer would have shown."""

    def __init__(self, data: NDArray[Any]) -> None:
        self.data = data
        self.refreshes = 0

    def refresh(self) -> None:
        self.refreshes += 1


class _Layers:
    """A layer list with the three operations the controller uses."""

    def __init__(self) -> None:
        self.items: dict[str, _Layer] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.items

    def __getitem__(self, name: str) -> _Layer:
        return self.items[name]

    def remove(self, name: str) -> None:
        del self.items[name]


class _Viewer:
    """A viewer double that records the layers added to it."""

    def __init__(self) -> None:
        self._layers = _Layers()

    @property
    def layers(self) -> _Layers:
        return self._layers

    def add_image(self, data: NDArray[Any], *, name: str) -> _Layer:
        layer = _Layer(data)
        self._layers.items[name] = layer
        return layer


def _frame(sequence: int, value: float) -> PreviewFrame:
    return PreviewFrame(
        image=np.full((2, 2), value, dtype=np.float32), sequence=sequence
    )


def test_the_controller_creates_updates_and_removes_a_layer() -> None:
    """Draw the first frame, skip a repeated sequence, redraw a new one, close."""
    viewer = _Viewer()
    controller = PreviewLayerController(viewer, name="preview")
    assert isinstance(_Viewer(), PreviewViewer)
    assert isinstance(_frame(0, 0.0), PreviewImage)

    assert controller.refresh() is False
    assert "preview" not in viewer.layers

    assert controller.refresh(_frame(1, 1.0)) is True
    layer = viewer.layers["preview"]
    np.testing.assert_array_equal(layer.data, np.ones((2, 2), dtype=np.float32))
    assert layer.refreshes == 0

    # The same sequence is a no-op: the viewer's own redraw is what happens.
    assert controller.refresh(_frame(1, 1.0)) is False
    assert layer.refreshes == 0

    assert controller.refresh(_frame(2, 2.0)) is True
    np.testing.assert_array_equal(layer.data, np.full((2, 2), 2.0, dtype=np.float32))
    assert layer.refreshes == 1

    controller.close()
    assert "preview" not in viewer.layers
    controller.close()


def test_a_controller_needs_a_layer_name() -> None:
    """Refuse an empty layer name rather than owning an unnamed layer."""
    with pytest.raises(ValueError, match="name must be nonempty"):
        PreviewLayerController(_Viewer(), name="")


def test_the_latest_preview_keeps_only_the_newest_frame() -> None:
    """Replace the held frame on every offer, so only one frame is retained."""
    preview = LatestPreview()
    assert preview.snapshot() is None
    assert preview.sequence is None

    assert preview.offer(np.zeros((2, 2), dtype=np.uint8)) == 0
    assert preview.offer(np.ones((2, 2), dtype=np.uint8)) == 1
    assert preview.offer(np.full((2, 2), 2, dtype=np.uint8)) == 2

    frame = preview.snapshot()
    assert frame is not None
    assert frame.sequence == 2
    np.testing.assert_array_equal(frame.image, np.full((2, 2), 2, dtype=np.uint8))
    assert preview.sequence == 2


def test_a_preview_sequence_must_be_a_non_negative_integer() -> None:
    """Refuse a frame or an offered sequence that cannot order frames."""
    preview = LatestPreview()
    with pytest.raises(ValueError, match="non-negative integer"):
        preview.offer(np.zeros((2, 2), dtype=np.uint8), sequence=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        PreviewFrame(image=np.zeros((2, 2), dtype=np.uint8), sequence=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        PreviewFrame(image=np.zeros((2, 2), dtype=np.uint8), sequence=1.5)  # type: ignore[arg-type]


def test_the_controller_draws_the_frames_the_source_holds() -> None:
    """Sample a latest-frame source the way the Qt driver does."""
    preview = LatestPreview()
    viewer = _Viewer()
    controller = PreviewLayerController(viewer, name="live")
    preview.offer(np.full((2, 2), 4, dtype=np.uint16))

    assert controller.refresh(preview.snapshot()) is True
    layer = viewer.layers["live"]
    np.testing.assert_array_equal(layer.data, np.full((2, 2), 4, dtype=np.uint16))
    # A second sample of the same frame is skipped.
    assert controller.refresh(preview.snapshot()) is False
