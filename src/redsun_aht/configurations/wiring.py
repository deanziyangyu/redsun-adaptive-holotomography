"""Links that compose the camera session, and its bounded preview buffers.

Modelled on ``_wiring.py`` from the IX83 imaging bundle at donor revision
``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed; copyright
the Redsun contributors. Each helper takes the components it links, not the
session, and yields each link for the session to make.

Changed for :mod:`redsun_aht`:

- Only what exists in this bundle is wired: the detector presenter, its
  settings view, the acquisition presenter and page, a bounded preview store
  and the plan locks. The donor's median, planner and RTASPT helpers have no
  component here yet.
- :func:`wire_acquisition` pairs the acquisition page with its presenter over
  :func:`redsun.links_between`, so the page's Run, Stop and parameter signals
  reach the presenter and the presenter's start, progress and outcome signals
  reach the page. :func:`wire_locks` forwards ``sig_locks_changed`` to a view
  whose ``set_locked`` disables the settings of the devices a run holds.
- The donor's ``ImageView`` carried the ROI box and drew the frames itself.
  Here :class:`redsun_aht.view.image.ImageView` owns the layers and the
  selection boxes, and :class:`DetectorPreview` keeps only the one-slot
  buffers. :class:`DetectorPreviewSource` adapts one detector's buffer to the
  source a :class:`redsun_aht.view.preview_driver.PreviewDriver` samples, so
  the session attaches one controller and one driver per detector. A lost
  preview never loses a stored frame, because the run is fed by Event
  documents rather than by these buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from redsun import links_between, slot

from ..view.preview import LatestPreview

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from bluesky.protocols import Reading
    from redsun import Link

    from ..presenter.acquisition import AcquisitionPresenter
    from ..presenter.detector import DetectorPresenter
    from ..view.acquisition import AcquisitionView
    from ..view.detector import DetectorView
    from ..view.image import ImageView
    from ..view.preview import PreviewFrame

__all__ = [
    "DetectorPreview",
    "DetectorPreviewSource",
    "HasLocked",
    "wire_acquisition",
    "wire_detector",
    "wire_locks",
    "wire_preview",
]


class HasLocked(Protocol):
    """A component told which devices a running plan holds."""

    def set_locked(self, names: frozenset[str]) -> None:
        """Disable the controls of *names*, and enable the rest."""
        ...


def wire_acquisition(
    ctrl: AcquisitionPresenter, view: AcquisitionView
) -> Iterator[Link]:
    """Link the acquisition presenter and its page in both directions.

    The page's launch, pause, resume, stop, action, base-directory and label
    signals reach the presenter's slots, and the presenter's started, done,
    stopped, failed, progress, action and base-directory signals reach the
    page's. Both sides name them through ``@slot``, so no pair is listed here.

    Parameters
    ----------
    ctrl : AcquisitionPresenter
        The presenter that gathers the plans and runs them.
    view : AcquisitionView
        The page that offers their controls and shows their outcome.

    Yields
    ------
    Link
        Each signal and the slot it reaches.
    """
    yield from links_between(view, ctrl)


def wire_locks(ctrl: AcquisitionPresenter, *holders: HasLocked) -> Iterator[Link]:
    """Tell each component which devices a running plan holds.

    A view disables their controls; a presenter refuses to move them by hand.

    Parameters
    ----------
    ctrl : AcquisitionPresenter
        The presenter whose engine reports the locks.
    *holders : HasLocked
        The components told the held device names.

    Yields
    ------
    Link
        The lock signal and each holder's slot.
    """
    for holder in holders:
        yield ctrl.sig_locks_changed, holder.set_locked


def wire_detector(
    ctrl: DetectorPresenter,
    view: DetectorView,
    image: ImageView | None = None,
) -> Iterator[Link]:
    """Link the detector presenter, its settings view and the image view.

    A property the view announces reaches the presenter as the canonical
    ``<detector>-<setting>`` name, and a value the device read back settles the
    edit in the view. A refused setting is logged by the presenter and never
    reaches the view, so the pending edit stays visible.

    When *image* is given, the applied configuration also reaches its layers,
    its selection box follows the panel, and a box dragged on the image fills
    the panel's editor.

    Parameters
    ----------
    ctrl : DetectorPresenter
        The presenter that owns the detectors and applies their settings.
    view : DetectorView
        The view that shows one settings panel per detector.
    image : ImageView | None
        The view that draws the detectors' frames, when one is declared.

    Yields
    ------
    Link
        Each signal and the slot it reaches.
    """
    yield view.sig_property_changed, ctrl.set_property
    yield ctrl.sig_new_configuration, view.on_new_configuration
    if image is not None and hasattr(image, "sig_roi_drawn"):
        yield ctrl.sig_new_configuration, image.on_new_configuration
        yield image.sig_roi_drawn, view.on_roi_drawn
        yield view.sig_roi_selection, image.set_roi_selection
        yield view.sig_roi_edited, image.set_roi_box


def wire_preview(ctrl: DetectorPresenter, preview: DetectorPreview) -> Iterator[Link]:
    """Route every live frame the presenter forwards into *preview*.

    Parameters
    ----------
    ctrl : DetectorPresenter
        The presenter forwarding each live reading.
    preview : DetectorPreview
        The one-slot buffer per detector the frames land in.

    Yields
    ------
    Link
        The presenter's frame signal and the preview's slot.
    """
    yield ctrl.sig_new_data, preview.update_layers


class DetectorPreview:
    """The newest live frame of each detector, in its own one-slot buffer.

    The presenter forwards a ``dict`` of buffer readings per Event. Only the
    frame is kept here:
    a viewer samples at its own cadence, so a slow viewer loses previews rather
    than growing without bound. Two detectors get two buffers, because a
    preview layer draws one sensor's frames.

    A page reads :meth:`latest` for the detector it wants to show and passes
    the returned :class:`~redsun_aht.view.preview.LatestPreview` to a
    :class:`~redsun_aht.view.preview_driver.PreviewDriver`.
    """

    def __init__(self) -> None:
        self._latest: dict[str, LatestPreview] = {}

    @property
    def detectors(self) -> tuple[str, ...]:
        """Return the detectors a frame has arrived for, in arrival order."""
        return tuple(self._latest)

    def latest(self, detector: str) -> LatestPreview | None:
        """Return *detector*'s buffer, or ``None`` before its first frame.

        Parameters
        ----------
        detector : str
            The detector name the presenter and the view use.

        Returns
        -------
        LatestPreview | None
            The one-slot buffer, once a frame has arrived.
        """
        return self._latest.get(detector)

    @slot
    def update_layers(self, readings: Mapping[str, Reading[Any]]) -> None:
        """Offer every ``<detector>-buffer`` reading to that detector's buffer.

        Parameters
        ----------
        readings : Mapping[str, Reading[Any]]
            One live Event's readings, as the presenter emits them. A reading
            that is not a detector buffer, or whose value is ``None``, is
            ignored.
        """
        for key, reading in readings.items():
            if not key.endswith("-buffer"):
                continue
            value: Any = reading["value"]
            if value is None:
                continue
            detector = key.removesuffix("-buffer")
            latest = self._latest.setdefault(detector, LatestPreview())
            latest.offer(value)


class DetectorPreviewSource:
    """One detector's newest frame, as a preview driver's source.

    Parameters
    ----------
    preview : DetectorPreview
        The buffers the presenter fills.
    detector : str
        The detector whose buffer is sampled.

    Notes
    -----
    The buffer does not exist until the detector's first frame arrives, so
    :meth:`snapshot` returns ``None`` until then and the driver's first useful
    tick draws that first frame.
    """

    def __init__(self, preview: DetectorPreview, detector: str) -> None:
        self._preview = preview
        self._detector = detector

    def snapshot(self) -> PreviewFrame | None:
        """Return the newest frame of the detector, or ``None`` before one."""
        latest = self._preview.latest(self._detector)
        return None if latest is None else latest.snapshot()


def wire_led_patterns(presenter, view):
    """Initialize the plot and connect the file/selection actions."""
    selected = None
    errors = []
    for filename in presenter.sources:
        try:
            selected = presenter.read(filename)
        except (OSError, ValueError) as error:
            errors.append(f"{filename}: {error}")
    names, panel, label, weights = (presenter.contents(selected) if selected is not None
                                    else presenter.contents())
    view.set_patterns(names, label)
    view.set_pattern(panel, label, weights)
    if errors:
        view.show_error("\n".join(errors))
    yield view.sig_load, presenter.load
    yield view.sig_select, presenter.select
    yield presenter.sig_patterns, view.set_patterns
    yield presenter.sig_pattern, view.set_pattern
    yield presenter.sig_error, view.show_error
