"""Container hook that runs a session on napari's application and viewer.

Adapted from ``redsun_searching_is_imaging/hooks.py`` at donor revision
``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed; copyright
the Redsun contributors. The hook point, the ``QApplication`` supplied by
napari, and the embedded ``ViewerModel``/``QtViewer`` pair are the donor's.

Changed for :mod:`redsun_aht`:

- The viewer is made here, once, under ``QtHook.CREATE_APPLICATION``, and kept
  on the hook. A session's image view hosts the widget this hook exposes
  instead of making a second viewer, so the application and the viewer have one
  owner each and closing the hook closes the viewer.
- The donor's dark style, scrollable docks and font-size hook are not carried
  over. A session that wants napari's stylesheet puts it on the application
  with a separate ``configure_application`` provider.
- napari is imported only when the hook runs, not when this module is imported.
  A headless process can therefore import the session module to read its
  declarations without loading napari.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QApplication

__all__ = ["NapariApplication", "current_application"]

_APPLICATION: NapariApplication | None = None
"""The hook that made the viewer of the session being built, if one did."""


class NapariApplication:
    """Hook running the session on napari's application.

    ``create_application`` returns the application napari itself would build
    and creates the viewer the session's image view hosts. A second call keeps
    the viewer it already made, so a rebuilt session does not orphan one.
    """

    def __init__(self) -> None:
        self._viewer: ViewerModel | None = None
        self._qt_viewer: Any | None = None

    def create_application(self, argv: list[str]) -> QApplication:
        """Return napari's application and make the session's viewer.

        Parameters
        ----------
        argv : list[str]
            The command line the session was started with. It is not
            forwarded: napari builds its own, so the macOS application menu
            shows the name it expects.

        Returns
        -------
        QApplication
            The running application, created by napari when none exists.
        """
        from napari._qt.qt_event_loop import get_qapp
        from napari._qt.qt_viewer import QtViewer
        from napari.components import ViewerModel

        application = cast("QApplication", get_qapp())
        if self._viewer is None:
            self._viewer = ViewerModel(
                title="aht-viewer", ndisplay=2, order=(), axis_labels=()
            )
            self._qt_viewer = QtViewer(self._viewer, show_welcome_screen=False)
        global _APPLICATION
        _APPLICATION = self
        return application

    @property
    def viewer(self) -> ViewerModel:
        """The viewer model the session's image view hosts.

        Raises
        ------
        RuntimeError
            If the hook has not created the viewer yet, or it was closed.
        """
        if self._viewer is None:
            raise RuntimeError("the napari viewer has not been created")
        return self._viewer

    @property
    def qt_viewer(self) -> Any:
        """The Qt widget that draws :attr:`viewer`.

        Raises
        ------
        RuntimeError
            If the hook has not created the viewer yet, or it was closed.
        """
        if self._qt_viewer is None:
            raise RuntimeError("the napari viewer has not been created")
        return self._qt_viewer

    def close(self) -> None:
        """Close the viewer and forget it, leaving the application alive."""
        global _APPLICATION
        viewer = self._viewer
        qt_viewer = self._qt_viewer
        self._viewer = None
        self._qt_viewer = None
        if viewer is not None:
            viewer.layers.clear()
        if qt_viewer is not None:
            qt_viewer.close()
        if _APPLICATION is self:
            _APPLICATION = None

    def shutdown(self) -> None:
        """Close the viewer when the session releases its hooks."""
        self.close()


def current_application() -> NapariApplication | None:
    """Return the hook that made the running session's viewer, if one did."""
    return _APPLICATION
