"""The application's acquisition page, and the device controls it always draws.

Ported from ``redsun_searching_is_imaging/view/acquisition.py`` at donor
revision ``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed;
copyright the Redsun contributors. The widget-factory extension point, the
empty device control and the install-once rule are the donor's.

Changed for :mod:`redsun_aht`:

- The page class is :class:`AcquisitionView`, the name this bundle's session
  declares, and the notice comes from
  :mod:`redsun_aht.presenter.acquisition`. The donor's app-specific class name
  is not carried over.
- The page is the built-in acquisition view unchanged: the plan list, the
  device checkboxes, the frame-count field, the Run/Stop controls and the
  outcome lines all come from :class:`redsun.view.qt.builtins.AcquisitionView`.
  This module adds only the readable control for a device role the session does
  not declare.
- Importing this module installs the entry, as in the donor; declaring the view
  in a session is what imports it, so the entry is in place before any page is
  built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import magicgui.widgets as mgw  # noqa: TC002
from qtpy import QtWidgets as QtW
from redsun.view.qt._device_sequence_edit import DeviceSequenceEdit
from redsun.view.qt._widget_factory import WIDGET_FACTORY_MAP
from redsun.view.qt.builtins import AcquisitionView as _BuiltinAcquisitionView

from redsun_aht.presenter.acquisition import missing_device_notice

if TYPE_CHECKING:
    from redsun.presenter.plan_spec import ParamDescription

__all__ = [
    "AcquisitionView",
    "MissingDeviceEdit",
    "install_missing_device_control",
    "is_unmatched_device",
    "make_missing_device_edit",
]


class MissingDeviceEdit(DeviceSequenceEdit):
    """The device control of a role the session does not declare.

    No session device matches the parameter's annotation, so the list holds no
    checkbox at all: the widget shows the sentence naming the missing role and
    selects nothing. No device is invented, and none is pre-checked.

    It reports no problem, so *Run* stays available: the page cannot supply the
    device, and the plan that needs it names the missing role when it runs.
    Asking for "choose at least one device" would demand something the page does
    not offer.
    """

    def __init__(self, name: str, notice: str) -> None:
        """Show *notice* in the control of the device parameter *name*.

        Parameters
        ----------
        name : str
            The plan parameter the control belongs to.
        notice : str
            The sentence naming the role the session does not declare.
        """
        super().__init__(name=name, choices=[], value=[])
        self._notice = notice
        native = self.native
        native.setToolTip(notice)
        # the widget's own "0 / 0 selected" count says nothing the notice does not
        for counted in native.findChildren(QtW.QLabel):
            if counted.text().endswith(" selected"):
                counted.hide()
        label = QtW.QLabel(notice)
        label.setObjectName("missing-device-notice")
        label.setWordWrap(True)
        layout = native.layout()
        if layout is not None:
            layout.insertWidget(0, label)

    @property
    def notice(self) -> str:
        """The sentence naming the role the session does not declare."""
        return self._notice

    @property
    def problems(self) -> list[str]:
        """Report no problem: nothing on this page can supply the missing device."""
        return []


def is_unmatched_device(parameter: ParamDescription) -> bool:
    """Return true for a device parameter the session declares no candidate for.

    Parameters
    ----------
    parameter : ParamDescription
        One parameter of a plan's specification.

    Returns
    -------
    bool
        Whether the parameter names a device role and offers no device.
    """
    return parameter.device_proto is not None and parameter.choices == []


def make_missing_device_edit(parameter: ParamDescription) -> mgw.Widget:
    """Return the empty control naming the role *parameter* asks for.

    Parameters
    ----------
    parameter : ParamDescription
        A device parameter no session device matches.

    Returns
    -------
    mgw.Widget
        The control that shows the missing role instead of a list.
    """
    return MissingDeviceEdit(
        name=parameter.name, notice=missing_device_notice(parameter.annotation)
    )


def install_missing_device_control() -> None:
    """Put the app's empty-device entry first in ``WIDGET_FACTORY_MAP``, once.

    The entry has to be installed before any page is built. Calling this more
    than once adds nothing, and the entry stays first so it wins over the
    widget a parameter shape would otherwise get.
    """
    if any(factory is make_missing_device_edit for _, factory in WIDGET_FACTORY_MAP):
        return
    WIDGET_FACTORY_MAP.insert(0, (is_unmatched_device, make_missing_device_edit))


install_missing_device_control()


class AcquisitionView(_BuiltinAcquisitionView):
    """The built-in acquisition page, with the app's empty-device control installed.

    The page behaves as the framework's own does: a plan list, the selected
    plan's device checkboxes, its frame count, the Run and Stop controls and
    the outcome of the last run. Declaring this class as the session's
    acquisition page is what imports the module, and so what installs the
    control above.
    """
