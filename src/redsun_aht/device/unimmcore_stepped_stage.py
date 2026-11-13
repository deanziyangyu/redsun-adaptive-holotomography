"""Optional UniMMCore adapter for the shared stepped PVA axis."""

from pymmcore_plus.experimental.unicore import StageDevice

from redsun_aht.device.pva_stepped_axis import PvaSteppedAxis


class SteppedPvaStage(PvaSteppedAxis, StageDevice):
    """Expose the shared host-stepped axis through UniMMCore."""
