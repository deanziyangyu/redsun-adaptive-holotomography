"""Typed PVA client for the independently owned Spinnaker service."""

from ophyd_async.core import SignalRW, SignalW

from redsun_aht.device.mmcore.camera import MMCamera


class SpinnakerCamera(MMCamera):
    """Common camera signals plus an explicit native software trigger."""

    software_trigger: SignalW[bool]
    immediate_frame_count: SignalRW[bool]
