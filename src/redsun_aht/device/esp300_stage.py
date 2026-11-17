"""ESP300 positioner client with XYZ axes in a hardware-free session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing import Annotated as A

import numpy as np
from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    SignalW,
    StandardReadableFormat,
    set_mock_value,
    soft_signal_rw,
)

from redsun_aht.device.mmcore._containers import ReadableDeviceMap  # noqa: TC001
from redsun_aht.device.mmcore.stage import AXIS_UNITS, MMAxis, MMStage

if TYPE_CHECKING:
    from ophyd_async.core import DeviceMock


class Esp300Axis(MMAxis):
    """Distinguish routine run cleanup from a latched operator stop."""

    stop_active_request: SignalW[bool]

    async def stop(self, success: bool = False) -> None:
        """Stop unresolved moves on cleanup and retain explicit stop semantics."""
        if not success:
            await super().stop()
            return
        try:
            await self.stop_active_request.set(True)
        except Exception:
            await super().stop(success=True)
            raise


class Esp300Stage(MMStage):
    """Use the shared stage PV contract and declare simulated XYZ for mock mode."""

    _mock_axes = False
    axis: A[ReadableDeviceMap[Esp300Axis], StandardReadableFormat.CHILD]

    async def connect(
        self,
        mock: bool | DeviceMock[Any] = False,
        timeout: float = DEFAULT_TIMEOUT,
        force_reconnect: bool = False,
    ) -> None:
        """Populate the mock device map, which otherwise needs a live PVI tree."""
        if self._mock_axes and not mock:
            self.axis.clear()
            self._mock_axes = False
        elif mock and not self.axis:
            for name in ("x", "y", "z"):
                axis = Esp300Axis()
                axis.position = soft_signal_rw(float, 0.0, units=AXIS_UNITS)
                axis.limits = soft_signal_rw(
                    np.ndarray, np.array([-100.0, 100.0], dtype=np.float64)
                )
                axis.precision = soft_signal_rw(float, 0.1, units=AXIS_UNITS)
                axis.stop_request = soft_signal_rw(bool, False)
                axis.stop_active_request = soft_signal_rw(bool, False)
                axis.add_readables([axis.position])
                self.axis[name] = axis
            self._mock_axes = True
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        if mock:
            set_mock_value(self.provenance, "simulated")
