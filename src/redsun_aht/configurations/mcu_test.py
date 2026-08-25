"""Inspectable human-readable MCU v1 simulator composition."""

from __future__ import annotations

from dataclasses import dataclass

from redsun_aht.device.mcu import McuCommandConsole, McuHostClient
from redsun_aht.mcu import Capabilities, SimulatedMcu, SimulatedMcuTransport


@dataclass(frozen=True, slots=True)
class SimulatedMcuTestSession:
    """Readable console plus its inspectable deterministic endpoint."""

    console: McuCommandConsole
    endpoint: SimulatedMcu


def build_simulated_mcu_test_session(
    *,
    led_count: int = 8,
    panel_revision: int = 1,
) -> SimulatedMcuTestSession:
    """Build a human-readable console over the native-v1 simulator."""
    if not 1 <= led_count <= 0xFFFF:
        raise ValueError("simulated MCU LED count must be in the range 1..65535")
    if not 0 <= panel_revision <= 0xFFFF:
        raise ValueError("simulated panel revision must be in the range 0..65535")
    capabilities = Capabilities(
        firmware_version="1.0.0-sim",
        target="simulator",
        panel_profile=f"aht-{led_count}",
        panel_revision=panel_revision,
        led_count=led_count,
        max_pattern_bytes=led_count,
        max_sequence_steps=256,
    )
    endpoint = SimulatedMcu(capabilities)
    host = McuHostClient(SimulatedMcuTransport(endpoint))
    return SimulatedMcuTestSession(McuCommandConsole(host), endpoint)


__all__ = ["SimulatedMcuTestSession", "build_simulated_mcu_test_session"]
