"""Separated, composable execution-profile declarations."""

from __future__ import annotations

from dataclasses import dataclass

from redsun_aht.domain import BackendKind


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """Per-device backend selection resolved before construction."""

    device_id: str
    backend: BackendKind
    profile: str | None = None


@dataclass(frozen=True, slots=True)
class ApplicationProfile:
    """Top-level execution profile without hidden hardware construction."""

    name: str
    description: str
    constructs_hardware: bool
    uses_gui: bool
    devices: tuple[DeviceSelection, ...] = ()


PROFILES = {
    "simulate": ApplicationProfile(
        name="simulate",
        description="hardware-free deterministic lifecycle and replay",
        constructs_hardware=False,
        uses_gui=False,
        devices=(DeviceSelection("detector", BackendKind.MOCK),),
    ),
    "detector-simulate": ApplicationProfile(
        name="detector-simulate",
        description="two isolated mock detector services and shared-memory rings",
        constructs_hardware=False,
        uses_gui=False,
        devices=(
            DeviceSelection("fluorescence", BackendKind.MOCK),
            DeviceSelection("dhm", BackendKind.MOCK),
        ),
    ),
    "process-headless": ApplicationProfile(
        name="process-headless",
        description="hardware-free detached processing CLI",
        constructs_hardware=False,
        uses_gui=False,
        devices=(),
    ),
    "process-gui": ApplicationProfile(
        name="process-gui",
        description="hardware-free Qt/Napari processing UI",
        constructs_hardware=False,
        uses_gui=True,
        devices=(),
    ),
}
