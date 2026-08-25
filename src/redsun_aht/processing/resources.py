"""Explicit NVIDIA GPU inventory and reservation admission."""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class GpuDiscoveryError(RuntimeError):
    """Raised when the NVIDIA inventory cannot be queried or parsed."""


class GpuAdmissionError(RuntimeError):
    """Raised when no device can satisfy a declared reservation."""


@dataclass(frozen=True, slots=True)
class GpuDeviceInfo:
    """One context-free NVIDIA inventory snapshot."""

    device_id: int
    name: str
    total_memory_bytes: int
    free_memory_bytes: int
    driver_version: str

    def __post_init__(self) -> None:
        """Validate identity and the bounded memory snapshot."""
        if self.device_id < 0 or not self.name or not self.driver_version:
            raise ValueError("GPU identity fields are invalid")
        if (
            self.total_memory_bytes <= 0
            or self.free_memory_bytes < 0
            or self.free_memory_bytes > self.total_memory_bytes
        ):
            raise ValueError("GPU memory inventory is invalid")


@dataclass(frozen=True, slots=True)
class GpuResourceRequest:
    """Placement and VRAM reservation required by one solver worker."""

    reservation_bytes: int
    allowed_device_ids: tuple[int, ...] = ()
    preferred_device_ids: tuple[int, ...] = ()
    exclusive: bool = True

    def __post_init__(self) -> None:
        """Validate placement sets and reservation size."""
        if self.reservation_bytes <= 0:
            raise ValueError("GPU reservation must be positive")
        if (
            len(set(self.allowed_device_ids)) != len(self.allowed_device_ids)
            or len(set(self.preferred_device_ids)) != len(self.preferred_device_ids)
            or any(value < 0 for value in self.allowed_device_ids)
            or any(value < 0 for value in self.preferred_device_ids)
        ):
            raise ValueError("GPU device preferences must be unique and non-negative")
        if self.allowed_device_ids and not set(self.preferred_device_ids).issubset(
            self.allowed_device_ids
        ):
            raise ValueError("preferred GPUs must be included in the allowed set")


@dataclass(frozen=True, slots=True)
class GpuLease:
    """One immutable supervisor-owned GPU placement lease."""

    lease_id: str
    solver_id: str
    device_id: int
    reservation_bytes: int
    exclusive: bool

    def __post_init__(self) -> None:
        """Validate immutable lease identity and placement."""
        if (
            not self.lease_id
            or not self.solver_id
            or self.device_id < 0
            or self.reservation_bytes <= 0
        ):
            raise ValueError("GPU lease fields are invalid")


def discover_nvidia_gpus(
    executable: str = "nvidia-smi", *, timeout_s: float = 5.0
) -> tuple[GpuDeviceInfo, ...]:
    """Query NVIDIA inventory without creating a CUDA context."""
    if not executable or timeout_s <= 0:
        raise ValueError("GPU discovery executable and timeout must be valid")
    command = [
        executable,
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        devices = tuple(
            _parse_inventory_line(line)
            for line in completed.stdout.splitlines()
            if line.strip()
        )
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise GpuDiscoveryError("NVIDIA GPU inventory query failed") from error
    if not devices or len({item.device_id for item in devices}) != len(devices):
        raise GpuDiscoveryError("NVIDIA GPU inventory is empty or ambiguous")
    return devices


def _parse_inventory_line(line: str) -> GpuDeviceInfo:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 5:
        raise ValueError("unexpected nvidia-smi inventory row")
    device_id, name, total_mib, free_mib, driver_version = fields
    mib = 1024 * 1024
    return GpuDeviceInfo(
        device_id=int(device_id),
        name=name,
        total_memory_bytes=int(total_mib) * mib,
        free_memory_bytes=int(free_mib) * mib,
        driver_version=driver_version,
    )


class GpuResourcePool:
    """Admit exclusive or explicitly shared leases against one inventory snapshot."""

    def __init__(
        self, devices: Sequence[GpuDeviceInfo], *, safety_margin_fraction: float = 0.1
    ) -> None:
        if not devices or len({item.device_id for item in devices}) != len(devices):
            raise ValueError("GPU resource pool requires unique devices")
        if not 0 <= safety_margin_fraction < 1:
            raise ValueError("GPU safety margin must be in [0, 1)")
        self._devices = {item.device_id: item for item in devices}
        self._safety_margin_fraction = safety_margin_fraction
        self._leases: dict[str, GpuLease] = {}

    @property
    def devices(self) -> Mapping[int, GpuDeviceInfo]:
        """Return an immutable inventory view."""
        return MappingProxyType(self._devices)

    @property
    def leases(self) -> Mapping[str, GpuLease]:
        """Return an immutable active-lease view."""
        return MappingProxyType(self._leases)

    def reserve(self, solver_id: str, request: GpuResourceRequest) -> GpuLease:
        """Select a deterministic eligible device and reserve declared VRAM."""
        if not solver_id:
            raise ValueError("GPU reservation solver identity cannot be empty")
        allowed = (
            set(request.allowed_device_ids)
            if request.allowed_device_ids
            else set(self._devices)
        )
        unknown = allowed.difference(self._devices)
        if unknown:
            raise GpuAdmissionError(f"unknown allowed GPU IDs: {sorted(unknown)}")
        order = [
            *request.preferred_device_ids,
            *(
                device_id
                for device_id in sorted(allowed)
                if device_id not in request.preferred_device_ids
            ),
        ]
        for device_id in order:
            existing = [
                lease for lease in self._leases.values() if lease.device_id == device_id
            ]
            if existing and (
                request.exclusive or any(lease.exclusive for lease in existing)
            ):
                continue
            device = self._devices[device_id]
            usable_total = int(
                device.total_memory_bytes * (1 - self._safety_margin_fraction)
            )
            already_reserved = sum(lease.reservation_bytes for lease in existing)
            available = min(device.free_memory_bytes, usable_total) - already_reserved
            if request.reservation_bytes > available:
                continue
            lease = GpuLease(
                lease_id=uuid.uuid4().hex,
                solver_id=solver_id,
                device_id=device_id,
                reservation_bytes=request.reservation_bytes,
                exclusive=request.exclusive,
            )
            self._leases[lease.lease_id] = lease
            return lease
        raise GpuAdmissionError(
            "no GPU can satisfy "
            f"{request.reservation_bytes} reserved bytes for {solver_id}"
        )

    def release(self, lease: GpuLease) -> None:
        """Release one lease idempotently while rejecting identity reuse."""
        current = self._leases.get(lease.lease_id)
        if current is None:
            return
        if current != lease:
            raise ValueError("GPU lease identity does not match active reservation")
        del self._leases[lease.lease_id]


__all__ = [
    "GpuAdmissionError",
    "GpuDeviceInfo",
    "GpuDiscoveryError",
    "GpuLease",
    "GpuResourcePool",
    "GpuResourceRequest",
    "discover_nvidia_gpus",
]
