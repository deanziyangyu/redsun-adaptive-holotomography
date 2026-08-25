"""Independent detector discovery and reconnect supervision."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from redsun_aht.domain import DetectorCapabilities, HardwareServiceStatus
    from redsun_aht.protocols import SupervisedDetector


class DetectorRegistry:
    """Register isolated detector services without a monolithic manager."""

    def __init__(self) -> None:
        self._services: dict[str, SupervisedDetector] = {}
        self._prefixes: set[str] = set()

    @property
    def services(self) -> Mapping[str, SupervisedDetector]:
        """Read-only registration view."""
        return dict(self._services)

    def register(self, service: SupervisedDetector) -> None:
        """Register one service while enforcing identity/prefix isolation."""
        if service.service_id in self._services:
            raise ValueError(f"duplicate detector service {service.service_id!r}")
        if service.epics_prefix in self._prefixes:
            raise ValueError(f"duplicate EPICS prefix {service.epics_prefix!r}")
        self._services[service.service_id] = service
        self._prefixes.add(service.epics_prefix)

    async def discover(self) -> dict[str, DetectorCapabilities]:
        """Discover every registered service independently."""
        return {
            service_id: await service.discover()
            for service_id, service in self._services.items()
        }

    async def connect(
        self, service_ids: Iterable[str] | None = None
    ) -> dict[str, Exception]:
        """Connect selected services, retaining per-device failures."""
        failures: dict[str, Exception] = {}
        targets = self._services if service_ids is None else service_ids
        for service_id in targets:
            service = self._services[service_id]
            try:
                await service.connect()
            except Exception as exc:
                failures[service_id] = exc
        return failures

    async def reconnect(
        self, service_id: str, *, attempts: int = 3
    ) -> HardwareServiceStatus:
        """Reconnect one failed service without disturbing its peer."""
        if attempts <= 0:
            raise ValueError("attempts must be positive")
        service = self._services[service_id]
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                await service.disconnect()
                await service.connect()
            except Exception as exc:
                last_error = exc
                continue
            return service.status
        raise ConnectionError(
            f"detector service {service_id!r} failed to reconnect"
        ) from last_error

    async def disconnect_all(self) -> None:
        """Disconnect each service and release its owned resources."""
        for service in self._services.values():
            await service.disconnect()

    def status(self) -> dict[str, HardwareServiceStatus]:
        """Return immutable health snapshots keyed by service ID."""
        return {
            service_id: service.status for service_id, service in self._services.items()
        }


__all__ = ["DetectorRegistry"]
