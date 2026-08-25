"""Optional RedSun-device compatibility seam for processing jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from redsun_aht.protocols import ProcessingFlyer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from redsun_aht.domain.models import JsonValue


@runtime_checkable
class ProcessingDeviceAdapter(ProcessingFlyer, Protocol):
    """A named processing flyer that may be composed as a RedSun device.

    This contract intentionally does not require processing implementations to
    be devices. Detached workers and supervisors remain the primary execution
    boundary; an adapter can satisfy this seam if RedSun device composition is
    beneficial later.
    """

    @property
    def name(self) -> str:
        """Stable RedSun composition name."""
        ...

    async def describe_configuration(self) -> Mapping[str, JsonValue]:
        """Describe the detached worker configuration exposed to a run."""
        ...


__all__ = ["ProcessingDeviceAdapter"]
