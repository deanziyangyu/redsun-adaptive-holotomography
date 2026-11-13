"""Device containers that report what they hold.

Ported from ``redsun_searching_is_imaging/device/containers.py`` at donor
revision ``3a3bf06`` (``redsun-searching-is-imaging``).

Changed: the container lives under ``device/mmcore`` rather than in a
bundle-wide ``device`` package, because the camera's property map and the
stage's axis map are its only users in this package for now. Behaviour is the
donor's, unchanged; it belongs in ophyd-async, and the class goes once the
container reports itself there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from ophyd_async.core import AsyncReadable, Device, DeviceMap, merge_gathered_dicts

if TYPE_CHECKING:
    from bluesky.protocols import Reading
    from event_model import DataKey

DeviceT = TypeVar("DeviceT", bound=Device)


class ReadableDeviceMap(DeviceMap[DeviceT], AsyncReadable):
    """A ``DeviceMap`` whose readable entries answer ``read`` and ``describe``.

    Entries are gathered when a verb runs, so a map a connector fills at
    connect reports what it holds. A plain ``DeviceMap`` implements neither
    verb, so a ``StandardReadable`` registering one reports nothing.
    """

    def _readable(self) -> list[AsyncReadable]:
        """Return the entries that can be read."""
        return [
            child for _, child in self.children() if isinstance(child, AsyncReadable)
        ]

    async def read(self) -> dict[str, Reading[Any]]:
        """Return the reading of every readable entry."""
        return await merge_gathered_dicts([child.read() for child in self._readable()])

    async def describe(self) -> dict[str, DataKey]:
        """Return the data key of every readable entry."""
        return await merge_gathered_dicts(
            [child.describe() for child in self._readable()]
        )


__all__ = ["ReadableDeviceMap"]
