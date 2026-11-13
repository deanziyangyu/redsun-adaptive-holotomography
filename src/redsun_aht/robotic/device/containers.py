"""Device containers that report what they hold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from ophyd_async.core import AsyncReadable, Device, DeviceMap, merge_gathered_dicts

if TYPE_CHECKING:
    from bluesky.protocols import Reading
    from event_model import DataKey

DeviceT = TypeVar("DeviceT", bound=Device)

# TODO: this belongs in ophyd-async, on DeviceMap and DeviceVector themselves,
# and the class goes once it lands there. `StandardReadable.add_readables`
# registers a container by testing the container for AsyncReadable, which
# neither implements, so an annotated container reports nothing; a container a
# connector fills at connect has no other registration point, since its entries
# arrive after __init__. Second choice is redsun, if more than one bundle needs
# it before ophyd-async takes it.


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
