"""Experimental headless software-stepped stack without hardware triggers."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from redsun_aht.storage.writer import RunFrameStore

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from redsun_aht.device.pva_stepped_axis import PvaSteppedAxis
    from redsun_aht.device.software_snap import SoftwareSnapCamera


class _Plane:
    name = "software_step"

    def __init__(
        self,
        plane: int,
        target_um: float,
        position_um: float,
        frame_count: int,
        mean_dn: float,
        stored_index: int,
    ) -> None:
        self.values = {
            "plane": plane,
            "target_um": target_um,
            "controller_position_um": position_um,
            "camera_frame_count": frame_count,
            "mean_dn": mean_dn,
            "stored_index": stored_index,
        }

    def read(self) -> dict[str, dict[str, Any]]:
        now = time.time()
        return {
            key: {"value": value, "timestamp": now}
            for key, value in self.values.items()
        }

    def describe(self) -> dict[str, dict[str, Any]]:
        return {
            key: {"source": "ARH:software-step", "dtype": "number", "shape": []}
            for key in self.values
        }


def software_stepped_plan(
    stage: PvaSteppedAxis,
    camera: SoftwareSnapCamera,
    targets_um: Sequence[float],
    *,
    reservation_held: bool,
    camera_exclusive: bool,
    timeout_s: float = 15.0,
    store_uri: str | None = None,
) -> Generator[Any, Any, None]:
    """Move then snap each plane; caller retains ownership on all outcomes.

    Positions are controller-reported and frames are associated by host order.
    The caller must reserve the MCU and exclusively control the camera.
    """
    if reservation_held is not True or camera_exclusive is not True:
        raise ValueError("MCU reservation and exclusive camera must be asserted")
    if not 1 <= len(targets_um) <= 32 or not 0 < timeout_s <= 300:
        raise ValueError("invalid finite target count or timeout")
    if store_uri is not None and (
        not isinstance(store_uri, str) or not store_uri.strip()
    ):
        raise ValueError("store URI must be nonempty")
    targets = tuple(float(value) for value in targets_um)
    for target in targets:
        stage.validate_position_um(target)

    @bpp.run_decorator(
        md={
            "targets_um": targets,
            "mode": "software_stepped",
            "association": "host_order_inferred",
            "store_uri": store_uri,
        }
    )  # type: ignore[untyped-decorator]
    def run() -> Generator[Any, Any, None]:
        previous_count = None
        source = _Plane(0, 0.0, 0.0, 0, 0.0, -1)
        store = None
        try:
            if store_uri is not None:
                templates = yield from bps.wait_for(
                    [
                        lambda: asyncio.wait_for(
                            camera.buffer.get_value(cached=False), timeout_s
                        )
                    ]
                )
                store = RunFrameStore(
                    store_uri,
                    "software_snap",
                    templates[0].result(),
                    planned_frames=len(targets),
                )
            for index, target in enumerate(targets):
                tasks = yield from bps.wait_for(
                    [
                        lambda target=target: asyncio.to_thread(
                            stage.set_position_um, target
                        )
                    ]
                )
                tasks[0].result()
                positions = yield from bps.wait_for(
                    [lambda: asyncio.to_thread(stage.get_position_um)]
                )
                position = positions[0].result()
                snaps = yield from bps.wait_for(
                    [lambda: asyncio.wait_for(camera.snap(), timeout_s)]
                )
                count, image = snaps[0].result()
                if previous_count is not None and count != previous_count + 1:
                    raise RuntimeError("software snap count is discontinuous")
                previous_count = count
                stored_index = store.append(image) if store is not None else -1
                source.values = _Plane(
                    index, target, position, count, float(image.mean()), stored_index
                ).values
                yield from bps.create(name="software_step")
                yield from bps.read(source)
                yield from bps.save()
        finally:
            if store is not None:
                store.close()

    yield from run()
