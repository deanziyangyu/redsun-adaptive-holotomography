"""Experimental headless session for separately owned camera and MCU services."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    Protocol,
    runtime_checkable,
)

from bluesky import plan_stubs as bps
from redsun import AsDevice, AsPresenter, Declare, DevicesOf, Session

from redsun_aht.device.mmcore.camera import MMCamera  # noqa: TC001
from redsun_aht.device.pva_rotation import PvaRotationMotor  # noqa: TC001
from redsun_aht.device.pva_unified_mcu import PvaUnifiedMcu  # noqa: TC001
from redsun_aht.device.spinnaker_camera import SpinnakerCamera  # noqa: TC001
from redsun_aht.plans.dpct import _bounded_call
from redsun_aht.presenter.pva_timelapse import (
    PvaTimeLapseBindings,
    PvaTimeLapsePresenter,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from redsun.registry import PlanEntry


@runtime_checkable
class SessionCamera(Protocol):
    """Camera capability owned by the session's PVA service."""

    exposure: Any
    acquire: Any
    capture: Any
    sequence_started: Any


@runtime_checkable
class SessionMcu(Protocol):
    """MCU reservation and run-ID capabilities."""

    reservation_generation: int

    async def set_reserved_illumination(self, command: str) -> None:
        """Set an illumination command while reserved."""
        ...

    async def reserve(self) -> None:
        """Reserve the MCU cooperatively."""
        ...

    async def allocate_run_ids(self, count: int) -> tuple[int, ...]:
        """Allocate controller command IDs while reserved."""
        ...

    async def release_reservation(self) -> None:
        """Release the cooperative reservation."""
        ...


@dataclass(frozen=True, slots=True)
class LedChannel:
    """LED color, exposure in milliseconds and firmware brightness command."""

    color: Literal["blue", "green", "red"]
    exposure_ms: float
    brightness_command: int = 64

    def __post_init__(self) -> None:
        """Reject unsupported LED and camera settings before acquisition."""
        if self.color not in ("blue", "green", "red"):
            raise ValueError("invalid LED color")
        if (
            isinstance(self.exposure_ms, bool)
            or not isinstance(self.exposure_ms, (int, float))
            or not math.isfinite(self.exposure_ms)
            or self.exposure_ms <= 0
        ):
            raise ValueError("exposure must be positive and finite")
        if type(self.brightness_command) is not int or self.brightness_command not in (
            61,
            62,
            63,
            64,
        ):
            raise ValueError("brightness command must be 61..64")


class SessionTimeLapsePresenter:
    """Bind the managed plan to the devices actually injected by the session."""

    def __init__(
        self,
        name: str,
        *,
        cameras: DevicesOf[SessionCamera],
        mcus: DevicesOf[SessionMcu],
    ) -> None:
        self.name = name
        self._cameras = dict(cameras)
        self._mcus = dict(mcus)
        self._delegate = PvaTimeLapsePresenter(name, bindings=None)

    def configure(
        self,
        *,
        store_for_plane: Callable[[int, int], str],
        move_position: Callable[[Any], Generator],
        configure_channel: Callable[[Any], Generator],
        open_gate: Callable[[], Generator],
        close_gate: Callable[[], Generator],
    ) -> None:
        """Bind instrument callbacks without constructing or opening devices."""
        if len(self._cameras) != 1 or len(self._mcus) != 1:
            raise RuntimeError("exactly one session camera and MCU are required")
        camera = next(iter(self._cameras.values()))
        owner = next(iter(self._mcus.values()))
        self._delegate.bindings = PvaTimeLapseBindings(
            owner=owner,
            camera=camera,
            stack_for_block=lambda index: None,
            store_for_plane=store_for_plane,
            move_position=move_position,
            configure_channel=configure_channel,
            open_gate=open_gate,
            close_gate=close_gate,
        )

    def plan_map(self) -> Mapping[str, PlanEntry]:
        """Offer only the managed plan after explicit instrument binding."""
        plans = self._delegate.plan_map()
        return (
            {
                name: plans[name]
                for name in (
                    "pva_managed_timelapse",
                    "pva_automatic_stack",
                    "pva_gated_stack",
                )
            }
            if plans
            else {}
        )

    def configure_stationary_led(
        self, *, store_for_plane: Callable[[int, int], str]
    ) -> None:
        """Bind one stationary position and the session's LED/camera services.

        Position must be ``None``. Each channel is a :class:`LedChannel` and
        the logical gate is a no-op. Camera trigger mode and exclusivity are
        still established by the caller.
        """
        if len(self._cameras) != 1 or len(self._mcus) != 1:
            raise RuntimeError("exactly one session camera and MCU are required")
        camera = next(iter(self._cameras.values()))
        owner = next(iter(self._mcus.values()))
        last_channel: tuple[int, str, int] | None = None

        def move_position(position: Any) -> Generator:
            if position is not None:
                raise ValueError("stationary LED binding accepts only position None")
            yield from bps.null()

        def configure_channel(channel: Any) -> Generator:
            nonlocal last_channel
            if not isinstance(channel, LedChannel):
                raise ValueError("stationary LED binding requires LedChannel")
            current = yield from _bounded_call(
                lambda: camera.exposure.get_value(cached=False),
                timeout_s=5,
                what="camera exposure readback",
            )
            if abs(current - channel.exposure_ms) > 0.01:
                yield from bps.abs_set(camera.exposure, channel.exposure_ms, wait=True)
            current_key = (
                owner.reservation_generation,
                channel.color,
                channel.brightness_command,
            )
            previous = (
                last_channel
                if last_channel and last_channel[0] == current_key[0]
                else None
            )
            commands = []
            if previous is None or previous[1] != channel.color:
                commands.append(
                    {"blue": "51", "green": "52", "red": "53"}[channel.color]
                )
            if previous is None or previous[2] != channel.brightness_command:
                commands.append(str(channel.brightness_command))
            for command in commands:
                yield from _bounded_call(
                    lambda command=command: owner.set_reserved_illumination(command),
                    timeout_s=5,
                    what="set LED channel",
                )
            last_channel = current_key

        def gate() -> Generator:
            yield from bps.null()

        self.configure(
            store_for_plane=store_for_plane,
            move_position=move_position,
            configure_channel=configure_channel,
            open_gate=gate,
            close_gate=gate,
        )


class PvaTimeLapseSession(Session):
    """Headless client session; camera and MCU services are launched separately."""

    config: ClassVar[dict[str, object]] = {"session": "pva-timelapse", "strict": True}
    camera: Annotated[AsDevice[MMCamera], Declare(prefix="TLCAM:")]
    mcu: Annotated[AsDevice[PvaUnifiedMcu], Declare(prefix="TLMCU:")]
    rotation: Annotated[AsDevice[PvaRotationMotor], Declare(prefix="TLMCU:")]
    presenter: AsPresenter[SessionTimeLapsePresenter]


class PvaSpinnakerTimeLapseSession(Session):
    """The same managed plans with a separately owned Spinnaker camera."""

    config: ClassVar[dict[str, object]] = {"session": "pva-timelapse", "strict": True}
    camera: Annotated[AsDevice[SpinnakerCamera], Declare(prefix="TLCAM:")]
    mcu: Annotated[AsDevice[PvaUnifiedMcu], Declare(prefix="TLMCU:")]
    rotation: Annotated[AsDevice[PvaRotationMotor], Declare(prefix="TLMCU:")]
    presenter: AsPresenter[SessionTimeLapsePresenter]
