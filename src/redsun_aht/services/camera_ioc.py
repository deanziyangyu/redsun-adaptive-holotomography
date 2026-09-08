"""Caproto control plane for an isolated camera service."""

from __future__ import annotations

import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar
from uuid import uuid4

import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, pvproperty, run

from redsun_aht.buffers import BufferFullError, SharedMemoryFrameRing
from redsun_aht.domain import BufferUsage, FrameBufferDescriptor

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from redsun_aht.device.camera.mmcore import CameraIdentity


T = TypeVar("T")


class CameraBackend(Protocol):
    """Lifecycle surface owned by one isolated camera IOC process."""

    def connect(self) -> CameraIdentity: ...

    def arm(self) -> None: ...

    def trigger(self) -> Any: ...

    def start_sequence(
        self,
        *,
        interval_ms: float,
        buffer_memory_mb: int,
        frame_count: int | None = None,
    ) -> None: ...

    def drain_sequence(self) -> tuple[np.ndarray[Any, Any], ...]: ...

    def sequence_overflowed(self) -> bool: ...

    def stop(self) -> None: ...

    def disconnect(self) -> None: ...


class CameraServiceIOC(PVGroup):  # type: ignore[misc]  # caproto has no type metadata
    """Publish camera health, commands, counters, and frame descriptors."""

    service_id = pvproperty(
        value="", dtype=ChannelType.STRING, name="SERVICE_ID", read_only=True
    )
    schema_generation = pvproperty(value=1, name="SCHEMA_GENERATION", read_only=True)
    heartbeat = pvproperty(value=0.0, name="HEARTBEAT", read_only=True)
    connection_state = pvproperty(
        value="disconnected",
        dtype=ChannelType.STRING,
        name="CONNECTION_STATE",
        read_only=True,
    )
    acquisition_state = pvproperty(
        value="idle",
        dtype=ChannelType.STRING,
        name="ACQUISITION_STATE",
        read_only=True,
    )
    command_state = pvproperty(
        value="idle", dtype=ChannelType.STRING, name="COMMAND_STATE", read_only=True
    )
    frames_published = pvproperty(value=0, name="FRAMES_PUBLISHED", read_only=True)
    frames_dropped = pvproperty(value=0, name="FRAMES_DROPPED", read_only=True)
    last_error = pvproperty(
        value="", dtype=ChannelType.STRING, name="LAST_ERROR", read_only=True
    )
    descriptor_generation = pvproperty(
        value=0, name="DESCRIPTOR_GENERATION", read_only=True
    )
    descriptor_sequence = pvproperty(
        value=-1, name="DESCRIPTOR_SEQUENCE", read_only=True
    )
    descriptor_slot = pvproperty(value=-1, name="DESCRIPTOR_SLOT", read_only=True)
    descriptor_name = pvproperty(
        value="", dtype=ChannelType.STRING, name="DESCRIPTOR_NAME", read_only=True
    )
    descriptor_run_id = pvproperty(
        value="", dtype=ChannelType.STRING, name="DESCRIPTOR_RUN_ID", read_only=True
    )
    descriptor_frame_id = pvproperty(
        value="", dtype=ChannelType.STRING, name="DESCRIPTOR_FRAME_ID", read_only=True
    )
    descriptor_shape = pvproperty(
        value="", dtype=ChannelType.STRING, name="DESCRIPTOR_SHAPE", read_only=True
    )
    descriptor_dtype = pvproperty(
        value="", dtype=ChannelType.STRING, name="DESCRIPTOR_DTYPE", read_only=True
    )
    descriptor_byte_order = pvproperty(
        value="", dtype=ChannelType.STRING, name="DESCRIPTOR_BYTE_ORDER", read_only=True
    )
    descriptor_timestamp_ns = pvproperty(
        value="",
        dtype=ChannelType.STRING,
        name="DESCRIPTOR_TIMESTAMP_NS",
        read_only=True,
    )
    descriptor_checksum_hi = pvproperty(
        value="",
        dtype=ChannelType.STRING,
        name="DESCRIPTOR_CHECKSUM_HI",
        read_only=True,
    )
    descriptor_checksum_lo = pvproperty(
        value="",
        dtype=ChannelType.STRING,
        name="DESCRIPTOR_CHECKSUM_LO",
        read_only=True,
    )
    camera_adapter = pvproperty(
        value="", dtype=ChannelType.STRING, name="CAMERA_ADAPTER", read_only=True
    )
    camera_device = pvproperty(
        value="", dtype=ChannelType.STRING, name="CAMERA_DEVICE", read_only=True
    )
    camera_serial = pvproperty(
        value="", dtype=ChannelType.STRING, name="CAMERA_SERIAL", read_only=True
    )
    camera_pixel_binning = pvproperty(
        value="", dtype=ChannelType.STRING, name="CAMERA_PIXEL_BINNING", read_only=True
    )
    camera_pixel_type = pvproperty(
        value="", dtype=ChannelType.STRING, name="CAMERA_PIXEL_TYPE", read_only=True
    )
    camera_exposure_ms = pvproperty(
        value="", dtype=ChannelType.STRING, name="CAMERA_EXPOSURE_MS", read_only=True
    )
    live_state = pvproperty(
        value="idle", dtype=ChannelType.STRING, name="LIVE_STATE", read_only=True
    )
    live_publish_hz = pvproperty(value=60.0, name="LIVE_PUBLISH_HZ", read_only=True)
    live_sequence = pvproperty(value=-1, name="LIVE_SEQUENCE", read_only=True)
    live_timestamp_ns = pvproperty(
        value="", dtype=ChannelType.STRING, name="LIVE_TIMESTAMP_NS", read_only=True
    )
    live_shape = pvproperty(
        value="", dtype=ChannelType.STRING, name="LIVE_SHAPE", read_only=True
    )
    live_dtype = pvproperty(
        value="", dtype=ChannelType.STRING, name="LIVE_DTYPE", read_only=True
    )
    live_byte_count = pvproperty(value=0, name="LIVE_BYTE_COUNT", read_only=True)
    live_frames_drained = pvproperty(
        value=0, name="LIVE_FRAMES_DRAINED", read_only=True
    )
    live_frames_published = pvproperty(
        value=0, name="LIVE_FRAMES_PUBLISHED", read_only=True
    )
    live_frames_skipped = pvproperty(
        value=0, name="LIVE_FRAMES_SKIPPED", read_only=True
    )
    live_buffer_overruns = pvproperty(
        value=0, name="LIVE_BUFFER_OVERRUNS", read_only=True
    )
    live_image = pvproperty(
        value=np.array([0], dtype=np.uint8),
        dtype=ChannelType.CHAR,
        max_length=16 * 1024 * 1024,
        name="LIVE_IMAGE",
        read_only=True,
    )

    command_connect = pvproperty(value=0, name="COMMAND:CONNECT")
    command_arm = pvproperty(value=0, name="COMMAND:ARM")
    command_trigger = pvproperty(value=0, name="COMMAND:TRIGGER")
    command_start_live = pvproperty(value=0, name="COMMAND:START_LIVE")
    command_stop = pvproperty(value=0, name="COMMAND:STOP")
    command_disconnect = pvproperty(value=0, name="COMMAND:DISCONNECT")
    command_acknowledge = pvproperty(value=-1, name="COMMAND:ACKNOWLEDGE")

    def __init__(
        self,
        service_id: str,
        prefix: str,
        generation: int = 1,
        backend: CameraBackend | None = None,
        run_id: str = "unassigned",
        acquisition_slots: int = 4,
        live_publish_hz: float = 60.0,
        live_buffer_memory_mb: int = 128,
    ) -> None:
        self._configured_service_id = service_id
        self._configured_generation = generation
        self._backend = backend
        self._run_id = run_id
        self._acquisition_slots = acquisition_slots
        self._acquisition_ring: SharedMemoryFrameRing | None = None
        self._latest_descriptor: FrameBufferDescriptor | None = None
        if not 0 < live_publish_hz <= 60:
            raise ValueError("live_publish_hz must be in the range (0, 60]")
        if live_buffer_memory_mb <= 0:
            raise ValueError("live_buffer_memory_mb must be positive")
        self._live_publish_hz = live_publish_hz
        self._live_buffer_memory_mb = live_buffer_memory_mb
        self._live_task: asyncio.Task[None] | None = None
        self._connect_task: asyncio.Task[None] | None = None
        self._backend_lock = asyncio.Lock()
        self._backend_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"camera-{service_id}",
        )
        super().__init__(prefix)

    @service_id.startup  # type: ignore[no-redef,untyped-decorator]
    async def service_id(self, instance: object, async_lib: object) -> None:
        """Initialize identity and maintain an observable heartbeat."""
        del instance
        await self.service_id.write(self._configured_service_id)
        await self.schema_generation.write(self._configured_generation)
        await self.live_publish_hz.write(self._live_publish_hz)
        while True:
            await self.heartbeat.write(time.time())
            await async_lib.library.sleep(0.1)  # type: ignore[attr-defined]

    @command_connect.putter  # type: ignore[no-redef,untyped-decorator]
    async def command_connect(self, instance: object, value: int) -> int:
        """Connect and admit the configured process-local camera backend."""
        del instance
        if value:
            if self._connect_task is not None:
                raise RuntimeError("camera service connection is already in progress")
            await self.connection_state.write("connecting")
            await self.command_state.write("connecting")
            await self.last_error.write("")
            self._connect_task = asyncio.create_task(self._connect_backend())
        return 0

    async def _connect_backend(self) -> None:
        """Admit the thread-affine camera backend without blocking CA puts."""
        try:
            if self._backend is not None:
                identity = await self._run_backend(self._backend.connect)
                await self.camera_adapter.write(identity.adapter)
                await self.camera_device.write(identity.device_name)
                await self.camera_serial.write(identity.serial or "")
                properties = await self._run_backend(
                    getattr(self._backend, "properties", lambda: ())
                )
                property_values = {
                    property_.name: property_.value for property_ in properties
                }
                await self.camera_pixel_binning.write(
                    property_values.get(
                        "PixelBinning", property_values.get("Binning", "")
                    )
                )
                await self.camera_pixel_type.write(
                    property_values.get(
                        "PixelType", property_values.get("Pixel Format", "")
                    )
                )
                exposure_ms = await self._run_backend(
                    getattr(
                        self._backend,
                        "configured_exposure_ms",
                        lambda: property_values.get("Exposure", ""),
                    )
                )
                await self.camera_exposure_ms.write(exposure_ms)
            await self.connection_state.write("ready")
            await self.command_state.write("connected")
            await self.last_error.write("")
        except Exception as exc:
            await self.connection_state.write("faulted")
            await self.command_state.write("connect_failed")
            await self.last_error.write(str(exc))
        finally:
            self._connect_task = None

    @command_arm.putter  # type: ignore[no-redef,untyped-decorator]
    async def command_arm(self, instance: object, value: int) -> int:
        """Arm descriptor publication."""
        del instance
        if value:
            if self._live_task is not None:
                await self.command_state.write("arm_rejected")
                await self.last_error.write("camera live sequence is active")
                raise RuntimeError("camera live sequence is active")
            if self.connection_state.value != "ready":
                await self.command_state.write("arm_rejected")
                await self.last_error.write("camera service is not connected")
                raise RuntimeError("camera service is not connected")
            if self._backend is not None:
                await self._run_backend(self._backend.arm)
            await self.acquisition_state.write("armed")
            await self.command_state.write("armed")
            await self.last_error.write("")
        return 0

    @command_start_live.putter  # type: ignore[no-redef,untyped-decorator]
    async def command_start_live(self, instance: object, value: int) -> int:
        """Start a continuous camera sequence for bounded-rate live viewing."""
        del instance
        if not value:
            return 0
        if self.connection_state.value != "ready":
            await self.command_state.write("live_rejected")
            await self.last_error.write("camera service is not connected")
            raise RuntimeError("camera service is not connected")
        if self._live_task is not None:
            await self.command_state.write("live_rejected")
            await self.last_error.write("camera live sequence is already active")
            raise RuntimeError("camera live sequence is already active")
        if self.acquisition_state.value != "idle":
            await self.command_state.write("live_rejected")
            await self.last_error.write("camera is not idle")
            raise RuntimeError("camera is not idle")
        try:
            if self._backend is not None:
                backend = self._backend
                await self._run_backend(
                    lambda: backend.start_sequence(
                        interval_ms=0.0,
                        buffer_memory_mb=self._live_buffer_memory_mb,
                    )
                )
            await self.live_state.write("running")
            await self.acquisition_state.write("live")
            await self.command_state.write("live_started")
            await self.last_error.write("")
            self._live_task = asyncio.create_task(self._run_live_loop())
        except Exception as exc:
            await self.live_state.write("faulted")
            await self.command_state.write("live_start_failed")
            await self.last_error.write(str(exc))
            raise
        return 0

    @command_trigger.putter  # type: ignore[no-redef,untyped-decorator]
    async def command_trigger(self, instance: object, value: int) -> int:
        """Publish the next simulated descriptor counter."""
        del instance
        if value:
            if self.acquisition_state.value != "armed":
                await self.command_state.write("trigger_rejected")
                await self.last_error.write("camera service is not armed")
                raise RuntimeError("camera service is not armed")
            try:
                sequence = int(self.frames_published.value)
                await self.acquisition_state.write("acquiring")
                if self._backend is not None:
                    if self._acquisition_ring is not None:
                        self._acquisition_ring.ensure_writable()
                    frame = np.asarray(await self._run_backend(self._backend.trigger))
                    descriptor = self._publish_frame(frame)
                    await self._write_descriptor(descriptor)
                else:
                    await self.descriptor_generation.write(
                        int(self.schema_generation.value)
                    )
                    await self.descriptor_sequence.write(sequence)
                    await self.descriptor_slot.write(sequence % 4)
                    await self.descriptor_name.write(
                        f"simulated-{self._configured_service_id}"
                    )
                await self.frames_published.write(sequence + 1)
                await self.acquisition_state.write("armed")
                await self.command_state.write("frame_committed")
                await self.last_error.write("")
            except BufferFullError as exc:
                await self.acquisition_state.write("armed")
                await self.command_state.write("buffer_full")
                await self.frames_dropped.write(int(self.frames_dropped.value) + 1)
                await self.last_error.write(str(exc))
                raise
            except Exception as exc:
                await self.acquisition_state.write("idle")
                await self.connection_state.write("faulted")
                await self.command_state.write("trigger_failed")
                await self.last_error.write(str(exc))
                raise
        return 0

    @command_stop.putter  # type: ignore[no-redef,untyped-decorator]
    async def command_stop(self, instance: object, value: int) -> int:
        """Return acquisition to idle."""
        del instance
        if value:
            await self._stop_live_task()
            if self._backend is not None and self.connection_state.value == "ready":
                await self._run_backend(self._backend.stop)
            await self.acquisition_state.write("idle")
            await self.live_state.write("idle")
            await self.command_state.write("stopped")
        return 0

    @command_disconnect.putter  # type: ignore[no-redef,untyped-decorator]
    async def command_disconnect(self, instance: object, value: int) -> int:
        """Publish disconnected state for supervisor recovery."""
        del instance
        if value:
            await self._stop_live_task()
            self._close_ring()
            try:
                if self._backend is not None:
                    await self._run_backend(self._backend.disconnect)
            finally:
                self._latest_descriptor = None
            await self.acquisition_state.write("idle")
            await self.live_state.write("idle")
            await self.connection_state.write("disconnected")
            await self.command_state.write("disconnected")
        return 0

    @command_acknowledge.putter  # type: ignore[no-redef,untyped-decorator]
    async def command_acknowledge(self, instance: object, value: int) -> int:
        """Acknowledge one persisted lossless frame by exact sequence."""
        del instance
        if value < 0:
            return -1
        descriptor = self._latest_descriptor
        ring = self._acquisition_ring
        if descriptor is None or ring is None or descriptor.sequence != value:
            raise RuntimeError(f"no current acquisition descriptor at sequence {value}")
        ring.acknowledge(descriptor)
        await self.command_state.write("frame_acknowledged")
        return -1

    async def _run_live_loop(self) -> None:
        """Drain MMCore's circular buffer and publish its newest frame only."""
        interval_s = 1.0 / self._live_publish_hz
        try:
            while True:
                started_ns = time.monotonic_ns()
                frames: tuple[np.ndarray[Any, Any], ...]
                if self._backend is None:
                    frames = (
                        np.full(
                            (2, 3),
                            int(self.live_frames_drained.value) + 1,
                            dtype=np.uint16,
                        ),
                    )
                    overflowed = False
                else:
                    backend = self._backend

                    def drain(
                        bound_backend: CameraBackend = backend,
                    ) -> tuple[tuple[np.ndarray[Any, Any], ...], bool]:
                        return (
                            bound_backend.drain_sequence(),
                            bound_backend.sequence_overflowed(),
                        )

                    frames, overflowed = await self._run_backend(
                        drain
                    )
                if overflowed:
                    await self.live_buffer_overruns.write(
                        int(self.live_buffer_overruns.value) + 1
                    )
                    await self.live_state.write("faulted")
                    await self.acquisition_state.write("idle")
                    await self.command_state.write("live_overrun")
                    await self.last_error.write("MMCore circular buffer overflowed")
                    if self._backend is not None:
                        await self._run_backend(self._backend.stop)
                    return
                if frames:
                    drained = int(self.live_frames_drained.value) + len(frames)
                    skipped = max(0, len(frames) - 1)
                    await self.live_frames_drained.write(drained)
                    if skipped:
                        await self.live_frames_skipped.write(
                            int(self.live_frames_skipped.value) + skipped
                        )
                    await self._write_live_frame(frames[-1], sequence=drained - 1)
                elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
                await asyncio.sleep(max(0.0, interval_s - elapsed_s))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.live_state.write("faulted")
            await self.acquisition_state.write("idle")
            await self.command_state.write("live_failed")
            await self.last_error.write(str(exc))
            if self._backend is not None:
                await self._run_backend(self._backend.stop)
        finally:
            if self._live_task is asyncio.current_task():
                self._live_task = None

    async def _write_live_frame(
        self, frame: np.ndarray[Any, Any], *, sequence: int
    ) -> None:
        """Publish one latest-wins preview image as a typed byte waveform."""
        array = np.ascontiguousarray(np.asarray(frame))
        if array.ndim != 2:
            raise ValueError(f"live view requires a 2D frame, got {array.shape!r}")
        payload = np.frombuffer(array.tobytes(order="C"), dtype=np.uint8)
        if payload.size > 16 * 1024 * 1024:
            raise ValueError(
                f"live frame is {payload.size} bytes, above the 16 MiB EPICS limit"
            )
        await self.live_image.write(payload)
        await self.live_sequence.write(sequence)
        await self.live_timestamp_ns.write(str(time.time_ns()))
        await self.live_shape.write(",".join(map(str, array.shape)))
        await self.live_dtype.write(array.dtype.str)
        await self.live_byte_count.write(int(payload.size))
        await self.live_frames_published.write(
            int(self.live_frames_published.value) + 1
        )

    async def _stop_live_task(self) -> None:
        """Cancel the live drain task without leaving an orphan sequence loop."""
        task, self._live_task = self._live_task, None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _run_backend(self, operation: Callable[[], T]) -> T:
        """Run one thread-affine camera-SDK operation without blocking CA."""
        async with self._backend_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._backend_executor, operation)

    def _publish_frame(self, frame: np.ndarray[Any, Any]) -> FrameBufferDescriptor:
        ring = self._acquisition_ring
        if ring is None:
            ring = SharedMemoryFrameRing(
                service_id=self._configured_service_id,
                shape=frame.shape,
                dtype=frame.dtype.str,
                slots=self._acquisition_slots,
                usage=BufferUsage.ACQUISITION,
                generation=self._configured_generation,
            )
            self._acquisition_ring = ring
        descriptor = ring.publish(
            frame,
            run_id=self._run_id,
            frame_id=str(uuid4()),
        )
        self._latest_descriptor = descriptor
        return descriptor

    async def _write_descriptor(self, descriptor: FrameBufferDescriptor) -> None:
        await self.descriptor_generation.write(descriptor.generation)
        await self.descriptor_sequence.write(descriptor.sequence)
        await self.descriptor_slot.write(descriptor.slot)
        await self.descriptor_name.write(descriptor.shared_memory_name)
        await self.descriptor_run_id.write(descriptor.run_id)
        await self.descriptor_frame_id.write(descriptor.frame_id)
        await self.descriptor_shape.write(",".join(map(str, descriptor.shape)))
        await self.descriptor_dtype.write(descriptor.dtype)
        await self.descriptor_byte_order.write(descriptor.byte_order)
        await self.descriptor_timestamp_ns.write(str(descriptor.timestamp_ns))
        await self.descriptor_checksum_hi.write(descriptor.checksum[:32])
        await self.descriptor_checksum_lo.write(descriptor.checksum[32:])

    def _close_ring(self) -> None:
        if self._acquisition_ring is not None:
            self._acquisition_ring.close()
        self._acquisition_ring = None


def build_camera_ioc(
    service_id: str,
    prefix: str,
    *,
    generation: int = 1,
    backend: CameraBackend | None = None,
    run_id: str = "unassigned",
    acquisition_slots: int = 4,
    live_publish_hz: float = 60.0,
    live_buffer_memory_mb: int = 128,
) -> CameraServiceIOC:
    """Build an inspectable IOC without starting sockets or hardware."""
    if not prefix.endswith(":"):
        raise ValueError("camera IOC prefix must end with ':'")
    if generation <= 0:
        raise ValueError("camera IOC generation must be positive")
    if not run_id:
        raise ValueError("camera IOC run_id must not be empty")
    if acquisition_slots <= 0:
        raise ValueError("camera IOC acquisition_slots must be positive")
    if not 0 < live_publish_hz <= 60:
        raise ValueError("camera IOC live_publish_hz must be in the range (0, 60]")
    if live_buffer_memory_mb <= 0:
        raise ValueError("camera IOC live_buffer_memory_mb must be positive")
    return CameraServiceIOC(
        service_id,
        prefix,
        generation,
        backend,
        run_id,
        acquisition_slots,
        live_publish_hz,
        live_buffer_memory_mb,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one camera IOC as an isolated service process."""
    parser = argparse.ArgumentParser(prog="aht-camera-ioc")
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--run-id", default="unassigned")
    parser.add_argument("--acquisition-slots", type=int, default=4)
    parser.add_argument("--live-publish-hz", type=float, default=60.0)
    parser.add_argument("--live-buffer-memory-mb", type=int, default=128)
    parser.add_argument(
        "--backend", choices=("simulation", "mmcore"), default="simulation"
    )
    parser.add_argument("--mm-config", type=Path)
    parser.add_argument("--mm-path", type=Path)
    parser.add_argument("--adapter")
    parser.add_argument("--device-name")
    parser.add_argument("--camera-label")
    parser.add_argument("--serial-property")
    parser.add_argument("--expected-serial")
    parser.add_argument(
        "--pvcam-fcs-profile",
        action="store_true",
        help=(
            "Apply the verified Device API 75 PVCAM 64x64 FCS property profile "
            "after loading the base MMCore configuration."
        ),
    )
    parser.add_argument(
        "--camera-property",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Explicit admission-time MMCore property, repeatable.",
    )
    args = parser.parse_args(argv)
    backend = None
    if args.backend == "mmcore":
        required = {
            "--mm-config": args.mm_config,
            "--mm-path": args.mm_path,
            "--adapter": args.adapter,
            "--device-name": args.device_name,
            "--camera-label": args.camera_label,
        }
        missing = [option for option, value in required.items() if value is None]
        if missing:
            parser.error(f"mmcore backend requires {', '.join(missing)}")
        from redsun_aht.device.camera.mmcore import (
            MMCoreCameraBackend,
            MMCoreCameraProfile,
        )

        properties: dict[str, str] = {}
        for item in args.camera_property:
            name, separator, value = item.partition("=")
            if not separator or not name or not value or name in properties:
                parser.error("--camera-property must be unique NAME=VALUE")
            properties[name] = value
        if args.pvcam_fcs_profile:
            if args.adapter != "PVCAM":
                parser.error("--pvcam-fcs-profile requires --adapter PVCAM")
            from redsun_aht.device.camera.profiles import PVCAM_FCS_PROPERTIES

            duplicates = sorted(set(properties).intersection(PVCAM_FCS_PROPERTIES))
            if duplicates:
                parser.error(
                    "--camera-property duplicates --pvcam-fcs-profile setting(s): "
                    f"{', '.join(duplicates)}"
                )
            properties.update(PVCAM_FCS_PROPERTIES)
        backend = MMCoreCameraBackend(
            MMCoreCameraProfile(
                service_id=args.service_id,
                config_path=args.mm_config,
                mm_path=args.mm_path,
                adapter=args.adapter,
                device_name=args.device_name,
                expected_label=args.camera_label,
                serial_property=args.serial_property,
                expected_serial=args.expected_serial,
                initial_properties=properties,
            )
        )
    ioc = build_camera_ioc(
        args.service_id,
        args.prefix,
        generation=args.generation,
        backend=backend,
        run_id=args.run_id,
        acquisition_slots=args.acquisition_slots,
        live_publish_hz=args.live_publish_hz,
        live_buffer_memory_mb=args.live_buffer_memory_mb,
    )
    try:
        run(ioc.pvdb, interfaces=["127.0.0.1"])
    finally:
        if backend is not None:
            backend.disconnect()
    return 0


__all__ = ["CameraServiceIOC", "build_camera_ioc", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
