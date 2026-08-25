"""Ophyd-async device for the isolated camera-service control plane."""

from __future__ import annotations

import asyncio
from typing import Annotated as A

from ophyd_async.core import AsyncStatus, SignalR, SignalW, TriggerableCommand
from ophyd_async.epics.core import EpicsDevice, PvSuffix

from redsun_aht.domain import (
    AcquisitionState,
    HardwareServiceStatus,
    ServiceConnectionState,
)


class CameraServiceDevice(EpicsDevice):
    """Typed application-side view of one camera service IOC."""

    service_id: A[SignalR[str], PvSuffix("SERVICE_ID")]
    schema_generation: A[SignalR[int], PvSuffix("SCHEMA_GENERATION")]
    heartbeat: A[SignalR[float], PvSuffix("HEARTBEAT")]
    connection_state: A[SignalR[str], PvSuffix("CONNECTION_STATE")]
    acquisition_state: A[SignalR[str], PvSuffix("ACQUISITION_STATE")]
    command_state: A[SignalR[str], PvSuffix("COMMAND_STATE")]
    frames_published: A[SignalR[int], PvSuffix("FRAMES_PUBLISHED")]
    frames_dropped: A[SignalR[int], PvSuffix("FRAMES_DROPPED")]
    last_error: A[SignalR[str], PvSuffix("LAST_ERROR")]
    descriptor_generation: A[SignalR[int], PvSuffix("DESCRIPTOR_GENERATION")]
    descriptor_sequence: A[SignalR[int], PvSuffix("DESCRIPTOR_SEQUENCE")]
    descriptor_slot: A[SignalR[int], PvSuffix("DESCRIPTOR_SLOT")]
    descriptor_name: A[SignalR[str], PvSuffix("DESCRIPTOR_NAME")]
    descriptor_run_id: A[SignalR[str], PvSuffix("DESCRIPTOR_RUN_ID")]
    descriptor_frame_id: A[SignalR[str], PvSuffix("DESCRIPTOR_FRAME_ID")]
    descriptor_shape: A[SignalR[str], PvSuffix("DESCRIPTOR_SHAPE")]
    descriptor_dtype: A[SignalR[str], PvSuffix("DESCRIPTOR_DTYPE")]
    descriptor_byte_order: A[SignalR[str], PvSuffix("DESCRIPTOR_BYTE_ORDER")]
    descriptor_timestamp_ns: A[SignalR[str], PvSuffix("DESCRIPTOR_TIMESTAMP_NS")]
    descriptor_checksum_hi: A[SignalR[str], PvSuffix("DESCRIPTOR_CHECKSUM_HI")]
    descriptor_checksum_lo: A[SignalR[str], PvSuffix("DESCRIPTOR_CHECKSUM_LO")]

    command_connect: A[TriggerableCommand, PvSuffix("COMMAND:CONNECT")]
    command_arm: A[TriggerableCommand, PvSuffix("COMMAND:ARM")]
    command_trigger: A[TriggerableCommand, PvSuffix("COMMAND:TRIGGER")]
    command_stop: A[TriggerableCommand, PvSuffix("COMMAND:STOP")]
    command_disconnect: A[TriggerableCommand, PvSuffix("COMMAND:DISCONNECT")]
    command_acknowledge: A[SignalW[int], PvSuffix("COMMAND:ACKNOWLEDGE")]

    def __init__(self, prefix: str, *, name: str) -> None:
        if not prefix.endswith(":"):
            raise ValueError("camera service prefix must end with ':'")
        self.epics_prefix = prefix
        super().__init__(prefix=prefix, name=name)

    async def request_connect(self) -> None:
        """Ask the service to connect its isolated backend."""
        await self.command_connect.trigger()

    async def arm(self) -> None:
        """Ask the service to arm acquisition."""
        await self.command_arm.trigger()

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        """Ask the service to publish one acquisition descriptor."""
        await self.command_trigger.trigger()

    @AsyncStatus.wrap
    async def stop(self, *, success: bool = False) -> None:
        """Ask the service to enter its idle acquisition state."""
        del success
        await self.command_stop.trigger()

    async def request_disconnect(self) -> None:
        """Ask the service to disconnect its isolated backend."""
        await self.command_disconnect.trigger()

    async def acknowledge(self, sequence: int) -> None:
        """Acknowledge one exact lossless descriptor after durable copy."""
        if sequence < 0:
            raise ValueError("acknowledgement sequence must be non-negative")
        await self.command_acknowledge.set(sequence)

    async def health(self) -> HardwareServiceStatus:
        """Read one typed health snapshot from the EPICS control plane."""
        (
            service_id,
            generation,
            heartbeat,
            connection,
            acquisition,
            command,
            published,
            dropped,
            error,
        ) = await asyncio.gather(
            self.service_id.get_value(),
            self.schema_generation.get_value(),
            self.heartbeat.get_value(),
            self.connection_state.get_value(),
            self.acquisition_state.get_value(),
            self.command_state.get_value(),
            self.frames_published.get_value(),
            self.frames_dropped.get_value(),
            self.last_error.get_value(),
        )
        return HardwareServiceStatus(
            service_id=service_id,
            epics_prefix=self.epics_prefix,
            schema_generation=generation,
            heartbeat_ns=int(heartbeat * 1_000_000_000),
            connection_state=ServiceConnectionState(connection),
            acquisition_state=AcquisitionState(acquisition),
            command_state=command,
            frames_published=published,
            frames_dropped=dropped,
            last_error=error or None,
        )


__all__ = ["CameraServiceDevice"]
