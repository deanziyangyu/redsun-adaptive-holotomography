"""Explicit, optional oxiserial transport primitives for a Ludl serial client.

This module contains optional OxiSerial transport primitives. Importing it
never opens a port; callers must explicitly construct a session after
arranging sole ownership of the controller connection.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

SUPPORTED_BAUDRATES = (2400, 4800, 9600, 19200, 28800, 38400, 57600, 115200)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class AsyncSerialPort(Protocol):
    """Small async port surface supplied by ``oxiserial.aio`` or a test double."""

    @property
    def is_open(self) -> bool: ...

    def write(self, data: bytes) -> Awaitable[int]: ...

    def read(self, size: int = 1) -> Awaitable[bytes]: ...

    def read_until(
        self, expected: bytes, size: int | None = None
    ) -> Awaitable[bytes]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SerialExchange:
    """Raw request/response bytes with host-side call timing brackets.

    The timestamps bracket the awaited transport calls. They are not UART wire
    timestamps, controller timestamps, or stage-position timestamps.
    """

    request_length: int
    response: bytes
    write_call_started_ns: int
    write_call_finished_ns: int
    read_call_started_ns: int
    read_call_finished_ns: int

    @property
    def round_trip_call_ns(self) -> int:
        return self.read_call_finished_ns - self.write_call_started_ns


@dataclass(frozen=True)
class SerialReadChunk:
    """One raw asynchronous read, preserving its host-call interval."""

    data: bytes
    call_started_ns: int
    call_finished_ns: int


@dataclass(frozen=True)
class MAC6000ReceivedFrame:
    """One complete frame with host-side read-call brackets."""

    data: bytes
    call_started_ns: int
    call_finished_ns: int


class OxiSerialSession:
    """Own one serial port and serialize complete request/response exchanges.

    Raw line exchanges, unframed reads and MAC6000 binary transactions are
    mutually exclusive modes for a session. A partial or failed binary
    transaction permanently marks that session unusable.
    """

    def __init__(
        self,
        port: AsyncSerialPort,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._port = port
        self._monotonic_ns = monotonic_ns
        self._exchange_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._unsolicited_frames: asyncio.Queue[MAC6000ReceivedFrame] = asyncio.Queue(
            maxsize=100_000
        )
        self._frame_stream_usable = True
        self._event_reader_task: asyncio.Task[None] | None = None
        self._pending_frame_matcher: Callable[[bytes], bool] | None = None
        self._pending_frame_future: asyncio.Future[bytes] | None = None
        self._captured_event_matcher: Callable[[bytes], bool] | None = None
        self._captured_events: asyncio.Queue[MAC6000ReceivedFrame] | None = None
        self._event_reader_error: BaseException | None = None

    @classmethod
    def open(
        cls,
        port: str,
        *,
        baudrate: int = 115200,
        timeout: float = 0.5,
        stopbits: int = 2,
        parity: str = "N",
    ) -> OxiSerialSession:
        """Explicitly open an oxiserial port for one owned controller link.

        The default 8N2 profile matches the qualified MAC6000 link. ASI
        controllers use 8N1, so callers select ``stopbits=1`` there. This opens
        the requested endpoint; it does not identify or validate a controller
        and must not be used while MMCore owns the port.
        """
        from oxiserial.aio import serial_for_url

        if (
            isinstance(baudrate, bool)
            or not isinstance(baudrate, int)
            or baudrate not in SUPPORTED_BAUDRATES
        ):
            raise ValueError(
                f"unsupported baud rate {baudrate!r}; "
                f"choose one of {SUPPORTED_BAUDRATES}"
            )
        if stopbits not in (1, 2):
            raise ValueError("stopbits must be 1 or 2")
        if parity not in ("N", "E", "O"):
            raise ValueError("parity must be 'N', 'E' or 'O'")

        serial = serial_for_url(
            port,
            baudrate=baudrate,
            bytesize=8,
            parity=parity,
            stopbits=stopbits,
            timeout=timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            do_not_open=True,
        )
        try:
            # Match the MMCore Ludl SerialManager profile without pulsing DTR
            # high during open; some controllers treat this line as reset.
            serial.dtr = False
            serial.open()
        except BaseException:
            serial.close()
            raise
        return cls(serial)

    async def exchange(self, request: bytes, *, terminator: bytes) -> bytes:
        """Send one already-encoded request and read its response atomically.

        Protocol encoding, response validation, timeouts, and retries belong
        to a later MAC6000 protocol layer; this transport performs no retries
        and does not discard unsolicited controller event messages.
        """
        return (await self.exchange_timed(request, terminator=terminator)).response

    async def exchange_line(
        self, request: bytes, *, terminator: bytes = b"\r"
    ) -> bytes:
        """Exchange one legacy ASCII command, without enabling binary events."""
        return await self.exchange(request, terminator=terminator)

    async def send_unframed(self, request: bytes) -> None:
        """Send an explicit protocol selector that has no response payload."""
        if not request:
            raise ValueError("request must not be empty")
        if self._event_reader_task is not None:
            raise RuntimeError(
                "unframed writes are disabled while the event reader runs"
            )
        async with self._exchange_lock:
            self._frame_stream_usable = False
            async with self._write_lock:
                written = await self._port.write(request)
                if written != len(request):
                    raise OSError(f"Short serial write: {written}/{len(request)} bytes")

    async def exchange_fixed(
        self, request: bytes, *, response_size: int, timeout_s: float = 1.0
    ) -> bytes:
        """Exchange a request whose protocol response has a fixed byte count."""
        if response_size < 1:
            raise ValueError("response_size must be positive")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if self._event_reader_task is not None:
            raise RuntimeError(
                "fixed-size exchanges are disabled while the event reader runs"
            )
        async with self._exchange_lock:
            self._frame_stream_usable = False
            started_ns = self._monotonic_ns()
            written = await self._port.write(request)
            if written != len(request):
                raise OSError(f"Short serial write: {written}/{len(request)} bytes")
            deadline_ns = started_ns + int(timeout_s * 1_000_000_000)
            response = bytearray()
            while len(response) < response_size:
                remaining_ns = deadline_ns - self._monotonic_ns()
                if remaining_ns <= 0:
                    raise TimeoutError("No fixed-size serial response before timeout")
                data = await asyncio.wait_for(
                    self._port.read(response_size - len(response)),
                    timeout=remaining_ns / 1_000_000_000,
                )
                if data:
                    response.extend(data)
            return bytes(response)

    async def exchange_timed(
        self, request: bytes, *, terminator: bytes
    ) -> SerialExchange:
        """Perform one serialized exchange and retain host call brackets.

        This remains a line-oriented request/response path. It is unavailable
        while the opt-in MAC6000 event reader owns the serial stream.
        """
        if self._event_reader_task is not None:
            raise RuntimeError(
                "line exchanges are disabled while the MAC6000 reader runs"
            )
        async with self._exchange_lock:
            self._frame_stream_usable = False
            write_started_ns = self._monotonic_ns()
            written = await self._port.write(request)
            write_finished_ns = self._monotonic_ns()
            if written != len(request):
                raise OSError(f"Short serial write: {written}/{len(request)} bytes")
            read_started_ns = self._monotonic_ns()
            response = await self._port.read_until(expected=terminator)
            read_finished_ns = self._monotonic_ns()
            if not response:
                raise TimeoutError("No serial response before the configured timeout")
            return SerialExchange(
                request_length=len(request),
                response=response,
                write_call_started_ns=write_started_ns,
                write_call_finished_ns=write_finished_ns,
                read_call_started_ns=read_started_ns,
                read_call_finished_ns=read_finished_ns,
            )

    async def read_line(self) -> bytes:
        """Read a follow-up ASCII line without sending another command."""
        async with self._exchange_lock:
            if self._event_reader_task is not None:
                raise RuntimeError("line reads are disabled while the event reader runs")
            self._frame_stream_usable = False
            return await self._port.read_until(expected=b"\n")

    async def read_chunk(self, size: int = 256) -> SerialReadChunk:
        """Read raw bytes without assuming a delimiter or binary frame shape.

        An empty read is retained as an empty chunk so the future event reader
        can distinguish a read timeout from a protocol-decoded controller
        event. This method does not parse or discard any bytes.
        """
        if size < 1:
            raise ValueError("read size must be positive")
        if self._event_reader_task is not None:
            raise RuntimeError("raw reads are disabled while the MAC6000 reader runs")
        async with self._exchange_lock:
            started_ns = self._monotonic_ns()
            data = await self._port.read(size)
            finished_ns = self._monotonic_ns()
            if data:
                self._frame_stream_usable = False
        return SerialReadChunk(data, started_ns, finished_ns)

    async def exchange_mac6000(
        self,
        request: bytes,
        *,
        matches_response: Callable[[bytes], bool],
        timeout_s: float = 1.0,
    ) -> bytes:
        """Exchange a framed MAC6000 request and retain unmatched complete frames.

        MAC6000 uses a binary payload length, so carriage return is not a safe
        line delimiter. This method reads the fixed header and its declared
        payload length. It fails closed if a partial or malformed frame leaves
        the serial stream position uncertain.
        """
        if self._event_reader_task is not None:
            return await self._exchange_mac6000_with_reader(
                request, matches_response=matches_response, timeout_s=timeout_s
            )
        async with self._exchange_lock:
            if not self._frame_stream_usable:
                raise RuntimeError("MAC6000 byte stream is unusable after a read error")
            deadline_ns = self._deadline(timeout_s)
            try:
                written = await self._write_mac6000(request)
                if written != len(request):
                    raise OSError(f"Short serial write: {written}/{len(request)} bytes")
                while True:
                    read_started_ns = self._monotonic_ns()
                    frame = await self._read_mac6000_frame(deadline_ns)
                    read_finished_ns = self._monotonic_ns()
                    if matches_response(frame):
                        return frame
                    self._queue_unsolicited(frame, read_started_ns, read_finished_ns)
            except BaseException:
                self._frame_stream_usable = False
                raise

    async def send_mac6000(self, request: bytes) -> None:
        """Write a binary command without waiting for a reply.

        This deliberately remains available after a read-side framing fault so
        a caller may attempt an explicit stop command. Success only means that
        the serial write completed; it does not confirm controller receipt or
        that motion stopped.
        """
        async with self._write_lock:
            try:
                written = await self._port.write(request)
                if written != len(request):
                    raise OSError(f"Short serial write: {written}/{len(request)} bytes")
            except BaseException:
                self._frame_stream_usable = False
                raise

    async def send_and_wait_mac6000(
        self,
        request: bytes,
        *,
        matches_event: Callable[[bytes], bool],
        timeout_s: float = 60.0,
    ) -> bytes:
        """Write a command and wait for its matching asynchronous event frame."""
        if self._event_reader_task is not None:
            return await self._exchange_mac6000_with_reader(
                request, matches_response=matches_event, timeout_s=timeout_s
            )
        async with self._exchange_lock:
            if not self._frame_stream_usable:
                raise RuntimeError("MAC6000 byte stream is unusable after a read error")
            deadline_ns = self._deadline(timeout_s)
            try:
                written = await self._write_mac6000(request)
                if written != len(request):
                    raise OSError(f"Short serial write: {written}/{len(request)} bytes")
                while True:
                    read_started_ns = self._monotonic_ns()
                    frame = await self._read_mac6000_frame(deadline_ns)
                    read_finished_ns = self._monotonic_ns()
                    if matches_event(frame):
                        return frame
                    self._queue_unsolicited(frame, read_started_ns, read_finished_ns)
            except BaseException:
                self._frame_stream_usable = False
                raise

    async def start_mac6000_event_reader(self) -> None:
        """Start the sole MAC6000 reader so idle reports retain receipt brackets.

        This mode is opt-in and cannot be mixed with line-oriented transport.
        Once stopped, the session is terminal and should be closed rather than
        reused because a partial frame may have been cancelled.
        """
        async with self._exchange_lock:
            if not self._frame_stream_usable:
                raise RuntimeError("MAC6000 byte stream is unusable after a read error")
            if self._event_reader_task is not None:
                raise RuntimeError("MAC6000 event reader is already running")
            if not self._unsolicited_frames.empty():
                raise RuntimeError(
                    "drain queued MAC6000 frames before starting the event reader"
                )
            self._event_reader_error = None
            task = asyncio.create_task(
                self._mac6000_event_reader(), name="mac6000-serial-event-reader"
            )
            task.add_done_callback(self._observe_event_reader)
            self._event_reader_task = task

    async def wait_mac6000_event(
        self,
        *,
        matches_event: Callable[[bytes], bool],
        timeout_s: float = 60.0,
    ) -> bytes:
        """Wait for a matching event without writing another command frame.

        This supports MAC6000 global-start operations whose completion reports
        arrive asynchronously after one broadcast command. Unmatched frames
        remain queued for the next waiter or telemetry consumer.
        """
        queued = self._take_queued_match(matches_event)
        if queued is not None:
            return queued
        if self._event_reader_task is not None:
            async with self._exchange_lock:
                if not self._frame_stream_usable:
                    reason = self._event_reader_error or "a read error"
                    raise RuntimeError(
                        f"MAC6000 byte stream is unusable after {reason}"
                    )
                if self._event_reader_task.done():
                    raise RuntimeError("MAC6000 event reader is not running")
                if self._pending_frame_future is not None:
                    raise RuntimeError("another MAC6000 response is already pending")
                future = asyncio.get_running_loop().create_future()
                self._pending_frame_matcher = matches_event
                self._pending_frame_future = future
                try:
                    return await asyncio.wait_for(future, timeout=timeout_s)
                except TimeoutError:
                    self._event_reader_error = TimeoutError(
                        "MAC6000 event timed out; a late frame may still arrive"
                    )
                    self._frame_stream_usable = False
                    raise
                finally:
                    if self._pending_frame_future is future:
                        self._pending_frame_future = None
                        self._pending_frame_matcher = None

        async with self._exchange_lock:
            if not self._frame_stream_usable:
                raise RuntimeError("MAC6000 byte stream is unusable after a read error")
            deadline_ns = self._deadline(timeout_s)
            try:
                while True:
                    queued = self._take_queued_match(matches_event)
                    if queued is not None:
                        return queued
                    started_ns = self._monotonic_ns()
                    frame = await self._read_mac6000_frame(deadline_ns)
                    finished_ns = self._monotonic_ns()
                    if matches_event(frame):
                        return frame
                    self._queue_unsolicited(frame, started_ns, finished_ns)
            except BaseException:
                self._frame_stream_usable = False
                raise

    def start_mac6000_event_capture(
        self, *, matches_event: Callable[[bytes], bool]
    ) -> asyncio.Queue[MAC6000ReceivedFrame]:
        """Reserve matching asynchronous reports while the event reader runs.

        Captured frames are also retained in the normal unsolicited queue for
        telemetry. The separate queue prevents a telemetry consumer from
        stealing a completion report before a multi-axis move has inspected it.
        """
        if self._event_reader_task is None or self._event_reader_task.done():
            raise RuntimeError("MAC6000 event reader is not running")
        if self._captured_events is not None:
            raise RuntimeError("another MAC6000 event capture is already active")
        self._captured_event_matcher = matches_event
        self._captured_events = asyncio.Queue()
        return self._captured_events

    def stop_mac6000_event_capture(
        self, events: asyncio.Queue[MAC6000ReceivedFrame]
    ) -> None:
        """Release one active reserved-event queue after its move has ended."""
        if events is not self._captured_events:
            raise RuntimeError("MAC6000 event capture handle is not active")
        self._captured_event_matcher = None
        self._captured_events = None

    def _take_queued_match(self, matcher: Callable[[bytes], bool]) -> bytes | None:
        """Take one queued matching frame and restore other frames in order."""
        retained: list[MAC6000ReceivedFrame] = []
        matched: bytes | None = None
        while True:
            try:
                item = self._unsolicited_frames.get_nowait()
            except asyncio.QueueEmpty:
                break
            if matched is None and matcher(item.data):
                matched = item.data
            else:
                retained.append(item)
        for item in retained:
            self._unsolicited_frames.put_nowait(item)
        return matched

    async def stop_mac6000_event_reader(self) -> None:
        """Stop the sole reader; this session must then be closed, not reused."""
        task = self._event_reader_task
        if task is None:
            return
        self._event_reader_task = None
        self._frame_stream_usable = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if self._event_reader_error is not exc:
                raise
        future = self._pending_frame_future
        if future is not None and not future.done():
            future.set_exception(RuntimeError("MAC6000 event reader stopped"))
        self._pending_frame_future = None
        self._pending_frame_matcher = None

    async def _exchange_mac6000_with_reader(
        self,
        request: bytes,
        *,
        matches_response: Callable[[bytes], bool],
        timeout_s: float,
    ) -> bytes:
        async with self._exchange_lock:
            if not self._frame_stream_usable:
                reason = self._event_reader_error or "a read error"
                raise RuntimeError(f"MAC6000 byte stream is unusable after {reason}")
            if self._event_reader_task is None or self._event_reader_task.done():
                raise RuntimeError("MAC6000 event reader is not running")
            if self._pending_frame_future is not None:
                raise RuntimeError("another MAC6000 response is already pending")
            loop = asyncio.get_running_loop()
            response_future: asyncio.Future[bytes] = loop.create_future()
            self._pending_frame_matcher = matches_response
            self._pending_frame_future = response_future
            deadline_ns = self._deadline(timeout_s)
            try:
                written = await self._write_mac6000(request)
                if written != len(request):
                    raise OSError(f"Short serial write: {written}/{len(request)} bytes")
                remaining = max(
                    0.0, (deadline_ns - self._monotonic_ns()) / 1_000_000_000
                )
                return await asyncio.wait_for(response_future, timeout=remaining)
            except TimeoutError:
                self._event_reader_error = TimeoutError(
                    "MAC6000 transaction timed out; late frame may still arrive"
                )
                self._frame_stream_usable = False
                raise TimeoutError(
                    "Timed out waiting for matching MAC6000 response/event"
                ) from None
            except asyncio.CancelledError:
                # The write may have reached the controller. Without a
                # transaction ID, a late reply could be attributed to a later
                # request, so this session must not accept another exchange.
                self._event_reader_error = RuntimeError(
                    "MAC6000 transaction cancelled; late frame may still arrive"
                )
                self._frame_stream_usable = False
                raise
            except BaseException as exc:
                self._event_reader_error = exc
                self._frame_stream_usable = False
                raise
            finally:
                if self._pending_frame_future is response_future:
                    self._pending_frame_future = None
                    self._pending_frame_matcher = None

    async def _mac6000_event_reader(self) -> None:
        """Continuously frame reports and route only the active reply."""
        try:
            while True:
                started_ns = self._monotonic_ns()
                frame = await self._read_mac6000_frame(None)
                finished_ns = self._monotonic_ns()
                matcher = self._pending_frame_matcher
                future = self._pending_frame_future
                matches = False
                if matcher is not None and future is not None and not future.done():
                    try:
                        matches = matcher(frame)
                    except (TypeError, ValueError):
                        matches = False
                if matches and future is not None:
                    future.set_result(frame)
                else:
                    captured_matcher = self._captured_event_matcher
                    captured_events = self._captured_events
                    if captured_matcher is not None and captured_events is not None:
                        try:
                            captured = captured_matcher(frame)
                        except (TypeError, ValueError):
                            captured = False
                        if captured:
                            captured_events.put_nowait(
                                MAC6000ReceivedFrame(frame, started_ns, finished_ns)
                            )
                    self._queue_unsolicited(frame, started_ns, finished_ns)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._event_reader_error = exc
            self._frame_stream_usable = False
            future = self._pending_frame_future
            if future is not None and not future.done():
                future.set_exception(exc)
            raise

    def _observe_event_reader(self, task: asyncio.Task[None]) -> None:
        """Retrieve background failures so they cannot become unhandled tasks."""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._event_reader_error = error
            self._frame_stream_usable = False

    async def read_unsolicited_frame(self) -> bytes:
        """Return the next complete nonmatching frame retained by an exchange."""
        return (await self.read_timed_frame()).data

    async def read_timed_frame(self) -> MAC6000ReceivedFrame:
        """Return an unmatched frame and the host-call bracket that received it."""
        return await self._unsolicited_frames.get()

    def read_timed_frame_nowait(self) -> MAC6000ReceivedFrame:
        """Return one queued frame immediately or raise ``asyncio.QueueEmpty``."""
        return self._unsolicited_frames.get_nowait()

    def _queue_unsolicited(
        self, frame: bytes, call_started_ns: int, call_finished_ns: int
    ) -> None:
        """Retain unmatched reports; overflow is a visible protocol failure."""
        try:
            self._unsolicited_frames.put_nowait(
                MAC6000ReceivedFrame(frame, call_started_ns, call_finished_ns)
            )
        except asyncio.QueueFull as exc:
            self._frame_stream_usable = False
            raise RuntimeError(
                "MAC6000 unmatched-frame queue capacity exceeded"
            ) from exc

    async def _write_mac6000(self, request: bytes) -> int:
        async with self._write_lock:
            return await self._port.write(request)

    def _deadline(self, timeout_s: float) -> int:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout must be a finite positive number")
        return self._monotonic_ns() + int(timeout_s * 1_000_000_000)

    async def _read_mac6000_frame(self, deadline_ns: int | None) -> bytes:
        header = await self._read_exactly(8, deadline_ns)
        if header[0] != 0x23:
            raise ValueError("MAC6000 frame has an invalid start byte")
        if header[3] != 0:
            raise ValueError("MAC6000 reserved byte must be zero")
        payload_length = int.from_bytes(header[6:8], "little")
        tail = await self._read_exactly(payload_length + 1, deadline_ns)
        if tail[-1] != 0x0D:
            raise ValueError("MAC6000 frame has an invalid terminator")
        return header + tail

    async def _read_exactly(self, size: int, deadline_ns: int | None) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            remaining_ns = (
                None if deadline_ns is None else deadline_ns - self._monotonic_ns()
            )
            if remaining_ns is not None and remaining_ns <= 0:
                raise TimeoutError("Incomplete MAC6000 frame before deadline")
            read = self._port.read(size - len(chunks))
            data = (
                await read
                if remaining_ns is None
                else await asyncio.wait_for(read, timeout=remaining_ns / 1_000_000_000)
            )
            if not data:
                sleep_s = (
                    0.005
                    if remaining_ns is None
                    else min(0.005, remaining_ns / 1_000_000_000)
                )
                await asyncio.sleep(sleep_s)
                continue
            if len(data) > size - len(chunks):
                raise OSError("Serial read returned more bytes than requested")
            chunks.extend(data)
        return bytes(chunks)

    def close(self) -> None:
        """Close this explicitly opened port."""
        if self._port.is_open:
            self._port.close()
