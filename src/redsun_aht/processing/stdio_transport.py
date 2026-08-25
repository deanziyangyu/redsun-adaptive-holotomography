"""Length-framed parent transport for persistent GPU subprocess workers."""

from __future__ import annotations

import os
import queue
import struct
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from typing import BinaryIO

_MAX_FRAME_BYTES = 16 * 1024 * 1024


def _write_frame(stream: BinaryIO, payload: bytes) -> None:
    if not isinstance(payload, bytes) or len(payload) > _MAX_FRAME_BYTES:
        raise ValueError("processing stdio frame is invalid or too large")
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def _read_frame(stream: BinaryIO) -> bytes | None:
    header = _read_exact(stream, 4)
    if header is None:
        return None
    size = struct.unpack(">I", header)[0]
    if size > _MAX_FRAME_BYTES:
        raise ValueError("processing stdio frame exceeds maximum size")
    payload = _read_exact(stream, size)
    if payload is None:
        raise EOFError("processing stdio frame ended before its payload")
    return payload


def _read_exact(stream: BinaryIO, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise EOFError("processing stdio frame ended unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _CommandQueue:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def put(self, payload: bytes) -> None:
        with self._lock:
            _write_frame(self._stream, payload)

    def close(self) -> None:
        self._stream.close()

    def join_thread(self) -> None:
        pass


class _ResponseQueue:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._queue: queue.Queue[bytes | BaseException] = queue.Queue()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="aht-gpu-worker-response",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        try:
            while (payload := _read_frame(self._stream)) is not None:
                self._queue.put(payload)
        except BaseException as error:
            self._queue.put(error)

    def get(self, *, timeout: float) -> bytes:
        payload = self._queue.get(timeout=timeout)
        if isinstance(payload, BaseException):
            raise RuntimeError("GPU worker response stream failed") from payload
        return payload

    def close(self) -> None:
        self._stream.close()

    def join_thread(self) -> None:
        self._thread.join(timeout=1)


class _FileCancelEvent:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.clear()

    def clear(self) -> None:
        self._path.unlink(missing_ok=True)

    def set(self) -> None:
        self._path.touch(exist_ok=True)

    def close(self) -> None:
        self.clear()


class _SubprocessHandle:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def join(self, timeout: float) -> None:
        with suppress(subprocess.TimeoutExpired):
            self._process.wait(timeout=timeout)

    def terminate(self) -> None:
        self._process.terminate()

    def close(self) -> None:
        if self.is_alive():
            raise RuntimeError("cannot close a running GPU worker handle")


def start_gpu_stdio_worker(
    solver_id: str,
    gpu_id: int,
    memory_reservation_bytes: int,
    cache_directory: str,
    restart_count: int,
) -> tuple[Any, Any, Any, Any]:
    """Start one non-shell GPU subprocess and return queue-compatible adapters."""
    cache_path = Path(cache_directory)
    if not cache_path.is_absolute():
        raise ValueError("GPU worker cache path must be absolute")
    cache_path.mkdir(parents=True, exist_ok=True)
    cancel_path = cache_path / f"cancel-{solver_id}.flag"
    cancel_event = _FileCancelEvent(cancel_path)
    command = [
        sys.executable,
        "-m",
        "redsun_aht.processing.stdio_worker",
        "--solver-id",
        solver_id,
        "--gpu-id",
        str(gpu_id),
        "--memory-reservation-bytes",
        str(memory_reservation_bytes),
        "--cache-directory",
        str(cache_path),
        "--cancel-file",
        str(cancel_path),
        "--restart-count",
        str(restart_count),
    ]
    environment = dict(os.environ)
    for variable in ("CUPY_CACHE_DIR", "TEMP", "TMP", "TMPDIR"):
        environment[variable] = str(cache_path)
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        creationflags=creationflags,
        env=environment,
    )
    if process.stdin is None or process.stdout is None:
        process.terminate()
        raise RuntimeError("GPU worker stdio pipes were not created")
    return (
        _CommandQueue(cast("BinaryIO", process.stdin)),
        _ResponseQueue(cast("BinaryIO", process.stdout)),
        cancel_event,
        _SubprocessHandle(process),
    )


__all__ = ["start_gpu_stdio_worker"]
