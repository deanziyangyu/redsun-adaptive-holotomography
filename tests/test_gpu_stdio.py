from __future__ import annotations

import io
import queue
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from redsun_aht.processing import stdio_transport, stdio_worker


def test_stdio_frames_round_trip_and_reject_truncation() -> None:
    stream = io.BytesIO()
    stdio_transport._write_frame(stream, b"payload")
    stream.seek(0)
    assert stdio_transport._read_frame(stream) == b"payload"
    assert stdio_transport._read_frame(stream) is None

    with pytest.raises(ValueError, match="too large"):
        stdio_transport._write_frame(io.BytesIO(), b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(EOFError, match="before its payload"):
        stdio_transport._read_frame(io.BytesIO(struct.pack(">I", 2)))
    with pytest.raises(EOFError, match="unexpectedly"):
        stdio_transport._read_frame(io.BytesIO(b"\x00\x00"))
    with pytest.raises(ValueError, match="exceeds maximum"):
        stdio_transport._read_frame(io.BytesIO(struct.pack(">I", 17 * 1024 * 1024)))


def test_stdio_queue_and_cancel_adapters(tmp_path: Path) -> None:
    command_stream = io.BytesIO()
    command = stdio_transport._CommandQueue(command_stream)
    command.put(b"command")
    command_stream.seek(0)
    assert stdio_transport._read_frame(command_stream) == b"command"
    command.join_thread()

    response_stream = io.BytesIO()
    stdio_transport._write_frame(response_stream, b"first")
    stdio_transport._write_frame(response_stream, b"second")
    response_stream.seek(0)
    response = stdio_transport._ResponseQueue(response_stream)
    assert response.get(timeout=1) == b"first"
    assert response.get(timeout=1) == b"second"
    with pytest.raises(queue.Empty):
        response.get(timeout=0.01)
    response.join_thread()

    invalid_response = stdio_transport._ResponseQueue(io.BytesIO(b"\x00"))
    with pytest.raises(RuntimeError, match="response stream failed"):
        invalid_response.get(timeout=1)
    invalid_response.join_thread()

    cancel = stdio_transport._FileCancelEvent(tmp_path / "cancel.flag")
    cancel.set()
    assert (tmp_path / "cancel.flag").is_file()
    cancel.clear()
    cancel.close()
    assert not (tmp_path / "cancel.flag").exists()


class _FakeProcess:
    def __init__(self, *, stdin: Any = None, stdout: Any = None) -> None:
        self.stdin = io.BytesIO() if stdin is None else stdin
        response = io.BytesIO()
        stdio_transport._write_frame(response, b"ready")
        response.seek(0)
        self.stdout = response if stdout is None else stdout
        self.pid = 123
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -1


def test_gpu_stdio_start_uses_non_shell_explicit_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    fake = _FakeProcess()

    def popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        captured["command"] = command
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(subprocess, "Popen", popen)
    cache = str((tmp_path / "cache").resolve())

    command, response, cancel, process = stdio_transport.start_gpu_stdio_worker(
        "gpu-mean-projection", 1, 1024, cache, 2
    )

    assert response.get(timeout=1) == b"ready"
    assert captured["command"][0]
    assert captured["command"][1:3] == ["-m", "redsun_aht.processing.stdio_worker"]
    assert "--gpu-id" in captured["command"]
    assert captured["env"]["CUPY_CACHE_DIR"] == cache
    assert all(captured["env"][name] == cache for name in ("TEMP", "TMP", "TMPDIR"))
    assert "shell" not in captured
    assert process.pid == 123
    assert process.is_alive()
    process.join(0.01)
    with pytest.raises(RuntimeError, match="cannot close"):
        process.close()
    process.terminate()
    assert not process.is_alive()
    process.close()
    command.close()
    response.close()
    response.join_thread()
    cancel.close()

    with pytest.raises(ValueError, match="must be absolute"):
        stdio_transport.start_gpu_stdio_worker("gpu", 0, 1, "relative", 0)

    no_pipes = _FakeProcess()
    no_pipes.stdin = None
    no_pipes.stdout = None
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: no_pipes)
    with pytest.raises(RuntimeError, match="pipes were not created"):
        stdio_transport.start_gpu_stdio_worker("gpu", 0, 1, cache, 0)
    assert no_pipes.terminated


def test_stdio_worker_adapters_and_argument_forwarding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    framed = io.BytesIO()
    stdio_transport._write_frame(framed, b"command")
    framed.seek(0)
    input_queue = stdio_worker._InputQueue(framed)
    assert input_queue.get() == b"command"
    with pytest.raises(EOFError, match="command stream closed"):
        input_queue.get()

    output_stream = io.BytesIO()
    stdio_worker._OutputQueue(output_stream).put(b"response")
    output_stream.seek(0)
    assert stdio_transport._read_frame(output_stream) == b"response"

    cancel_file = tmp_path / "cancel.flag"
    probe = stdio_worker._FileCancelProbe(cancel_file)
    assert not probe.is_set()
    cancel_file.touch()
    assert probe.is_set()

    captured: tuple[object, ...] | None = None

    def worker(*args: object) -> None:
        nonlocal captured
        captured = args

    monkeypatch.setattr(stdio_worker, "processing_worker_main", worker)
    monkeypatch.setattr(
        sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO()),
    )
    monkeypatch.setattr(
        sys,
        "stdout",
        SimpleNamespace(buffer=io.BytesIO()),
    )
    cache = str((tmp_path / "cache").resolve())
    assert (
        stdio_worker.main(
            [
                "--solver-id",
                "gpu-mean-projection",
                "--gpu-id",
                "1",
                "--memory-reservation-bytes",
                "1024",
                "--cache-directory",
                cache,
                "--cancel-file",
                str(cancel_file),
                "--restart-count",
                "3",
            ]
        )
        == 0
    )
    assert captured is not None
    assert captured[0] == "gpu-mean-projection"
    assert captured[4:] == (3, 1, 1024, cache)
