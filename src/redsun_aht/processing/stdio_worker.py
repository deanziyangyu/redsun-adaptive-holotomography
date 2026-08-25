"""Command-line entry point for one length-framed GPU processing worker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from redsun_aht.processing.stdio_transport import _read_frame, _write_frame
from redsun_aht.processing.worker import processing_worker_main

if TYPE_CHECKING:
    from typing import BinaryIO


class _InputQueue:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def get(self) -> bytes:
        payload = _read_frame(self._stream)
        if payload is None:
            raise EOFError("GPU worker command stream closed")
        return payload


class _OutputQueue:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def put(self, payload: bytes) -> None:
        _write_frame(self._stream, payload)


class _FileCancelProbe:
    def __init__(self, path: Path) -> None:
        self._path = path

    def is_set(self) -> bool:
        return self._path.is_file()


def main(argv: list[str] | None = None) -> int:
    """Run one persistent GPU solver over binary stdin/stdout frames."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--solver-id", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--memory-reservation-bytes", type=int, required=True)
    parser.add_argument("--cache-directory", required=True)
    parser.add_argument("--cancel-file", type=Path, required=True)
    parser.add_argument("--restart-count", type=int, required=True)
    args = parser.parse_args(argv)
    processing_worker_main(
        args.solver_id,
        _InputQueue(sys.stdin.buffer),
        _OutputQueue(sys.stdout.buffer),
        _FileCancelProbe(args.cancel_file),
        args.restart_count,
        args.gpu_id,
        args.memory_reservation_bytes,
        args.cache_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
