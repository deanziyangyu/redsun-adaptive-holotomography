"""Isolate native external-trigger DCAM calls in a disposable process."""

import asyncio
import hashlib
import logging
import os
import pickle
import struct
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .unimmcore_triggered_camera import DCAM_STACK_MAX_BYTES, dcam_external_stack_camera

logger = logging.getLogger(__name__)


class DcamProcess:
    def __init__(self, config, label, polarity, mm_path, *, config_digest=""):
        self._args = (str(config), label, polarity, str(mm_path or ""), config_digest)
        self._process = None
        self._launch = None
        self._interrupted = False
        self._closed = False
        self._closing = None
        self._request_lock = asyncio.Lock()

    @property
    def released(self):
        return self._closed and (
            (self._launch is None and self._process is None)
            or (self._process is not None and self._process.returncode is not None)
        )

    async def _request(self, command, count=0):
        async with self._request_lock:
            if self._closed and command != "close":
                raise RuntimeError("External DCAM checkout is closed")
            if self._interrupted:
                raise RuntimeError("External DCAM worker exchange was interrupted")
            return await self._exchange(command, count)

    async def _exchange(self, command, count):
        process = self._process
        try:
            process.stdin.write(f"{command} {count}\n".encode("ascii"))
            await process.stdin.drain()
            ok, result = await self._read_response()
            if ok and command == "collect":
                if type(result) is not int or not 0 <= result <= count:
                    raise RuntimeError("Invalid frame count from external DCAM worker")
                frames = []
                remaining = DCAM_STACK_MAX_BYTES
                for _ in range(result):
                    frame = await self._read_response(remaining + 4096)
                    remaining -= frame.data.nbytes
                    if remaining < 0:
                        raise RuntimeError("External DCAM response exceeded its frame budget")
                    frame.data.setflags(write=False)
                    frames.append(frame)
                result = tuple(frames)
        except (asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError) as error:
            self._interrupted = True
            raise RuntimeError(
                f"External DCAM worker disconnected during {command}"
            ) from error
        except BaseException:
            self._interrupted = True
            raise
        if not ok:
            raise RuntimeError(result)
        return result

    async def _read_response(self, max_bytes=65536):
        stream = self._process.stdout
        size, = struct.unpack("!I", await stream.readexactly(4))
        if size > max_bytes:
            raise RuntimeError("External DCAM response exceeded its frame budget")
        # The private pipe is connected only to the child started below.
        return pickle.loads(await stream.readexactly(size))

    async def arm(self, frame_count):
        if self._closed or self._launch is not None or self._process is not None:
            raise RuntimeError("External DCAM checkout is one-use")
        self._launch = asyncio.create_task(asyncio.create_subprocess_exec(
            sys.executable, "-m", "redsun_aht.device.dcam_process", *self._args,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        ))
        try:
            self._process = await asyncio.shield(self._launch)
        except asyncio.CancelledError:
            self._interrupted = True
            raise
        if self._closed:
            raise RuntimeError("External DCAM checkout closed during worker startup")
        return await self._request("arm", frame_count)

    async def confirm_ready(self):
        await self._request("ready")

    async def collect(self, frame_count):
        return await self._request("collect", frame_count)

    async def finish(self):
        try:
            async with asyncio.timeout(3):
                await self._request("finish")
        except TimeoutError as error:
            raise RuntimeError("External DCAM sequence did not stop within three seconds") from error

    async def close(self):
        self._closed = True
        if self._closing is None:
            self._closing = asyncio.create_task(self._close_worker())
            self._closing.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )
        await asyncio.shield(self._closing)

    async def _close_worker(self):
        if self._launch is not None and self._process is None:
            try:
                self._process = await self._launch
            except OSError:
                self._launch = None
                return
        process = self._process
        if process is None:
            return
        if process.returncode is not None:
            if process.returncode and not self._interrupted:
                raise RuntimeError(f"External DCAM worker exited with code {process.returncode} before close")
            return
        try:
            if not self._request_lock.locked() and not self._interrupted and not process.stdin.is_closing():
                async with asyncio.timeout(1):
                    await self._request("close")
                    code = await process.wait()
                    if code:
                        raise RuntimeError(f"External DCAM worker exited with code {code} after acknowledging close")
        except TimeoutError as error:
            raise RuntimeError("External DCAM did not close cleanly. Terminating its worker.") from error
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                # Drain any buffered response so the pipe cannot delay reaping.
                process.stdin.close()
                async with self._request_lock:
                    while await process.stdout.read(65536):
                        pass
                await process.wait()
                logger.warning("Terminated and reaped external DCAM worker %s", process.pid)


async def _serve(config, label, polarity, mm_path, config_digest):
    wire = sys.stdout.buffer
    sys.stdout = sys.stderr
    camera = None
    try:
        for line in sys.stdin:
            command, count = line.split()
            try:
                if command == "arm":
                    config_bytes = Path(config).read_bytes()
                    if config_digest and hashlib.sha256(config_bytes).hexdigest() != config_digest:
                        raise RuntimeError("External DCAM config changed after validation; restart the session")
                    from pymmcore_plus import CMMCorePlus

                    core = CMMCorePlus(mm_path=mm_path) if mm_path else CMMCorePlus()
                    camera = dcam_external_stack_camera(core, device=label, polarity=polarity, owns_core=True)
                    with TemporaryDirectory(prefix="arh-dcam-") as directory:
                        snapshot = Path(directory) / "external.cfg"
                        snapshot.write_bytes(config_bytes)
                        core.loadSystemConfiguration(str(snapshot))
                    result = await camera.arm(int(count))
                elif command == "collect":
                    await camera.collect(int(count))
                    result = camera.take_frames()
                elif command == "ready":
                    result = await camera.confirm_ready()
                elif command == "finish":
                    result = await camera.finish()
                elif command == "close":
                    result = await camera.close() if camera is not None else None
                else:
                    raise ValueError(f"Unknown DCAM command: {command}")
                response = True, result
            except Exception as error:
                response = False, " ".join((str(error), *getattr(error, "__notes__", ())))
            if command == "collect" and response[0]:
                _write_response(wire, (True, len(result)))
                result.reverse()
                while result:
                    _write_response(wire, result.pop())
            else:
                _write_response(wire, response)
            wire.flush()
            if command == "close":
                return
    finally:
        if camera is not None and not camera.released:
            try:
                async with asyncio.timeout(1):
                    await camera.close()
            except TimeoutError:
                # The parent pipe may be gone and native cleanup can block indefinitely.
                os._exit(1)


def _write_response(wire, response):
    payload = pickle.dumps(response, protocol=5)
    wire.write(struct.pack("!I", len(payload)))
    wire.write(payload)


if __name__ == "__main__":
    asyncio.run(_serve(*sys.argv[1:]))
