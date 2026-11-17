"""Record finite sample-handler videos through an OpenCV capture source."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastcs.attributes import AttrR, AttrW, AttributeIO, AttributeIORef
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, String
from redsun.services import configure_logging

from ._process import controller_id, identity_arguments, run, serve


@dataclass
class WebcamRef(AttributeIORef):
    start: bool = False
    snap: bool = False


class WebcamIO(AttributeIO[bool, WebcamRef]):
    def __init__(self, webcam: WorkcellWebcamController):
        super().__init__()
        self.webcam = webcam

    async def send(self, attr: AttrW[bool, WebcamRef], value: bool) -> None:
        if value:
            if attr.io_ref.snap:
                task = await self.webcam.start_recording(still=True)
                await asyncio.shield(task)
            elif attr.io_ref.start:
                await self.webcam.start_recording()
            else:
                await self.webcam.stop_recording()


def record_clip(cv, source: int | str, backend: int, final: Path, fps: float,
                duration: float, stop: Path, *, exposure=None, flip=None) -> int:
    """Own one capture and writer, holding at most one frame at a time."""
    partial = final.with_suffix(".partial.avi")
    capture = cv.VideoCapture(source, backend)
    writer = None
    count = 0
    try:
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the workcell video source")
        if isinstance(source, int):
            if backend == cv.CAP_DSHOW:
                capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
            capture.set(cv.CAP_PROP_FPS, fps)
            if exposure is not None and not capture.set(cv.CAP_PROP_EXPOSURE, exposure):
                raise RuntimeError("Workcell camera rejected the requested exposure")
            for _ in range(7):
                capture.grab()
        started = time.monotonic()
        limit = math.ceil(duration * fps)
        while count < limit and not stop.exists() and time.monotonic() < started + duration:
            ok, frame = capture.read()
            if not ok:
                if isinstance(source, str) and count:
                    break
                raise RuntimeError("OpenCV could not read a workcell frame")
            height, width = frame.shape[:2]
            if frame.ndim != 3 or frame.shape[2] != 3 or not (0 < width <= 1920 and 0 < height <= 1080):
                raise RuntimeError("Workcell frames must be BGR and at most 1920 × 1080")
            if writer is None:
                if isinstance(source, int) and (width, height) != (1920, 1080):
                    raise RuntimeError(
                        f"Workcell camera delivered {width}x{height}, expected 1920x1080"
                    )
                shape = frame.shape
                writer = cv.VideoWriter(str(partial), cv.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not open the workcell recording file")
            if frame.shape != shape:
                raise RuntimeError("Workcell frame layout changed during recording")
            writer.write(cv.flip(frame, flip) if flip is not None else frame)
            count += 1
            del frame
            time.sleep(max(0, started + count / fps - time.monotonic()))
        if not count:
            raise RuntimeError("No workcell frames were recorded")
        writer.release()
        writer = None
        if partial.stat().st_size == 0:
            raise RuntimeError("OpenCV produced an empty workcell recording")
        partial.rename(final)
        return count
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        partial.unlink(missing_ok=True)


class WorkcellWebcamController(Controller):
    def __init__(self, source: int | str, backend: int, directory: Path,
                 *, fps: float, duration: float, exposure=None, flip=None):
        self._source = source
        self._backend = backend
        self._directory = directory
        self._fps = fps
        self._duration = duration
        self._exposure = exposure
        self._flip = flip
        self._stop: Path | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        super().__init__(ios=[WebcamIO(self)])
        self.start = AttrW(Bool(), io_ref=WebcamRef(start=True))
        self.snap = AttrW(Bool(), io_ref=WebcamRef(snap=True))
        self.finish = AttrW(Bool(), io_ref=WebcamRef())
        self.status = AttrR(String(), initial_value=json.dumps({"state": "idle"}))

    async def start_recording(self, *, still: bool = False) -> asyncio.Task[None]:
        if self._closing or (self._task is not None and not self._task.done()):
            raise RuntimeError("Workcell recorder is busy or closing")
        self._directory.mkdir(parents=True, exist_ok=True)
        stem = datetime.now(timezone.utc).strftime("workcell-%Y%m%dT%H%M%SZ-") + uuid4().hex
        final = self._directory / (stem + (".png" if still else ".avi"))
        self._stop = final.with_suffix(".stop")
        self._task = asyncio.create_task(self._record(final, self._stop))
        return self._task

    async def _record(self, path: Path, stop: Path) -> None:
        still = path.suffix == ".png"
        await self.status.update(json.dumps({"state": "snapping" if still else "recording"}))
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "redsun_aht.robotic.services.workcell_webcam",
                "--record-worker", json.dumps(self._source), str(self._backend),
                str(path), str(self._fps), str(self._duration), str(stop),
                json.dumps(self._exposure), json.dumps(self._flip),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            )
            self._process = process
            async with asyncio.timeout(8 if still else self._duration + 8):
                output, _ = await process.communicate()
            if not output:
                raise RuntimeError(f"Workcell recording worker exited with code {process.returncode}")
            result = json.loads(output)
            if process.returncode:
                raise RuntimeError(result["error"])
            count = result["frames"]
        except Exception as error:
            message = str(error) or type(error).__name__
            status = {"state": "error", "error": message}
        else:
            status = {"state": "complete", "path": str(path), "frames": count}
        finally:
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.communicate()
            path.with_suffix(f".partial{path.suffix}").unlink(missing_ok=True)
            stop.unlink(missing_ok=True)
            self._process = None
        await self.status.update(json.dumps(status))

    async def stop_recording(self) -> None:
        task = self._task
        if task is None or task.done():
            return
        self._stop.touch()
        try:
            await asyncio.wait_for(asyncio.shield(task), 3)
        except TimeoutError:
            process = self._process
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await asyncio.shield(task)

    async def disconnect(self) -> None:
        self._closing = True
        try:
            await self.stop_recording()
        finally:
            await super().disconnect()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera-index", type=int)
    source.add_argument("--replay-file", type=Path)
    parser.add_argument("--backend", choices=("any", "dshow", "msmf"), default="any")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--exposure", type=float, help="OpenCV backend-specific exposure value")
    parser.add_argument("--h-flip", action="store_true")
    parser.add_argument("--v-flip", action="store_true")
    parser.add_argument("--duration", type=float, default=60, help="Maximum clip duration in seconds")
    identity_arguments(parser, "workcell-webcam")
    options = parser.parse_args(argv)
    if options.exposure is not None and (
        options.camera_index is None or not math.isfinite(options.exposure)
    ):
        parser.error("Exposure requires a live camera and a finite value")
    if (not 1 <= options.fps <= 30 or not 0.1 <= options.duration <= 600
            or (options.camera_index is not None and options.camera_index < 0)):
        parser.error("Use 1–30 fps, 0.1–600 seconds and a nonnegative camera index")
    if options.replay_file is not None and not options.replay_file.is_file():
        parser.error("Replay input must be an existing local video file")
    try:
        import cv2
    except ImportError:
        parser.error("Install the camera-webcam extra to use the workcell recorder")
    configure_logging()
    backend = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[options.backend]
    video_source = options.camera_index if options.replay_file is None else str(options.replay_file.resolve())
    controller = WorkcellWebcamController(
        video_source, backend, options.output_dir.resolve(),
        fps=options.fps, duration=options.duration,
        exposure=options.exposure,
        flip=(-1 if options.v_flip else 1) if options.h_flip else (0 if options.v_flip else None),
    )

    async def run_service():
        try:
            await serve(controller, controller_id(options))
        finally:
            await controller.disconnect()

    return run(run_service())


def _record_worker(source: str, backend: str, output: str, fps: str, duration: str,
                   stop: str, exposure_json: str = "null", flip_json: str = "null") -> int:
    try:
        import cv2

        path = Path(output)
        video_source = json.loads(source)
        exposure = json.loads(exposure_json)
        flip = json.loads(flip_json)
        if path.suffix == ".png":
            capture = cv2.VideoCapture(video_source, int(backend))
            partial = path.with_suffix(".partial.png")
            try:
                if not capture.isOpened():
                    raise RuntimeError("OpenCV could not open the workcell video source")
                if isinstance(video_source, int):
                    if int(backend) == cv2.CAP_DSHOW:
                        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    if exposure is not None and not capture.set(cv2.CAP_PROP_EXPOSURE, exposure):
                        raise RuntimeError("Workcell camera rejected the requested exposure")
                    for _ in range(7):
                        capture.grab()
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("OpenCV could not read a workcell still frame")
                if isinstance(video_source, int) and frame.shape[:2] != (1080, 1920):
                    raise RuntimeError("Workcell still capture did not deliver 1920x1080")
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] > 1080 or frame.shape[1] > 1920:
                    raise RuntimeError("Workcell still must be BGR and at most 1920 × 1080")
                if not cv2.imwrite(str(partial), cv2.flip(frame, flip) if flip is not None else frame):
                    raise RuntimeError("OpenCV could not write the workcell still")
                partial.replace(path)
                count = 1
            finally:
                capture.release()
                partial.unlink(missing_ok=True)
        else:
            count = record_clip(
                cv2, video_source, int(backend), path, float(fps), float(duration), Path(stop),
                exposure=exposure, flip=flip,
            )
    except Exception as error:
        print(json.dumps({"error": str(error)}), flush=True)
        return 1
    print(json.dumps({"frames": count}), flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--record-worker"]:
        raise SystemExit(_record_worker(*sys.argv[2:]))
    raise SystemExit(main())
