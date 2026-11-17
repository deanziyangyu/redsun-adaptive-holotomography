"""OpenCV backend for the Redsun MM camera PV contract."""

import argparse

from redsun.services import configure_logging

from ..robotic.services._process import controller_id, identity_arguments, run, serve
from .mmcore_camera import MMCameraController


class OpenCVCameraCore:
    """Supply the MM camera controller's capture and fixed-layout settings."""

    def __init__(self, cv, index: int, backend: int, fps: float):
        self.cv = cv
        self.index = index
        self.backend = backend
        self.fps = fps
        self.capture = None
        self.roi = (0, 0, 1920, 1080)

    def _open(self):
        capture = self.cv.VideoCapture(self.index, self.backend)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open OpenCV camera {self.index}")
            if self.backend == self.cv.CAP_DSHOW:
                capture.set(self.cv.CAP_PROP_FOURCC, self.cv.VideoWriter_fourcc(*"MJPG"))
            capture.set(self.cv.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(self.cv.CAP_PROP_FRAME_HEIGHT, 1080)
            capture.set(self.cv.CAP_PROP_FPS, self.fps)
            for _ in range(7):
                capture.grab()
            return capture
        except BaseException:
            capture.release()
            raise

    def _read(self, capture):
        ok, frame = capture.read()
        if not ok or frame is None or frame.shape != (1080, 1920, 3):
            raise RuntimeError("OpenCV camera did not deliver a 1920x1080 color frame")
        x, y, width, height = self.roi
        return self.cv.cvtColor(frame[y:y + height, x:x + width], self.cv.COLOR_BGR2GRAY)

    def snap(self):
        capture = self._open()
        try:
            return self._read(capture)
        finally:
            capture.release()

    def startContinuousSequenceAcquisition(self, interval):
        if self.capture is not None:
            raise RuntimeError("OpenCV camera is already streaming")
        self.capture = self._open()

    def stopSequenceAcquisition(self):
        capture, self.capture = self.capture, None
        if capture is not None:
            capture.release()

    def isSequenceRunning(self):
        return self.capture is not None

    def getRemainingImageCount(self):
        return int(self.isSequenceRunning())

    def popNextImage(self):
        return self._read(self.capture)

    def popNextImageAndMD(self):
        return self.popNextImage(), {}

    def getROI(self):
        return self.roi

    def clearROI(self):
        self.roi = (0, 0, 1920, 1080)

    def setROI(self, x, y, width, height):
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1920 or y + height > 1080:
            raise ValueError("ROI is outside the OpenCV sensor area")
        self.roi = (x, y, width, height)

    def getImageWidth(self):
        return self.roi[2]

    def getImageHeight(self):
        return self.roi[3]

    def getCameraDevice(self):
        return "OpenCV"

    def getExposure(self):
        return 0.0

    def setExposure(self, value):
        raise ValueError("Exposure in milliseconds is unavailable for this OpenCV backend")

    def getDevicePropertyNames(self, label):
        return ("PixelType",)

    def hasProperty(self, label, name):
        return name == "PixelType"

    def isPropertyReadOnly(self, label, name):
        return True

    def getAllowedPropertyValues(self, label, name):
        return (self.getProperty(label, name),)

    def getProperty(self, label, name):
        if not self.hasProperty(label, name):
            raise ValueError(f"Unknown OpenCV property {name}")
        return "8bit"

    def setProperty(self, label, name, value):
        if value != self.getProperty(label, name):
            raise ValueError(f"OpenCV property {name} is fixed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    identity_arguments(parser, "opencv-camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--backend", choices=("any", "dshow", "msmf"), default="dshow")
    parser.add_argument("--fps", type=float, default=10)
    options = parser.parse_args(argv)
    if options.camera_index < 0 or not 1 <= options.fps <= 30:
        parser.error("Use a nonnegative camera index and 1–30 fps")
    import cv2

    configure_logging()
    backend = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[options.backend]
    core = OpenCVCameraCore(cv2, options.camera_index, backend, options.fps)
    controller = MMCameraController(core, "OpenCV", options.name, defer_initial_frame=True)
    return run(serve(controller, controller_id(options)))


if __name__ == "__main__":
    raise SystemExit(main())
