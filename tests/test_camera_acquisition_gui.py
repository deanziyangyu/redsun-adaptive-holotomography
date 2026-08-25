from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import pytest
from caproto.sync.client import read, write

from redsun_aht.configurations import run_processing_simulation
from redsun_aht.presenter.camera import AcquiredCameraFrame, LiveCameraFrame
from redsun_aht.storage import CameraCaptureZarrStore

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def acquire_one(self) -> AcquiredCameraFrame:
        self.calls += 1
        return AcquiredCameraFrame(
            service_id="camera",
            run_id="gui-test",
            frame_id="frame-1",
            sequence=3,
            timestamp_ns=123,
            checksum="a" * 64,
            array=np.arange(12, dtype=np.uint16).reshape(3, 4),
        )


class _LiveClient:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.frames = [
            LiveCameraFrame(
                sequence=8,
                timestamp_ns=456,
                array=np.arange(12, dtype=np.uint16).reshape(3, 4),
            )
        ]

    def start(self) -> None:
        self.started += 1

    def read_latest(self) -> LiveCameraFrame | None:
        return self.frames.pop(0) if self.frames else None

    def stop(self) -> None:
        self.stopped += 1


def test_camera_acquisition_widget_captures_one_detached_frame() -> None:
    from qtpy import QtWidgets

    from redsun_aht.view.camera import CameraAcquisitionWidget

    application_instance = QtWidgets.QApplication.instance()
    application = (
        application_instance
        if isinstance(application_instance, QtWidgets.QApplication)
        else QtWidgets.QApplication([])
    )
    client = _Client()
    widget = CameraAcquisitionWidget("AHT:CAM:", client=client)

    widget.capture_once()

    assert application is not None
    assert client.calls == 1
    assert widget.last_frame is not None
    assert widget.last_frame.shape == (3, 4)
    assert "Captured camera sequence 3" in widget.status_label.text()
    assert widget.preview_label.pixmap() is not None
    widget.close()


def test_camera_acquisition_widget_surfaces_capture_failure() -> None:
    from qtpy import QtWidgets

    from redsun_aht.view.camera import CameraAcquisitionWidget

    class FailingClient:
        def acquire_one(self) -> AcquiredCameraFrame:
            raise RuntimeError("camera unavailable")

    application_instance = QtWidgets.QApplication.instance()
    application = (
        application_instance
        if isinstance(application_instance, QtWidgets.QApplication)
        else QtWidgets.QApplication([])
    )
    widget = CameraAcquisitionWidget("AHT:CAM:", client=FailingClient())

    widget.capture_once()

    assert application is not None
    assert widget.status_label.text() == "Failed: RuntimeError: camera unavailable"
    assert widget.capture_button.isEnabled()
    widget.close()


def test_camera_acquisition_widget_uses_sequence_backed_live_client() -> None:
    from qtpy import QtWidgets

    from redsun_aht.view.camera import CameraAcquisitionWidget

    application_instance = QtWidgets.QApplication.instance()
    application = (
        application_instance
        if isinstance(application_instance, QtWidgets.QApplication)
        else QtWidgets.QApplication([])
    )
    snap_client = _Client()
    live_client = _LiveClient()
    widget = CameraAcquisitionWidget(
        "AHT:CAM:", client=snap_client, live_client=live_client
    )

    widget.start_live()
    widget._poll_live()

    assert application is not None
    assert live_client.started == 1
    assert snap_client.calls == 0
    assert widget.last_frame is not None
    assert widget.last_frame.shape == (3, 4)
    assert widget.status_label.text() == "Live sequence 8"
    assert not widget.capture_button.isEnabled()
    widget.stop_live()
    assert live_client.stopped == 1
    assert widget.capture_button.isEnabled()
    widget.close()


def test_camera_gui_prefix_is_explicit() -> None:
    from redsun_aht.presenter.camera import EpicsCameraGuiClient

    try:
        EpicsCameraGuiClient("AHT:CAM")
    except ValueError as error:
        assert "end with" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("missing trailing colon was accepted")


def test_epics_camera_client_performs_ordered_acknowledged_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from redsun_aht.presenter.camera import EpicsCameraGuiClient

    client = EpicsCameraGuiClient("AHT:CAM:")
    values: dict[str, object] = {
        "CONNECTION_STATE": b"ready",
        "ACQUISITION_STATE": b"armed",
        "FRAMES_PUBLISHED": 0,
        "COMMAND_STATE": b"frame_acknowledged",
    }
    writes: list[tuple[str, int]] = []

    def fake_read(suffix: str) -> object:
        return values[suffix]

    def fake_write(suffix: str, value: int) -> None:
        writes.append((suffix, value))
        if suffix == "COMMAND:TRIGGER":
            values["FRAMES_PUBLISHED"] = 1

    frame = AcquiredCameraFrame(
        service_id="camera",
        run_id="run",
        frame_id="frame",
        sequence=0,
        timestamp_ns=4,
        checksum="a" * 64,
        array=np.ones((2, 3), dtype=np.uint16),
    )
    monkeypatch.setattr(client, "_read", fake_read)
    monkeypatch.setattr(client, "_write", fake_write)
    monkeypatch.setattr(client, "_copy_verified_frame", lambda: frame)

    assert client.acquire_one() is frame
    assert writes == [
        ("COMMAND:CONNECT", 1),
        ("COMMAND:ARM", 1),
        ("COMMAND:TRIGGER", 1),
        ("COMMAND:ACKNOWLEDGE", 0),
        ("COMMAND:STOP", 1),
        ("COMMAND:DISCONNECT", 1),
    ]
    assert client._text("CONNECTION_STATE") == "ready"
    assert client._integer("FRAMES_PUBLISHED") == 1


def test_camera_gui_configuration_delegates_to_optional_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import redsun_aht.view.camera as camera_view
    from redsun_aht.configurations.camera_gui import run_camera_acquisition_gui

    expected = object()
    monkeypatch.setattr(
        camera_view,
        "launch_camera_acquisition_gui",
        lambda prefix, *, run_event_loop: (
            expected if (prefix, run_event_loop) == ("AHT:CAM:", False) else None
        ),
    )

    assert run_camera_acquisition_gui("AHT:CAM:", run_event_loop=False) is expected


def _available_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _start_ioc(
    service_id: str, prefix: str, server_port: int, *, arguments: list[str]
) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
    environment["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    environment["EPICS_CA_SERVER_PORT"] = str(server_port)
    return subprocess.Popen(
        [sys.executable, "-m", "redsun_aht.services.camera_ioc", *arguments],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _await_value(pv: str, expected: object, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = read(pv, timeout=0.5, repeater=False).data[0]
            value = value.decode() if isinstance(value, bytes) else value
        except Exception:
            time.sleep(0.05)
            continue
        if value == expected:
            return
        time.sleep(0.05)
    raise TimeoutError(f"{pv} did not become {expected!r}")


def _stop_ioc(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.hardware
@pytest.mark.skipif(
    os.environ.get("AHT_TEST_REAL_GUI_ACQUIRE") != "1",
    reason="set AHT_TEST_REAL_GUI_ACQUIRE=1 for one GUI-mediated frame",
)
def test_camera_acquisition_widget_real_frame_beside_simulated_peer(
    tmp_path: Path,
) -> None:
    from qtpy import QtWidgets

    from redsun_aht.view.camera import CameraAcquisitionWidget

    required = (
        "AHT_TEST_MMCORE_PATH",
        "AHT_TEST_REAL_MMCORE_CONFIG",
        "AHT_TEST_REAL_ADAPTER",
        "AHT_TEST_REAL_DEVICE_NAME",
        "AHT_TEST_REAL_CAMERA_LABEL",
        "AHT_TEST_REAL_SERIAL_PROPERTY",
        "AHT_TEST_REAL_EXPECTED_SERIAL",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(f"missing GUI hardware variables: {', '.join(missing)}")
    real_prefix = "AHT:HW:GUI:"
    simulated_prefix = "AHT:HW:GUI:SIM:"
    real_port = _available_udp_port()
    simulated_port = _available_udp_port()
    os.environ["EPICS_CA_ADDR_LIST"] = (
        f"127.0.0.1:{real_port} 127.0.0.1:{simulated_port}"
    )
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    os.environ["PYMM_LOG_FILE"] = str(tmp_path / "gui-real-camera.log")
    real = _start_ioc(
        "real-gui",
        real_prefix,
        real_port,
        arguments=[
            "--service-id",
            "real-gui",
            "--prefix",
            real_prefix,
            "--backend",
            "mmcore",
            "--run-id",
            "gui-hardware-smoke",
            "--acquisition-slots",
            "1",
            "--mm-path",
            os.environ["AHT_TEST_MMCORE_PATH"],
            "--mm-config",
            os.environ["AHT_TEST_REAL_MMCORE_CONFIG"],
            "--adapter",
            os.environ["AHT_TEST_REAL_ADAPTER"],
            "--device-name",
            os.environ["AHT_TEST_REAL_DEVICE_NAME"],
            "--camera-label",
            os.environ["AHT_TEST_REAL_CAMERA_LABEL"],
            "--serial-property",
            os.environ["AHT_TEST_REAL_SERIAL_PROPERTY"],
            "--expected-serial",
            os.environ["AHT_TEST_REAL_EXPECTED_SERIAL"],
        ],
    )
    simulated = _start_ioc(
        "simulated-gui",
        simulated_prefix,
        simulated_port,
        arguments=["--service-id", "simulated-gui", "--prefix", simulated_prefix],
    )
    try:
        _await_value(f"{real_prefix}SCHEMA_GENERATION", 1)
        _await_value(f"{simulated_prefix}SCHEMA_GENERATION", 1)
        application_instance = QtWidgets.QApplication.instance()
        application = (
            application_instance
            if isinstance(application_instance, QtWidgets.QApplication)
            else QtWidgets.QApplication([])
        )
        widget = CameraAcquisitionWidget(real_prefix)
        widget.show()
        widget.capture_once()
        application.processEvents()
        assert widget.last_frame is not None, widget.status_label.text()
        assert widget.last_frame.size > 0
        assert widget.status_label.text().startswith("Captured real-gui sequence 0")
        capture = widget.last_capture
        assert capture is not None
        capture_root = tmp_path / "captured-frame"
        CameraCaptureZarrStore(capture_root).write(
            run_id=capture.run_id,
            service_id=capture.service_id,
            frame_id=capture.frame_id,
            sequence=capture.sequence,
            timestamp_ns=capture.timestamp_ns,
            array=capture.array,
        )
        processing = run_processing_simulation(capture_root, tmp_path / "products")
        assert set(processing.results) == {"mean-projection", "quality-metrics"}
        assert not processing.failures
        _await_value(f"{real_prefix}CONNECTION_STATE", "disconnected")
        simulated_count = read(
            f"{simulated_prefix}FRAMES_PUBLISHED", timeout=0.5, repeater=False
        ).data[0]
        assert int(simulated_count) == 0
        widget.close()
    finally:
        for process, prefix in ((real, real_prefix), (simulated, simulated_prefix)):
            if process.poll() is None:
                with suppress(Exception):
                    write(
                        f"{prefix}COMMAND:DISCONNECT",
                        1,
                        notify=True,
                        repeater=False,
                        timeout=5.0,
                    )
            _stop_ioc(process)
