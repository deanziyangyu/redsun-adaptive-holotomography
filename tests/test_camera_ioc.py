from __future__ import annotations

import asyncio
import hashlib
import os
import socket
import subprocess
import sys
import time
from contextlib import suppress
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import aioca
import numpy as np
import pytest
from caproto.sync.client import read, write
from ophyd_async.core import wait_for_value
from redsun.device.epics import EpicsServiceDevice

from redsun_aht.device import CameraServiceDevice
from redsun_aht.device.camera.profiles import PVCAM_FCS_PROPERTIES
from redsun_aht.domain import ServiceConnectionState
from redsun_aht.services import camera_ioc
from redsun_aht.services.camera_ioc import build_camera_ioc


def test_two_camera_iocs_have_disjoint_control_planes() -> None:
    fluorescence = build_camera_ioc("fluorescence", "AHT:PVCAM:SIM:")
    dhm = build_camera_ioc("dhm", "AHT:FLIR:SIM:")

    assert set(fluorescence.pvdb).isdisjoint(dhm.pvdb)
    assert "AHT:PVCAM:SIM:SERVICE_ID" in fluorescence.pvdb
    assert "AHT:FLIR:SIM:COMMAND:TRIGGER" in dhm.pvdb
    assert len(fluorescence.pvdb) == len(dhm.pvdb) == 46


def test_camera_ioc_prefix_is_explicit() -> None:
    with pytest.raises(ValueError, match="end with"):
        build_camera_ioc("camera", "AMBIGUOUS")
    with pytest.raises(ValueError, match="generation"):
        build_camera_ioc("camera", "AHT:CAM:", generation=0)


def test_camera_ioc_lifecycle_hooks() -> None:
    async def scenario() -> None:
        ioc = build_camera_ioc("camera", "AHT:HOOK:", generation=3)

        class StopAfterHeartbeat:
            library: StopAfterHeartbeat

            def __init__(self) -> None:
                self.library = self

            async def sleep(self, delay: float) -> None:
                del delay
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await ioc.service_id.pvspec.startup(
                ioc, ioc.service_id, StopAfterHeartbeat()
            )
        assert ioc.service_id.value == "camera"
        assert ioc.schema_generation.value == 3

        with pytest.raises(RuntimeError, match="not connected"):
            await ioc.command_arm.pvspec.put(ioc, ioc.command_arm, 1)
        await ioc.command_connect.pvspec.put(ioc, ioc.command_connect, 1)
        with pytest.raises(RuntimeError, match="not armed"):
            await ioc.command_trigger.pvspec.put(ioc, ioc.command_trigger, 1)
        for _ in range(100):
            if ioc.connection_state.value == "ready":
                break
            await asyncio.sleep(0)
        else:
            raise TimeoutError("camera service did not enter its ready state")
        await ioc.command_arm.pvspec.put(ioc, ioc.command_arm, 1)
        await ioc.command_trigger.pvspec.put(ioc, ioc.command_trigger, 1)
        assert ioc.connection_state.value == "ready"
        assert ioc.acquisition_state.value == "armed"
        assert ioc.frames_published.value == 1
        assert ioc.descriptor_generation.value == 3
        assert ioc.descriptor_sequence.value == 0

        await ioc.command_stop.pvspec.put(ioc, ioc.command_stop, 1)
        await ioc.command_disconnect.pvspec.put(ioc, ioc.command_disconnect, 1)
        assert ioc.connection_state.value == "disconnected"
        assert ioc.acquisition_state.value == "idle"

    asyncio.run(scenario())


def test_camera_ioc_entry_point_builds_without_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(pvdb: object, **options: object) -> None:
        captured["pvdb"] = pvdb
        captured["options"] = options

    monkeypatch.setattr(camera_ioc, "run", fake_run)
    assert (
        camera_ioc.main(
            ["--service-id", "camera", "--prefix", "AHT:CLI:", "--generation", "2"]
        )
        == 0
    )
    assert "AHT:CLI:SERVICE_ID" in captured["pvdb"]


def test_camera_ioc_cli_applies_verified_pvcam_fcs_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from redsun_aht.device.camera import mmcore

    captured: dict[str, Any] = {}

    class RecordingBackend:
        def __init__(self, profile: object) -> None:
            captured["profile"] = profile

        def disconnect(self) -> None:
            pass

    monkeypatch.setattr(mmcore, "MMCoreCameraBackend", RecordingBackend)
    monkeypatch.setattr(camera_ioc, "run", lambda *args, **kwargs: None)
    config_path = tmp_path / "pvcam_base.cfg"
    config_path.write_text("# fake", encoding="utf-8")

    assert (
        camera_ioc.main(
            [
                "--service-id",
                "pvcam-fcs",
                "--prefix",
                "AHT:PVCAM:FCS:",
                "--backend",
                "mmcore",
                "--mm-path",
                str(tmp_path),
                "--mm-config",
                str(config_path),
                "--adapter",
                "PVCAM",
                "--device-name",
                "Camera-1",
                "--camera-label",
                "Camera-1",
                "--pvcam-fcs-profile",
            ]
        )
        == 0
    )
    profile = captured["profile"]
    assert profile.initial_properties == PVCAM_FCS_PROPERTIES


def _start_ioc(
    service_id: str,
    prefix: str,
    server_port: int,
    generation: int = 1,
) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
    environment["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    environment["EPICS_CA_SERVER_PORT"] = str(server_port)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "redsun_aht.services.camera_ioc",
            "--service-id",
            service_id,
            "--prefix",
            prefix,
            "--generation",
            str(generation),
        ],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _start_mmcore_ioc(
    service_id: str,
    prefix: str,
    server_port: int,
    *,
    mm_path: Path,
    config_path: Path,
    log_path: Path,
    adapter: str = "DemoCamera",
    device_name: str = "DCam",
    camera_label: str = "Camera",
    serial_property: str | None = None,
    expected_serial: str | None = None,
    run_id: str = "unassigned",
    acquisition_slots: int = 4,
) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
    environment["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    environment["EPICS_CA_SERVER_PORT"] = str(server_port)
    environment["PYMM_LOG_FILE"] = str(log_path)
    arguments = [
        sys.executable,
        "-m",
        "redsun_aht.services.camera_ioc",
        "--service-id",
        service_id,
        "--prefix",
        prefix,
        "--backend",
        "mmcore",
        "--run-id",
        run_id,
        "--acquisition-slots",
        str(acquisition_slots),
        "--mm-path",
        str(mm_path),
        "--mm-config",
        str(config_path),
        "--adapter",
        adapter,
        "--device-name",
        device_name,
        "--camera-label",
        camera_label,
    ]
    if serial_property is not None and expected_serial is not None:
        arguments.extend(
            ("--serial-property", serial_property, "--expected-serial", expected_serial)
        )
    return subprocess.Popen(
        arguments,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _read_value(pv: str) -> Any:
    value = read(pv, timeout=0.5, repeater=False).data[0]
    return value.decode() if isinstance(value, bytes) else value


def _await_value(pv: str, expected: object, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = _read_value(pv)
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


def _available_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def test_simulated_live_ioc_publishes_epics_latest_frame() -> None:
    prefix = "AHT:TEST:LIVE:"
    port = _available_udp_port()
    os.environ["EPICS_CA_ADDR_LIST"] = f"127.0.0.1:{port}"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    process = _start_ioc("live", prefix, port)
    try:
        _await_value(f"{prefix}SCHEMA_GENERATION", 1)
        write(f"{prefix}COMMAND:CONNECT", 1, notify=True, repeater=False)
        _await_value(f"{prefix}CONNECTION_STATE", "ready")
        write(f"{prefix}COMMAND:START_LIVE", 1, notify=True, repeater=False)
        _await_value(f"{prefix}LIVE_STATE", "running")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if int(_read_value(f"{prefix}LIVE_FRAMES_PUBLISHED")) > 0:
                break
            time.sleep(0.02)
        else:
            raise TimeoutError("simulated live IOC did not publish a frame")
        payload = read(f"{prefix}LIVE_IMAGE", timeout=0.5, repeater=False).data
        byte_count = int(_read_value(f"{prefix}LIVE_BYTE_COUNT"))
        assert byte_count == 12
        assert len(np.asarray(payload).tobytes()) >= byte_count
        assert int(_read_value(f"{prefix}LIVE_SEQUENCE")) >= 0
        assert int(_read_value(f"{prefix}LIVE_FRAMES_DRAINED")) >= 1
        write(f"{prefix}COMMAND:STOP", 1, notify=True, repeater=False)
        _await_value(f"{prefix}LIVE_STATE", "idle")
    finally:
        _stop_ioc(process)


def test_epics_live_gui_client_decodes_simulated_latest_frame() -> None:
    from redsun_aht.presenter.camera import EpicsLiveCameraGuiClient

    prefix = "AHT:TEST:LIVEGUI:"
    port = _available_udp_port()
    os.environ["EPICS_CA_ADDR_LIST"] = f"127.0.0.1:{port}"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    process = _start_ioc("live-gui", prefix, port)
    client = EpicsLiveCameraGuiClient(prefix, timeout=5)
    try:
        _await_value(f"{prefix}SCHEMA_GENERATION", 1)
        client.start()
        deadline = time.monotonic() + 3.0
        frame = None
        while time.monotonic() < deadline and frame is None:
            frame = client.read_latest()
            time.sleep(0.02)
        assert frame is not None
        assert frame.sequence >= 0
        assert frame.array.shape == (2, 3)
        assert frame.array.dtype == np.uint16
        client.stop()
        _await_value(f"{prefix}CONNECTION_STATE", "disconnected")
    finally:
        _stop_ioc(process)


def test_two_live_iocs_command_independently_and_restart() -> None:
    first_prefix = "AHT:TEST:A:"
    second_prefix = "AHT:TEST:B:"
    first_port = _available_udp_port()
    second_port = _available_udp_port()
    os.environ["EPICS_CA_ADDR_LIST"] = f"127.0.0.1:{first_port} 127.0.0.1:{second_port}"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    first = _start_ioc("first", first_prefix, first_port)
    second = _start_ioc("second", second_prefix, second_port)
    try:
        _await_value(f"{first_prefix}SCHEMA_GENERATION", 1)
        _await_value(f"{second_prefix}SCHEMA_GENERATION", 1)
        write(f"{first_prefix}COMMAND:CONNECT", 1, notify=True, repeater=False)
        _await_value(f"{first_prefix}CONNECTION_STATE", "ready")
        write(f"{first_prefix}COMMAND:ARM", 1, notify=True, repeater=False)
        write(f"{first_prefix}COMMAND:TRIGGER", 1, notify=True, repeater=False)
        _await_value(f"{first_prefix}FRAMES_PUBLISHED", 1)
        assert _read_value(f"{second_prefix}FRAMES_PUBLISHED") == 0

        async def exercise_ophyd_device() -> None:
            try:
                device = CameraServiceDevice(second_prefix, name="second")
                assert isinstance(device, EpicsServiceDevice)
                await device.connect(timeout=3)
                await device.request_connect()
                await wait_for_value(
                    device.connection_state,
                    ServiceConnectionState.READY.value,
                    timeout=3,
                )
                await device.arm()
                await device.trigger()
                status = await device.health()
                assert status.service_id == "second"
                assert status.connection_state is ServiceConnectionState.READY
                assert status.frames_published == 1
                await device.shutdown()
                assert await device.connection_state.get_value() == "disconnected"
            finally:
                aioca.purge_channel_caches()

        asyncio.run(exercise_ophyd_device())
    finally:
        _stop_ioc(first)
        _stop_ioc(second)

    restarted = _start_ioc("first", first_prefix, first_port, generation=2)
    try:
        _await_value(f"{first_prefix}SCHEMA_GENERATION", 2)
        assert _read_value(f"{first_prefix}FRAMES_PUBLISHED") == 0
    finally:
        _stop_ioc(restarted)


@pytest.mark.integration
@pytest.mark.skipif(
    "AHT_TEST_MMCORE_PATH" not in os.environ,
    reason="set AHT_TEST_MMCORE_PATH for the full descriptor integration test",
)
def test_lossless_detector_routes_caproto_ioc_through_ophyd_async(
    tmp_path: Path,
) -> None:
    prefix = "AHT:TEST:DETECTOR:"
    port = _available_udp_port()
    os.environ["EPICS_CA_ADDR_LIST"] = f"127.0.0.1:{port}"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    config_path = tmp_path / "detector.cfg"
    config_path.write_text(
        "\n".join(
            (
                "Property,Core,Initialize,0",
                "Device,Camera,DemoCamera,DCam",
                "Property,Core,Initialize,1",
                "Property,Core,Camera,Camera",
                "Property,Core,AutoShutter,0",
            )
        ),
        encoding="utf-8",
    )
    process = _start_mmcore_ioc(
        "detector",
        prefix,
        port,
        mm_path=Path(os.environ["AHT_TEST_MMCORE_PATH"]),
        config_path=config_path,
        log_path=tmp_path / "detector.log",
    )
    try:
        _await_value(f"{prefix}SCHEMA_GENERATION", 1)

        environment = os.environ.copy()
        environment["EPICS_CA_ADDR_LIST"] = f"127.0.0.1:{port}"
        environment["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
        client = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("epics_detector_client.py")),
                "detector",
                prefix,
            ],
            cwd=Path(__file__).parents[1],
            env=environment,
            capture_output=True,
            text=True,
        )
        assert client.returncode == 0, client.stderr
    finally:
        _stop_ioc(process)


@pytest.mark.skipif(
    "AHT_TEST_MMCORE_PATH" not in os.environ,
    reason="set AHT_TEST_MMCORE_PATH for installed DemoCamera adapters",
)
def test_two_live_mmcore_demo_iocs_use_separate_processes(tmp_path: Path) -> None:
    mm_path = Path(os.environ["AHT_TEST_MMCORE_PATH"])
    configured = os.environ.get("AHT_TEST_MMCORE_CONFIG")
    config_path = Path(configured) if configured else tmp_path / "camera-only.cfg"
    if configured is None:
        config_path.write_text(
            "\n".join(
                (
                    "Property,Core,Initialize,0",
                    "Device,Camera,DemoCamera,DCam",
                    "Property,Core,Initialize,1",
                    "Property,Core,Camera,Camera",
                    "Property,Core,AutoShutter,0",
                )
            ),
            encoding="utf-8",
        )
    first_prefix = "AHT:MMDEMO:A:"
    second_prefix = "AHT:MMDEMO:B:"
    first_port = _available_udp_port()
    second_port = _available_udp_port()
    os.environ["EPICS_CA_ADDR_LIST"] = f"127.0.0.1:{first_port} 127.0.0.1:{second_port}"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    first = _start_mmcore_ioc(
        "first",
        first_prefix,
        first_port,
        mm_path=mm_path,
        config_path=config_path,
        log_path=tmp_path / "first.log",
    )
    second = _start_mmcore_ioc(
        "second",
        second_prefix,
        second_port,
        mm_path=mm_path,
        config_path=config_path,
        log_path=tmp_path / "second.log",
    )
    try:
        _await_value(f"{first_prefix}SCHEMA_GENERATION", 1)
        _await_value(f"{second_prefix}SCHEMA_GENERATION", 1)
        write(f"{first_prefix}COMMAND:CONNECT", 1, notify=True, repeater=False)
        write(f"{second_prefix}COMMAND:CONNECT", 1, notify=True, repeater=False)
        _await_value(f"{first_prefix}CAMERA_ADAPTER", "DemoCamera")
        _await_value(f"{second_prefix}CAMERA_ADAPTER", "DemoCamera")
        _await_value(f"{first_prefix}CONNECTION_STATE", "ready")
        _await_value(f"{second_prefix}CONNECTION_STATE", "ready")
        write(f"{first_prefix}COMMAND:ARM", 1, notify=True, repeater=False)
        write(f"{first_prefix}COMMAND:TRIGGER", 1, notify=True, repeater=False)
        _await_value(f"{first_prefix}FRAMES_PUBLISHED", 1)
        assert _read_value(f"{second_prefix}FRAMES_PUBLISHED") == 0
        write(f"{second_prefix}COMMAND:ARM", 1, notify=True, repeater=False)
        write(f"{second_prefix}COMMAND:TRIGGER", 1, notify=True, repeater=False)
        _await_value(f"{second_prefix}FRAMES_PUBLISHED", 1)
        write(f"{first_prefix}COMMAND:DISCONNECT", 1, notify=True, repeater=False)
        write(f"{second_prefix}COMMAND:DISCONNECT", 1, notify=True, repeater=False)
        _await_value(f"{first_prefix}CONNECTION_STATE", "disconnected")
        _await_value(f"{second_prefix}CONNECTION_STATE", "disconnected")
    finally:
        _stop_ioc(first)
        _stop_ioc(second)


@pytest.mark.hardware
@pytest.mark.skipif(
    "AHT_TEST_REAL_MMCORE_CONFIG" not in os.environ,
    reason="set AHT_TEST_REAL_MMCORE_CONFIG for explicit camera admission",
)
def test_one_real_mmcore_camera_beside_simulated_peer(tmp_path: Path) -> None:
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
        pytest.fail(f"missing hardware admission variables: {', '.join(missing)}")

    real_prefix = "AHT:HW:REAL:"
    simulated_prefix = "AHT:HW:SIM:"
    real_port = _available_udp_port()
    simulated_port = _available_udp_port()
    os.environ["EPICS_CA_ADDR_LIST"] = (
        f"127.0.0.1:{real_port} 127.0.0.1:{simulated_port}"
    )
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    real = _start_mmcore_ioc(
        "real",
        real_prefix,
        real_port,
        mm_path=Path(os.environ["AHT_TEST_MMCORE_PATH"]),
        config_path=Path(os.environ["AHT_TEST_REAL_MMCORE_CONFIG"]),
        log_path=tmp_path / "real-camera.log",
        adapter=os.environ["AHT_TEST_REAL_ADAPTER"],
        device_name=os.environ["AHT_TEST_REAL_DEVICE_NAME"],
        camera_label=os.environ["AHT_TEST_REAL_CAMERA_LABEL"],
        serial_property=os.environ["AHT_TEST_REAL_SERIAL_PROPERTY"],
        expected_serial=os.environ["AHT_TEST_REAL_EXPECTED_SERIAL"],
    )
    simulated = _start_ioc("simulated", simulated_prefix, simulated_port)
    try:
        _await_value(f"{real_prefix}SCHEMA_GENERATION", 1)
        _await_value(f"{simulated_prefix}SCHEMA_GENERATION", 1)
        write(
            f"{real_prefix}COMMAND:CONNECT",
            1,
            notify=True,
            repeater=False,
            timeout=15.0,
        )
        _await_value(
            f"{real_prefix}CAMERA_ADAPTER", os.environ["AHT_TEST_REAL_ADAPTER"]
        )
        _await_value(
            f"{real_prefix}CAMERA_DEVICE",
            os.environ["AHT_TEST_REAL_DEVICE_NAME"],
        )
        _await_value(
            f"{real_prefix}CAMERA_SERIAL",
            os.environ["AHT_TEST_REAL_EXPECTED_SERIAL"],
        )
        _await_value(f"{real_prefix}CONNECTION_STATE", "ready", timeout=15.0)
        write(f"{simulated_prefix}COMMAND:CONNECT", 1, notify=True, repeater=False)
        _await_value(f"{simulated_prefix}CONNECTION_STATE", "ready")
        write(f"{simulated_prefix}COMMAND:ARM", 1, notify=True, repeater=False)
        write(f"{simulated_prefix}COMMAND:TRIGGER", 1, notify=True, repeater=False)
        _await_value(f"{simulated_prefix}FRAMES_PUBLISHED", 1)
        assert _read_value(f"{real_prefix}FRAMES_PUBLISHED") == 0
        write(
            f"{real_prefix}COMMAND:DISCONNECT",
            1,
            notify=True,
            repeater=False,
            timeout=15.0,
        )
        _await_value(f"{real_prefix}CONNECTION_STATE", "disconnected")
    finally:
        if real.poll() is None:
            with suppress(Exception):
                write(
                    f"{real_prefix}COMMAND:DISCONNECT",
                    1,
                    notify=True,
                    repeater=False,
                    timeout=15.0,
                )
        _stop_ioc(real)
        _stop_ioc(simulated)


@pytest.mark.hardware
@pytest.mark.skipif(
    os.environ.get("AHT_TEST_REAL_ACQUIRE") != "1",
    reason="set AHT_TEST_REAL_ACQUIRE=1 for one unilluminated frame",
)
def test_real_mmcore_frame_round_trip_beside_simulated_peer(tmp_path: Path) -> None:
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
        pytest.fail(f"missing hardware acquisition variables: {', '.join(missing)}")

    real_prefix = "AHT:HW:FRAME:"
    simulated_prefix = "AHT:HW:FRAME:SIM:"
    real_port = _available_udp_port()
    simulated_port = _available_udp_port()
    os.environ["EPICS_CA_ADDR_LIST"] = (
        f"127.0.0.1:{real_port} 127.0.0.1:{simulated_port}"
    )
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    real = _start_mmcore_ioc(
        "real",
        real_prefix,
        real_port,
        mm_path=Path(os.environ["AHT_TEST_MMCORE_PATH"]),
        config_path=Path(os.environ["AHT_TEST_REAL_MMCORE_CONFIG"]),
        log_path=tmp_path / "real-frame.log",
        adapter=os.environ["AHT_TEST_REAL_ADAPTER"],
        device_name=os.environ["AHT_TEST_REAL_DEVICE_NAME"],
        camera_label=os.environ["AHT_TEST_REAL_CAMERA_LABEL"],
        serial_property=os.environ["AHT_TEST_REAL_SERIAL_PROPERTY"],
        expected_serial=os.environ["AHT_TEST_REAL_EXPECTED_SERIAL"],
        run_id="hardware-smoke",
        acquisition_slots=1,
    )
    simulated = _start_ioc("simulated", simulated_prefix, simulated_port)
    memory_name = ""
    try:
        _await_value(f"{real_prefix}SCHEMA_GENERATION", 1)
        _await_value(f"{simulated_prefix}SCHEMA_GENERATION", 1)
        write(
            f"{real_prefix}COMMAND:CONNECT",
            1,
            notify=True,
            repeater=False,
            timeout=15.0,
        )
        _await_value(f"{real_prefix}CONNECTION_STATE", "ready", timeout=15.0)
        write(f"{real_prefix}COMMAND:ARM", 1, notify=True, repeater=False)
        write(
            f"{real_prefix}COMMAND:TRIGGER",
            1,
            notify=True,
            repeater=False,
            timeout=15.0,
        )
        _await_value(f"{real_prefix}FRAMES_PUBLISHED", 1)
        assert _read_value(f"{simulated_prefix}FRAMES_PUBLISHED") == 0
        assert _read_value(f"{real_prefix}DESCRIPTOR_RUN_ID") == "hardware-smoke"

        memory_name = str(_read_value(f"{real_prefix}DESCRIPTOR_NAME"))
        shape = tuple(
            int(item)
            for item in str(_read_value(f"{real_prefix}DESCRIPTOR_SHAPE")).split(",")
        )
        dtype = np.dtype(str(_read_value(f"{real_prefix}DESCRIPTOR_DTYPE")))
        slot = int(_read_value(f"{real_prefix}DESCRIPTOR_SLOT"))
        sequence = int(_read_value(f"{real_prefix}DESCRIPTOR_SEQUENCE"))
        frame_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        memory = shared_memory.SharedMemory(name=memory_name)
        try:
            copied = np.ndarray(
                shape,
                dtype=dtype,
                buffer=memory.buf,
                offset=slot * frame_bytes,
            ).copy()
        finally:
            memory.close()
        checksum = hashlib.sha256(copied.tobytes(order="C")).hexdigest()
        published_checksum = str(
            _read_value(f"{real_prefix}DESCRIPTOR_CHECKSUM_HI")
        ) + str(_read_value(f"{real_prefix}DESCRIPTOR_CHECKSUM_LO"))
        assert checksum == published_checksum
        assert copied.size > 0
        write(
            f"{real_prefix}COMMAND:ACKNOWLEDGE",
            sequence,
            notify=True,
            repeater=False,
        )
        _await_value(f"{real_prefix}COMMAND_STATE", "frame_acknowledged")
        write(f"{real_prefix}COMMAND:STOP", 1, notify=True, repeater=False)
        write(
            f"{real_prefix}COMMAND:DISCONNECT",
            1,
            notify=True,
            repeater=False,
            timeout=15.0,
        )
        _await_value(f"{real_prefix}CONNECTION_STATE", "disconnected")
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=memory_name)
    finally:
        if real.poll() is None:
            with suppress(Exception):
                write(
                    f"{real_prefix}COMMAND:DISCONNECT",
                    1,
                    notify=True,
                    repeater=False,
                    timeout=15.0,
                )
        _stop_ioc(real)
        _stop_ioc(simulated)
