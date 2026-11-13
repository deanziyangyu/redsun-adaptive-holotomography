"""Tests for the ``MMCamera`` device over the camera service's signal set.

The device is connected in mock mode, so what is checked is its own wiring and
logic: which settings it reports, when ``trigger`` completes, what the optional
dark-frame retry does, and what the plan-facing ``redsun_aht.contracts`` adapter
makes of the published signals. The service behind the signals is qualified in
``test_mmcore_services.py``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from ophyd_async.core import get_mock_put, set_mock_value

from redsun_aht.contracts import (
    Camera,
    CameraSettings,
    Roi,
    SettingRefused,
    TriggerMode,
)
from redsun_aht.device.mmcore import MMCamera
from redsun_aht.device.mmcore import camera as camera_device
from redsun_aht.device.mmcore.camera import (
    DEFAULT_DARK_FRAME_STRUCTURE_DN,
    FAULTED,
    frame_has_signal,
    frame_shape_and_dtype,
    trigger_until_lit,
)
from redsun_aht.device.spinnaker_camera import SpinnakerCamera
from redsun_aht.protocols import LiveFlyer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

SHAPE = (4, 6)


@pytest.fixture(params=[MMCamera, SpinnakerCamera], ids=["mmcore", "spinnaker"])
async def camera(
    tmp_path: Path, request: pytest.FixtureRequest
) -> AsyncGenerator[MMCamera, None]:
    """Return a camera connected to mock signals instead of a service."""
    from redsun.path_provider import SessionPathProvider

    device = request.param(
        "PREFIX:CAM:",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="test"),
        name="cam",
    )
    await device.connect(mock=True)
    set_mock_value(device.state, "acquiring")
    set_mock_value(device.last_error, "")
    set_mock_value(device.frame_count, 0)
    set_mock_value(device.roi, "0,0,6,4")
    set_mock_value(device.pixel_dtype, "uint8")
    set_mock_value(device.pixel_dtypes, "uint8,uint16")
    set_mock_value(device.binning_factors, "1,2")
    set_mock_value(device.sensor_size_binning, "1x1")
    set_mock_value(device.sensor_size, np.array([6, 4], dtype=np.int32))
    set_mock_value(device.exposure, 10.0)
    set_mock_value(device.exposure_range, np.array([0.1, 1000.0]))
    set_mock_value(device.settings_revision, 7)
    set_mock_value(device.buffer, np.zeros(SHAPE, dtype=np.uint8))
    yield device


def pitched_frame(count: int = 200) -> np.ndarray:
    """Return a frame with enough spread to count as lit."""
    frame = np.zeros(SHAPE, dtype=np.uint16)
    frame[0, 0] = count
    return frame


async def publish_frames(camera: MMCamera, values: list[int]) -> None:
    """Publish the frame count a service would publish for each frame."""
    for value in values:
        set_mock_value(camera.frame_count, value)
        await asyncio.sleep(0.02)


async def test_frame_shape_and_dtype_come_from_the_roi_and_pixel_dtype(
    camera: MMCamera,
) -> None:
    """Size frames from the ROI text and the dtype the camera reads out in."""
    set_mock_value(camera.roi, "10,20,100,50")
    set_mock_value(camera.pixel_dtype, "uint16")

    assert await frame_shape_and_dtype(camera) == ((50, 100), "uint16")


async def test_the_configuration_names_the_settings_a_reader_needs(
    camera: MMCamera,
) -> None:
    """Describe the settings by name, and read the ones a record carries."""
    described = set(await camera.describe_configuration())

    assert described == {
        "cam-exposure",
        "cam-pixel_dtype",
        "cam-record_frame_timing",
        "cam-roi",
        "cam-sensor_size",
        "cam-sensor_size_binning",
        "cam-last_setting_change",
    }

    readings = await camera.read_configuration()

    assert set(readings) == described
    assert readings["cam-roi"]["value"] == "0,0,6,4"
    assert readings["cam-pixel_dtype"]["value"] == "uint8"


async def test_a_trigger_completes_on_a_frame_taken_after_it(
    camera: MMCamera,
) -> None:
    """Complete the trigger on a frame count newer than the one it saw."""
    task = asyncio.ensure_future(camera.trigger())

    await publish_frames(camera, [1, 2])

    await asyncio.wait_for(task, 2.0)


async def test_a_trigger_takes_one_frame_by_default(
    camera: MMCamera, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Take exactly one exposure per trigger, even when the frame carries no light."""
    monkeypatch.setattr(camera_device, "DEFAULT_TIMEOUT", 0.1)

    task = asyncio.ensure_future(camera.trigger())
    await asyncio.sleep(0.02)
    # the buffer holds no light, so a retaking trigger would wait for a third
    # frame that never comes and time out instead
    await publish_frames(camera, [1])
    await asyncio.wait_for(task, 1.0)

    assert camera._dark_frame_retries == 0


async def test_a_trigger_times_out_and_names_the_frame_count(
    camera: MMCamera, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail with the count it was waiting past when the service sends nothing."""
    monkeypatch.setattr(camera_device, "DEFAULT_TIMEOUT", 0.05)

    with pytest.raises(TimeoutError, match="still at 0"):
        await camera.trigger()


async def test_a_faulted_camera_raises_its_own_error(camera: MMCamera) -> None:
    """Raise the camera's error rather than waiting for a frame that is not coming."""
    set_mock_value(camera.state, FAULTED)
    set_mock_value(camera.last_error, "ValueError: the camera was unplugged")

    with pytest.raises(RuntimeError, match="the camera was unplugged"):
        await camera.trigger()


def test_an_unlit_frame_is_told_from_a_lit_one() -> None:
    """Separate the offset noise of a dark frame from the spread of a lit frame."""
    dark = np.full(SHAPE, 100, dtype=np.uint16)
    noise = np.full(SHAPE, 100, dtype=np.uint16)
    noise[0, 0] = 104

    assert not frame_has_signal(dark, DEFAULT_DARK_FRAME_STRUCTURE_DN)
    assert not frame_has_signal(noise, DEFAULT_DARK_FRAME_STRUCTURE_DN)
    assert frame_has_signal(pitched_frame(), DEFAULT_DARK_FRAME_STRUCTURE_DN)


async def test_the_dark_frame_retry_retakes_until_a_frame_is_lit() -> None:
    """Retake an unlit frame, and stop at the first one that carries light."""
    triggered = 0
    frames: list[np.ndarray] = [
        np.full(SHAPE, 100, dtype=np.uint16),
        np.full(SHAPE, 100, dtype=np.uint16),
        pitched_frame(),
    ]

    async def trigger_once() -> None:
        nonlocal triggered
        triggered += 1

    async def read_frame() -> np.ndarray:
        return frames[triggered - 1]

    attempts = await trigger_until_lit(
        trigger_once,
        read_frame,
        retries=2,
        structure_dn=DEFAULT_DARK_FRAME_STRUCTURE_DN,
    )

    assert attempts == 3
    assert triggered == 3


async def test_the_last_retry_is_kept_even_when_it_is_still_dark() -> None:
    """Keep the last frame of a retried trigger rather than looping for ever."""
    triggered = 0

    async def trigger_once() -> None:
        nonlocal triggered
        triggered += 1

    async def read_frame() -> Any:
        return np.full(SHAPE, 100, dtype=np.uint16)

    attempts = await trigger_until_lit(
        trigger_once,
        read_frame,
        retries=2,
        structure_dn=DEFAULT_DARK_FRAME_STRUCTURE_DN,
    )

    assert attempts == 3
    assert triggered == 3


async def test_describe_settings_reports_what_the_service_publishes(
    camera: MMCamera,
) -> None:
    """Turn the published signals into the contract's camera description."""
    description = await camera.describe_settings()

    assert description.detector_id == "cam"
    assert description.shape == (4, 6)
    assert description.pixel_dtypes == ("uint8", "uint16")
    assert description.pixel_dtype == "uint8"
    assert description.exposure_range_s == pytest.approx((0.0001, 1.0))
    assert description.roi_supported is True
    assert description.binning == (1, 2)
    assert description.trigger_modes == (TriggerMode.FREE_RUN, TriggerMode.SOFTWARE)
    assert description.revision == 7


async def test_describe_settings_refuses_an_unpublished_exposure_range(
    camera: MMCamera,
) -> None:
    """Refuse to describe a camera whose exposure bound the adapter does not name."""
    set_mock_value(camera.exposure_range, np.array([0.0, 0.0]))

    with pytest.raises(SettingRefused, match="exposure_out_of_range"):
        await camera.describe_settings()


async def test_apply_settings_writes_the_settings_and_returns_the_revision(
    camera: MMCamera,
) -> None:
    """Write the settings in the service's units, and report the revision it holds."""
    revision = await camera.apply_settings(
        CameraSettings(
            exposure_s=0.02,
            roi=Roi(1, 1, 2, 2),
            binning=1,
            pixel_dtype="uint16",
        )
    )

    # seconds become the milliseconds Micro-Manager holds, and only here
    get_mock_put(camera.exposure).assert_called_once_with(20.0)
    get_mock_put(camera.roi).assert_called_once_with("1,1,2,2")
    get_mock_put(camera.pixel_dtype).assert_called_once_with("uint16")
    # the fixture publishes no writable Binning property and the requested
    # factor is the one already in force, so nothing is written for it
    assert camera.properties.get("Binning") is None
    assert revision == 7


async def test_apply_settings_refuses_before_it_writes(camera: MMCamera) -> None:
    """Refuse a setting the description does not declare, without writing any."""
    with pytest.raises(SettingRefused, match="exposure_out_of_range"):
        await camera.apply_settings(
            CameraSettings(
                exposure_s=5.0,
                roi=Roi(0, 0, 6, 4),
                binning=1,
                pixel_dtype="uint8",
            )
        )

    get_mock_put(camera.exposure).assert_not_called()
    get_mock_put(camera.roi).assert_not_called()

    with pytest.raises(SettingRefused, match="binning_unsupported"):
        await camera.apply_settings(
            CameraSettings(
                exposure_s=0.02,
                roi=Roi(0, 0, 6, 4),
                binning=2,
                pixel_dtype="uint8",
            )
        )

    get_mock_put(camera.roi).assert_not_called()


async def test_capture_frame_returns_the_frame_with_its_sequence(
    camera: MMCamera,
) -> None:
    """Read one frame with the sequence, revision and exposure it was taken under."""
    task = asyncio.ensure_future(camera.capture_frame())
    await asyncio.sleep(0.02)
    set_mock_value(camera.buffer, np.full(SHAPE, 3, dtype=np.uint8))
    await publish_frames(camera, [11])

    readout = await asyncio.wait_for(task, 2.0)

    assert readout.shape == SHAPE
    assert readout.dtype == "uint8"
    assert int(readout.data[0, 0]) == 3
    assert readout.sequence == 11
    assert readout.setting_revision == 7
    assert readout.exposure_s == pytest.approx(0.01)


async def test_stop_clears_acquiring_and_capturing(camera: MMCamera) -> None:
    """Stop the capture and then the camera, as a RunEngine stop expects."""
    set_mock_value(camera.acquire, True)
    set_mock_value(camera.capture, True)

    await camera.stop()

    get_mock_put(camera.capture).assert_called_once_with(False)
    get_mock_put(camera.acquire).assert_called_once_with(False)


def test_the_camera_is_a_live_flyer_and_a_contract_camera(camera: MMCamera) -> None:
    """Satisfy the plan-facing protocols the live and scan plans accept."""
    assert isinstance(camera, LiveFlyer)
    assert isinstance(camera, Camera)
