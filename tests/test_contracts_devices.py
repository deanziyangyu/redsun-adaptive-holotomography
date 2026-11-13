"""Device-role contracts: protocols, setting refusals and command statuses."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any

import numpy as np
import pytest

from redsun_aht.contracts.devices import (
    CENTER_PATTERN_ID,
    DPCT_PATTERN_IDS,
    ILLUMINATION_PATTERN_IDS,
    AxisReading,
    AxisSpec,
    Camera,
    CameraDescription,
    CameraReadout,
    CameraSettings,
    CommandReceipt,
    CommandStatus,
    Illumination,
    IlluminationAction,
    IlluminationCapability,
    Roi,
    SettingRefusal,
    SettingRefused,
    Stage,
    StageReading,
    TriggerMode,
)
from redsun_aht.contracts.frames import QualityLabel


def _description(
    *,
    revision: int = 1,
    roi_supported: bool = True,
    shape: tuple[int, int] = (1024, 2048),
) -> CameraDescription:
    return CameraDescription(
        detector_id="dhm",
        shape=shape,
        pixel_dtypes=("uint16", "uint8"),
        exposure_range_s=(0.001, 10.0),
        roi_supported=roi_supported,
        binning=(1, 2, 4),
        trigger_modes=(TriggerMode.EXTERNAL, TriggerMode.SOFTWARE),
        revision=revision,
    )


def _settings(**overrides: Any) -> CameraSettings:
    values: dict[str, Any] = {
        "exposure_s": 0.01,
        "roi": Roi(0, 0, 2048, 1024),
        "pixel_dtype": "uint16",
        "binning": 1,
    }
    values.update(overrides)
    return CameraSettings(**values)


class ServiceStatus(StrEnum):
    """A stand-in for the serial service's own status enum."""

    SENT = "sent"
    ECHOED = "echoed"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class FakeCamera:
    """A camera that declares one description and captures on demand."""

    def __init__(self, description: CameraDescription) -> None:
        self.description = description
        self.revision = description.revision
        self.captures = 0
        self.stopped = False

    async def describe_settings(self) -> CameraDescription:
        return self.description

    async def apply_settings(self, settings: CameraSettings) -> int:
        settings.validate_for(self.description)
        self.revision += 1
        return self.revision

    async def capture_frame(self) -> CameraReadout:
        self.captures += 1
        height, width = self.description.shape
        return CameraReadout(
            data=np.zeros((height, width), dtype=np.uint16),
            sequence=self.captures,
            setting_revision=self.revision,
            exposure_s=0.01,
            captured_monotonic_ns=1_000 + self.captures,
        )

    async def stop(self) -> None:
        self.stopped = True


class PartialCamera:
    """A camera-shaped object that never captures a frame."""

    async def describe_settings(self) -> CameraDescription:
        return _description()

    async def apply_settings(self, settings: CameraSettings) -> int:
        return 1

    async def stop(self) -> None:
        return None


class FakeStage:
    """A three-axis stage whose axes are all measured readbacks."""

    def __init__(self) -> None:
        self.axes: dict[str, AxisSpec] = {
            "z": AxisSpec("z", "um", -50.0, 50.0, 0.1),
            "y": AxisSpec("y", "um", 0.0, 100.0, 0.5),
            "x": AxisSpec("x", "um", 0.0, 100.0, 0.5),
        }
        self.positions = {"z": 0.0, "y": 0.0, "x": 0.0}
        self.stopped = False

    async def move(self, position: Mapping[str, float]) -> StageReading:
        for name, value in position.items():
            self.positions[name] = self.axes[name].check(value)
        return await self.read_position()

    async def read_position(self) -> StageReading:
        return StageReading(
            axes=tuple(
                AxisReading(name, value, value, True)
                for name, value in self.positions.items()
            ),
            monotonic_ns=2_000,
            quality=QualityLabel.MEASURED,
        )

    async def stop(self) -> None:
        self.stopped = True


class PartialStage:
    """A stage-shaped object that cannot move."""

    def __init__(self) -> None:
        self.axes = {"z": AxisSpec("z", "um", -50.0, 50.0, 0.1)}

    async def read_position(self) -> StageReading:
        return StageReading(
            axes=(AxisReading("z", 0.0, 0.0, True),),
            monotonic_ns=2_000,
            quality=QualityLabel.MEASURED,
        )

    async def stop(self) -> None:
        return None


class FakeIllumination:
    """An MM-compatible illumination service with no echo."""

    def __init__(self) -> None:
        self.capability = IlluminationCapability()
        self.commands: list[str] = []

    def _receipt(
        self,
        action: IlluminationAction,
        command: str,
        pattern_id: int | None = None,
    ) -> CommandReceipt:
        self.commands.append(command)
        return CommandReceipt.from_transport(
            action=action,
            command=command,
            status=CommandStatus.SENT,
            pattern_id=pattern_id,
            sent_monotonic_ns=3_000 + len(self.commands),
        )

    async def blank(self) -> CommandReceipt:
        return self._receipt(IlluminationAction.BLANK, "00")

    async def open(self) -> CommandReceipt:
        return self._receipt(IlluminationAction.OPEN, "01")

    async def pattern(self, pattern_id: int) -> CommandReceipt:
        return self._receipt(
            IlluminationAction.PATTERN, f"{10 + pattern_id:02d}", pattern_id
        )


class PartialIllumination:
    """An illumination-shaped object that cannot blank."""

    def __init__(self) -> None:
        self.capability = IlluminationCapability()

    async def open(self) -> CommandReceipt:
        return CommandReceipt.from_transport(
            action=IlluminationAction.OPEN,
            command="01",
            status=CommandStatus.SENT,
            sent_monotonic_ns=3_000,
        )

    async def pattern(self, pattern_id: int) -> CommandReceipt:
        return CommandReceipt.from_transport(
            action=IlluminationAction.PATTERN,
            command=f"{10 + pattern_id:02d}",
            status=CommandStatus.SENT,
            pattern_id=pattern_id,
            sent_monotonic_ns=3_000,
        )


def use_camera(camera: Camera) -> Camera:
    """Check a camera structurally without running it."""
    return camera


def use_stage(stage: Stage) -> Stage:
    """Check a stage structurally without running it."""
    return stage


def use_illumination(illumination: Illumination) -> Illumination:
    """Check an illumination service structurally without running it."""
    return illumination


async def test_conforming_fake_satisfies_the_camera_protocol() -> None:
    fake = FakeCamera(_description())
    assert isinstance(fake, Camera)
    assert use_camera(fake) is fake
    revision = await fake.apply_settings(_settings())
    assert revision == 2
    readout = await fake.capture_frame()
    assert readout.shape == (1024, 2048)
    assert readout.dtype == "uint16"
    assert readout.sequence == 1
    assert readout.setting_revision == 2
    await fake.stop()
    assert fake.stopped is True


def test_the_camera_contract_does_not_shadow_bluesky_methods() -> None:
    for bluesky_name in ("trigger", "read", "describe", "stage", "unstage"):
        assert not hasattr(Camera, bluesky_name)
    for layered_name in (
        "describe_settings",
        "apply_settings",
        "capture_frame",
        "stop",
    ):
        assert hasattr(Camera, layered_name)


def test_partial_camera_is_not_a_camera() -> None:
    assert not isinstance(PartialCamera(), Camera)


async def test_conforming_fake_satisfies_the_stage_protocol() -> None:
    fake = FakeStage()
    assert isinstance(fake, Stage)
    assert use_stage(fake) is fake
    reading = await fake.move({"z": 12.5})
    assert reading.positions["z"] == 12.5
    assert reading.quality is QualityLabel.MEASURED
    await fake.stop()
    assert fake.stopped is True


def test_partial_stage_is_not_a_stage() -> None:
    assert not isinstance(PartialStage(), Stage)


def test_the_stage_contract_does_not_shadow_bluesky_methods() -> None:
    for bluesky_name in ("trigger", "read", "describe", "stage", "unstage"):
        assert not hasattr(Stage, bluesky_name)
    for layered_name in ("axes", "move", "read_position", "stop"):
        assert hasattr(Stage, layered_name)


async def test_conforming_fake_satisfies_the_illumination_protocol() -> None:
    fake = FakeIllumination()
    assert isinstance(fake, Illumination)
    assert use_illumination(fake) is fake
    blank = await fake.blank()
    opened = await fake.open()
    pattern = await fake.pattern(2)
    centre = await fake.pattern(CENTER_PATTERN_ID)
    assert (blank.command, opened.command, pattern.command, centre.command) == (
        "00",
        "01",
        "12",
        "15",
    )
    assert fake.commands == ["00", "01", "12", "15"]
    assert pattern.pattern_id == 2
    assert centre.pattern_id == CENTER_PATTERN_ID
    assert centre.status is CommandStatus.SENT


def test_partial_illumination_is_not_an_illumination_service() -> None:
    assert not isinstance(PartialIllumination(), Illumination)


def test_the_panel_declares_four_semicircles_and_a_centre() -> None:
    assert DPCT_PATTERN_IDS == (1, 2, 3, 4)
    assert ILLUMINATION_PATTERN_IDS == (1, 2, 3, 4, 5)
    assert CENTER_PATTERN_ID == 5
    assert set(DPCT_PATTERN_IDS) < set(ILLUMINATION_PATTERN_IDS)


@pytest.mark.parametrize("exposure", [0.0, -1e-9, math.nan, math.inf])
def test_camera_settings_refuses_a_non_positive_or_unbounded_exposure(
    exposure: float,
) -> None:
    with pytest.raises(SettingRefused) as error:
        _settings(exposure_s=exposure)
    assert error.value.refusal is SettingRefusal.EXPOSURE_NOT_POSITIVE


@pytest.mark.parametrize(
    "arguments",
    [(-1, 0, 8, 8), (0, -1, 8, 8), (0, 0, 0, 8), (0, 0, 8, 0)],
)
def test_a_roi_refuses_a_region_no_sensor_can_address(
    arguments: tuple[int, int, int, int],
) -> None:
    with pytest.raises(SettingRefused) as error:
        Roi(*arguments)
    assert error.value.refusal is SettingRefusal.ROI_INVALID


def test_camera_settings_requires_a_valid_roi() -> None:
    with pytest.raises(SettingRefused) as error:
        _settings(roi=(0, 0, 8, 8))
    assert error.value.refusal is SettingRefusal.ROI_INVALID


def test_roi_refuses_a_non_integer_extent_and_non_text_parse() -> None:
    with pytest.raises(SettingRefused) as extent:
        Roi(0, 0, 8.0, 8)  # type: ignore[arg-type]
    assert extent.value.refusal is SettingRefusal.ROI_INVALID
    with pytest.raises(SettingRefused) as parsed:
        Roi.parse("0,0,8")
    assert parsed.value.refusal is SettingRefusal.ROI_INVALID
    with pytest.raises(SettingRefused) as text:
        Roi.parse("0,0,eight,8")
    assert text.value.refusal is SettingRefusal.ROI_INVALID


def test_roi_text_round_trips() -> None:
    roi = Roi(4, 8, 512, 256)
    assert roi.as_text() == "4,8,512,256"
    assert Roi.parse(roi.as_text()) == roi
    assert roi.extent == (512, 256)


def _refuse_exposure_not_positive() -> None:
    _settings(exposure_s=0.0)


def _refuse_exposure_out_of_range() -> None:
    _settings(exposure_s=99.0).validate_for(_description())


def _refuse_roi_invalid() -> None:
    Roi(0, 0, 0, 0)


def _refuse_roi_unsupported() -> None:
    _settings(roi=Roi(0, 0, 4, 4)).validate_for(_description(roi_supported=False))


def _refuse_roi_outside_sensor() -> None:
    _settings(roi=Roi(0, 0, 4096, 4096)).validate_for(_description())


def _refuse_binning_invalid() -> None:
    _settings(binning=0)


def _refuse_binning_unsupported() -> None:
    _settings(binning=3).validate_for(_description())


def _refuse_pixel_dtype_invalid() -> None:
    _settings(pixel_dtype="not-a-dtype")


def _refuse_pixel_dtype_unsupported() -> None:
    _settings(pixel_dtype="int8").validate_for(_description())


SETTING_CASES: dict[SettingRefusal, Callable[[], object]] = {
    SettingRefusal.EXPOSURE_NOT_POSITIVE: _refuse_exposure_not_positive,
    SettingRefusal.EXPOSURE_OUT_OF_RANGE: _refuse_exposure_out_of_range,
    SettingRefusal.ROI_INVALID: _refuse_roi_invalid,
    SettingRefusal.ROI_UNSUPPORTED: _refuse_roi_unsupported,
    SettingRefusal.ROI_OUTSIDE_SENSOR: _refuse_roi_outside_sensor,
    SettingRefusal.BINNING_INVALID: _refuse_binning_invalid,
    SettingRefusal.BINNING_UNSUPPORTED: _refuse_binning_unsupported,
    SettingRefusal.PIXEL_DTYPE_INVALID: _refuse_pixel_dtype_invalid,
    SettingRefusal.PIXEL_DTYPE_UNSUPPORTED: _refuse_pixel_dtype_unsupported,
}


def test_every_setting_refusal_has_a_reachable_case() -> None:
    assert set(SETTING_CASES) == set(SettingRefusal)


@pytest.mark.parametrize("refusal", list(SettingRefusal))
def test_every_setting_refusal_is_reachable(refusal: SettingRefusal) -> None:
    with pytest.raises(SettingRefused) as error:
        SETTING_CASES[refusal]()
    assert error.value.refusal is refusal
    assert error.value.evidence


def test_camera_settings_refuses_unsupported_binning() -> None:
    description = _description()
    with pytest.raises(SettingRefused) as unsupported:
        _settings(binning=3).validate_for(description)
    assert unsupported.value.refusal is SettingRefusal.BINNING_UNSUPPORTED
    with pytest.raises(SettingRefused) as invalid:
        _settings(binning=0)
    assert invalid.value.refusal is SettingRefusal.BINNING_INVALID


def test_camera_settings_refuses_an_undeclared_pixel_dtype() -> None:
    description = _description()
    with pytest.raises(SettingRefused) as unsupported:
        _settings(pixel_dtype="int8").validate_for(description)
    assert unsupported.value.refusal is SettingRefusal.PIXEL_DTYPE_UNSUPPORTED
    with pytest.raises(SettingRefused) as invalid:
        _settings(pixel_dtype="not-a-dtype")
    assert invalid.value.refusal is SettingRefusal.PIXEL_DTYPE_INVALID


def test_camera_settings_refuses_roi_outside_the_sensor() -> None:
    with pytest.raises(SettingRefused) as error:
        _settings(roi=Roi(0, 0, 2049, 1024)).validate_for(_description())
    assert error.value.refusal is SettingRefusal.ROI_OUTSIDE_SENSOR
    assert error.value.evidence["shape"] == [1024, 2048]


def test_camera_settings_refuses_a_roi_on_a_camera_that_has_none() -> None:
    description = _description(roi_supported=False)
    with pytest.raises(SettingRefused) as unsupported:
        _settings(roi=Roi(0, 0, 512, 256)).validate_for(description)
    assert unsupported.value.refusal is SettingRefusal.ROI_UNSUPPORTED
    _settings().validate_for(description)


def test_camera_settings_refuses_an_exposure_outside_the_declared_range() -> None:
    for exposure in (0.0001, 11.0):
        with pytest.raises(SettingRefused) as error:
            _settings(exposure_s=exposure).validate_for(_description())
        assert error.value.refusal is SettingRefusal.EXPOSURE_OUT_OF_RANGE


def test_camera_settings_accepts_a_declared_configuration() -> None:
    description = _description()
    settings = _settings(exposure_s=0.05, binning=2, pixel_dtype="uint8")
    settings.validate_for(description)
    assert description.pixel_dtype == "uint16"
    assert description.exposure_range_s == (0.001, 10.0)


def test_camera_description_refuses_an_inverted_exposure_range() -> None:
    with pytest.raises(ValueError, match="exposure range"):
        CameraDescription(
            detector_id="dhm",
            shape=(8, 8),
            pixel_dtypes=("uint16",),
            exposure_range_s=(10.0, 1.0),
            roi_supported=True,
            binning=(1,),
            trigger_modes=(TriggerMode.SOFTWARE,),
            revision=0,
        )


def test_camera_readout_carries_its_own_shape_and_dtype() -> None:
    readout = CameraReadout(
        data=np.zeros((4, 6), dtype=np.uint8),
        sequence=3,
        setting_revision=2,
        exposure_s=0.02,
        captured_monotonic_ns=1_500,
    )
    assert readout.shape == (4, 6)
    assert readout.dtype == "uint8"
    with pytest.raises(ValueError, match="two-dimensional"):
        CameraReadout(
            data=np.zeros(4, dtype=np.uint8),
            sequence=0,
            setting_revision=0,
            exposure_s=0.02,
            captured_monotonic_ns=0,
        )


def test_axis_spec_checks_limits_and_precision() -> None:
    axis = AxisSpec("z", "um", -50.0, 50.0, 0.1)
    assert axis.limits == (-50.0, 50.0)
    assert axis.check(-50.0) == -50.0
    with pytest.raises(ValueError, match="outside"):
        axis.check(50.1)
    with pytest.raises(ValueError, match="precision"):
        AxisSpec("z", "um", 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="lower limit"):
        AxisSpec("z", "um", 1.0, 0.0, 0.1)


def test_stage_reading_cannot_claim_measured_from_a_setpoint() -> None:
    with pytest.raises(ValueError, match="every axis measured"):
        StageReading(
            axes=(
                AxisReading("z", 1.0, 1.0, True),
                AxisReading("y", 2.0, 2.0, False),
            ),
            monotonic_ns=10,
            quality=QualityLabel.MEASURED,
        )
    assert StageReading(
        axes=(
            AxisReading("z", 1.0, 1.0, True),
            AxisReading("y", 2.0, 2.0, False),
        ),
        monotonic_ns=10,
        quality=QualityLabel.COMMANDED,
    ).positions == {"z": 1.0, "y": 2.0}


def test_stage_reading_unavailable_carries_no_axes() -> None:
    empty = StageReading(axes=(), monotonic_ns=0, quality=QualityLabel.UNAVAILABLE)
    assert empty.positions == {}
    with pytest.raises(ValueError, match="carries no axes"):
        StageReading(
            axes=(AxisReading("z", 0.0, 0.0, False),),
            monotonic_ns=0,
            quality=QualityLabel.UNAVAILABLE,
        )


def test_command_receipt_distinguishes_sent_echoed_uncertain_and_failed() -> None:
    sent = CommandReceipt(
        action=IlluminationAction.PATTERN,
        command="11",
        status=CommandStatus.SENT,
        pattern_id=1,
        sent_monotonic_ns=10,
    )
    echoed = CommandReceipt(
        action=IlluminationAction.PATTERN,
        command="11",
        status=CommandStatus.ECHOED,
        pattern_id=1,
        echo="11",
        sent_monotonic_ns=10,
    )
    uncertain = CommandReceipt(
        action=IlluminationAction.PATTERN,
        command="12",
        status=CommandStatus.UNCERTAIN,
        pattern_id=2,
        sent_monotonic_ns=11,
        detail="no echo before the deadline; do not resend",
    )
    failed = CommandReceipt(
        action=IlluminationAction.BLANK,
        command="00",
        status=CommandStatus.FAILED,
        detail="port refused the write",
    )
    assert sent.wrote_bytes is True
    assert echoed.wrote_bytes is True
    assert uncertain.wrote_bytes is True
    assert failed.wrote_bytes is False
    assert echoed.echo == "11"
    assert sent.echo is None
    assert {member.value for member in CommandStatus} == {
        "sent",
        "echoed",
        "uncertain",
        "failed",
    }


def test_command_receipt_is_built_from_the_serial_transport_report() -> None:
    receipt = CommandReceipt.from_transport(
        action=IlluminationAction.PATTERN,
        command="13",
        status=CommandStatus.ECHOED,
        detail="echo matched",
        echo="13",
        pattern_id=3,
        sent_monotonic_ns=42,
    )
    assert receipt.status is CommandStatus.ECHOED
    assert receipt.pattern_id == 3
    assert receipt.echo == "13"
    assert receipt.detail == "echo matched"
    assert receipt.sent_monotonic_ns == 42
    # a transport report without a clock is recorded without inventing one
    quiet = CommandReceipt.from_transport(
        action=IlluminationAction.BLANK,
        command="00",
        status=CommandStatus.SENT,
    )
    assert quiet.sent_monotonic_ns is None
    assert quiet.wrote_bytes is True


def test_the_transport_adapter_takes_the_service_status_by_value() -> None:
    # a serial module publishes its own status enum; the adapter converts the
    # bare word, so the plan never imports that module
    receipt = CommandReceipt.from_transport(
        action=IlluminationAction.PATTERN,
        command="14",
        status=ServiceStatus.ECHOED,
        echo="14",
        pattern_id=4,
    )
    assert receipt.status is CommandStatus.ECHOED
    assert (
        CommandReceipt.from_transport(
            action=IlluminationAction.BLANK,
            command="00",
            status="uncertain",
        ).status
        is CommandStatus.UNCERTAIN
    )
    with pytest.raises(ValueError, match="unknown transport status"):
        CommandReceipt.from_transport(
            action=IlluminationAction.BLANK,
            command="00",
            status="delivered",
        )


def test_command_receipt_refuses_invented_echo_evidence() -> None:
    with pytest.raises(ValueError, match="own echo"):
        CommandReceipt(
            action=IlluminationAction.PATTERN,
            command="11",
            status=CommandStatus.ECHOED,
            pattern_id=1,
            sent_monotonic_ns=10,
        )
    with pytest.raises(ValueError, match="own echo"):
        CommandReceipt(
            action=IlluminationAction.PATTERN,
            command="11",
            status=CommandStatus.ECHOED,
            pattern_id=1,
            echo="12",
            sent_monotonic_ns=10,
        )
    with pytest.raises(ValueError, match="only an echoed command"):
        CommandReceipt(
            action=IlluminationAction.PATTERN,
            command="11",
            status=CommandStatus.SENT,
            pattern_id=1,
            echo="11",
            sent_monotonic_ns=10,
        )
    with pytest.raises(ValueError, match="non-negative"):
        CommandReceipt(
            action=IlluminationAction.PATTERN,
            command="11",
            status=CommandStatus.SENT,
            pattern_id=1,
            sent_monotonic_ns=-1,
        )


def test_command_receipt_ties_a_pattern_to_its_action() -> None:
    with pytest.raises(ValueError, match="needs the pattern"):
        CommandReceipt(
            action=IlluminationAction.PATTERN,
            command="11",
            status=CommandStatus.SENT,
            sent_monotonic_ns=10,
        )
    with pytest.raises(ValueError, match="only a pattern command"):
        CommandReceipt(
            action=IlluminationAction.BLANK,
            command="00",
            status=CommandStatus.SENT,
            pattern_id=1,
            sent_monotonic_ns=10,
        )
    with pytest.raises(ValueError, match="pattern id"):
        CommandReceipt(
            action=IlluminationAction.PATTERN,
            command="16",
            status=CommandStatus.SENT,
            pattern_id=6,
            sent_monotonic_ns=10,
        )
    with pytest.raises(ValueError, match="two-digit"):
        CommandReceipt(
            action=IlluminationAction.BLANK,
            command="0",
            status=CommandStatus.SENT,
            sent_monotonic_ns=10,
        )


def test_a_centre_pattern_receipt_is_a_valid_illumination_command() -> None:
    receipt = CommandReceipt(
        action=IlluminationAction.PATTERN,
        command="15",
        status=CommandStatus.SENT,
        pattern_id=CENTER_PATTERN_ID,
        sent_monotonic_ns=10,
    )
    assert receipt.pattern_id == 5
    assert receipt.wrote_bytes is True


def test_illumination_capability_defaults_describe_the_91_led_panel() -> None:
    capability = IlluminationCapability()
    assert ILLUMINATION_PATTERN_IDS == (1, 2, 3, 4, 5)
    assert capability.pattern_ids == ILLUMINATION_PATTERN_IDS
    assert CENTER_PATTERN_ID == 5
    assert capability.center_pattern_id == CENTER_PATTERN_ID
    assert capability.panel_profile == "panel91"
    assert capability.panel_revision == 1
    assert capability.led_count == 91
    assert capability.dialect == "mm-ascii"


def test_a_dpct_only_capability_is_still_valid() -> None:
    capability = IlluminationCapability(
        pattern_ids=DPCT_PATTERN_IDS,
        center_pattern_id=None,
        panel_profile="dpct-semicircles",
    )
    assert capability.center_pattern_id is None
    assert capability.pattern_ids == (1, 2, 3, 4)
    assert (
        IlluminationCapability(
            pattern_ids=(1, 3), center_pattern_id=3
        ).center_pattern_id
        == 3
    )


def test_illumination_capability_refuses_a_pattern_set_it_cannot_render() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        IlluminationCapability(pattern_ids=(0, 1))
    with pytest.raises(ValueError, match="positive integers"):
        IlluminationCapability(pattern_ids=(1, -2))
    with pytest.raises(ValueError, match="unique"):
        IlluminationCapability(pattern_ids=(1, 1))
    with pytest.raises(ValueError, match="ascending"):
        IlluminationCapability(pattern_ids=(2, 1))


def test_illumination_capability_refuses_a_centre_it_does_not_advertise() -> None:
    # the panel's default centre is not part of a DPCT-only pattern set
    with pytest.raises(ValueError, match="centre pattern"):
        IlluminationCapability(pattern_ids=DPCT_PATTERN_IDS)
    with pytest.raises(ValueError, match="centre pattern"):
        IlluminationCapability(pattern_ids=(1, 2, 3), center_pattern_id=4)
    with pytest.raises(ValueError, match="centre pattern"):
        IlluminationCapability(center_pattern_id="5")  # type: ignore[arg-type]


def test_illumination_capability_refuses_a_panel_it_cannot_describe() -> None:
    for count in (0, -1):
        with pytest.raises(ValueError, match="led count"):
            IlluminationCapability(led_count=count)
    for revision in (0, -1):
        with pytest.raises(ValueError, match="panel revision"):
            IlluminationCapability(panel_revision=revision)
    with pytest.raises(ValueError, match="panel profile"):
        IlluminationCapability(panel_profile=" ")


def test_a_camera_description_accepts_a_zero_exposure_floor() -> None:
    """Some adapters report 0 as "no floor"; that must not block description."""
    description = _description()
    assert description.exposure_range_s == (0.001, 10.0)
    with_zero_floor = CameraDescription(
        detector_id=description.detector_id,
        shape=description.shape,
        pixel_dtypes=description.pixel_dtypes,
        exposure_range_s=(0.0, 10.0),
        roi_supported=description.roi_supported,
        binning=description.binning,
        trigger_modes=description.trigger_modes,
        revision=description.revision,
    )
    assert with_zero_floor.exposure_range_s == (0.0, 10.0)


def test_a_camera_description_refuses_a_negative_exposure_floor() -> None:
    description = _description()
    with pytest.raises(ValueError, match="non-negative"):
        CameraDescription(
            detector_id=description.detector_id,
            shape=description.shape,
            pixel_dtypes=description.pixel_dtypes,
            exposure_range_s=(-0.5, 10.0),
            roi_supported=description.roi_supported,
            binning=description.binning,
            trigger_modes=description.trigger_modes,
            revision=description.revision,
        )
