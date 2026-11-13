"""Device-role contracts: camera, stage and MM-compatible illumination.

Three roles are frozen here as runtime-checkable Protocols with immutable value
types. A device package implements them and a plan consumes only these names, so
the acquisition application never imports a device class to run a plan.

``Camera`` and ``Stage`` describe the *settings and capture surface* that layers
on top of an ophyd-async ``StandardDetector`` or ``StandardMovable``; they are
not a replacement for the Bluesky device protocols. Nothing here is called
``trigger``, ``read`` or ``describe``, because those names already mean a
status, a reading and a descriptor on the device itself. A device satisfies both
surfaces at once: Bluesky drives triggering and Event documents, while these
methods carry the typed settings, the description revision and the raw readout
the plan and the store need.

Two rules run through the module:

* a *description* is what the device declares it can do, and a *setting* is what
  a caller asks for. Asking for something the description does not declare is
  refused by name, before any command reaches hardware.
* an illumination *status* records transport evidence only. No
  :class:`CommandStatus` member means that light was emitted or that a frame
  arrived. In the installed sketch a pattern command also pulses the camera
  trigger, so the frame is separate evidence that a coordinator must obtain from
  the camera, not from the illumination service.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .frames import QualityLabel

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

DPCT_PATTERN_IDS: tuple[int, ...] = (1, 2, 3, 4)
"""The four semicircle patterns the DPCT plan acquires, in frame order."""

ILLUMINATION_PATTERN_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
"""The patterns the 91-LED v1 panel renders: four semicircles and a centre."""

CENTER_PATTERN_ID: int = 5
"""The centre pattern of the 91-LED v1 panel, which the sketch commands as 15."""

_TWO_DIGIT_COMMAND = re.compile(r"[0-9]{2}")


class SettingRefusal(StrEnum):
    """Every way a requested camera setting can be refused."""

    EXPOSURE_NOT_POSITIVE = "exposure_not_positive"
    EXPOSURE_OUT_OF_RANGE = "exposure_out_of_range"
    ROI_INVALID = "roi_invalid"
    ROI_UNSUPPORTED = "roi_unsupported"
    ROI_OUTSIDE_SENSOR = "roi_outside_sensor"
    BINNING_INVALID = "binning_invalid"
    BINNING_UNSUPPORTED = "binning_unsupported"
    PIXEL_DTYPE_INVALID = "pixel_dtype_invalid"
    PIXEL_DTYPE_UNSUPPORTED = "pixel_dtype_unsupported"


class SettingRefused(ValueError):
    """A camera setting the device's description does not accept."""

    def __init__(self, refusal: SettingRefusal, evidence: Mapping[str, Any]) -> None:
        """Keep the machine-readable refusal beside the values that caused it."""
        self.refusal = refusal
        self.evidence = dict(evidence)
        super().__init__(
            f"{refusal.value}: {json.dumps(self.evidence, sort_keys=True)}"
        )


class TriggerMode(StrEnum):
    """How a frame is started, as the camera declares it.

    ``EXTERNAL`` is the mode the initial DPCT path needs: the illumination
    controller's TTL pulse starts the readout, and the plan issues no software
    trigger of its own.
    """

    SOFTWARE = "software"
    EXTERNAL = "external"
    FREE_RUN = "free_run"


@dataclass(frozen=True, slots=True)
class Roi:
    """A rectangular sensor region, in unbinned sensor pixels.

    Parameters
    ----------
    x, y
        The region's top-left corner on the full sensor.
    width, height
        The region's extent; both are positive.

    Notes
    -----
    :meth:`as_text` and :meth:`parse` use the ``"x,y,width,height"`` form the
    PVA camera service carries, because four integers do not travel as a
    portable scalar over the control plane.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Refuse a region that cannot be addressed on a sensor."""
        if any(
            type(value) is not int
            for value in (self.x, self.y, self.width, self.height)
        ):
            raise SettingRefused(
                SettingRefusal.ROI_INVALID,
                {"roi": [self.x, self.y, self.width, self.height]},
            )
        if self.x < 0 or self.y < 0:
            raise SettingRefused(
                SettingRefusal.ROI_INVALID,
                {"roi": [self.x, self.y, self.width, self.height], "reason": "origin"},
            )
        if self.width <= 0 or self.height <= 0:
            raise SettingRefused(
                SettingRefusal.ROI_INVALID,
                {"roi": [self.x, self.y, self.width, self.height], "reason": "extent"},
            )

    def as_text(self) -> str:
        """Return the ``"x,y,width,height"`` text the service carries."""
        return f"{self.x},{self.y},{self.width},{self.height}"

    @classmethod
    def parse(cls, text: str) -> Roi:
        """Parse the ``"x,y,width,height"`` form, refusing anything else."""
        parts = text.split(",")
        if len(parts) != 4:
            raise SettingRefused(SettingRefusal.ROI_INVALID, {"text": text})
        try:
            values = [int(part) for part in parts]
        except ValueError as error:
            raise SettingRefused(SettingRefusal.ROI_INVALID, {"text": text}) from error
        return cls(values[0], values[1], values[2], values[3])

    @property
    def extent(self) -> tuple[int, int]:
        """Return the region as ``(width, height)``."""
        return (self.width, self.height)


@dataclass(frozen=True, slots=True)
class CameraDescription:
    """What a camera declares about itself before it is configured.

    Parameters
    ----------
    detector_id
        The detector identity frames are recorded under. It is not the Bluesky
        device name: one session may name a device differently from the detector
        it declares.
    shape
        The full sensor as a numpy shape, ``(height, width)``. A ROI is
        expressed against it.
    pixel_dtypes
        The readout dtypes the detector can produce, native first.
    exposure_range_s
        The shortest and longest exposure the camera accepts, in seconds.
    roi_supported
        Whether the camera can read out a sub-region.
    binning
        The binning factors the camera accepts, ascending.
    trigger_modes
        The trigger modes the camera accepts.
    revision
        The camera's description revision; it rises whenever the declared
        capabilities change.
    """

    detector_id: str
    shape: tuple[int, int]
    pixel_dtypes: tuple[str, ...]
    exposure_range_s: tuple[float, float]
    roi_supported: bool
    binning: tuple[int, ...]
    trigger_modes: tuple[TriggerMode, ...]
    revision: int

    def __post_init__(self) -> None:
        """Refuse a description a plan could not act on."""
        if not isinstance(self.detector_id, str) or not self.detector_id.strip():
            raise ValueError("camera description needs a detector id")
        if not isinstance(self.shape, tuple) or len(self.shape) != 2:
            raise ValueError("camera shape must be two integers, (height, width)")
        if any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError("camera shape entries must be positive integers")
        if not isinstance(self.pixel_dtypes, tuple) or not self.pixel_dtypes:
            raise ValueError("camera description needs at least one pixel dtype")
        if len(set(self.pixel_dtypes)) != len(self.pixel_dtypes):
            raise ValueError("camera pixel dtypes must be unique")
        for dtype in self.pixel_dtypes:
            if not isinstance(dtype, str) or not dtype:
                raise ValueError("camera pixel dtypes must be non-empty text")
            try:
                np.dtype(dtype)
            except TypeError as error:
                raise ValueError(f"unknown camera pixel dtype {dtype!r}") from error
        lowest, highest = self._range()
        if not lowest <= highest:
            raise ValueError("camera exposure range must run from low to high")
        if not isinstance(self.roi_supported, bool):
            raise ValueError("camera roi support must be a boolean")
        if not isinstance(self.binning, tuple) or not self.binning:
            raise ValueError("camera description needs at least one binning factor")
        if any(type(factor) is not int or factor < 1 for factor in self.binning):
            raise ValueError("camera binning factors must be positive integers")
        if len(set(self.binning)) != len(self.binning):
            raise ValueError("camera binning factors must be unique")
        if not isinstance(self.trigger_modes, tuple) or not self.trigger_modes:
            raise ValueError("camera description needs at least one trigger mode")
        if any(not isinstance(item, TriggerMode) for item in self.trigger_modes):
            raise ValueError("camera trigger modes must be TriggerMode members")
        if len(set(self.trigger_modes)) != len(self.trigger_modes):
            raise ValueError("camera trigger modes must be unique")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("camera description revision must be non-negative")

    def _range(self) -> tuple[float, float]:
        """Return the exposure range as finite non-negative floats."""
        if not isinstance(self.exposure_range_s, tuple):
            raise ValueError("camera exposure range must be a (low, high) pair")
        if len(self.exposure_range_s) != 2:
            raise ValueError("camera exposure range must be a (low, high) pair")
        values = []
        for bound in self.exposure_range_s:
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError("camera exposure bounds must be real numbers")
            number = float(bound)
            if not math.isfinite(number) or number < 0:
                raise ValueError(
                    "camera exposure bounds must be finite and non-negative"
                )
            values.append(number)
        object.__setattr__(self, "exposure_range_s", (values[0], values[1]))
        return (values[0], values[1])

    @property
    def pixel_dtype(self) -> str:
        """Return the native readout dtype, which is the first declared one."""
        return self.pixel_dtypes[0]


@dataclass(frozen=True, slots=True)
class CameraSettings:
    """A bounded configuration request for one camera.

    Parameters
    ----------
    exposure_s
        The requested exposure, in seconds; finite and positive.
    roi
        The requested sensor region, in unbinned sensor pixels.
    binning
        The requested binning factor; one or more.
    pixel_dtype
        The requested readout dtype, as a numpy dtype string.

    Notes
    -----
    Construction checks only what is self-consistent. Whether a camera accepts
    the request is decided by :meth:`validate_for` against that camera's own
    :class:`CameraDescription`, which is the only authority on its range, its
    region and its binnings.
    """

    exposure_s: float
    roi: Roi
    binning: int
    pixel_dtype: str

    def __post_init__(self) -> None:
        """Refuse a setting that is not even self-consistent."""
        if (
            isinstance(self.exposure_s, bool)
            or not isinstance(self.exposure_s, (int, float))
            or not math.isfinite(float(self.exposure_s))
            or float(self.exposure_s) <= 0
        ):
            raise SettingRefused(
                SettingRefusal.EXPOSURE_NOT_POSITIVE,
                {"exposure_s": self.exposure_s},
            )
        object.__setattr__(self, "exposure_s", float(self.exposure_s))
        if not isinstance(self.roi, Roi):
            raise SettingRefused(SettingRefusal.ROI_INVALID, {"roi": repr(self.roi)})
        if type(self.binning) is not int or self.binning < 1:
            raise SettingRefused(
                SettingRefusal.BINNING_INVALID, {"binning": self.binning}
            )
        if not isinstance(self.pixel_dtype, str) or not self.pixel_dtype:
            raise SettingRefused(
                SettingRefusal.PIXEL_DTYPE_INVALID, {"pixel_dtype": self.pixel_dtype}
            )
        try:
            np.dtype(self.pixel_dtype)
        except TypeError as error:
            raise SettingRefused(
                SettingRefusal.PIXEL_DTYPE_INVALID,
                {"pixel_dtype": self.pixel_dtype},
            ) from error

    def validate_for(self, description: CameraDescription) -> None:
        """Check this request against a camera's declared capabilities.

        Raises
        ------
        SettingRefused
            Naming the refusal and the values that produced it. Nothing here
            commands a device: a camera may still refuse at configure time,
            which is why ``apply_settings`` returns a revision rather than
            nothing.
        """
        if not isinstance(description, CameraDescription):
            raise TypeError("camera settings are validated against a description")
        if not description.roi_supported and self.roi != Roi(
            0, 0, description.shape[1], description.shape[0]
        ):
            raise SettingRefused(
                SettingRefusal.ROI_UNSUPPORTED,
                {"detector_id": description.detector_id, "roi": self.roi.as_text()},
            )
        height, width = description.shape
        if self.roi.x + self.roi.width > width or self.roi.y + self.roi.height > height:
            raise SettingRefused(
                SettingRefusal.ROI_OUTSIDE_SENSOR,
                {
                    "detector_id": description.detector_id,
                    "roi": self.roi.as_text(),
                    "shape": [height, width],
                },
            )
        if self.binning not in description.binning:
            raise SettingRefused(
                SettingRefusal.BINNING_UNSUPPORTED,
                {
                    "detector_id": description.detector_id,
                    "binning": self.binning,
                    "declared": list(description.binning),
                },
            )
        if self.pixel_dtype not in description.pixel_dtypes:
            raise SettingRefused(
                SettingRefusal.PIXEL_DTYPE_UNSUPPORTED,
                {
                    "detector_id": description.detector_id,
                    "pixel_dtype": self.pixel_dtype,
                    "declared": list(description.pixel_dtypes),
                },
            )
        lowest, highest = description.exposure_range_s
        if not lowest <= self.exposure_s <= highest:
            raise SettingRefused(
                SettingRefusal.EXPOSURE_OUT_OF_RANGE,
                {
                    "detector_id": description.detector_id,
                    "exposure_s": self.exposure_s,
                    "declared": [lowest, highest],
                },
            )


@dataclass(frozen=True, slots=True, eq=False)
class CameraReadout:
    """One frame as the camera hands it over, with the revision it was taken under.

    Parameters
    ----------
    data
        The frame, a two-dimensional array.
    sequence
        The detector's own readout sequence for this frame.
    setting_revision
        The camera description revision in force when the frame was read.
    exposure_s
        The exposure the camera applied, which may differ from the requested
        value if the device clamped it.
    captured_monotonic_ns
        When the service took the frame, on a monotonic clock.

    Notes
    -----
    Readouts compare by identity, not by pixel value, so holding two of them
    never compares arrays. A consumer that needs the pixels copies them and
    builds a :class:`~redsun_aht.contracts.frames.FrameRecord`.
    """

    data: npt.NDArray[Any]
    sequence: int
    setting_revision: int
    exposure_s: float
    captured_monotonic_ns: int

    def __post_init__(self) -> None:
        """Refuse a readout that cannot be described as a frame."""
        if not isinstance(self.data, np.ndarray) or self.data.ndim != 2:
            raise ValueError("a camera readout must be a two-dimensional array")
        if self.data.size <= 0:
            raise ValueError("a camera readout must carry at least one pixel")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("camera readout sequence must be non-negative")
        if type(self.setting_revision) is not int or self.setting_revision < 0:
            raise ValueError("camera readout revision must be non-negative")
        if (
            isinstance(self.exposure_s, bool)
            or not isinstance(self.exposure_s, (int, float))
            or not math.isfinite(float(self.exposure_s))
            or float(self.exposure_s) <= 0
        ):
            raise ValueError("camera readout exposure must be finite and positive")
        object.__setattr__(self, "exposure_s", float(self.exposure_s))
        if (
            type(self.captured_monotonic_ns) is not int
            or self.captured_monotonic_ns < 0
        ):
            raise ValueError("camera readout time must be non-negative")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the frame as ``(height, width)``."""
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    @property
    def dtype(self) -> str:
        """Return the frame's numpy dtype string."""
        return str(self.data.dtype)


@runtime_checkable
class Camera(Protocol):
    """The settings and capture surface layered on the camera device.

    A camera is an ophyd-async ``StandardDetector`` whose Bluesky methods are
    ``describe``, ``read``, ``trigger`` and their statuses; those keep their
    Bluesky meanings and their Bluesky return shapes. This Protocol adds the
    four operations a plan and a store need beside them, and deliberately
    defines no member whose name would shadow a device protocol:

    ``describe_settings``
        The typed capabilities and description revision.
    ``apply_settings``
        Apply bounded settings and return the revision the camera now reports.
    ``capture_frame``
        The raw readout of one frame, with the revision it was taken under.
    ``stop``
        Stop acquisition and release the camera.

    Triggering is not here. The ophyd-async device owns it, and in the initial
    DPCT path the external TTL from the illumination service is what starts the
    exposure, so a plan must not issue a second trigger of its own.
    """

    async def describe_settings(self) -> CameraDescription:
        """Return the camera's declared capabilities and revision."""
        ...

    async def apply_settings(self, settings: CameraSettings) -> int:
        """Apply settings and return the revision the camera now reports."""
        ...

    async def capture_frame(self) -> CameraReadout:
        """Return the raw readout of one frame, with its settings revision."""
        ...

    async def stop(self) -> None:
        """Stop acquisition and release the camera for another owner."""
        ...


@dataclass(frozen=True, slots=True)
class AxisSpec:
    """One named stage axis, as the stage declares it.

    Parameters
    ----------
    name
        The axis name the plan and the frame identity use, such as ``"z"``.
    units
        The unit of the limits and the readings, such as ``"um"``.
    lower, upper
        The inclusive travel limits, in ``units``.
    precision
        The finest reliable step, in ``units``; positive and finite.
    """

    name: str
    units: str
    lower: float
    upper: float
    precision: float

    def __post_init__(self) -> None:
        """Refuse an axis that cannot be addressed or read back."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("axis spec needs a name")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("axis spec needs units")
        lower = _finite(self.lower, "axis lower limit")
        upper = _finite(self.upper, "axis upper limit")
        precision = _finite(self.precision, "axis precision")
        if lower > upper:
            raise ValueError("axis lower limit must not exceed its upper limit")
        if precision <= 0:
            raise ValueError("axis precision must be positive")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "precision", precision)

    @property
    def limits(self) -> tuple[float, float]:
        """Return the inclusive travel limits."""
        return (self.lower, self.upper)

    def check(self, value: float) -> float:
        """Return *value* when it is within the limits, otherwise refuse it."""
        number = _finite(value, f"axis {self.name!r} target")
        if not self.lower <= number <= self.upper:
            raise ValueError(
                f"axis {self.name!r} target {number} is outside "
                f"[{self.lower}, {self.upper}] {self.units}"
            )
        return number


@dataclass(frozen=True, slots=True)
class AxisReading:
    """One axis position, with the evidence behind it.

    Parameters
    ----------
    name
        The axis name, matching an :class:`AxisSpec`.
    value
        The position a consumer may use.
    setpoint
        The position the stage was asked for.
    measured
        Whether ``value`` came from the controller's readback. A controller
        that cannot be queried reports ``setpoint`` as ``value`` and sets this
        to ``False``.
    """

    name: str
    value: float
    setpoint: float
    measured: bool

    def __post_init__(self) -> None:
        """Refuse a reading that cannot be labelled honestly."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("axis reading needs a name")
        object.__setattr__(self, "value", _finite(self.value, "axis reading"))
        object.__setattr__(self, "setpoint", _finite(self.setpoint, "axis setpoint"))
        if not isinstance(self.measured, bool):
            raise ValueError("axis reading measured flag must be a boolean")


@dataclass(frozen=True, slots=True)
class StageReading:
    """Every axis position of one read, and how the positions were obtained.

    Parameters
    ----------
    axes
        One reading per axis, unordered but uniquely named.
    monotonic_ns
        When the reading was taken.
    quality
        How the set of positions was obtained. ``MEASURED`` requires every axis
        reading to be measured; ``UNAVAILABLE`` means nothing was read at all
        and therefore carries no axes.

    Notes
    -----
    The label describes the weakest evidence in the set. A stage that reports
    one axis from a readback and one from its setpoint cannot label the pair
    ``MEASURED``.
    """

    axes: tuple[AxisReading, ...]
    monotonic_ns: int
    quality: QualityLabel

    def __post_init__(self) -> None:
        """Refuse a reading whose label overstates its axes."""
        if not isinstance(self.axes, tuple):
            raise ValueError("stage reading axes must be a tuple")
        if any(not isinstance(item, AxisReading) for item in self.axes):
            raise ValueError("stage reading axes must be AxisReading values")
        if len({item.name for item in self.axes}) != len(self.axes):
            raise ValueError("stage reading axis names must be unique")
        if type(self.monotonic_ns) is not int or self.monotonic_ns < 0:
            raise ValueError("stage reading time must be non-negative")
        if not isinstance(self.quality, QualityLabel):
            raise ValueError("stage reading needs a valid quality label")
        if self.quality is QualityLabel.MEASURED and not all(
            item.measured for item in self.axes
        ):
            raise ValueError("a measured stage reading needs every axis measured")
        if self.quality is QualityLabel.UNAVAILABLE and self.axes:
            raise ValueError("an unavailable stage reading carries no axes")

    @property
    def positions(self) -> Mapping[str, float]:
        """Return the positions by axis name."""
        return MappingProxyType({item.name: item.value for item in self.axes})


@runtime_checkable
class Stage(Protocol):
    """The typed position surface layered on the stage device.

    A stage is an ophyd-async ``StandardMovable`` whose ``read`` already returns
    a Bluesky reading of its signals; that method keeps its meaning. This
    Protocol adds what a plan needs beside it — declared axes with units and
    limits, a move that reports the positions it settled at, and an explicit
    cleanup — and defines no member whose name would shadow a device protocol.
    """

    @property
    def axes(self) -> Mapping[str, AxisSpec]:
        """Return the declared axes by name."""
        ...

    async def move(self, position: Mapping[str, float]) -> StageReading:
        """Move every named axis and return the reading after it settles."""
        ...

    async def read_position(self) -> StageReading:
        """Return the current axis positions and their evidence."""
        ...

    async def stop(self) -> None:
        """Stop motion and release the stage for another owner."""
        ...


class CommandStatus(StrEnum):
    """What the transport did with one illumination command.

    The values are the strings the serial service publishes, so a status can be
    compared and logged without a translation table.

    No member claims that light was emitted or that a frame arrived. ``SENT``
    means the bytes were written and no echo was taken; ``ECHOED`` means the
    service also read back the matching command text. An echo is diagnostic
    only: the validated MMStudio configuration ignored responses, so an echo
    proves neither a capture nor an optically correct pattern. ``UNCERTAIN``
    covers a timeout or a short write: the command may or may not have been
    acted on, and it must not be resent blindly. ``FAILED`` means the transport
    refused the write before it completed.
    """

    SENT = "sent"
    ECHOED = "echoed"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class IlluminationAction(StrEnum):
    """The command surface the initial illumination service exposes."""

    BLANK = "blank"
    OPEN = "open"
    PATTERN = "pattern"


@dataclass(frozen=True, slots=True)
class IlluminationCapability:
    """What an illumination service declares it can be asked to do.

    The default describes the 91-LED v1 panel: four semicircle patterns plus a
    centre pattern. Which LEDs each pattern lights lives in the firmware; the
    host records only the panel profile, its revision and the pattern identity,
    so a changed light map does not silently change what a recorded run means.

    Parameters
    ----------
    pattern_ids
        The pattern groups the service renders, ascending. The initial service
        declares the four semicircles and the centre of the 91-LED panel.
    center_pattern_id
        The pattern that lights the centre LED, or ``None`` when the panel has
        no centre pattern. When set it is one of ``pattern_ids``.
    panel_profile
        The panel profile the service advertises, so a store can record which
        layout a frame was taken on without reading firmware.
    panel_revision
        The revision of that profile; a positive integer. A run that cannot name
        the revision cannot say which LED map produced its frames.
    led_count
        How many LEDs the panel has; positive.
    blank_supported
        Whether the service can blank its outputs.
    open_supported
        Whether the service can clear the shutter-closed state.
    echo_expected
        Whether the service reads a command echo back. The validated MMStudio
        configuration does not, so it defaults to ``False``.
    dialect
        The command dialect the service speaks, so a later firmware or profile
        revision can be named without changing the plan's meaning.
    """

    pattern_ids: tuple[int, ...] = ILLUMINATION_PATTERN_IDS
    center_pattern_id: int | None = CENTER_PATTERN_ID
    panel_profile: str = "panel91"
    panel_revision: int = 1
    led_count: int = 91
    blank_supported: bool = True
    open_supported: bool = True
    echo_expected: bool = False
    dialect: str = "mm-ascii"

    def __post_init__(self) -> None:
        """Refuse a capability that advertises a pattern it cannot render."""
        if not isinstance(self.pattern_ids, tuple):
            raise ValueError("illumination pattern ids must be a tuple")
        if any(
            type(pattern) is not int or pattern <= 0 for pattern in self.pattern_ids
        ):
            raise ValueError("illumination pattern ids must be positive integers")
        if len(set(self.pattern_ids)) != len(self.pattern_ids):
            raise ValueError("illumination pattern ids must be unique")
        if list(self.pattern_ids) != sorted(self.pattern_ids):
            raise ValueError("illumination pattern ids must be ascending")
        if self.center_pattern_id is not None:
            if type(self.center_pattern_id) is not int:
                raise ValueError("illumination centre pattern id must be an integer")
            if self.center_pattern_id not in self.pattern_ids:
                raise ValueError(
                    "illumination centre pattern id must be one of the advertised "
                    f"patterns {self.pattern_ids}"
                )
        if not isinstance(self.panel_profile, str) or not self.panel_profile.strip():
            raise ValueError("illumination capability needs a panel profile name")
        if type(self.panel_revision) is not int or self.panel_revision <= 0:
            raise ValueError("illumination panel revision must be a positive integer")
        if type(self.led_count) is not int or self.led_count <= 0:
            raise ValueError("illumination led count must be a positive integer")
        if not isinstance(self.blank_supported, bool):
            raise ValueError("illumination blank support must be a boolean")
        if not isinstance(self.open_supported, bool):
            raise ValueError("illumination open support must be a boolean")
        if not isinstance(self.echo_expected, bool):
            raise ValueError("illumination echo expectation must be a boolean")
        if not isinstance(self.dialect, str) or not self.dialect.strip():
            raise ValueError("illumination capability needs a dialect name")


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    """What one illumination command did on the wire, as a plan may see it.

    This is the plan-facing record. The serial service publishes its own
    transport dataclass; :meth:`from_transport` is the single, named adapter
    from it, so a plan never reads service internals and a service change lands
    in one place.

    Parameters
    ----------
    action
        Which command was issued.
    command
        The ASCII command text written, without its terminator, such as
        ``"11"``. The service owns the mapping from action to bytes.
    status
        What the transport reported, under the same names the service uses.
    pattern_id
        The pattern group for a :attr:`IlluminationAction.PATTERN` command, and
        ``None`` otherwise. It may be any pattern an illumination service
        advertises, including the 91-LED panel's centre pattern; the DPCT plan
        itself requests only the four semicircles.
    echo
        The matching text read back, present only for an ``ECHOED`` status.
    sent_monotonic_ns
        When the write was attempted, when the service reports a clock. It is
        absent when the transport report carries no timestamp.
    detail
        Free text from the service, for a log rather than for a decision.
    """

    action: IlluminationAction
    command: str
    status: CommandStatus
    pattern_id: int | None = None
    echo: str | None = None
    sent_monotonic_ns: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        """Refuse a receipt whose evidence contradicts its own status."""
        if not isinstance(self.action, IlluminationAction):
            raise ValueError("command receipt needs an illumination action")
        if not isinstance(self.command, str) or not _TWO_DIGIT_COMMAND.fullmatch(
            self.command
        ):
            raise ValueError("command receipt needs a two-digit ASCII command")
        if not isinstance(self.status, CommandStatus):
            raise ValueError("command receipt needs a transport status")
        if self.pattern_id is not None and (
            type(self.pattern_id) is not int
            or self.pattern_id not in ILLUMINATION_PATTERN_IDS
        ):
            raise ValueError(
                "command receipt pattern id must be one of the illumination "
                f"patterns {ILLUMINATION_PATTERN_IDS}"
            )
        if self.action is IlluminationAction.PATTERN and self.pattern_id is None:
            raise ValueError("a pattern command needs the pattern it rendered")
        if (
            self.action is not IlluminationAction.PATTERN
            and self.pattern_id is not None
        ):
            raise ValueError("only a pattern command names a pattern")
        if not isinstance(self.detail, str):
            raise ValueError("command receipt detail must be text")
        if self.status is CommandStatus.ECHOED:
            if self.echo != self.command:
                raise ValueError("an echoed command needs its own echo text")
        elif self.echo is not None:
            raise ValueError("only an echoed command carries echo text")
        if self.sent_monotonic_ns is not None and (
            type(self.sent_monotonic_ns) is not int or self.sent_monotonic_ns < 0
        ):
            raise ValueError("command receipt time must be non-negative")

    @classmethod
    def from_transport(
        cls,
        *,
        action: IlluminationAction,
        command: str,
        status: CommandStatus | str,
        detail: str = "",
        echo: str | None = None,
        pattern_id: int | None = None,
        sent_monotonic_ns: int | None = None,
    ) -> CommandReceipt:
        """Build the plan-facing receipt from the serial service's report.

        The keyword names mirror the transport dataclass the service publishes
        (``command``, ``status``, ``detail``, ``echo``), so the adapter is a
        field-for-field call. ``action``, ``pattern_id`` and the optional clock
        are what the plan adds, because the service sends bytes and does not
        know which pattern a plan asked for.

        *status* also accepts the service's own status member or the bare word
        it serialises to, and converts it by value: a plan maps a transport
        report without importing the service's module, and an unrecognised word
        is refused here rather than becoming an untyped string downstream.
        """
        try:
            known = CommandStatus(status)
        except ValueError as error:
            raise ValueError(f"unknown transport status {status!r}") from error
        return cls(
            action=action,
            command=command,
            status=known,
            pattern_id=pattern_id,
            echo=echo,
            sent_monotonic_ns=sent_monotonic_ns,
            detail=detail,
        )

    @property
    def wrote_bytes(self) -> bool:
        """Whether the transport wrote anything at all."""
        return self.status in (
            CommandStatus.SENT,
            CommandStatus.ECHOED,
            CommandStatus.UNCERTAIN,
        )


@runtime_checkable
class Illumination(Protocol):
    """The MM-compatible illumination command surface, as a plan sees it.

    Each method returns a :class:`CommandReceipt`, never the serial service's
    own transport dataclass, so a plan depends on this module alone.
    """

    @property
    def capability(self) -> IlluminationCapability:
        """Return what this service declares it can be asked to do."""
        ...

    async def blank(self) -> CommandReceipt:
        """Blank all outputs without triggering a frame."""
        ...

    async def open(self) -> CommandReceipt:
        """Clear the shutter-closed state without triggering a frame."""
        ...

    async def pattern(self, pattern_id: int) -> CommandReceipt:
        """Render one pattern, which also pulses the camera trigger."""
        ...


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


__all__ = [
    "CENTER_PATTERN_ID",
    "DPCT_PATTERN_IDS",
    "ILLUMINATION_PATTERN_IDS",
    "AxisReading",
    "AxisSpec",
    "Camera",
    "CameraDescription",
    "CameraReadout",
    "CameraSettings",
    "CommandReceipt",
    "CommandStatus",
    "Illumination",
    "IlluminationAction",
    "IlluminationCapability",
    "Roi",
    "SettingRefusal",
    "SettingRefused",
    "Stage",
    "StageReading",
    "TriggerMode",
]
