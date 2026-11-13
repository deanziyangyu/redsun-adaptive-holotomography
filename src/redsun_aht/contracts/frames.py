"""Frame identity, exposure evidence and never-upgraded quality labels.

A frame is not identified by the position it happened to occupy in a document
stream. Bluesky assigns an event ``seq_num`` within a run, and that number is
reassigned when a run is restarted or recorded twice, so a stored frame keyed by
it cannot be found again from what was actually acquired. This module freezes
the acquisition identity instead: the run, the detector, the plan-assigned frame
ordinal, the plane-local ordinal, the illumination pattern and the scan index.
The producer's own readout sequence travels beside that identity as
``producer_sequence`` and is never promoted to it.

The second rule is that a :class:`QualityLabel` is never upgraded. A frame
labelled ``measured`` must carry a measured exposure window; a frame whose
pixels came from a simulator stays ``simulated`` when it is copied, replaced or
read back from a document. A payload that claims a stronger label than its own
evidence supports is refused rather than believed. Nothing in this module
relabels a value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

MAX_RUN_ID_BYTES = 32
"""Longest run identity accepted, counted in UTF-8 bytes."""

_SHA256_LENGTH = 64


class QualityLabel(StrEnum):
    """How a frame's pixels were obtained.

    The members run from the most to the least trustworthy, but the order is
    documentation only: the contract enforces that a label never *rises*
    without supporting evidence, not that a caller may compare two labels and
    relabel a record.

    ``MEASURED`` means the pixels were read from a detector. ``COMMANDED``
    means the frame answers a command but its arrival was not observed.
    ``SIMULATED`` means a simulator produced the pixels. ``DERIVED`` means the
    pixels were computed from other frames. ``UNAVAILABLE`` means the
    provenance is unknown, which is also the label for a legacy record that
    carries no evidence at all.
    """

    MEASURED = "measured"
    COMMANDED = "commanded"
    SIMULATED = "simulated"
    DERIVED = "derived"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """One acquisition run, named the way the store will index it.

    Parameters
    ----------
    run_id
        The store key of the run: non-empty text of at most
        :data:`MAX_RUN_ID_BYTES` UTF-8 bytes, with no whitespace. A run id is a
        token, not a sentence, because it names a directory.
    plan_name
        The plan that produced the run, for provenance.
    purpose
        What the operator acquired the run for, in one line.
    scan_id
        The plan's scan number, non-negative and independent of Bluesky's own
        counters.
    """

    run_id: str
    plan_name: str
    purpose: str
    scan_id: int

    def __post_init__(self) -> None:
        """Refuse a run identity the store cannot key on."""
        run_id = _text(self.run_id, "run id")
        if len(run_id.encode("utf-8")) > MAX_RUN_ID_BYTES:
            raise ValueError(
                f"run id must fit {MAX_RUN_ID_BYTES} UTF-8 bytes, "
                f"found {len(run_id.encode('utf-8'))}"
            )
        if any(character.isspace() for character in run_id):
            raise ValueError("run id must not contain whitespace")
        _text(self.plan_name, "plan name")
        _text(self.purpose, "purpose")
        _non_negative(self.scan_id, "scan id")

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {
            "run_id": self.run_id,
            "plan_name": self.plan_name,
            "purpose": self.purpose,
            "scan_id": self.scan_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RunIdentity:
        """Rebuild a run identity from :meth:`as_payload` output."""
        return cls(
            run_id=payload["run_id"],
            plan_name=payload["plan_name"],
            purpose=payload["purpose"],
            scan_id=payload["scan_id"],
        )


@dataclass(frozen=True, slots=True)
class FrameIdentity:
    """One frame's acquisition identity, independent of any document stream.

    Parameters
    ----------
    run
        The run the frame belongs to.
    detector_id
        The declared detector identity, which is not necessarily the Bluesky
        device name.
    sequence
        The plan-assigned ordinal of the frame within the run, counting from
        zero. It is assigned before acquisition, so a requested frame and the
        produced frame that answers it share one number.
    plane_ordinal
        The frame's position within its plane, counting from zero.
    pattern_id
        The illumination pattern the frame was taken with, or ``0`` when the
        frame was taken without a declared pattern.
    scan_index
        The scan point the frame belongs to, counting from zero.
    producer_sequence
        The detector's own readout sequence, when it has one. It is evidence
        about transport, never the identity.

    Notes
    -----
    Bluesky's event ``seq_num`` is absent by design. It counts events in a run,
    so a restarted, resumed or re-recorded run reassigns it to the same frames.
    """

    run: RunIdentity
    detector_id: str
    sequence: int
    plane_ordinal: int
    pattern_id: int
    scan_index: int
    producer_sequence: int | None = None

    def __post_init__(self) -> None:
        """Refuse an identity that cannot be matched to a frame."""
        if not isinstance(self.run, RunIdentity):
            raise ValueError("frame identity needs a run identity")
        _text(self.detector_id, "detector id")
        for name in ("sequence", "plane_ordinal", "pattern_id", "scan_index"):
            _non_negative(getattr(self, name), name.replace("_", " "))
        if self.producer_sequence is not None:
            _non_negative(self.producer_sequence, "producer sequence")

    @property
    def identity_key(self) -> tuple[str, str, int, int, int, int]:
        """Return the tuple that decides whether two frames are the same frame.

        ``producer_sequence`` is excluded: a requested frame has no readout
        sequence yet, and the produced frame that answers it must still compare
        equal.
        """
        return (
            self.run.run_id,
            self.detector_id,
            self.sequence,
            self.plane_ordinal,
            self.pattern_id,
            self.scan_index,
        )

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {
            "run": self.run.as_payload(),
            "detector_id": self.detector_id,
            "sequence": self.sequence,
            "plane_ordinal": self.plane_ordinal,
            "pattern_id": self.pattern_id,
            "scan_index": self.scan_index,
            "producer_sequence": self.producer_sequence,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FrameIdentity:
        """Rebuild a frame identity from :meth:`as_payload` output."""
        return cls(
            run=RunIdentity.from_payload(payload["run"]),
            detector_id=payload["detector_id"],
            sequence=payload["sequence"],
            plane_ordinal=payload["plane_ordinal"],
            pattern_id=payload["pattern_id"],
            scan_index=payload["scan_index"],
            producer_sequence=payload["producer_sequence"],
        )


@dataclass(frozen=True, slots=True)
class ExposureWindow:
    """When an exposure happened, as far as anyone actually knows.

    Parameters
    ----------
    duration_s
        The exposure duration. It is always known, because it is the value the
        camera was asked for.
    start_ns, end_ns
        Measured endpoints on one monotonic clock, or both absent. A declared
        duration with no measured window is not a measurement.
    quality
        How the window itself was obtained.
    source
        Free text naming where the window came from, or empty when the caller
        supplied none.

    Notes
    -----
    A ``MEASURED`` window needs both endpoints, and the measured interval must
    agree with the declared duration within 0.1 percent or one microsecond,
    whichever is larger. That tolerance exists because a service reports
    nanoseconds while a requested exposure is a float; it is not a licence to
    reconcile two unrelated clocks.
    """

    duration_s: float
    start_ns: int | None = None
    end_ns: int | None = None
    quality: QualityLabel = QualityLabel.UNAVAILABLE
    source: str = ""

    def __post_init__(self) -> None:
        """Refuse a window whose evidence contradicts its own label."""
        duration = _finite(self.duration_s, "exposure duration")
        if duration <= 0:
            raise ValueError("exposure duration must be positive")
        object.__setattr__(self, "duration_s", duration)
        if not isinstance(self.quality, QualityLabel):
            raise ValueError("exposure window needs a valid quality label")
        if not isinstance(self.source, str):
            raise ValueError("exposure window source must be text")
        if (self.start_ns is None) != (self.end_ns is None):
            raise ValueError("exposure start and end are both known or both absent")
        if self.start_ns is not None and self.end_ns is not None:
            _non_negative(self.start_ns, "exposure start")
            _non_negative(self.end_ns, "exposure end")
            if self.end_ns <= self.start_ns:
                raise ValueError("exposure end must follow its start")
            observed_s = (self.end_ns - self.start_ns) / 1e9
            tolerance_s = max(1e-6, 1e-3 * duration)
            if abs(observed_s - duration) > tolerance_s:
                raise ValueError(
                    "exposure duration disagrees with its measured window: "
                    f"{duration} s declared, {observed_s} s observed"
                )
        if self.quality is QualityLabel.MEASURED and self.start_ns is None:
            raise ValueError("a measured exposure needs measured start and end")

    @property
    def is_measured(self) -> bool:
        """Whether both endpoints came from a clock reading."""
        return self.quality is QualityLabel.MEASURED and self.start_ns is not None

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {
            "duration_s": self.duration_s,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "quality": str(self.quality),
            "source": self.source,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExposureWindow:
        """Rebuild a window from :meth:`as_payload` output."""
        return cls(
            duration_s=payload["duration_s"],
            start_ns=payload["start_ns"],
            end_ns=payload["end_ns"],
            quality=QualityLabel(payload["quality"]),
            source=payload["source"],
        )


@dataclass(frozen=True, slots=True)
class SettingChange:
    """One camera setting as it was applied, with the value read back.

    Parameters
    ----------
    name
        The setting's canonical key, such as ``"exposure"`` or ``"roi"``.
    value
        The value that was written.
    revision
        The setting revision the camera returned for the applied change. It is
        the number a :class:`FrameRecord` carries, so a frame's settings can be
        recovered from the timeline.
    applied_monotonic_ns
        When the service applied the change, on a monotonic clock.
    readback
        The value the device reported after applying the change, or ``None``
        when no readback was taken. A readback that differs from ``value`` is
        recorded as it is: a camera may clamp an exposure, and hiding that
        would make the timeline a wish rather than a record.
    """

    name: str
    value: bool | int | float | str
    revision: int
    applied_monotonic_ns: int
    readback: bool | int | float | str | None = None

    def __post_init__(self) -> None:
        """Refuse a change that cannot be placed in the timeline."""
        _text(self.name, "setting name")
        _setting_value(self.value, "setting value")
        _non_negative(self.revision, "setting revision")
        _non_negative(self.applied_monotonic_ns, "setting change time")
        if self.readback is not None:
            _setting_value(self.readback, "setting readback")

    @property
    def confirmed(self) -> bool:
        """Whether the device read the setting back at all."""
        return self.readback is not None

    @property
    def clamped(self) -> bool:
        """Whether the device read back a different value than it was given."""
        return self.readback is not None and self.readback != self.value

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {
            "name": self.name,
            "value": self.value,
            "revision": self.revision,
            "applied_monotonic_ns": self.applied_monotonic_ns,
            "readback": self.readback,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SettingChange:
        """Rebuild a setting change from :meth:`as_payload` output."""
        return cls(
            name=payload["name"],
            value=payload["value"],
            revision=payload["revision"],
            applied_monotonic_ns=payload["applied_monotonic_ns"],
            readback=payload["readback"],
        )


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One frame as the store will know it: identity and evidence, not pixels.

    Parameters
    ----------
    identity
        The acquisition identity of the frame.
    shape
        The frame's numpy shape, ``(height, width)``.
    dtype
        The readout dtype, such as ``"uint16"``.
    checksum
        Lowercase hexadecimal SHA-256 of the frame's bytes.
    uri
        Scheme-qualified location of the stored frame.
    exposure
        The exposure window the frame was taken with.
    setting_revision
        The camera setting revision in force for this frame.
    quality
        How the pixels were obtained.

    Notes
    -----
    A ``MEASURED`` record requires a ``MEASURED`` exposure window. That single
    implication is what stops a copy or a re-read document from promoting a
    simulated frame: replacing only the label, or only the window, leaves the
    pair inconsistent and is refused.
    """

    identity: FrameIdentity
    shape: tuple[int, int]
    dtype: str
    checksum: str
    uri: str
    exposure: ExposureWindow
    setting_revision: int
    quality: QualityLabel

    def __post_init__(self) -> None:
        """Refuse a record that cannot be audited, or overstates its quality."""
        if not isinstance(self.identity, FrameIdentity):
            raise ValueError("frame record needs a frame identity")
        _shape_2d(self.shape)
        object.__setattr__(self, "shape", (int(self.shape[0]), int(self.shape[1])))
        _dtype(self.dtype)
        _checksum(self.checksum)
        uri = _text(self.uri, "frame uri")
        if "://" not in uri:
            raise ValueError("frame uri must be scheme-qualified, as in file://")
        if not isinstance(self.exposure, ExposureWindow):
            raise ValueError("frame record needs an exposure window")
        _non_negative(self.setting_revision, "setting revision")
        if not isinstance(self.quality, QualityLabel):
            raise ValueError("frame record needs a valid quality label")
        if self.quality is QualityLabel.MEASURED and not self.exposure.is_measured:
            raise ValueError(
                "a measured frame needs a measured exposure window; "
                f"the window is {self.exposure.quality}"
            )

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {
            "identity": self.identity.as_payload(),
            "shape": [self.shape[0], self.shape[1]],
            "dtype": self.dtype,
            "checksum": self.checksum,
            "uri": self.uri,
            "exposure": self.exposure.as_payload(),
            "setting_revision": self.setting_revision,
            "quality": str(self.quality),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FrameRecord:
        """Rebuild a record from :meth:`as_payload` output.

        The quality label is read from the document and then checked against
        the document's own evidence, so a hand-edited payload cannot raise it.
        """
        return cls(
            identity=FrameIdentity.from_payload(payload["identity"]),
            shape=(int(payload["shape"][0]), int(payload["shape"][1])),
            dtype=payload["dtype"],
            checksum=payload["checksum"],
            uri=payload["uri"],
            exposure=ExposureWindow.from_payload(payload["exposure"]),
            setting_revision=payload["setting_revision"],
            quality=QualityLabel(payload["quality"]),
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _non_negative(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _shape_2d(value: object) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("frame shape must be two integers, (height, width)")
    if any(type(size) is not int or size <= 0 for size in value):
        raise ValueError("frame shape entries must be positive integers")


def _dtype(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("frame dtype must be a numpy dtype string")
    try:
        np.dtype(value)
    except TypeError as error:
        raise ValueError(f"unknown frame dtype {value!r}") from error


def _checksum(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("frame checksum must be a lowercase hexadecimal SHA-256")


def _setting_value(value: object, name: str) -> None:
    if isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        _finite(value, name)
        return
    raise ValueError(f"{name} must be a boolean, number or text")


__all__ = [
    "MAX_RUN_ID_BYTES",
    "ExposureWindow",
    "FrameIdentity",
    "FrameRecord",
    "QualityLabel",
    "RunIdentity",
    "SettingChange",
]
