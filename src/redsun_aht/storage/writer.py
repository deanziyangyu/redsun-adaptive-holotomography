"""A finite-run frame store: Zarr v3 on disk, and the documents it leaves.

Adapted, with modification, from two modules of the IX83 app
``redsun-searching-is-imaging`` at donor revision ``3a3bf06``
(``redsun-searching-is-imaging``, Apache-2.0):

* ``acquisition/_zarr_store.py`` -- ``ZarrFrameStore``, the append-per-frame
  array behind the camera service's ``FrameWriter`` seam;
* the store section of ``services/mmcore_camera.py`` -- ``local_store_path``
  and the bounded service-arrival frame-timing sidecar, whose schema, clock and
  timestamp semantics this module reproduces exactly.

Changed for this bundle:

* ``acquire-zarr`` is gone. The array is a plain ``zarr`` v3 array, so a reader
  opens it with ``zarr`` alone: no streaming writer, no shared-memory ring and
  no C++ extension are involved in reading a recorded run.
* The store is create-only. A location that already holds frames, a manifest,
  chunk data or a foreign file tree is refused with a named
  :class:`StoreRefusal`; nothing is ever truncated or overwritten.
* Every appended frame can carry its acquisition identity -- run, detector,
  sequence, plane ordinal, pattern and scan index -- and the camera setting
  revision in force, and the manifest records each frame's SHA-256. The camera
  service still only needs ``append`` and ``close``, so identity is optional
  and is bound before the first frame when a caller knows it.
* ``close`` writes a manifest beside the array, so a standalone reader can
  verify shape, dtype, chunking, frame count and every frame's checksum without
  the writer object that produced them.
* The timing sidecar is written from the writer's own records, in the camera
  service's ``redsun-camera-frame-timing-v1`` schema and
  ``python.time.monotonic_ns`` clock. A capture that records no timing writes
  no sidecar.

One plane is laid out as::

    <store>/<data-key>/zarr.json             Zarr v3 array metadata
    <store>/<data-key>/c/0/0/0               one chunk per frame, in order
    <store>/<data-key>.manifest.json         shape, dtype, chunks, count, hashes
    <store>/<data-key>.timing.json           service arrival stamps, when timed

The manifest and timing sidecar are beside the array rather than inside it, and
are named after the data key, so a store directory may hold more than one plane
without ambiguity.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import zarr

from ..contracts.frames import FrameIdentity, RunIdentity

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from ..services.mmcore_camera import FrameWriter

MANIFEST_SCHEMA_V1: Final = "redsun-aht-frame-store-manifest-v1"
"""Previous manifest schema, with a required setting revision on every frame."""

MANIFEST_SCHEMA: Final = "redsun-aht-frame-store-manifest-v2"
"""Schema of the manifest a closed store carries."""

SUPPORTED_MANIFEST_SCHEMAS: Final = (MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA)
"""Manifest schemas this package can read."""

MANIFEST_SUFFIX: Final = ".manifest.json"
"""Suffix of a plane's manifest, appended to its data key."""

ARRAY_FORMAT: Final = "zarr-v3"
"""The array format this package writes and reads."""

CHECKSUM_ALGORITHM: Final = "sha256"
"""The digest used for every frame and array checksum."""

FRAME_CHECKSUM_SCOPE: Final = "c_order_raw_dtype_bytes"
"""What a frame checksum covers: its pixels, C-ordered, in their own dtype."""

TIMING_SCHEMA: Final = "redsun-camera-frame-timing-v1"
"""The camera service's schema for the frame-timing sidecar."""

TIMING_CLOCK: Final = "python.time.monotonic_ns"
"""The clock domain the timing sidecar's stamps are expressed in."""

TIMING_SEMANTICS: Final = "service_arrival_after_core_frame_read"
"""What a timing record's stamp means, so a reader does not over-read it."""

TIMING_FRAME_METADATA_SOURCE: Final = (
    "per-record; null when the camera is not sequencing"
)
"""Where a record's raw adapter metadata came from."""

TIMING_FRAME_METADATA_SEMANTICS: Final = (
    "raw per-frame MMCore tags; timestamps are not assumed to be "
    "exposure intervals or mapped to the host monotonic clock"
)
"""What a record's raw adapter metadata does and does not prove."""

TIMING_RECORD_LIMIT: Final = 100_000
"""Most timing records one store accepts, matching the camera service."""


class StoreRefusal(StrEnum):
    """Every way a store operation is refused, named rather than described.

    A refusal names the condition the caller has to handle. The camera service
    never inspects one, but a plan that receives a partial-run refusal has to
    tell it from a checksum failure without parsing a sentence, so each member
    is a stable token.
    """

    STORE_NOT_EMPTY = "store_not_empty"
    STORE_ALREADY_CLOSED = "store_already_closed"
    STORE_UNCOMMITTED = "store_uncommitted"
    STORE_CLOSED = "store_closed"
    STORE_LAYOUT_MISMATCH = "store_layout_mismatch"
    MANIFEST_UNREADABLE = "manifest_unreadable"
    MANIFEST_AMBIGUOUS = "manifest_ambiguous"
    FRAME_SHAPE_MISMATCH = "frame_shape_mismatch"
    FRAME_DTYPE_MISMATCH = "frame_dtype_mismatch"
    PLANNED_FRAMES_EXCEEDED = "planned_frames_exceeded"
    IDENTITY_LOCKED = "identity_locked"
    TIMING_NOT_RECORDED = "timing_not_recorded"
    TIMING_CAPACITY_EXCEEDED = "timing_capacity_exceeded"


class StoreRefused(ValueError):
    """A store operation refused for a named reason, with the values behind it.

    Raised by the writer, which returns an index rather than a result object.
    The receipts module converts one into a typed refusal, so a caller that
    prefers a value over an exception never has to catch this class.
    """

    def __init__(self, reason: StoreRefusal, evidence: Mapping[str, Any]) -> None:
        """Keep the machine-readable reason beside the values behind it."""
        self.reason = reason
        self.evidence = dict(evidence)
        serialised = json.dumps(self.evidence, sort_keys=True, default=str)
        super().__init__(f"{reason.value}: {serialised}")


def frame_checksum(frame: NDArray[Any]) -> str:
    """Return the SHA-256 of one frame's pixels, C-ordered, in its own dtype.

    Parameters
    ----------
    frame
        A two-dimensional frame.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256, exactly what a manifest records.
    """
    pixels = np.ascontiguousarray(np.asarray(frame))
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def frames_checksum(frames: Iterable[NDArray[Any]]) -> str:
    """Return the SHA-256 of *frames*' bytes in the order they are given.

    The digest is the frame checksums' common parent: a reader can recompute it
    from the stored array alone, and a store whose frames are reordered or
    altered produces a different one. Feeding a frame at a time keeps a large
    plane off the heap, which is how the receipts module verifies a store.
    """
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(np.ascontiguousarray(np.asarray(frame)).tobytes())
    return digest.hexdigest()


def local_store_path(uri: str) -> Path:
    """Return the filesystem path *uri* names, refusing a non-local store.

    Mirrors ``services.mmcore_camera.local_store_path`` so a store written
    through the service and one written through this factory agree about what
    a location means, without the storage package importing a transport.
    """
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    if "://" in uri:
        raise ValueError("a frame store must be a local file path")
    return Path(uri)


def store_uri(path: Path) -> str:
    """Return the scheme-qualified URI of *path*, absolute and readable."""
    return path.resolve().as_uri()


def array_path(uri: str, data_key: str) -> Path:
    """Return the filesystem path of the array *data_key* inside *uri*."""
    _require_data_key(data_key)
    return local_store_path(uri) / data_key


def manifest_path(uri: str, data_key: str) -> Path:
    """Return the path of the manifest that closes *data_key* inside *uri*."""
    _require_data_key(data_key)
    return local_store_path(uri) / f"{data_key}{MANIFEST_SUFFIX}"


def timing_sidecar_path(uri: str, data_key: str) -> Path:
    """Return one plane's timing sidecar inside its store directory."""
    _require_data_key(data_key)
    return local_store_path(uri) / f"{data_key}.timing.json"


@dataclass(frozen=True, slots=True)
class FrameStamp:
    """What the caller knows about a frame beyond its pixels.

    Parameters
    ----------
    pattern_id
        The illumination pattern the frame was taken with, or ``0``.
    scan_index
        The scan point the frame belongs to, counting from zero.
    producer_sequence
        The detector's own readout sequence, when it has one. It is evidence
        about transport and never becomes the frame's identity.
    setting_revision
        The camera setting revision in force for this frame. Left as ``None``
        the writer's current revision applies, which is the usual case: a
        revision changes between frames, not with every frame.
    """

    pattern_id: int = 0
    scan_index: int = 0
    producer_sequence: int | None = None
    setting_revision: int | None = None

    def __post_init__(self) -> None:
        """Refuse a stamp that cannot be placed in a run."""
        _require_index(self.pattern_id, "pattern id")
        _require_index(self.scan_index, "scan index")
        if self.producer_sequence is not None:
            _require_index(self.producer_sequence, "producer sequence")
        if self.setting_revision is not None:
            _require_index(self.setting_revision, "setting revision")


@dataclass(frozen=True, slots=True)
class SettingMark:
    """The setting revision that came into force at one frame boundary.

    Parameters
    ----------
    first_frame
        The index of the first frame recorded under *revision*.
    revision
        The camera setting revision, as the camera service reported it, or
        ``None`` when pixels cannot be associated with a revision.
    """

    first_frame: int
    revision: int | None

    def __post_init__(self) -> None:
        """Refuse a mark that names no usable boundary."""
        _require_index(self.first_frame, "setting mark frame")
        if self.revision is not None:
            _require_index(self.revision, "setting revision")

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {"first_frame": self.first_frame, "revision": self.revision}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SettingMark:
        """Rebuild a mark from :meth:`as_payload` output."""
        return cls(first_frame=payload["first_frame"], revision=payload["revision"])


@dataclass(frozen=True, slots=True)
class FrameEntry:
    """One stored frame, as the manifest records it.

    Parameters
    ----------
    index
        The frame's zero-based index in the array, which is its append order.
    checksum
        Lowercase hexadecimal SHA-256 of the frame's pixels.
    setting_revision
        The camera setting revision in force when the frame was appended.
    pattern_id
        The illumination pattern the frame was taken with, or ``0``.
    scan_index
        The scan point the frame belongs to.
    identity
        The frame's acquisition identity, or ``None`` when the store was opened
        without a run and no caller bound one before the frame was written.
    """

    index: int
    checksum: str
    setting_revision: int | None
    pattern_id: int
    scan_index: int
    identity: FrameIdentity | None = None

    def __post_init__(self) -> None:
        """Refuse an entry that cannot be audited."""
        _require_index(self.index, "frame index")
        _require_digest(self.checksum, "frame checksum")
        if self.setting_revision is not None:
            _require_index(self.setting_revision, "setting revision")
        _require_index(self.pattern_id, "pattern id")
        _require_index(self.scan_index, "scan index")
        if self.identity is not None and not isinstance(self.identity, FrameIdentity):
            raise TypeError("a frame entry holds a frame identity or nothing")

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {
            "index": self.index,
            "checksum": self.checksum,
            "setting_revision": self.setting_revision,
            "pattern_id": self.pattern_id,
            "scan_index": self.scan_index,
            "identity": None if self.identity is None else self.identity.as_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FrameEntry:
        """Rebuild an entry from :meth:`as_payload` output."""
        identity = payload.get("identity")
        return cls(
            index=payload["index"],
            checksum=payload["checksum"],
            setting_revision=payload["setting_revision"],
            pattern_id=payload["pattern_id"],
            scan_index=payload["scan_index"],
            identity=None if identity is None else FrameIdentity.from_payload(identity),
        )


@dataclass(frozen=True, slots=True)
class StoreManifest:
    """What a closed store contains, verifiable without the writer.

    Parameters
    ----------
    schema
        The document schema, which a reader refuses to reinterpret.
    array_format
        The array format, which must be :data:`ARRAY_FORMAT`.
    store_uri
        The scheme-qualified location of the store directory.
    array_uri
        The scheme-qualified location of the array itself, ``store_uri`` plus
        the data key. This is the asset a receipt names.
    data_key
        The key the array sits under inside the store.
    shape
        One frame's shape, ``(height, width)``.
    dtype
        The array's numpy dtype name, such as ``"uint16"``.
    chunks
        The array's chunk shape, always ``(1, height, width)``: one frame per
        chunk, so a reader fetches exactly one frame per chunk.
    frame_count
        How many frames the store holds.
    planned_frames
        How many frames the run asked for, when it was opened with a bound, or
        ``None`` when the store is not a finite run. A store holding fewer
        frames than this is a partial run.
    frames
        One entry per stored frame, in append order.
    array_checksum
        SHA-256 of every frame's bytes, in append order.
    run, detector_id
        The run and detector the frames belong to, or ``None`` when the store
        was opened without them.
    plane_start
        The plane ordinal of the frames this store holds, within its run.
        Every frame in the store carries it: a plane is the set of frames a
        receipt admits together, and
        :meth:`~redsun_aht.contracts.assets.PlaneReceipt.admits` compares a
        frame's ordinal with the receipt's rather than counting inside it.
    setting_timeline
        The setting revisions the frames were recorded under, in frame order.
    created_utc, closed_utc
        When the store was opened and when it was closed, ISO 8601 in UTC.
    timing_sidecar_uri
        The scheme-qualified location of the timing sidecar, or ``None`` when
        the capture did not record timing.
    """

    schema: str
    array_format: str
    store_uri: str
    array_uri: str
    data_key: str
    shape: tuple[int, int]
    dtype: str
    chunks: tuple[int, int, int]
    frame_count: int
    planned_frames: int | None
    frames: tuple[FrameEntry, ...]
    array_checksum: str
    run: RunIdentity | None
    detector_id: str | None
    plane_start: int
    setting_timeline: tuple[SettingMark, ...]
    created_utc: str
    closed_utc: str
    timing_sidecar_uri: str | None = None

    def __post_init__(self) -> None:
        """Refuse a manifest that cannot describe a closed store."""
        _require_text(self.schema, "manifest schema")
        _require_text(self.array_format, "array format")
        _require_text(self.store_uri, "store uri")
        _require_text(self.array_uri, "array uri")
        _require_data_key(self.data_key)
        _require_frame_shape(self.shape)
        if np.dtype(self.dtype).name != self.dtype:
            raise ValueError("manifest dtype must be a numpy dtype name")
        if tuple(self.chunks) != (1, *self.shape):
            raise ValueError("a finite-run store chunks one frame at a time")
        _require_index(self.frame_count, "frame count")
        if len(self.frames) != self.frame_count:
            raise ValueError(
                "manifest frame count disagrees with its frame entries: "
                f"{self.frame_count} declared, {len(self.frames)} listed"
            )
        for expected, entry in enumerate(self.frames):
            if entry.index != expected:
                raise ValueError(
                    "manifest frames must be listed in append order: "
                    f"{entry.index} where {expected} was expected"
                )
        _require_digest(self.array_checksum, "array checksum")
        if self.planned_frames is not None:
            _require_index(self.planned_frames, "planned frames")
        if (self.run is None) != (self.detector_id is None):
            raise ValueError(
                "a manifest names both its run and its detector, or neither"
            )
        if self.run is not None and not isinstance(self.run, RunIdentity):
            raise TypeError("a manifest run is a run identity")
        _require_index(self.plane_start, "plane start")
        if not isinstance(self.setting_timeline, tuple) or not self.setting_timeline:
            raise ValueError("a manifest records at least one setting revision")
        previous = -1
        for mark in self.setting_timeline:
            if mark.first_frame <= previous:
                raise ValueError("setting marks must advance with the frames")
            previous = mark.first_frame
        _require_text(self.created_utc, "creation time")
        _require_text(self.closed_utc, "closing time")
        if self.timing_sidecar_uri is not None:
            _require_text(self.timing_sidecar_uri, "timing sidecar uri")

    @property
    def setting_revision(self) -> int | None:
        """Return the last frame's setting revision, or ``None`` if unknown."""
        return self.setting_timeline[-1].revision

    @property
    def is_complete(self) -> bool:
        """Whether the store holds every frame the run planned for."""
        return self.planned_frames is None or self.frame_count >= self.planned_frames

    def as_payload(self) -> dict[str, Any]:
        """Return the JSON-compatible document the manifest file holds."""
        return {
            "schema": self.schema,
            "array_format": self.array_format,
            "store_uri": self.store_uri,
            "array_uri": self.array_uri,
            "data_key": self.data_key,
            "shape": [self.shape[0], self.shape[1]],
            "dtype": self.dtype,
            "chunks": [self.chunks[0], self.chunks[1], self.chunks[2]],
            "frame_count": self.frame_count,
            "planned_frames": self.planned_frames,
            "checksum_algorithm": CHECKSUM_ALGORITHM,
            "frame_checksum_scope": FRAME_CHECKSUM_SCOPE,
            "frames": [entry.as_payload() for entry in self.frames],
            "array_checksum": self.array_checksum,
            "run": None if self.run is None else self.run.as_payload(),
            "detector_id": self.detector_id,
            "plane_start": self.plane_start,
            "setting_timeline": [mark.as_payload() for mark in self.setting_timeline],
            "created_utc": self.created_utc,
            "closed_utc": self.closed_utc,
            "timing_sidecar_uri": self.timing_sidecar_uri,
        }

    def as_json(self) -> str:
        """Return the manifest text, sorted and indented, ready to write."""
        return json.dumps(self.as_payload(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StoreManifest:
        """Rebuild a manifest from :meth:`as_payload` output."""
        run = payload.get("run")
        shape = payload["shape"]
        chunks = payload["chunks"]
        return cls(
            schema=payload["schema"],
            array_format=payload["array_format"],
            store_uri=payload["store_uri"],
            array_uri=payload["array_uri"],
            data_key=payload["data_key"],
            shape=(int(shape[0]), int(shape[1])),
            dtype=payload["dtype"],
            chunks=(int(chunks[0]), int(chunks[1]), int(chunks[2])),
            frame_count=payload["frame_count"],
            planned_frames=payload.get("planned_frames"),
            frames=tuple(FrameEntry.from_payload(item) for item in payload["frames"]),
            array_checksum=payload["array_checksum"],
            run=None if run is None else RunIdentity.from_payload(run),
            detector_id=payload.get("detector_id"),
            plane_start=payload["plane_start"],
            setting_timeline=tuple(
                SettingMark.from_payload(item) for item in payload["setting_timeline"]
            ),
            created_utc=payload["created_utc"],
            closed_utc=payload["closed_utc"],
            timing_sidecar_uri=payload.get("timing_sidecar_uri"),
        )

    @classmethod
    def from_json(cls, text: str) -> StoreManifest:
        """Rebuild a manifest from :meth:`as_json` output."""
        return cls.from_payload(json.loads(text))


@dataclass(frozen=True, slots=True)
class TimingRecord:
    """One frame's service-arrival stamp, in the camera service's schema.

    Parameters
    ----------
    written_index
        The frame's zero-based index in the array, which joins the record to
        its manifest entry.
    frame_id
        The store's own one-based ordinal for the frame. The camera service
        records its grab ordinal under this key; both are diagnostic, and
        ``written_index`` is what a reader joins on.
    service_monotonic_ns
        When the service had the frame in hand, on the monotonic clock named by
        the sidecar's ``clock``.
    camera_frame_counter
        The camera's own frame counter, when the camera publishes one.
    mmcore_metadata_source, mmcore_metadata
        The raw adapter tags taken with the frame, unmodified, and where they
        came from. Neither is an exposure interval.
    """

    written_index: int
    frame_id: int
    service_monotonic_ns: int
    camera_frame_counter: int | None = None
    mmcore_metadata_source: str | None = None
    mmcore_metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Refuse a record that cannot be joined to a frame."""
        _require_index(self.written_index, "timing written index")
        _require_index(self.frame_id, "timing frame id")
        _require_index(self.service_monotonic_ns, "timing stamp")
        if self.camera_frame_counter is not None:
            _require_index(self.camera_frame_counter, "camera frame counter")
        if self.mmcore_metadata is not None:
            if not isinstance(self.mmcore_metadata, Mapping):
                raise TypeError("timing metadata must be a mapping of text to text")
            object.__setattr__(
                self,
                "mmcore_metadata",
                {str(key): str(value) for key, value in self.mmcore_metadata.items()},
            )

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {
            "written_index": self.written_index,
            "frame_id": self.frame_id,
            "service_monotonic_ns": self.service_monotonic_ns,
            "camera_frame_counter": self.camera_frame_counter,
            "mmcore_metadata_source": self.mmcore_metadata_source,
            "mmcore_metadata": (
                None if self.mmcore_metadata is None else dict(self.mmcore_metadata)
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TimingRecord:
        """Rebuild a record from :meth:`as_payload` output."""
        return cls(
            written_index=payload["written_index"],
            frame_id=payload["frame_id"],
            service_monotonic_ns=payload["service_monotonic_ns"],
            camera_frame_counter=payload.get("camera_frame_counter"),
            mmcore_metadata_source=payload.get("mmcore_metadata_source"),
            mmcore_metadata=payload.get("mmcore_metadata"),
        )


@dataclass(frozen=True, slots=True)
class TimingSidecar:
    """The frame-timing document written beside a store that recorded timing.

    The field order and the payload keys are the camera service's, so a reader
    written for that service reads this document unchanged. ``clock`` and
    ``timestamp_semantics`` are what stop the stamps from being mistaken for
    exposure intervals or for an absolute time. ``store_uri`` is the store's
    filesystem path, spelled the way the camera service writes it, rather than
    a ``file://`` URI.
    """

    schema: str
    timestamp_semantics: str
    clock: str
    frame_metadata_source: str
    frame_metadata_semantics: str
    store_uri: str
    data_key: str
    written_count: int
    records: tuple[TimingRecord, ...]
    camera_frame_counter: int | None = None

    def __post_init__(self) -> None:
        """Refuse a sidecar that cannot be joined to its store."""
        _require_text(self.schema, "timing schema")
        _require_text(self.timestamp_semantics, "timestamp semantics")
        _require_text(self.clock, "timing clock")
        _require_text(self.store_uri, "timing store uri")
        _require_data_key(self.data_key)
        _require_index(self.written_count, "timing written count")
        if len(self.records) != self.written_count:
            raise ValueError(
                "timing record count disagrees with its entries: "
                f"{self.written_count} declared, {len(self.records)} listed"
            )
        for expected, record in enumerate(self.records):
            if record.written_index != expected:
                raise ValueError(
                    "timing records must be listed in append order: "
                    f"{record.written_index} where {expected} was expected"
                )
        if self.camera_frame_counter is not None:
            _require_index(self.camera_frame_counter, "camera frame counter")

    def as_payload(self) -> dict[str, Any]:
        """Return the JSON-compatible document the sidecar file holds."""
        return {
            "schema": self.schema,
            "timestamp_semantics": self.timestamp_semantics,
            "clock": self.clock,
            "frame_metadata_source": self.frame_metadata_source,
            "frame_metadata_semantics": self.frame_metadata_semantics,
            "store_uri": self.store_uri,
            "data_key": self.data_key,
            "written_count": self.written_count,
            "records": [record.as_payload() for record in self.records],
            "camera_frame_counter": self.camera_frame_counter,
        }

    def as_json(self) -> str:
        """Return the sidecar text, sorted and indented, ready to write."""
        return json.dumps(self.as_payload(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TimingSidecar:
        """Rebuild a sidecar from :meth:`as_payload` output."""
        return cls(
            schema=payload["schema"],
            timestamp_semantics=payload["timestamp_semantics"],
            clock=payload["clock"],
            frame_metadata_source=payload["frame_metadata_source"],
            frame_metadata_semantics=payload["frame_metadata_semantics"],
            store_uri=payload["store_uri"],
            data_key=payload["data_key"],
            written_count=payload["written_count"],
            records=tuple(
                TimingRecord.from_payload(item) for item in payload["records"]
            ),
            camera_frame_counter=payload.get("camera_frame_counter"),
        )

    @classmethod
    def from_json(cls, text: str) -> TimingSidecar:
        """Rebuild a sidecar from :meth:`as_json` output."""
        return cls.from_payload(json.loads(text))


class RunFrameStore:
    """Append YX frames to a create-only Zarr v3 array, then close it.

    The class satisfies the camera service's ``FrameWriter`` protocol, so the
    service can write through it with ``append`` and ``close`` alone. A caller
    that knows the acquisition identity binds it before the first frame and
    passes a :class:`FrameStamp` with each frame; the manifest then carries the
    identity, the setting revision and the checksum of every frame.

    Parameters
    ----------
    uri
        The store directory, as a path or a ``file://`` URI. It is created if
        absent and never truncated.
    data_key
        The key the array sits under inside the store. It is also the stem of
        the manifest's file name.
    frame
        One frame, which fixes the store's frame shape and dtype. Only the
        layout is taken from it; no pixels are stored.
    run
        The run the frames belong to. With *detector_id* it makes every
        appended frame carry a :class:`~redsun_aht.contracts.frames.FrameIdentity`.
    detector_id
        The detector identity the frames were recorded for.
    sequence_offset
        The run-wide ordinal of this store's first frame. A second plane of the
        same run starts at the number of frames already recorded.
    plane_start
        The plane ordinal of the frames this store holds. Every frame in the
        store carries it, because a plane is the set of frames one receipt
        admits together; a run with two planes records two stores.
    pattern_id
        The illumination pattern of the frames, used until a stamp overrides it.
    scan_index
        The scan point of the frames, used until a stamp overrides it.
    planned_frames
        How many frames the run is expected to write. A store that reaches it
        refuses another frame rather than growing past the run it belongs to,
        and a store closed short of it is a partial run.
    record_timing
        Whether to record one service-arrival stamp per frame and write the
        timing sidecar on close. The sidecar is bounded to
        :data:`TIMING_RECORD_LIMIT` records, as the camera service bounds its
        own.
    setting_revision
        The camera setting revision in force before the first frame.

    Raises
    ------
    StoreRefused
        If the location already holds frames, a manifest, chunk data or
        something that is not an array, or if an empty array left there
        disagrees with *frame*'s layout. Nothing is truncated.
    ValueError
        If *data_key* is not a single path component or *frame* is not a
        two-dimensional array.
    """

    def __init__(
        self,
        uri: str,
        data_key: str,
        frame: NDArray[Any],
        *,
        run: RunIdentity | None = None,
        detector_id: str | None = None,
        sequence_offset: int = 0,
        plane_start: int = 0,
        pattern_id: int = 0,
        scan_index: int = 0,
        planned_frames: int | None = None,
        record_timing: bool = False,
        setting_revision: int | None = 0,
    ) -> None:
        """Open the store, creating its array only if the location is unused."""
        _require_data_key(data_key)
        _require_frame_shape(tuple(np.shape(frame)))
        if (run is None) != (detector_id is None):
            raise ValueError("a store binds both its run and its detector, or neither")
        if run is not None and not isinstance(run, RunIdentity):
            raise TypeError("a store run is a run identity")
        if detector_id is not None:
            _require_text(detector_id, "detector id")
        _require_index(sequence_offset, "sequence offset")
        _require_index(plane_start, "plane start")
        _require_index(pattern_id, "pattern id")
        _require_index(scan_index, "scan index")
        if setting_revision is not None:
            _require_index(setting_revision, "setting revision")
        if planned_frames is not None:
            _require_index(planned_frames, "planned frames")
            if planned_frames == 0:
                raise ValueError("a finite run plans at least one frame")
        if not isinstance(record_timing, bool):
            raise TypeError("a store records timing or does not")

        pixels = np.asarray(frame)
        if pixels.dtype.kind == "O":
            raise ValueError("a frame store cannot hold object arrays")

        self._uri = uri
        self._data_key = data_key
        self._store_path = local_store_path(uri)
        self._shape = (int(pixels.shape[0]), int(pixels.shape[1]))
        self._run = run
        self._detector_id = detector_id
        self._sequence_offset = sequence_offset
        self._plane_start = plane_start
        self._pattern_id = pattern_id
        self._scan_index = scan_index
        self._planned_frames = planned_frames
        self._record_timing = record_timing
        self._setting_revision = setting_revision
        self._setting_timeline: list[SettingMark] = [
            SettingMark(first_frame=0, revision=setting_revision)
        ]
        self._entries: list[FrameEntry] = []
        self._timings: list[TimingRecord] = []
        self._digest = hashlib.sha256()
        self._frames_written = 0
        self._closed = False
        self._write_uncertain = False
        self._manifest: StoreManifest | None = None
        self._created_utc = _utc_now()

        try:
            self._store_path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise StoreRefused(
                StoreRefusal.STORE_NOT_EMPTY,
                {
                    "store": str(self._store_path),
                    "reason": "the store location is not a writable directory",
                    "error": f"{type(error).__name__}: {error}",
                },
            ) from error
        self._lock_path = self._store_path / f".{data_key}.writer.lock"
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            try:
                held = int(
                    zarr.open_array(
                        store=str(self._store_path / data_key), mode="r"
                    ).shape[0]
                )
            except (zarr.errors.BaseZarrError, OSError, ValueError):
                held = 0
            raise StoreRefused(
                StoreRefusal.STORE_NOT_EMPTY,
                {
                    "store": str(self._store_path),
                    "frames_held": held,
                    "reason": "another writer owns this plane",
                },
            ) from error
        try:
            os.close(fd)
            self._array = self._open_array(pixels)
        except BaseException:
            self._lock_path.unlink(missing_ok=True)
            raise
        self._array_uri = store_uri(self._store_path / data_key)

    def _open_array(self, frame: NDArray[Any]) -> zarr.Array[Any]:
        """Create the array, or adopt one an interrupted open left empty."""
        location = self._store_path / self._data_key
        if manifest_path(self._uri, self._data_key).exists():
            raise StoreRefused(
                StoreRefusal.STORE_ALREADY_CLOSED,
                {
                    "manifest": str(manifest_path(self._uri, self._data_key)),
                    "reason": "the store already carries a closed run's manifest",
                },
            )
        if not location.exists():
            try:
                return zarr.open_array(
                    store=str(location),
                    mode="w-",
                    shape=(0, *self._shape),
                    chunks=(1, *self._shape),
                    dtype=frame.dtype,
                    zarr_format=3,
                )
            except (zarr.errors.BaseZarrError, OSError, ValueError) as error:
                raise StoreRefused(
                    StoreRefusal.STORE_NOT_EMPTY,
                    {
                        "array": str(location),
                        "reason": "the location could not be created as an array",
                        "error": f"{type(error).__name__}: {error}",
                    },
                ) from error

        try:
            existing = zarr.open_array(store=str(location), mode="r")
        except (zarr.errors.BaseZarrError, OSError, ValueError) as error:
            raise StoreRefused(
                StoreRefusal.STORE_NOT_EMPTY,
                {
                    "array": str(location),
                    "reason": "the location holds no readable zarr array",
                    "error": f"{type(error).__name__}: {error}",
                },
            ) from error
        held = int(existing.shape[0])
        if held:
            raise StoreRefused(
                StoreRefusal.STORE_NOT_EMPTY,
                {
                    "array": str(location),
                    "frames_held": held,
                    "reason": "a store that already holds frames is never written",
                },
            )
        if (location / "c").exists():
            raise StoreRefused(
                StoreRefusal.STORE_NOT_EMPTY,
                {
                    "array": str(location),
                    "reason": "chunk data is present although the shape is empty",
                },
            )
        layout = (
            tuple(existing.shape[1:]),
            existing.dtype.name,
            tuple(existing.chunks),
        )
        offered = (self._shape, frame.dtype.name, (1, *self._shape))
        if layout != offered:
            raise StoreRefused(
                StoreRefusal.STORE_LAYOUT_MISMATCH,
                {
                    "array": str(location),
                    "held": {
                        "shape": list(layout[0]),
                        "dtype": layout[1],
                        "chunks": list(layout[2]),
                    },
                    "offered": {
                        "shape": list(offered[0]),
                        "dtype": offered[1],
                        "chunks": list(offered[2]),
                    },
                },
            )
        return zarr.open_array(store=str(location), mode="a")

    @property
    def uri(self) -> str:
        """Return the store location as it was given."""
        return self._uri

    @property
    def data_key(self) -> str:
        """Return the key the array sits under inside the store."""
        return self._data_key

    @property
    def array_uri(self) -> str:
        """Return the scheme-qualified location of the array itself."""
        return self._array_uri

    @property
    def shape(self) -> tuple[int, int]:
        """Return one frame's shape as ``(height, width)``."""
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return the array's dtype."""
        return self._array.dtype

    @property
    def planned_frames(self) -> int | None:
        """Return how many frames the run asked for, if it is bounded."""
        return self._planned_frames

    def set_planned_frames(self, count: int) -> None:
        """Bind a finite count once, before any frame has been appended."""
        _require_index(count, "planned frames")
        if count == 0:
            raise ValueError("planned frames must be positive")
        if self._closed or self._frames_written or self._planned_frames is not None:
            raise RuntimeError("planned frame count is already fixed")
        self._planned_frames = count

    @property
    def frames_written(self) -> int:
        """Number of frames successfully appended to the store."""
        return self._frames_written

    @property
    def closed(self) -> bool:
        """Whether the store has been closed and its manifest written."""
        return self._closed

    @property
    def manifest(self) -> StoreManifest | None:
        """Return the manifest :meth:`close` wrote, or ``None`` while open."""
        return self._manifest

    def bind(
        self,
        run: RunIdentity,
        detector_id: str,
        *,
        sequence_offset: int | None = None,
        plane_start: int | None = None,
    ) -> None:
        """Name the run and detector the frames appended from now on belong to.

        A capture opened through ``frame_store_factory`` is created before its
        run is known, so this is how a caller that receives only a
        ``FrameWriter`` records identity. Calling it after a frame is written
        is refused: frames already in the array would keep no identity, and a
        manifest must not describe half a plane.

        Raises
        ------
        StoreRefused
            If a frame has already been appended, or the store is closed.
        """
        if not isinstance(run, RunIdentity):
            raise TypeError("a store run is a run identity")
        _require_text(detector_id, "detector id")
        if self._write_uncertain:
            raise RuntimeError("store write outcome uncertain")
        if self._closed:
            raise StoreRefused(
                StoreRefusal.STORE_CLOSED,
                {
                    "store": str(self._store_path),
                    "frames_written": self._frames_written,
                },
            )
        if self._frames_written:
            raise StoreRefused(
                StoreRefusal.IDENTITY_LOCKED,
                {
                    "store": str(self._store_path),
                    "frames_written": self._frames_written,
                    "reason": (
                        "frames were appended before the run was bound; a store "
                        "holds the identity of every frame or of none"
                    ),
                },
            )
        if sequence_offset is not None:
            _require_index(sequence_offset, "sequence offset")
            self._sequence_offset = sequence_offset
        if plane_start is not None:
            _require_index(plane_start, "plane start")
            self._plane_start = plane_start
        self._run = run
        self._detector_id = detector_id

    def set_setting_revision(self, revision: int) -> None:
        """Record the camera setting revision in force from the next frame on.

        A revision that is already in force adds no mark, so a caller may call
        this before every frame without growing the timeline. A revision that
        changes more than once before the next frame replaces the mark for that
        boundary rather than adding a second one: the timeline is a step
        function of frame index, and two steps at one index cannot be ordered.

        Raises
        ------
        StoreRefused
            If the store is closed.
        """
        _require_index(revision, "setting revision")
        if self._closed:
            raise StoreRefused(
                StoreRefusal.STORE_CLOSED,
                {
                    "store": str(self._store_path),
                    "frames_written": self._frames_written,
                },
            )
        if revision == self._setting_revision:
            return
        self._setting_revision = revision
        mark = SettingMark(first_frame=self._frames_written, revision=revision)
        if self._setting_timeline[-1].first_frame == self._frames_written:
            self._setting_timeline[-1] = mark
        else:
            self._setting_timeline.append(mark)

    def append(
        self,
        frame: NDArray[Any],
        *,
        stamp: FrameStamp | None = None,
        arrival_ns: int | None = None,
        mmcore_metadata: Mapping[str, str] | None = None,
    ) -> int:
        """Persist one YX frame and return its zero-based index.

        Parameters
        ----------
        frame
            The frame to append. Its shape and dtype must be the store's.
        stamp
            What the caller knows about this frame beyond its pixels. Left out,
            the writer's pattern, scan index and setting revision apply.
        arrival_ns
            When the service had this frame in hand, on the monotonic clock.
            Only a store opened with *record_timing* accepts one; without it
            the writer stamps the append time.
        mmcore_metadata
            The raw adapter tags taken with the frame, kept unmodified as
            evidence about transport.

        Returns
        -------
        int
            The frame's index in the array, which is its append order.

        Raises
        ------
        StoreRefused
            If the store is closed or full, the frame disagrees with the
            store's layout, or timing was offered to a store not recording it.
        """
        if self._write_uncertain:
            raise RuntimeError("store write outcome uncertain")
        if self._closed:
            raise StoreRefused(
                StoreRefusal.STORE_CLOSED,
                {
                    "store": str(self._store_path),
                    "frames_written": self._frames_written,
                },
            )
        pixels = np.asarray(frame)
        if tuple(pixels.shape) != self._shape:
            raise StoreRefused(
                StoreRefusal.FRAME_SHAPE_MISMATCH,
                {
                    "expected": list(self._shape),
                    "found": list(pixels.shape),
                    "frames_written": self._frames_written,
                },
            )
        if pixels.dtype != self._array.dtype:
            raise StoreRefused(
                StoreRefusal.FRAME_DTYPE_MISMATCH,
                {
                    "expected": self._array.dtype.name,
                    "found": pixels.dtype.name,
                    "frames_written": self._frames_written,
                },
            )
        if (
            self._planned_frames is not None
            and self._frames_written >= self._planned_frames
        ):
            raise StoreRefused(
                StoreRefusal.PLANNED_FRAMES_EXCEEDED,
                {
                    "planned_frames": self._planned_frames,
                    "frames_written": self._frames_written,
                    "reason": "a finite run accepts no more frames than it planned",
                },
            )
        offered_timing = arrival_ns is not None or mmcore_metadata is not None
        if offered_timing and not self._record_timing:
            raise StoreRefused(
                StoreRefusal.TIMING_NOT_RECORDED,
                {
                    "store": str(self._store_path),
                    "reason": (
                        "open the store with record_timing to keep arrival stamps"
                    ),
                },
            )
        if self._record_timing and len(self._timings) >= TIMING_RECORD_LIMIT:
            raise StoreRefused(
                StoreRefusal.TIMING_CAPACITY_EXCEEDED,
                {
                    "limit": TIMING_RECORD_LIMIT,
                    "frames_written": self._frames_written,
                },
            )
        if arrival_ns is not None:
            _require_index(arrival_ns, "arrival stamp")

        index = self._frames_written
        record = stamp if stamp is not None else FrameStamp()
        if not isinstance(record, FrameStamp):
            raise TypeError("a frame stamp must be a FrameStamp")
        revision = (
            self._setting_revision
            if record.setting_revision is None
            else record.setting_revision
        )
        contiguous = np.ascontiguousarray(pixels)
        pixels_bytes = contiguous.tobytes()
        entry = FrameEntry(
            index=index,
            checksum=hashlib.sha256(pixels_bytes).hexdigest(),
            setting_revision=revision,
            pattern_id=record.pattern_id,
            scan_index=record.scan_index,
            identity=self._identity_for(index, record),
        )
        timing = (
            TimingRecord(
                written_index=index,
                frame_id=index + 1,
                service_monotonic_ns=(
                    time.monotonic_ns() if arrival_ns is None else arrival_ns
                ),
                mmcore_metadata_source=(
                    "CMMCorePlus.popNextImageAndMD"
                    if mmcore_metadata is not None
                    else None
                ),
                mmcore_metadata=mmcore_metadata,
            )
            if self._record_timing
            else None
        )
        self._array.append(contiguous[None], axis=0)
        self._digest.update(pixels_bytes)
        self._entries.append(entry)
        if revision is not None and revision != self._setting_revision:
            self.set_setting_revision(revision)
        if timing is not None:
            self._timings.append(timing)
        self._frames_written += 1
        return index

    @property
    def supports_bulk_append(self) -> bool:
        """Whether buffered frames can be written without per-frame timing."""
        return not self._record_timing

    def append_batch(self, frames: list[NDArray[Any]]) -> int:
        """Append one buffered batch, returning its first index.

        A failed bulk write leaves the on-disk array outcome uncertain; the
        store must not be retried or closed with a potentially false manifest.
        Stores recording individual timing use the regular append path.
        """
        if self._write_uncertain:
            raise RuntimeError("store write outcome uncertain")
        if self._closed:
            raise StoreRefused(
                StoreRefusal.STORE_CLOSED, {"store": str(self._store_path)}
            )
        if not frames:
            return self._frames_written
        if self._record_timing:
            raise ValueError("bulk append requires an untimed store")
        if self._planned_frames is not None and (
            self._frames_written + len(frames) > self._planned_frames
        ):
            raise StoreRefused(
                StoreRefusal.PLANNED_FRAMES_EXCEEDED,
                {
                    "planned_frames": self._planned_frames,
                    "frames_written": self._frames_written,
                },
            )
        arrays = [np.asarray(frame) for frame in frames]
        for pixels in arrays:
            if tuple(pixels.shape) != self._shape:
                raise StoreRefused(
                    StoreRefusal.FRAME_SHAPE_MISMATCH,
                    {
                        "expected": list(self._shape),
                        "found": list(pixels.shape),
                        "frames_written": self._frames_written,
                    },
                )
            if pixels.dtype != self._array.dtype:
                raise StoreRefused(
                    StoreRefusal.FRAME_DTYPE_MISMATCH,
                    {
                        "expected": self._array.dtype.name,
                        "found": pixels.dtype.name,
                        "frames_written": self._frames_written,
                    },
                )
        first = self._frames_written
        pixels = np.ascontiguousarray(np.stack(arrays))
        stamp = FrameStamp()
        raw_frames = [pixels[i].tobytes() for i in range(len(arrays))]
        entries = [
            FrameEntry(
                index=first + i,
                checksum=hashlib.sha256(raw).hexdigest(),
                setting_revision=self._setting_revision,
                pattern_id=stamp.pattern_id,
                scan_index=stamp.scan_index,
                identity=self._identity_for(first + i, stamp),
            )
            for i, raw in enumerate(raw_frames)
        ]
        try:
            self._array.append(pixels, axis=0)
            for raw in raw_frames:
                self._digest.update(raw)
            self._entries.extend(entries)
            self._frames_written += len(entries)
        except BaseException:
            self._write_uncertain = True
            raise
        return first

    def _identity_for(self, index: int, stamp: FrameStamp) -> FrameIdentity | None:
        """Return the frame's identity, or ``None`` while the run is unknown."""
        if self._run is None or self._detector_id is None:
            return None
        return FrameIdentity(
            run=self._run,
            detector_id=self._detector_id,
            sequence=self._sequence_offset + index,
            plane_ordinal=self._plane_start,
            pattern_id=stamp.pattern_id,
            scan_index=stamp.scan_index,
            producer_sequence=stamp.producer_sequence,
        )

    def close(self) -> None:
        """Write the manifest and the timing sidecar, and stop accepting frames.

        Closing twice is harmless: the second call writes nothing and the
        manifest stays what the first call recorded. The manifest is written
        last and atomically, so a reader that finds one finds a store that
        closed.
        """
        if self._closed:
            return
        if self._write_uncertain:
            raise RuntimeError("store write outcome uncertain; manifest withheld")
        manifest = self._build_manifest()
        if self._record_timing:
            timing = self._build_timing()
            _write_text(
                timing_sidecar_path(self._uri, self._data_key), timing.as_json()
            )
        # a manifest is the store's own statement that it closed: it is
        # written after everything it describes, so a reader that finds one
        # finds frames that are already on disk
        _write_text(manifest_path(self._uri, self._data_key), manifest.as_json())
        self._manifest = manifest
        self._closed = True
        self._lock_path.unlink(missing_ok=True)

    def _build_manifest(self) -> StoreManifest:
        """Return the manifest describing the frames written so far."""
        return StoreManifest(
            schema=MANIFEST_SCHEMA,
            array_format=ARRAY_FORMAT,
            store_uri=store_uri(self._store_path),
            array_uri=self._array_uri,
            data_key=self._data_key,
            shape=self._shape,
            dtype=self._array.dtype.name,
            chunks=(1, *self._shape),
            frame_count=self._frames_written,
            planned_frames=self._planned_frames,
            frames=tuple(self._entries),
            array_checksum=self._digest.hexdigest(),
            run=self._run,
            detector_id=self._detector_id,
            plane_start=self._plane_start,
            setting_timeline=tuple(self._setting_timeline),
            created_utc=self._created_utc,
            closed_utc=_utc_now(),
            timing_sidecar_uri=(
                store_uri(timing_sidecar_path(self._uri, self._data_key))
                if self._record_timing
                else None
            ),
        )

    def _build_timing(self) -> TimingSidecar:
        """Return the timing sidecar for the frames written so far."""
        return TimingSidecar(
            schema=TIMING_SCHEMA,
            timestamp_semantics=TIMING_SEMANTICS,
            clock=TIMING_CLOCK,
            frame_metadata_source=TIMING_FRAME_METADATA_SOURCE,
            frame_metadata_semantics=TIMING_FRAME_METADATA_SEMANTICS,
            store_uri=str(self._store_path),
            data_key=self._data_key,
            written_count=len(self._timings),
            records=tuple(self._timings),
        )

    def __enter__(self) -> RunFrameStore:
        """Return this store, so a capture can close it by leaving the block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the store, whether the block succeeded or failed.

        A run that failed still closes: the manifest then records the frames
        that did arrive, which is what lets a later receipt name the partial
        run instead of leaving an unreadable directory behind.
        """
        self.close()


def open_run_store(
    uri: str,
    data_key: str,
    frame: NDArray[Any],
    *,
    run: RunIdentity,
    detector_id: str,
    sequence_offset: int = 0,
    plane_start: int = 0,
    pattern_id: int = 0,
    scan_index: int = 0,
    planned_frames: int | None = None,
    record_timing: bool = False,
    setting_revision: int | None = 0,
) -> RunFrameStore:
    """Open a create-only store that knows the run and detector it records.

    This is the entry point a plan or a live capture uses: it returns the
    concrete store, so the caller can bind, stamp and set the camera's setting
    revision, rather than the narrow ``FrameWriter`` the camera service needs.
    """
    return RunFrameStore(
        uri,
        data_key,
        frame,
        run=run,
        detector_id=detector_id,
        sequence_offset=sequence_offset,
        plane_start=plane_start,
        pattern_id=pattern_id,
        scan_index=scan_index,
        planned_frames=planned_frames,
        record_timing=record_timing,
        setting_revision=setting_revision,
    )


def frame_store_factory(uri: str, data_key: str, frame: NDArray[Any]) -> FrameWriter:
    """Open the store a camera capture writes into, as the service calls it.

    The signature is the camera service's ``FrameStoreFactory``, so this
    function can replace the service's default binding without the service
    learning anything about runs, receipts or manifests.
    """
    return RunFrameStore(uri, data_key, frame)


def _utc_now() -> str:
    """Return the current instant as ISO 8601 text in UTC."""
    return datetime.now(UTC).isoformat()


def _write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _require_text(value: object, name: str) -> None:
    """Refuse a value that is not non-empty text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _require_index(value: object, name: str) -> None:
    """Refuse a value that is not a non-negative integer."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_digest(value: object, name: str) -> None:
    """Refuse a value that is not a lowercase hexadecimal SHA-256."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")


def _require_frame_shape(shape: tuple[int, ...]) -> None:
    """Refuse a frame shape that is not two positive dimensions."""
    if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("a frame store holds two-dimensional frames of positive size")


def _require_data_key(data_key: object) -> None:
    """Refuse a data key that is not a single path component."""
    if not isinstance(data_key, str) or not data_key.strip():
        raise ValueError("a store data key must be non-empty text")
    if data_key in {".", ".."} or any(
        separator in data_key for separator in ("/", "\\")
    ):
        raise ValueError("a store data key must be a single path component")


__all__ = [
    "ARRAY_FORMAT",
    "CHECKSUM_ALGORITHM",
    "FRAME_CHECKSUM_SCOPE",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_V1",
    "MANIFEST_SUFFIX",
    "SUPPORTED_MANIFEST_SCHEMAS",
    "TIMING_CLOCK",
    "TIMING_FRAME_METADATA_SEMANTICS",
    "TIMING_FRAME_METADATA_SOURCE",
    "TIMING_RECORD_LIMIT",
    "TIMING_SCHEMA",
    "TIMING_SEMANTICS",
    "FrameEntry",
    "FrameStamp",
    "RunFrameStore",
    "SettingMark",
    "StoreManifest",
    "StoreRefusal",
    "StoreRefused",
    "TimingRecord",
    "TimingSidecar",
    "array_path",
    "frame_checksum",
    "frame_store_factory",
    "frames_checksum",
    "local_store_path",
    "manifest_path",
    "open_run_store",
    "store_uri",
    "timing_sidecar_path",
]
