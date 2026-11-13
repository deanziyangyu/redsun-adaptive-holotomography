"""Receipts for closed frame stores: assets, refusals and the timing document.

The store format and its timing sidecar belong to
:mod:`redsun_aht.storage.writer`, which adapts them from
``redsun-searching-is-imaging`` at donor revision ``3a3bf06``; this module only
reads what that writer and the camera service leave on disk. Nothing here is
copied from the donor: it is the bridge from that format to the contracts'
asset vocabulary.

A store on disk is a directory; the acquisition application talks about assets.
This module is the translation. It opens a closed store without the writer that
produced it, rebuilds the asset reference from what is actually on disk, checks
the manifest against the array, and answers with one of two things:

* a :class:`~redsun_aht.contracts.assets.PlaneReceipt` for a plane whose every
  declared frame is present and whose bytes match the checksums they were
  recorded with, or
* a :class:`RefusedStore`, which names the condition and the frames behind it
  instead of raising.

Nothing here writes, and nothing here re-derives a receipt's identity: the
receipt is built from the manifest's content digest, so a copy of a store is
the same plane at another location. The refusal vocabulary is the contracts'
wherever the contracts can name the condition -- including the two storage
lifecycle conditions it now names, a plane short of its frames and an occupied
store; only an uncommitted store and an ambiguous data key remain
:class:`~redsun_aht.storage.writer.StoreRefusal` members.

The mapping from condition to refusal is:

===================  ============================  =============================
condition            reason                        meaning
===================  ============================  =============================
``no_store``         ``AssetRefusal.MISSING_URI``  nothing is there
``no_manifest``      ``AssetRefusal.MISSING_URI``  a directory no run closed
``uncommitted_store`` ``StoreRefusal.STORE_UNCOMMITTED``  frames, but no manifest
``several_manifests`` ``StoreRefusal.MANIFEST_AMBIGUOUS``  one key must be named
``store_not_empty``  ``AssetRefusal.STORE_NOT_EMPTY``  the location holds a run
``array_absent``     ``AssetRefusal.MISSING_URI``  the manifest's array is gone
``mixed_schema``     ``AssetRefusal.CONTRACT_VERSION_MISMATCH``  another schema
``array_dtype``      ``AssetRefusal.DTYPE_MISMATCH``  the array was retyped
``array_shape``      ``AssetRefusal.SHAPE_MISMATCH``  the array was reshaped
``missing_frames``   ``AssetRefusal.MISSING_FRAMES``  declared frames are absent
``extra_frames``     ``AssetRefusal.SHAPE_MISMATCH``  the array is longer
``zero_frames``      ``AssetRefusal.ZERO_FRAME_COUNT``  nothing was recorded
``partial_run``      ``AssetRefusal.MISSING_FRAMES``  fewer frames than requested
``overfull_run``     ``AssetRefusal.SHAPE_MISMATCH``  more frames than requested
``frame_checksum``   ``AssetRefusal.CHECKSUM_MISMATCH``  a frame's bytes changed
``array_checksum``   ``AssetRefusal.CHECKSUM_MISMATCH``  the plane's bytes changed
``unbound_store``    ``AssetRefusal.WRONG_RUN``    no run or detector to name
``wrong_run``        ``AssetRefusal.WRONG_RUN``    the caller named another run
``wrong_detector``   ``AssetRefusal.WRONG_DETECTOR``  another detector
``wrong_plane``      ``AssetRefusal.WRONG_PLANE``  another plane ordinal
``unreadable_json``  ``StoreRefusal.MANIFEST_UNREADABLE``  not a JSON document
``malformed_manifest`` ``StoreRefusal.MANIFEST_UNREADABLE``  missing fields
``write_refused``    ``StoreRefusal.*``           another writer refusal
===================  ============================  =============================

Only a genuine shape disagreement keeps ``SHAPE_MISMATCH``: the array's frame
shape, an array longer than the manifest declares, and a run that recorded more
frames than it requested. A plane short of its frames is
``MISSING_FRAMES``, so a caller never has to read a shape refusal as a missing
one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import zarr

from ..contracts.assets import AssetRef, AssetRefusal, PlaneReceipt
from .writer import (
    ARRAY_FORMAT,
    MANIFEST_SUFFIX,
    SUPPORTED_MANIFEST_SCHEMAS,
    TIMING_SCHEMA,
    StoreManifest,
    StoreRefusal,
    StoreRefused,
    TimingSidecar,
    frame_checksum,
    local_store_path,
    manifest_path,
    store_uri,
    timing_sidecar_path,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ..contracts.frames import RunIdentity

STORE_MIMETYPE: Final = "application/x-zarr"
"""Media type of a Zarr v3 array, the object a plane's receipt names."""

TIMING_MIMETYPE: Final = "application/json"
"""Media type of the frame-timing sidecar."""


@dataclass(frozen=True, slots=True)
class RefusedStore:
    """A store a caller cannot use, named rather than thrown.

    Parameters
    ----------
    reason
        The contract's reason where the contract can name the condition, or a
        store-lifecycle reason where it cannot.
    condition
        The precise condition behind *reason*. Two refusals with the same
        reason are told apart here, so a caller never parses ``detail``.
    detail
        One line for a log, for a human rather than for a decision.
    missing_ordinals
        The frame ordinals that are absent, for a partial, short or lost run.
        Empty when the refusal is about the store rather than about frames.
    evidence
        The values the refusal was decided on.
    """

    reason: AssetRefusal | StoreRefusal
    condition: str
    detail: str = ""
    missing_ordinals: tuple[int, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a refusal that cannot be asserted on or logged."""
        if not isinstance(self.reason, (AssetRefusal, StoreRefusal)):
            raise TypeError("a store refusal needs a contract or store reason")
        if not isinstance(self.condition, str) or not self.condition.strip():
            raise ValueError("a store refusal needs a condition")
        if not isinstance(self.detail, str):
            raise ValueError("a store refusal detail is text")
        if any(
            type(ordinal) is not int or ordinal < 0
            for ordinal in (self.missing_ordinals)
        ):
            raise ValueError("missing ordinals must be non-negative integers")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("a store refusal keeps its evidence as a mapping")

    @property
    def is_partial_run(self) -> bool:
        """Whether the refusal is a run that stopped short of its frames."""
        return self.reason is AssetRefusal.MISSING_FRAMES

    @property
    def message(self) -> str:
        """Return the reason, condition and detail as one line."""
        prefix = f"{self.reason.value}/{self.condition}"
        return prefix if not self.detail else f"{prefix}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ClosedStore:
    """A closed store opened for reading, without the writer that produced it.

    Parameters
    ----------
    manifest
        The manifest the store closed with.
    array
        The array itself, opened read-only.
    array_uri
        Where the array was opened. A copy of a store names its own location
        here, so a receipt built from it points at bytes that exist, while the
        manifest keeps the location the frames were originally written to.
    """

    manifest: StoreManifest
    array: zarr.Array[Any]
    array_uri: str

    @property
    def data_key(self) -> str:
        """Return the key the array sits under inside the store."""
        return self.manifest.data_key

    @property
    def frame_count(self) -> int:
        """Return how many frames the store holds."""
        return self.manifest.frame_count

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Return one frame's shape as ``(height, width)``."""
        return self.manifest.shape

    @property
    def dtype(self) -> str:
        """Return the array's numpy dtype name."""
        return self.manifest.dtype

    @property
    def checksums(self) -> tuple[str, ...]:
        """Return the checksums the manifest recorded, in append order."""
        return tuple(entry.checksum for entry in self.manifest.frames)

    def frame(self, index: int) -> NDArray[Any]:
        """Return one frame, loaded from disk."""
        return np.asarray(self.array[index])

    def frames(self) -> NDArray[Any]:
        """Return every frame as one ``(frames, height, width)`` array."""
        return np.asarray(self.array[:])


def open_closed_store(
    uri: str,
    *,
    data_key: str | None = None,
    verify_checksums: bool = False,
) -> ClosedStore | RefusedStore:
    """Open a closed store for reading, and check it describes itself.

    Parameters
    ----------
    uri
        The store directory, as a path or a ``file://`` URI.
    data_key
        The plane to open. Left out, the store must hold exactly one manifest.
    verify_checksums
        Whether to re-hash every frame against the manifest. Off by default:
        opening a plane to look at it should not read every byte a second time.

    Returns
    -------
    ClosedStore or RefusedStore
        The readable store, or the reason it is not one.
    """
    located = _locate_manifest(uri, data_key)
    if isinstance(located, RefusedStore):
        return located
    manifest = _read_manifest(located)
    if isinstance(manifest, RefusedStore):
        return manifest
    expected_key = located.name.removesuffix(MANIFEST_SUFFIX)
    if manifest.data_key != expected_key:
        return RefusedStore(
            StoreRefusal.MANIFEST_UNREADABLE,
            "manifest_key_mismatch",
            detail="the manifest names a different plane than its file",
            evidence={
                "manifest": str(located),
                "expected": expected_key,
                "found": manifest.data_key,
            },
        )
    opened = _open_array(uri, manifest.data_key, manifest)
    if isinstance(opened, RefusedStore):
        return opened
    store = ClosedStore(
        manifest=manifest,
        array=opened,
        array_uri=store_uri(local_store_path(uri) / manifest.data_key),
    )
    if verify_checksums:
        problem = _verify_checksums(store)
        if problem is not None:
            return problem
    return store


def read_manifest(
    uri: str, *, data_key: str | None = None
) -> StoreManifest | RefusedStore:
    """Read a closed store's manifest without opening its array.

    Returns
    -------
    StoreManifest or RefusedStore
        The manifest, or the reason none can be read. A manifest whose schema
        is not this package's is refused rather than reinterpreted, which is
        what stops a pre-reset dataset being read as a new run.
    """
    located = _locate_manifest(uri, data_key)
    if isinstance(located, RefusedStore):
        return located
    manifest = _read_manifest(located)
    if isinstance(manifest, RefusedStore):
        return manifest
    expected_key = located.name.removesuffix(MANIFEST_SUFFIX)
    if manifest.data_key != expected_key:
        return RefusedStore(
            StoreRefusal.MANIFEST_UNREADABLE,
            "manifest_key_mismatch",
            detail="the manifest names a different plane than its file",
            evidence={
                "manifest": str(located),
                "expected": expected_key,
                "found": manifest.data_key,
            },
        )
    return manifest


def _locate_timing(uri: str, data_key: str | None) -> Path | RefusedStore:
    """Find one plane's sidecar; require a key when several are present."""
    store = local_store_path(uri)
    if data_key is None:
        candidates = sorted(store.glob("*.timing.json")) if store.is_dir() else []
        if len(candidates) > 1:
            return RefusedStore(
                StoreRefusal.MANIFEST_AMBIGUOUS,
                "several_timing_sidecars",
                detail="the store holds more than one timed plane; name its data key",
                evidence={
                    "store": str(store),
                    "sidecars": [path.name for path in candidates],
                },
            )
        path = candidates[0] if candidates else store / "unknown.timing.json"
    else:
        path = timing_sidecar_path(uri, data_key)
    if not path.is_file():
        return RefusedStore(
            AssetRefusal.MISSING_URI,
            "no_timing_sidecar",
            detail="the capture recorded no frame timing",
            evidence={"sidecar": str(path)},
        )
    return path


def read_timing(
    uri: str, *, data_key: str | None = None
) -> TimingSidecar | RefusedStore:
    """Read the frame-timing sidecar a timed capture left beside its store."""
    path = _locate_timing(uri, data_key)
    if isinstance(path, RefusedStore):
        return path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return RefusedStore(
            StoreRefusal.MANIFEST_UNREADABLE,
            "unreadable_json",
            detail=f"{type(error).__name__}: {error}",
            evidence={"sidecar": str(path)},
        )
    if not isinstance(payload, dict):
        return RefusedStore(
            StoreRefusal.MANIFEST_UNREADABLE,
            "unreadable_json",
            detail="a sidecar is a JSON object",
            evidence={"sidecar": str(path)},
        )
    recorded = payload.get("schema")
    if recorded != TIMING_SCHEMA:
        return RefusedStore(
            AssetRefusal.CONTRACT_VERSION_MISMATCH,
            "mixed_schema",
            detail="the sidecar was written under another schema",
            evidence={
                "expected": TIMING_SCHEMA,
                "found": recorded,
                "sidecar": str(path),
            },
        )
    try:
        sidecar = TimingSidecar.from_payload(payload)
        expected_key = path.name.removesuffix(".timing.json")
        if sidecar.data_key != expected_key:
            raise ValueError("timing sidecar names a different plane than its file")
        return sidecar
    except (KeyError, TypeError, ValueError) as error:
        return RefusedStore(
            StoreRefusal.MANIFEST_UNREADABLE,
            "malformed_manifest",
            detail=f"{type(error).__name__}: {error}",
            evidence={"sidecar": str(path)},
        )


def build_asset(
    uri: str,
    *,
    data_key: str | None = None,
    verify_checksums: bool = True,
) -> AssetRef | RefusedStore:
    """Return the plane's asset reference, identified by its content.

    The reference names the array where it is now, not where the manifest says
    it was written, so a relocated copy is referenced where it lives. Its
    checksum is the manifest's, which is only issued after the frames verify.
    """
    opened = open_closed_store(
        uri, data_key=data_key, verify_checksums=verify_checksums
    )
    if isinstance(opened, RefusedStore):
        return opened
    return _asset_for(opened)


def timing_asset(uri: str, *, data_key: str | None = None) -> AssetRef | RefusedStore:
    """Return the asset reference of a timed capture's sidecar.

    The sidecar is a JSON document rather than an array, so its reference
    names its bytes as one ``uint8`` row and is admitted by checksum alone.
    """
    path = _locate_timing(uri, data_key)
    if isinstance(path, RefusedStore):
        return path
    try:
        payload = path.read_bytes()
    except OSError as error:
        return RefusedStore(
            AssetRefusal.MISSING_URI,
            "unreadable_sidecar",
            detail=f"{type(error).__name__}: {error}",
            evidence={"sidecar": str(path)},
        )
    return AssetRef(
        uri=store_uri(path),
        checksum=hashlib.sha256(payload).hexdigest(),
        shape=(1,),
        dtype="uint8",
        mimetype=TIMING_MIMETYPE,
    )


def assemble_receipt(
    uri: str,
    *,
    data_key: str | None = None,
    run: RunIdentity | None = None,
    detector_id: str | None = None,
    plane_ordinal: int | None = None,
    expected_frames: int | None = None,
    verify_checksums: bool = True,
) -> PlaneReceipt | RefusedStore:
    """Assemble the receipt for one closed store, from the frames it holds.

    Parameters
    ----------
    uri
        The store directory, as a path or a ``file://`` URI.
    data_key
        The plane to receipt. Left out, the store must hold exactly one.
    run, detector_id
        The run and detector the caller believes the plane belongs to. They
        supply what a store opened through the camera service cannot know; a
        store that already names them is checked against them rather than
        silently overridden.
    plane_ordinal
        The plane ordinal to receipt. A bound store's own ordinal wins, and a
        disagreement is refused.
    expected_frames
        How many frames the caller's run asked for. It overrides the store's
        own planned count, so a plan can report a short run against what it
        requested even when the store was opened without a bound.
    verify_checksums
        Whether to re-hash every frame and the plane. On by default: a receipt
        is an attestation, so it costs one pass over the stored bytes.

    Returns
    -------
    PlaneReceipt or RefusedStore
        The receipt, or the named reason no receipt can be issued.
    """
    opened = open_closed_store(uri, data_key=data_key, verify_checksums=False)
    if isinstance(opened, RefusedStore):
        return opened
    manifest = opened.manifest
    if manifest.frame_count == 0:
        return RefusedStore(
            AssetRefusal.ZERO_FRAME_COUNT,
            "zero_frames",
            detail="the store closed before it recorded a frame",
            evidence={"store": manifest.store_uri, "data_key": manifest.data_key},
        )

    expected = (
        expected_frames if expected_frames is not None else manifest.planned_frames
    )
    if expected is not None and manifest.frame_count < expected:
        return RefusedStore(
            AssetRefusal.MISSING_FRAMES,
            "partial_run",
            detail=(
                f"{manifest.frame_count} of {expected} requested frames were recorded"
            ),
            missing_ordinals=tuple(range(manifest.frame_count, expected)),
            evidence={
                "requested_frames": expected,
                "stored_frames": manifest.frame_count,
                "store": manifest.store_uri,
                "uri": manifest.array_uri,
                "data_key": manifest.data_key,
            },
        )
    if expected is not None and manifest.frame_count > expected:
        return RefusedStore(
            AssetRefusal.SHAPE_MISMATCH,
            "overfull_run",
            detail=f"{manifest.frame_count} frames were recorded for {expected}",
            evidence={
                "requested_frames": expected,
                "stored_frames": manifest.frame_count,
                "store": manifest.store_uri,
            },
        )
    if verify_checksums:
        problem = _verify_checksums(opened)
        if problem is not None:
            return problem

    identity = _resolve_identity(
        manifest,
        run=run,
        detector_id=detector_id,
        plane_ordinal=plane_ordinal,
    )
    if isinstance(identity, RefusedStore):
        return identity
    run_identity, detector, plane = identity
    asset = _asset_for(opened)
    if isinstance(asset, RefusedStore):
        return asset
    return PlaneReceipt(
        run=run_identity,
        detector=detector,
        plane_ordinal=plane,
        asset=asset,
        frame_count=manifest.frame_count,
    )


def refusal_from_exception(
    error: StoreRefused, *, condition: str | None = None
) -> RefusedStore:
    """Wrap a writer's exception as the same refusal a reader would return.

    A caller that opens a store for writing may prefer the value form, so the
    create-only refusal is available both ways without a second vocabulary.

    The two lifecycle conditions the contracts now name are reported in their
    own members: an occupied location becomes ``AssetRefusal.STORE_NOT_EMPTY``
    with the store's URI, its plane count and the frames it already holds, and
    every other writer refusal keeps its
    :class:`~redsun_aht.storage.writer.StoreRefusal`.

    Parameters
    ----------
    error
        The refusal the writer raised.
    condition
        The condition to report. Left out, an occupied location reports
        ``store_not_empty`` and anything else reports ``write_refused``.
    """
    if not isinstance(error, StoreRefused):
        raise TypeError("only a store refusal becomes a refused store")
    evidence = dict(error.evidence)
    reason: AssetRefusal | StoreRefusal = error.reason
    if error.reason is StoreRefusal.STORE_NOT_EMPTY:
        reason = AssetRefusal.STORE_NOT_EMPTY
        evidence.update(_occupied_evidence(evidence))
    if condition is None:
        condition = (
            "store_not_empty"
            if reason is AssetRefusal.STORE_NOT_EMPTY
            else "write_refused"
        )
    return RefusedStore(
        reason=reason,
        condition=condition,
        detail=str(error),
        evidence=evidence,
    )


def _occupied_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Name the occupied store and how much it already holds."""
    store = _store_directory(evidence)
    held = evidence.get("frames_held")
    if type(held) is not int or held < 0:
        held = 0 if store is None else _frames_on_disk(store)
    report: dict[str, Any] = {"frames_held": held}
    if store is None:
        report["plane_count"] = 1
    else:
        report["uri"] = store_uri(store)
        report["plane_count"] = max(1, _plane_count(store))
    return report


def _store_directory(evidence: Mapping[str, Any]) -> Path | None:
    """Return the store directory a writer's evidence points at."""
    for key in ("store", "array", "manifest"):
        value = evidence.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = local_store_path(value)
        return path.parent if key == "array" else path
    return None


def _plane_count(store: Path) -> int:
    """Count the plane arrays a store directory already holds."""
    if not store.is_dir():
        return 0
    return sum(
        1
        for child in store.iterdir()
        if child.is_dir() and (child / "zarr.json").is_file()
    )


def _asset_for(store: ClosedStore) -> AssetRef | RefusedStore:
    """Build the content-addressed reference for an opened plane."""
    manifest = store.manifest
    if manifest.frame_count == 0:
        return RefusedStore(
            AssetRefusal.ZERO_FRAME_COUNT,
            "zero_frames",
            detail="the store closed before it recorded a frame",
            evidence={"store": manifest.store_uri, "data_key": manifest.data_key},
        )
    return AssetRef(
        uri=store.array_uri,
        checksum=manifest.array_checksum,
        shape=(manifest.frame_count, *manifest.shape),
        dtype=manifest.dtype,
        mimetype=STORE_MIMETYPE,
    )


def _resolve_identity(
    manifest: StoreManifest,
    *,
    run: RunIdentity | None,
    detector_id: str | None,
    plane_ordinal: int | None,
) -> tuple[RunIdentity, str, int] | RefusedStore:
    """Return the run, detector and plane a receipt may name."""
    bound = manifest.run is not None
    if run is not None and manifest.run is not None and run != manifest.run:
        return RefusedStore(
            AssetRefusal.WRONG_RUN,
            "wrong_run",
            detail="the store records another run",
            evidence={
                "expected": manifest.run.run_id,
                "found": run.run_id,
                "store": manifest.store_uri,
            },
        )
    if (
        detector_id is not None
        and manifest.detector_id is not None
        and detector_id != manifest.detector_id
    ):
        return RefusedStore(
            AssetRefusal.WRONG_DETECTOR,
            "wrong_detector",
            detail="the store records another detector",
            evidence={
                "expected": manifest.detector_id,
                "found": detector_id,
                "store": manifest.store_uri,
            },
        )
    if plane_ordinal is not None and bound and plane_ordinal != manifest.plane_start:
        return RefusedStore(
            AssetRefusal.WRONG_PLANE,
            "wrong_plane",
            detail="the store records another plane ordinal",
            evidence={
                "expected": manifest.plane_start,
                "found": plane_ordinal,
                "store": manifest.store_uri,
            },
        )
    resolved_run = manifest.run if run is None else run
    resolved_detector = manifest.detector_id if detector_id is None else detector_id
    if resolved_run is None or resolved_detector is None:
        return RefusedStore(
            AssetRefusal.WRONG_RUN,
            "unbound_store",
            detail="the store names no run and no detector; pass both",
            evidence={"store": manifest.store_uri, "data_key": manifest.data_key},
        )
    resolved_plane = manifest.plane_start if plane_ordinal is None else plane_ordinal
    return resolved_run, resolved_detector, resolved_plane


def _locate_manifest(uri: str, data_key: str | None) -> Path | RefusedStore:
    """Return the manifest file of the plane *data_key* names."""
    store = local_store_path(uri)
    if not store.is_dir():
        return RefusedStore(
            AssetRefusal.MISSING_URI,
            "no_store",
            detail="no store directory is there",
            evidence={"store": str(store)},
        )
    if data_key is not None:
        candidate = manifest_path(uri, data_key)
        candidates = [candidate] if candidate.is_file() else []
    else:
        candidates = sorted(store.glob(f"*{MANIFEST_SUFFIX}"))
    if not candidates:
        held = _frames_on_disk(store)
        if held:
            return RefusedStore(
                StoreRefusal.STORE_UNCOMMITTED,
                "uncommitted_store",
                detail=(
                    "the store holds frames but no manifest; a capture is open "
                    "or was interrupted"
                ),
                evidence={
                    "store": str(store),
                    "uri": store_uri(store),
                    "data_key": data_key,
                    "frames_on_disk": held,
                    "plane_count": _plane_count(store),
                },
            )
        return RefusedStore(
            AssetRefusal.MISSING_URI,
            "no_manifest",
            detail="no closed run is recorded in the store",
            evidence={"store": str(store), "data_key": data_key},
        )
    if len(candidates) > 1:
        return RefusedStore(
            StoreRefusal.MANIFEST_AMBIGUOUS,
            "several_manifests",
            detail="the store holds more than one plane; name its data key",
            evidence={
                "store": str(store),
                "manifests": [path.name for path in candidates],
            },
        )
    return candidates[0]


def _read_manifest(path: Path) -> StoreManifest | RefusedStore:
    """Read one manifest file, refusing a schema this package cannot read."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        return RefusedStore(
            StoreRefusal.MANIFEST_UNREADABLE,
            "unreadable_json",
            detail=f"{type(error).__name__}: {error}",
            evidence={"manifest": str(path)},
        )
    if not isinstance(payload, dict):
        return RefusedStore(
            StoreRefusal.MANIFEST_UNREADABLE,
            "unreadable_json",
            detail="a manifest is a JSON object",
            evidence={"manifest": str(path)},
        )
    recorded = payload.get("schema")
    if recorded not in SUPPORTED_MANIFEST_SCHEMAS:
        return RefusedStore(
            AssetRefusal.CONTRACT_VERSION_MISMATCH,
            "mixed_schema",
            detail="the store was written under another manifest schema",
            evidence={
                "expected": list(SUPPORTED_MANIFEST_SCHEMAS),
                "found": recorded,
                "manifest": str(path),
            },
        )
    array_format = payload.get("array_format")
    if array_format != ARRAY_FORMAT:
        return RefusedStore(
            AssetRefusal.CONTRACT_VERSION_MISMATCH,
            "array_format",
            detail="the store's array is not the format this package reads",
            evidence={
                "expected": ARRAY_FORMAT,
                "found": array_format,
                "manifest": str(path),
            },
        )
    try:
        return StoreManifest.from_payload(payload)
    except (KeyError, TypeError, ValueError) as error:
        return RefusedStore(
            StoreRefusal.MANIFEST_UNREADABLE,
            "malformed_manifest",
            detail=f"{type(error).__name__}: {error}",
            evidence={"manifest": str(path)},
        )


def _open_array(
    uri: str, data_key: str, manifest: StoreManifest
) -> zarr.Array[Any] | RefusedStore:
    """Open the manifest's array read-only, and check its layout."""
    location = local_store_path(uri) / data_key
    try:
        array = zarr.open_array(store=str(location), mode="r")
    except (zarr.errors.BaseZarrError, OSError, ValueError) as error:
        return RefusedStore(
            AssetRefusal.MISSING_URI,
            "array_absent",
            detail=f"the manifest's array cannot be opened: {error}",
            evidence={"array": str(location), "manifest": manifest.store_uri},
        )
    if array.dtype.name != manifest.dtype:
        return RefusedStore(
            AssetRefusal.DTYPE_MISMATCH,
            "array_dtype",
            detail="the array's dtype is not the manifest's",
            evidence={
                "expected": manifest.dtype,
                "found": array.dtype.name,
                "array": str(location),
            },
        )
    if tuple(array.shape[1:]) != manifest.shape:
        return RefusedStore(
            AssetRefusal.SHAPE_MISMATCH,
            "array_shape",
            detail="the array's frame shape is not the manifest's",
            evidence={
                "expected": list(manifest.shape),
                "found": list(array.shape[1:]),
                "array": str(location),
            },
        )
    if tuple(array.chunks) != manifest.chunks:
        return RefusedStore(
            StoreRefusal.STORE_LAYOUT_MISMATCH,
            "array_chunks",
            detail="the array's chunk layout is not the manifest's",
            evidence={
                "expected": list(manifest.chunks),
                "found": list(array.chunks),
                "array": str(location),
            },
        )
    held = int(array.shape[0])
    if held < manifest.frame_count:
        return RefusedStore(
            AssetRefusal.MISSING_FRAMES,
            "missing_frames",
            detail=(
                f"the manifest declares {manifest.frame_count} frames but the "
                f"array holds {held}"
            ),
            missing_ordinals=tuple(range(held, manifest.frame_count)),
            evidence={
                "declared_frames": manifest.frame_count,
                "stored_frames": held,
                "array": str(location),
            },
        )
    if held > manifest.frame_count:
        return RefusedStore(
            AssetRefusal.SHAPE_MISMATCH,
            "extra_frames",
            detail=(
                f"the array holds {held} frames but the manifest declares "
                f"{manifest.frame_count}"
            ),
            evidence={
                "declared_frames": manifest.frame_count,
                "stored_frames": held,
                "array": str(location),
            },
        )
    return array


def _verify_checksums(store: ClosedStore) -> RefusedStore | None:
    """Re-hash every frame, and the plane, against the manifest."""
    manifest = store.manifest
    digest = hashlib.sha256()
    changed: list[int] = []
    for index, entry in enumerate(manifest.frames):
        pixels = np.ascontiguousarray(np.asarray(store.array[index]))
        payload = pixels.tobytes()
        digest.update(payload)
        if hashlib.sha256(payload).hexdigest() != entry.checksum:
            changed.append(index)
    if changed:
        return RefusedStore(
            AssetRefusal.CHECKSUM_MISMATCH,
            "frame_checksum",
            detail=f"{len(changed)} frame(s) no longer match the manifest",
            missing_ordinals=tuple(changed),
            evidence={
                "changed_ordinals": changed[:10],
                "changed_count": len(changed),
                "store": manifest.store_uri,
            },
        )
    recorded = manifest.array_checksum
    computed = digest.hexdigest()
    if computed != recorded:
        return RefusedStore(
            AssetRefusal.CHECKSUM_MISMATCH,
            "array_checksum",
            detail="the plane's digest is not the manifest's",
            evidence={
                "expected": recorded,
                "found": computed,
                "store": manifest.store_uri,
            },
        )
    return None


def _frames_on_disk(store: Path) -> int:
    """Count the frames in any arrays a store directory holds, ignoring faults."""
    held = 0
    for candidate in sorted(store.iterdir()):
        if not candidate.is_dir() or not (candidate / "zarr.json").is_file():
            continue
        try:
            held += int(zarr.open_array(store=str(candidate), mode="r").shape[0])
        except (zarr.errors.BaseZarrError, OSError, ValueError):
            continue
    return held


__all__ = [
    "STORE_MIMETYPE",
    "TIMING_MIMETYPE",
    "ClosedStore",
    "RefusedStore",
    "assemble_receipt",
    "build_asset",
    "frame_checksum",
    "open_closed_store",
    "read_manifest",
    "read_timing",
    "refusal_from_exception",
    "timing_asset",
]
