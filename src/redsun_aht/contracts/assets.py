"""Asset identity and plane receipts: what a store accepted, and what it did not.

The control plane must not carry pixels. A producer offers a frame as an asset
reference -- a content digest, a shape, a dtype, a media type and a location --
and the storage owner answers with a receipt for the plane it committed. This
module freezes both halves as data so the camera, live, plan and storage
packages can exchange them without sharing a class.

Two properties are deliberate:

* an asset is admitted by *content*. A receipt admits a byte-identical copy at
  another URI, and refuses a document whose checksum, shape or dtype disagrees
  with what was committed.
* a receipt's fingerprint covers identity and nothing else. Re-serialising it
  with the keys in another order, or annotating it for a human, changes the
  bytes but not the fingerprint, so a stored index can be compared across
  rewrites.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from .frames import FrameIdentity, RunIdentity

if TYPE_CHECKING:
    from collections.abc import Mapping

ASSET_CONTRACT_VERSION = "1.0.0"
"""Version of the asset reference and plane receipt documents."""

_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


class AssetRefusal(StrEnum):
    """Every way an offered asset, receipt or store can be refused.

    The members are the closed vocabulary a storage owner may refuse with. A
    caller that needs to react differently to two refusals must be able to
    tell them apart without parsing a message, so no refusal is a bare string.

    Most members describe an asset or a receipt that cannot be accepted, and are
    raised by this module's own validation. The last two name storage lifecycle
    conditions the owner discovers while opening or closing a run — a plane that
    holds fewer frames than were requested, and a store that already holds a
    run — and are raised by the owner, so neither is ever reported as a shape or
    a checksum problem.
    """

    MISSING_URI = "missing_uri"
    MISSING_CHECKSUM = "missing_checksum"
    MALFORMED_CHECKSUM = "malformed_checksum"
    EMPTY_SHAPE = "empty_shape"
    NON_POSITIVE_DIMENSION = "non_positive_dimension"
    UNSUPPORTED_DTYPE = "unsupported_dtype"
    MISSING_MIMETYPE = "missing_mimetype"
    WRONG_RUN = "wrong_run"
    WRONG_DETECTOR = "wrong_detector"
    WRONG_PLANE = "wrong_plane"
    ZERO_FRAME_COUNT = "zero_frame_count"
    SHAPE_MISMATCH = "shape_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    CONTRACT_VERSION_MISMATCH = "contract_version_mismatch"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    MISSING_FRAMES = "missing_frames"
    STORE_NOT_EMPTY = "store_not_empty"


class AssetContractViolation(ValueError):
    """An offered asset or receipt that cannot be accepted, with its reason."""

    def __init__(self, reason: AssetRefusal, evidence: Mapping[str, Any]) -> None:
        """Keep the machine-readable reason beside the values behind it."""
        self.reason = reason
        self.evidence = dict(evidence)
        super().__init__(f"{reason.value}: {json.dumps(self.evidence, sort_keys=True)}")


@dataclass(frozen=True, slots=True)
class AssetRef:
    """One stored asset, identified by its content rather than its location.

    Parameters
    ----------
    uri
        Scheme-qualified location, such as ``"file:///runs/run-1/dhm.zarr"``.
        Two references that differ only here are the same asset.
    checksum
        Lowercase hexadecimal SHA-256 of the stored bytes.
    shape
        The array's numpy shape, which may be two or three dimensional.
    dtype
        The array's numpy dtype string, such as ``"uint16"``.
    mimetype
        The media type of the stored object, such as ``"image/tiff"``.
    """

    uri: str
    checksum: str
    shape: tuple[int, ...]
    dtype: str
    mimetype: str

    def __post_init__(self) -> None:
        """Refuse a reference that cannot be found or verified again."""
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise AssetContractViolation(AssetRefusal.MISSING_URI, {"uri": self.uri})
        if "://" not in self.uri:
            raise AssetContractViolation(
                AssetRefusal.MISSING_URI, {"uri": self.uri, "reason": "no scheme"}
            )
        if any(character.isspace() for character in self.uri):
            raise AssetContractViolation(
                AssetRefusal.MISSING_URI, {"uri": self.uri, "reason": "whitespace"}
            )
        if not isinstance(self.checksum, str) or not self.checksum:
            raise AssetContractViolation(
                AssetRefusal.MISSING_CHECKSUM, {"uri": self.uri}
            )
        if len(self.checksum) != _SHA256_LENGTH or any(
            character not in _HEX_DIGITS for character in self.checksum
        ):
            raise AssetContractViolation(
                AssetRefusal.MALFORMED_CHECKSUM,
                {"uri": self.uri, "checksum": self.checksum},
            )
        if not isinstance(self.shape, tuple) or not self.shape:
            raise AssetContractViolation(
                AssetRefusal.EMPTY_SHAPE, {"uri": self.uri, "shape": list(self.shape)}
            )
        if any(type(size) is not int or size <= 0 for size in self.shape):
            raise AssetContractViolation(
                AssetRefusal.NON_POSITIVE_DIMENSION,
                {"uri": self.uri, "shape": list(self.shape)},
            )
        if not isinstance(self.dtype, str) or not self.dtype:
            raise AssetContractViolation(
                AssetRefusal.UNSUPPORTED_DTYPE, {"uri": self.uri, "dtype": self.dtype}
            )
        try:
            np.dtype(self.dtype)
        except TypeError as error:
            raise AssetContractViolation(
                AssetRefusal.UNSUPPORTED_DTYPE, {"uri": self.uri, "dtype": self.dtype}
            ) from error
        if not isinstance(self.mimetype, str) or not self.mimetype.strip():
            raise AssetContractViolation(
                AssetRefusal.MISSING_MIMETYPE,
                {"uri": self.uri, "mimetype": self.mimetype},
            )
        if self.mimetype.count("/") != 1:
            raise AssetContractViolation(
                AssetRefusal.MISSING_MIMETYPE,
                {"uri": self.uri, "mimetype": self.mimetype},
            )

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible document."""
        return {
            "uri": self.uri,
            "checksum": self.checksum,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "mimetype": self.mimetype,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AssetRef:
        """Rebuild a reference from :meth:`as_payload` output."""
        return cls(
            uri=payload["uri"],
            checksum=payload["checksum"],
            shape=tuple(int(size) for size in payload["shape"]),
            dtype=payload["dtype"],
            mimetype=payload["mimetype"],
        )


@dataclass(frozen=True, slots=True)
class PlaneReceipt:
    """What a storage owner committed for one detector plane.

    Parameters
    ----------
    run
        The run the plane belongs to.
    detector
        The detector identity the plane was recorded for.
    plane_ordinal
        The plane's ordinal in the run, counting from zero.
    asset
        The committed asset, identified by content.
    frame_count
        How many frames the plane holds; strictly positive.
    contract_version
        The document version this receipt was written under. A receipt from
        another version is refused rather than reinterpreted.
    note
        Human-facing annotation. It is presentation, not identity, and takes no
        part in :attr:`fingerprint`.
    """

    run: RunIdentity
    detector: str
    plane_ordinal: int
    asset: AssetRef
    frame_count: int
    contract_version: str = ASSET_CONTRACT_VERSION
    note: str = ""

    def __post_init__(self) -> None:
        """Refuse a receipt that cannot be audited."""
        if not isinstance(self.run, RunIdentity):
            raise AssetContractViolation(
                AssetRefusal.WRONG_RUN, {"run": repr(self.run)}
            )
        if not isinstance(self.detector, str) or not self.detector.strip():
            raise AssetContractViolation(
                AssetRefusal.WRONG_DETECTOR, {"detector": self.detector}
            )
        if type(self.plane_ordinal) is not int or self.plane_ordinal < 0:
            raise AssetContractViolation(
                AssetRefusal.WRONG_PLANE, {"plane_ordinal": self.plane_ordinal}
            )
        if not isinstance(self.asset, AssetRef):
            raise TypeError("a plane receipt needs an asset reference")
        if type(self.frame_count) is not int or self.frame_count <= 0:
            raise AssetContractViolation(
                AssetRefusal.ZERO_FRAME_COUNT,
                {"plane_ordinal": self.plane_ordinal, "frame_count": self.frame_count},
            )
        if self.contract_version != ASSET_CONTRACT_VERSION:
            raise AssetContractViolation(
                AssetRefusal.CONTRACT_VERSION_MISMATCH,
                {
                    "expected": ASSET_CONTRACT_VERSION,
                    "found": self.contract_version,
                },
            )
        if not isinstance(self.note, str):
            raise TypeError("a plane receipt note must be text")

    @property
    def plane_key(self) -> tuple[str, str, int]:
        """Return the ``(run, detector, plane)`` a store indexes the receipt by."""
        return (self.run.run_id, self.detector, self.plane_ordinal)

    def admits(self, identity: FrameIdentity) -> None:
        """Check that *identity* belongs to the plane this receipt committed.

        Raises
        ------
        AssetContractViolation
            Naming whether the run, the detector or the plane disagreed.
        """
        if not isinstance(identity, FrameIdentity):
            raise TypeError("a plane receipt admits frame identities")
        if identity.run != self.run:
            raise AssetContractViolation(
                AssetRefusal.WRONG_RUN,
                {
                    "expected": self.run.run_id,
                    "found": identity.run.run_id,
                    "plane_ordinal": self.plane_ordinal,
                },
            )
        if identity.detector_id != self.detector:
            raise AssetContractViolation(
                AssetRefusal.WRONG_DETECTOR,
                {
                    "expected": self.detector,
                    "found": identity.detector_id,
                    "plane_ordinal": self.plane_ordinal,
                },
            )
        if identity.plane_ordinal != self.plane_ordinal:
            raise AssetContractViolation(
                AssetRefusal.WRONG_PLANE,
                {
                    "expected": self.plane_ordinal,
                    "found": identity.plane_ordinal,
                    "detector": self.detector,
                },
            )

    def admits_asset(self, asset: AssetRef) -> None:
        """Check that an offered asset is the content this receipt committed.

        The URI is not compared: the same bytes at another location are the same
        asset, and copying a store must not invalidate its receipts.

        Raises
        ------
        AssetContractViolation
            Naming whether the shape, the dtype or the checksum disagreed.
        """
        if not isinstance(asset, AssetRef):
            raise TypeError("a plane receipt admits asset references")
        if tuple(asset.shape) != tuple(self.asset.shape):
            raise AssetContractViolation(
                AssetRefusal.SHAPE_MISMATCH,
                {
                    "expected": list(self.asset.shape),
                    "found": list(asset.shape),
                    "plane_ordinal": self.plane_ordinal,
                },
            )
        if asset.dtype != self.asset.dtype:
            raise AssetContractViolation(
                AssetRefusal.DTYPE_MISMATCH,
                {
                    "expected": self.asset.dtype,
                    "found": asset.dtype,
                    "plane_ordinal": self.plane_ordinal,
                },
            )
        if asset.checksum != self.asset.checksum:
            raise AssetContractViolation(
                AssetRefusal.CHECKSUM_MISMATCH,
                {
                    "expected": self.asset.checksum,
                    "found": asset.checksum,
                    "uri": asset.uri,
                },
            )

    def _identity_payload(self) -> dict[str, Any]:
        """Return the fields a fingerprint covers: identity, not presentation."""
        return {
            "contract_version": self.contract_version,
            "run": self.run.as_payload(),
            "detector": self.detector,
            "plane_ordinal": self.plane_ordinal,
            "asset": self.asset.as_payload(),
            "frame_count": self.frame_count,
        }

    @property
    def fingerprint(self) -> str:
        """Return the stable identity of this receipt, independent of `note`.

        Any change to the run, the detector, the plane, the asset's content or
        the frame count changes the fingerprint. Re-serialising the receipt, or
        annotating it, does not.
        """
        text = _canonical(self._identity_payload())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def as_payload(self) -> dict[str, Any]:
        """Return the JSON-compatible document, fingerprint included."""
        return {
            **self._identity_payload(),
            "note": self.note,
            "fingerprint": self.fingerprint,
        }

    def as_json(self) -> str:
        """Return the canonical JSON text of the document.

        The text is byte-stable: keys are sorted and separators are fixed, so
        :meth:`from_json` followed by :meth:`as_json` reproduces it exactly.
        """
        return _canonical(self.as_payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlaneReceipt:
        """Rebuild a receipt, refusing a document whose fingerprint disagrees."""
        receipt = cls(
            run=RunIdentity.from_payload(payload["run"]),
            detector=payload["detector"],
            plane_ordinal=payload["plane_ordinal"],
            asset=AssetRef.from_payload(payload["asset"]),
            frame_count=payload["frame_count"],
            contract_version=payload["contract_version"],
            # an annotation is presentation: a document that omits it is the
            # same receipt as one that carries an empty one
            note=payload.get("note", ""),
        )
        recorded = payload.get("fingerprint")
        if recorded is not None and recorded != receipt.fingerprint:
            raise AssetContractViolation(
                AssetRefusal.FINGERPRINT_MISMATCH,
                {"recorded": recorded, "computed": receipt.fingerprint},
            )
        return receipt

    @classmethod
    def from_json(cls, text: str) -> PlaneReceipt:
        """Rebuild a receipt from :meth:`as_json` output."""
        return cls.from_payload(json.loads(text))


def _canonical(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON: sorted keys, fixed separators, no NaN."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


__all__ = [
    "ASSET_CONTRACT_VERSION",
    "AssetContractViolation",
    "AssetRef",
    "AssetRefusal",
    "PlaneReceipt",
]
