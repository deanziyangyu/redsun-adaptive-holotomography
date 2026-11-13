"""Finite-run frame storage: the writer, its documents and its receipts.

The camera service captures through one protocol -- ``append`` and ``close`` --
and this package owns everything behind it. A recorded plane is a Zarr v3 array
plus a manifest that names its shape, dtype, chunking, frame count and each
frame's checksum, with an optional frame-timing sidecar in the service's own
schema. The receipts module turns such a store into the contracts' asset
vocabulary without the writer that produced it, and answers with a named
refusal rather than an exception when a plane is short, altered or from another
schema.

Not here, deliberately: a read-only adapter for archived pre-reset AHT
datasets, and any reconstruction kernel. Both are deferred, and neither the
legacy DPCT schema nor ``redsun_aht.processing`` is reachable from this package.
"""

from .receipts import (
    STORE_MIMETYPE,
    TIMING_MIMETYPE,
    ClosedStore,
    RefusedStore,
    assemble_receipt,
    build_asset,
    open_closed_store,
    read_manifest,
    read_timing,
    refusal_from_exception,
    timing_asset,
)
from .writer import (
    ARRAY_FORMAT,
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_V1,
    MANIFEST_SUFFIX,
    SUPPORTED_MANIFEST_SCHEMAS,
    TIMING_CLOCK,
    TIMING_RECORD_LIMIT,
    TIMING_SCHEMA,
    TIMING_SEMANTICS,
    FrameEntry,
    FrameStamp,
    RunFrameStore,
    SettingMark,
    StoreManifest,
    StoreRefusal,
    StoreRefused,
    TimingRecord,
    TimingSidecar,
    frame_checksum,
    frame_store_factory,
    frames_checksum,
    local_store_path,
    manifest_path,
    open_run_store,
    timing_sidecar_path,
)

__all__ = [
    "ARRAY_FORMAT",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_V1",
    "MANIFEST_SUFFIX",
    "STORE_MIMETYPE",
    "SUPPORTED_MANIFEST_SCHEMAS",
    "TIMING_CLOCK",
    "TIMING_MIMETYPE",
    "TIMING_RECORD_LIMIT",
    "TIMING_SCHEMA",
    "TIMING_SEMANTICS",
    "ClosedStore",
    "FrameEntry",
    "FrameStamp",
    "RefusedStore",
    "RunFrameStore",
    "SettingMark",
    "StoreManifest",
    "StoreRefusal",
    "StoreRefused",
    "TimingRecord",
    "TimingSidecar",
    "assemble_receipt",
    "build_asset",
    "frame_checksum",
    "frame_store_factory",
    "frames_checksum",
    "local_store_path",
    "manifest_path",
    "open_closed_store",
    "open_run_store",
    "read_manifest",
    "read_timing",
    "refusal_from_exception",
    "timing_asset",
    "timing_sidecar_path",
]
