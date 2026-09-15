# RedSun multidimensional storage feature-request note

This is a completed reconnaissance note, not an issue ready to file. Testing
against RedSun `0.13.0rc0` found no RedSun framework gap that justifies a
RedSun-owned `MultidimensionalOpenStore` contract. RedSun launches and owns the
service lifecycle, composes an ordinary ophyd-async EPICS device, preserves
standard Bluesky Resource/Datum documents, and shuts the service down with the
application container. The remaining mismatch is in generic
Bluesky-to-Tiled normalization of a pre-existing multidimensional array inside
a Zarr group. That work belongs in an AHT/Tiled adapter or, if generalized, in
`bluesky-tiled-plugins`; this note should not be filed against RedSun as a
multidimensional storage request.

## Suggested title

Document the service-owned multidimensional external-asset boundary

## Issue body

**Is your feature request related to a problem? Please describe.**

RedSun `0.13.0rc0` adds the service layer needed to declare, launch, attach,
configure, and stop a backing acquisition service while composing an ordinary
ophyd-async device over it. This resolves the earlier service lifecycle and
device-prefix problem without an AHT-specific RedSun device subclass.

RedSun's retained live-acquisition storage shim assumes that two-dimensional
frames enter the application and are appended through RedSun's
`StreamSpec`/`FrameSink`/`OpenStore` path. That model becomes restrictive when
the hardware is owned by a separately managed service and the corresponding
ophyd-async device needs to write a multidimensional acquisition directly.

For example, an acquisition may contain multiple cameras and dimensions such
as illumination pattern, scan position, channel, time, and detector. The
service or device may need to choose OME-Zarr, OME-TIFF, another file format,
or a facility-specific writer. RedSun should not require those frames to pass
back through a central Zarr-oriented application shim merely to participate in
a RedSun/Bluesky run.

The remaining question was not how RedSun writes each array. It was whether a
service-backed ophyd-async device could own that decision while participating
fully in RedSun composition and emitting standard Bluesky external-asset
documents. Validation shows that it can. Direct Tiled interpretation requires
an additional format-aware catalog adapter because the stored object is a
multidimensional array nested in an OME-Zarr group rather than an
event-appended flat array.

**Describe the solution you'd like**

No new RedSun multidimensional storage API is requested. Document and retain
the tested behavior that live-acquisition storage can be owned by a declared
service and its ordinary ophyd-async device, without requiring `BaseStorage`,
`FrameSink`, or a RedSun storage backend.

The intended division of responsibility is:

- RedSun declares, configures, starts, checks, and stops the backing service.
- The ophyd-async device prepares the acquisition and delegates writing to its
  service-side or device-side writer.
- The service/device chooses its file format, multidimensional schema, frame
  routing, chunking, durability, and completion behavior.
- Bluesky plans emit run metadata and actual motor, illumination, and other
  scan-state readings as Events.
- The device emits standard Resource/Datum or StreamResource/StreamDatum
  documents linking those Events to the externally stored arrays.
- A format-aware Tiled adapter indexes the Bluesky documents and points each
  data key at its actual array within the service/device-owned asset for GUI
  and downstream access.

Some backend-neutral schema concepts from the earlier
`MultidimensionalOpenStore` experiment remain useful inside AHT at the
boundary between its service, ophyd-async device, and writer. They do not make
RedSun responsible for writing frames and are not proposed for RedSun's public
namespace. For example, the AHT service/device may use internal models
equivalent to:

```python
@dataclass(frozen=True)
class DimensionSpec:
    name: str
    size: int | None
    kind: Literal["space", "time", "channel", "position", "other"]
    unit: str | None = None
    coordinates: tuple[str | int | float, ...] | None = None


@dataclass(frozen=True)
class MultidimensionalAssetSpec:
    data_key: str
    mimetype: str
    uri: str
    frame_shape: tuple[int, ...]
    dtype: str
    dimensions: tuple[DimensionSpec, ...]
    chunks: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FramePlacement:
    axes: tuple[str, ...]
    index: tuple[int, ...]


@dataclass(frozen=True)
class PersistedFrame:
    data_key: str
    placement: FramePlacement
    locator: Mapping[str, JsonValue]
    checksum: str | None = None
```

`MultidimensionalAssetSpec` describes an asset selected and created by the
service/device. It does not select a writer or require Zarr. Its serializable
fields can become Resource or StreamResource parameters. `FramePlacement`
provides the explicit event-to-array-position record needed when acquisition
order differs from storage order. A `PersistedFrame` is an internal
service/device receipt whose locator can become Datum metadata; RedSun does
not need to receive the NumPy frame that produced it.

The earlier transaction lifecycle is also still useful as a semantic
requirement owned by the service/device:

- a persisted receipt is not issued merely because a frame entered a queue;
- successful completion is published only after the device's chosen writer is
  finalized;
- completion and abort are mutually exclusive terminal outcomes; and
- an interrupted acquisition may remain discoverable as incomplete, but must
  not be represented as complete.

An `IndexedWrite`-like object may also remain useful *inside* a service or
device implementation:

```python
@dataclass(frozen=True)
class IndexedWrite:
    data_key: str
    frame: NDArray[Any]
    placement: FramePlacement
    source_checksum: str | None
    context: Mapping[str, JsonValue]
```

This is not proposed as a RedSun frame-ingress API. It is an implementation
pattern by which a service/device can keep the payload, logical placement,
source identity, and per-frame provenance together until its selected writer
returns a `PersistedFrame` receipt.

An illustrative Bluesky integration would be:

```python
yield from bps.open_run(
    md={
        "plan_name": "dpct_grid_scan",
        "purpose": "dpct",
        "scan_mode": "grid",
        "recipe_id": recipe.id,
    }
)
yield from bps.prepare(
    detector,
    TriggerInfo(
        number_of_events=point_count,
        collections_per_event=pattern_count,
        trigger=DetectorTrigger.INTERNAL,
    ),
)
```

During `prepare`, the ophyd-async device would pass its resolved
`MultidimensionalAssetSpec` to its own service-side or device-side writer. The
declaration establishes the data key, dimensions, expected extents, frame
shape, dtype, MIME type, and optional chunk layout before acquisition starts.
RedSun only needs to allow this normal device preparation to occur and preserve
the documents the device later emits.

For each acquired frame, the service/device—not the RedSun storage shim—would
derive a logical placement from the resolved scan point:

```python
write = IndexedWrite(
    data_key="dhm_image",
    frame=frame,
    placement=FramePlacement(
        axes=("pattern", "z"),
        index=(pattern_ordinal, z_ordinal),
    ),
    source_checksum=checksum,
    context={
        "acquisition_id": acquisition_id,
        "camera_sequence": camera_sequence,
        "configuration_revision": configuration_revision,
        "timestamp_ns": timestamp_ns,
    },
)
receipt = await device_writer.write_indexed(write)
```

Where the service uses a retained camera buffer, it should acknowledge that
buffer only after the receipt satisfies the service/device's declared
durability policy. RedSun does not prescribe whether the writer is Zarr, TIFF,
OME-based, or something else.

The ophyd-async device can then translate the asset specification and receipt
into standard external-asset documents:

- for step scans, `read()` can expose a datum ID for the committed frame while
  `collect_asset_docs()` supplies the corresponding Resource and Datum;
- for fly or streamed acquisition, `collect_asset_docs()` can supply
  StreamResource/StreamDatum documents where sequential ranges describe the
  data faithfully; and
- for non-linear, sparse, snake, retried, or reordered acquisition, the
  resource parameters or Datum locator must retain an explicit placement map
  rather than treating Event sequence as array position.

The placement derivation differs by scan mode while using the same schema:

```text
count                     ("frame",)       -> (event_ordinal,)
list_scan(z=[...])        ("z",)           -> (z_ordinal,)
grid_scan(x, y)           ("y", "x")      -> (y_ordinal, x_ordinal)
snake grid_scan(x, y)     ("y", "x")      -> logical grid coordinate
DPCT                      ("pattern", "z") -> (pattern_ordinal, z_ordinal)
```

Each detector data key may have its own asset specification, URI, frame shape,
dtype, and placement record. Bluesky Events should still contain the actual
motor, illumination, timing, and detector readbacks. `FramePlacement` records
the intended array cell; the Event records what the hardware actually did.

These models are illustrative AHT implementation concepts rather than a demand
for these exact classes in RedSun. Standard ophyd-async and event-model
structures should carry the externally visible information wherever they can
do so without loss. The validation target is preservation of dimension
declarations, explicit placement, asset location, and terminal state through
the RedSun RunEngine and into Tiled.

The `0.13.0rc0` validation established the following:

- a device can be declared without registering a RedSun frame sink;
- its standard ophyd-async `prepare`, `trigger` or `kickoff`/`complete`,
  `read`/`collect`, and `collect_asset_docs` behavior passes through the RedSun
  RunEngine unchanged;
- external-asset documents are associated with the correct run and data key;
- declared dimensions, physical coordinates, dtype, shape, chunks, and asset
  format remain present in Resource parameters and the stored manifest;
- a Datum or equivalent record preserves explicit named array placement when
  event order is not sufficient to locate the frame;
- motor and other scan readbacks can be recorded in the same Event stream as
  the external image reference;
- multiple devices may independently write different formats, shapes, or
  dtypes in one run; and
- the container terminates its launched service cleanly after success; and
- the FLIR-only hardware path writes and checksum-verifies two physical camera
  frames using the same service, ophyd-async device, and external-asset path.

The direct generic Tiled path does *not* yet satisfy the final access step. In
the tested `bluesky-tiled-plugins` writer, `RunNormalizer` converts legacy
Datum documents to StreamDatum documents with one-dimensional sequential
`indices`. Per-frame multidimensional placement in `datum_kwargs` is not
retained on each resulting StreamDatum. The default consolidator for
`application/x-zarr` then models the event frames as an appended array and
registers the OME-Zarr group URI as if it were an array URI. For the minimal
two-pattern, two-scan case it advertised `(32, 8)` instead of the physical
`(2, 2, 8, 8)` array, and reads failed because Tiled opened the Zarr group as
an array.

This is not evidence for a RedSun storage interface. The narrow next step is
an AHT-owned Bluesky–Tiled integration that:

- registers one Tiled array node per detector data key;
- resolves the Resource's group URI plus `array_path` to the actual Zarr
  array;
- takes full shape, chunks, dtype, and dimensions from the declared asset
  schema rather than deriving them by stacking Event sequence numbers;
- keeps Event-to-logical-placement records queryable alongside actual motor
  and illumination readbacks; and
- validates that the Tiled array equals the checksum-verified DPCT bundle.

If that mechanism proves broadly reusable, a separate minimal proposal may be
made to `bluesky-tiled-plugins` for a group-component Zarr consolidator and a
lossless placement representation. It should not be coupled to RedSun's
service or storage APIs.

The maintainer has separately indicated an intention to remove the current
live-acquisition storage shim and push writing down to services/devices. The
shim is still present in `0.13.0rc0`, but its removal is not part of this
request and it should not be replaced with a RedSun-owned multidimensional
version. I am not requesting that RedSun standardize a multidimensional array
schema or add a Zarr/TIFF writer.

Standard ophyd-async and Bluesky interfaces cover the RedSun composition and
document path. Therefore no RedSun multidimensional storage feature request is
needed. A RedSun documentation example or integration test remains optional,
but the demonstrated catalog mismatch should be handled outside RedSun.

**Describe alternatives you've considered**

1. **Extend the existing RedSun storage shim.** An experimental
   `MultidimensionalOpenStore` protocol added named frame placement, persisted
   receipts, and complete/abort operations. This still makes RedSun own a
   storage abstraction that the service/device is better placed to own, and
   risks duplicating writer libraries. The metadata-only concepts above retain
   the useful schema and traceability portions without routing frame payloads
   through RedSun.

2. **Keep application-side shared-memory storage.** A camera service can expose
   a retained shared-memory frame, after which the application copies, writes,
   verifies, and acknowledges it. This provides strong backpressure and is a
   useful compatibility path, but it adds a process-boundary copy and keeps
   storage orchestration in the application.

3. **Adopt `ome-writers` directly in RedSun.** `ome-writers` already maps
   streams of 2-D microscopy frames into multidimensional OME-Zarr or OME-TIFF
   data. It is a promising implementation choice for an individual service or
   device, but RedSun should not require OME formats or make that package its
   universal storage model.

4. **Put image arrays directly in EPICS or Bluesky Events.** This is useful for
   bounded latest-frame preview data, but it is not an appropriate durable
   transport for high-rate raw acquisition. External asset documents are the
   better fit.

5. **Use generic `TiledWriter` as the acquisition writer without a format
   adapter.** Tiled is well suited to cataloging and serving completed or
   in-progress external assets, but its generic legacy Resource/Datum
   normalizer assumes sequential ranges and does not infer an array component
   inside a multidimensional Zarr group. A small format-aware adapter is still
   required, and Tiled does not define the service's camera-buffer
   acknowledgement or durable-write semantics.

**Additional context**

Reconnaissance against RedSun `0.13.0rc0` (`0146e27`) confirms that the new
service layer starts services before devices, injects the configured service
prefix into ordinary ophyd-async devices, supports launched and attached
services, and closes launched subprocesses with container teardown. A
hardware-free run preserved non-sequential scan IDs `17` and `3`, explicit
pattern/scan array placement, motor and illumination readbacks, and
checksum-resolvable Resource/Datum references. A RedSun-launched simulated IOC
also completed the same external-asset flow and stopped with the container.

The FLIR-only hardware validation used the available Teledyne FLIR Blackfly S
`BFS-U3-20S4M` (serial `20154173`) with the supplied `flir_high.cfg`. It wrote
two `uint16` frames of shape `(620, 808)` at scan ID `9` and pattern IDs `10`
and `11`. Source and stored SHA-256 checksums matched for both frames, the
manifest finalized as complete, and the launched camera service exited with
the RedSun container. No PVCAM hardware was used.

The remaining local Tiled integration test proved that run creation,
descriptor/Event ingestion, and external node registration succeed. Reading
the node fails because generic normalization registers the whole
`data.ome.zarr` group with an event-derived array shape rather than registering
the detector component named by `array_path`. This localizes the next work to
the AHT/Tiled boundary.

The current AHT camera service owns MMCore/the vendor SDK and a temporary
lossless shared-memory ring. EPICS carries commands and immutable frame
descriptors; it does not currently provide the durable raw-data store. The
application copies and checksum-verifies a retained frame, commits it to its
DPCT bundle, and only then acknowledges the exact service sequence. A separate
EPICS waveform is used for latest-frame GUI preview and may intentionally skip
frames.

A future service-owned writer could eliminate the raw-frame copy by writing
directly from the camera service and exposing only external asset references
through its ophyd-async device. That writer might use `ome-writers`, tifffile,
Zarr, or another backend. Its format and durability policy should remain a
device/service concern.

Motor readings in Bluesky Events provide the measured physical coordinates,
but the external asset documentation must also preserve the association
between an Event and its stored image location. Event sequence alone is not a
safe array index for snake scans, adaptive scans, retries, skipped frames, or
independent multi-camera streams.

An earlier local prototype exists at
`deanziyangyu/redsun:feat/pre-upstream@03009a5`. It should be treated only as a
requirements experiment and should not be opened as a PR in its present form,
because its RedSun-owned storage protocol conflicts with the service/device
ownership described here and it is stacked on an unrelated EPICS lifecycle
candidate.

Disposition and next steps:

1. Do not file this note as a RedSun multidimensional storage issue.
2. Keep the schema, indexed-write, receipt, and durability contracts in AHT's
   service/device implementation.
3. Implement and test an AHT-owned Tiled consolidator/registration adapter for
   an array component inside the DPCT OME-Zarr group.
4. Preserve the original Resource/Datum placement records in the run export or
   publish a companion placement table; do not reduce snake, adaptive,
   reordered, or sparse acquisition to Event sequence.
5. If the adapter exposes a generally missing capability, prepare a separate
   `bluesky-tiled-plugins` issue with the minimal reproducer and without a
   RedSun API request.
