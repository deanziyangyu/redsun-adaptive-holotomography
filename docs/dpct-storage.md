# DPCT OME-Zarr storage and replay

Status: the durable provenance/storage slice and standard Bluesky
Resource/Datum publication are implemented. They pass hardware-free,
RedSun-launched-service, and FLIR-only physical acquisition tests. Completed
bundle registration through Tiled is implemented; a format-aware live-run
Tiled adapter, remote/object storage, partial-run resume, and TIFF export
remain pending.

## Bundle contract

`aht simulate-dpct --output RUN_DIRECTORY` creates a fresh, self-contained run
bundle:

- `run-manifest.json` is the immutable resolved configuration snapshot;
- `data.ome.zarr/` is a Zarr v3 hierarchy with OME-Zarr 0.5 metadata;
- `frame-commits.jsonl` is the append-only durable frame/provenance journal; and
- `acquisition-manifest.json` is the atomic completion marker.

The completion manifest is deliberately absent from failed or interrupted
runs. A non-empty output directory is never reused or overwritten.

## Array and coordinate layout

Each detector owns one independent `detectors/<detector>/0` array. The stored
axis order is `C,Z,Y,X`, conforming to the OME-Zarr axis ordering, while AHT
records that the recipe's acquisition order is logically `Z,C,Y,X`. `C` indexes
the declared illumination pattern IDs; `Z` indexes the ordered scan positions.
Exact requested and measured stage coordinates, units, monotonic pose times,
uncertainties, exposure times, configuration revisions, detector/frame IDs, and
frame sequences remain explicit metadata rather than being inferred from array
indices.

The initial layout accepts 2-D detector frames and stores one frame per chunk.
Every chunk is read back and SHA-256 checked before its commit record is fsynced.
Only after all detector/position/pattern combinations exist is the hierarchy
marked complete and the checksum-bearing acquisition manifest atomically
published.

## Multi-slice measurement extension

The same OME-Zarr bundle is the canonical target for multi-slice migration.
Its detector group records `aht.multislice` metadata and uses CZYX arrays for
the measurement domain:

- `aht_real_amplitude_focal_stack` stores real-amplitude data with physical
  focal positions on the acquisition `Z` axis. It has no imaginary component
  and no numerical refocusing.
- `interferometric_complex_field` stores paired float32 real and imaginary
  arrays. It may declare numerical focus offsets and is the representation used
  by the supplied MATLAB bridge.

The split-component representation is intentional: CuPy can retain the two
float32 arrays without requiring a persistent complex64 allocation. HDF5/MAT
files are compatibility inputs that will be converted into this contract by a
later importer; they are not the canonical bundle format.

## Lossless acknowledgement boundary

The RedSun DPCT detector-group facade copies each exact retained detector
sequence, validates its shape, dtype, and source checksum, and passes it to the
durable sink from the Bluesky generator plan. Detector acknowledgement occurs
only after storage write, readback, checksum, and commit journal fsync succeed.
Storage failure therefore leaves the acquisition buffer unacknowledged until
ordinary plan cleanup disconnects the service.

## Replay

`DpctZarrReplay.verify()` checks the run/acquisition identities, completion
state, commit-journal checksum, frame count, duplicate coverage, array layout,
and every stored frame checksum before returning the completion manifest.
Detector arrays are exposed for hardware-free processing only after the same
verification.

## Bluesky external assets and Tiled

The DPCT detector group emits one Resource per detector array and one Datum per
committed frame. Resource parameters declare the group URI, internal
`array_path`, dimensions, full shape, chunks, dtype, pattern IDs, and ordered
scan IDs. Datum metadata records the logical `(pattern, scan)` index, source
identity, sequence, and checksum. Events reference those datum IDs while also
recording actual stage and illumination readbacks.

RedSun `0.13.0rc0` preserves these documents unchanged. The generic
`bluesky-tiled-plugins` legacy normalizer does not yet provide direct array
access for this layout: it converts placements into sequential StreamDatum
ranges, derives an event-stacked shape, and points Tiled at the enclosing
OME-Zarr group rather than the detector array. The remaining catalog task is
an AHT-owned format-aware registration/consolidation adapter. This does not
require a RedSun multidimensional storage interface.

## Standards baseline

- [OME-Zarr 0.5](https://ngff.openmicroscopy.org/0.5/) defines the Zarr v3
  metadata namespace, multiscales metadata, axis ordering, and matching array
  dimension names used here.
- [Zarr-Python storage](https://zarr.readthedocs.io/en/latest/user-guide/storage/)
  defines the local hierarchy implementation. AHT pins `zarr==3.3.0` rather
  than relying on an undeclared transitive dependency.
