# DPCT OME-Zarr storage and replay

Status: implemented hardware-free Phase 4 provenance/storage slice. Tiled
registration, remote/object storage, partial-run resume, TIFF export, and
physical acquisition remain pending.

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

## Lossless acknowledgement boundary

The DPCT runner now copies each exact retained detector sequence, validates its
shape, dtype, and source checksum, and passes it to the durable sink. Detector
acknowledgement occurs only after storage write, readback, checksum, and commit
journal fsync succeed. Storage failure therefore leaves the acquisition buffer
unacknowledged until ordinary cleanup disconnects the service.

## Replay

`DpctZarrReplay.verify()` checks the run/acquisition identities, completion
state, commit-journal checksum, frame count, duplicate coverage, array layout,
and every stored frame checksum before returning the completion manifest.
Detector arrays are exposed for hardware-free processing only after the same
verification.

## Standards baseline

- [OME-Zarr 0.5](https://ngff.openmicroscopy.org/0.5/) defines the Zarr v3
  metadata namespace, multiscales metadata, axis ordering, and matching array
  dimension names used here.
- [Zarr-Python storage](https://zarr.readthedocs.io/en/latest/user-guide/storage/)
  defines the local hierarchy implementation. AHT pins `zarr==3.3.0` rather
  than relying on an undeclared transitive dependency.
