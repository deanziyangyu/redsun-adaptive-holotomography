# XY-tiled reconstruction

Status: deterministic planning, full-Z/full-shot tile execution, overlap
blending, cancellation, diagnostics, CPU Tikhonov DPCT/qOBT adapters, spawned
offline `ProcessingJob` execution, and checksum-bound provenance are
implemented. Typed headless CLI and Qt/Napari controls select either the NumPy
reference or a worker-owned CuPy quantitative adapter. Both backends are
implemented and the CUDA frontend path is physically characterized.

## Donor provenance

The behavioral donor is
`pyhololab/src/core/tiled_reconstruction.py` from commit
`405e53b35498898dc7e973f9840c0259ace7671a` (`feat: tiled GPU recon for fast
full field recon`, 2026-08-12). The donor checkout was audited again at
`e8dd96bfa2d54784e00209769d594f305863b01a` on 2026-08-20. The tiling file is
unchanged from `405e53b`, but the checked-out donor repository does not contain
the previously described `tests/test_tiled_reconstruction.py`; AHT therefore
uses new characterization tests rather than claiming to have migrated that
missing test file.

The port retains the deterministic geometry, serial row-major execution,
full-Z/full-shot input contract, reflect/edge padding, complementary
raised-cosine weights, final coverage normalization, progress callback, and
stable smaller-tile GPU-OOM recommendation. It omits the donor's TIFF-first
publication and direct dependency on donor solver classes.

The AHT-owned NumPy adapters retain only the donor's offline four-shot
Tikhonov DPCT and qOBT paths. On the frozen 3x4x16x16 characterization input,
qOBT is byte-identical to the donor and DPCT differs by at most one `float32`
ULP (`1.1920928955078125e-7`). The committed hashes, extrema, and means are
covered by AHT tests. TV/ADMM and production optical calibration gates are not
claimed by these reference adapters.

## Precise meaning

This feature is **computational XY tiling with overlap blending**. Input is one
already aligned `Z,S,Y,X` volume. Every tile retains the full Z range and every
illumination shot, and each placement is a translation in the same source
pixel grid. It does not estimate transforms, register stage fields, stitch a
sample mosaic, or perform multi-view bundle adjustment.

`resolve_tile_plan` produces immutable row-major placements. Auto geometry is
bounded by a whole-volume voxel budget, XY min/max sizes, and alignment. Z is
never subdivided. Explicit mode accepts a tile shape and overlap; off mode
produces exactly one full-field tile. Overlap must remain smaller than its tile
dimension.

## Solver boundary

`TileSolver` accepts a prepared `Z,Y,X` tile shape and reconstructs one
`Z,S,Y,X` sample tile, plus an optional `S,Y,X` background, into `Z,Y,X`.
Only the explicit `dpct` and `qobt` model identities are admitted. The
reconstructor retains expensive solver state across tiles, releases per-tile
inputs after each placement, and closes persistent state idempotently.

The completed result carries:

- requested and resolved tile/overlap geometry;
- every translation, valid extent, padding, and neighbor overlap;
- row-major order and raised-cosine blend policy;
- full input/output axis and shape declarations;
- per-tile and total timing;
- worker-owned GPU telemetry supplied by the scientific adapter.

This layer runs only over completed arrays and raises `ProcessingCancelled`
between tiles. It is not admitted to a live acquisition `ProcessingFlyer`.

`TiledProcessingRequest` integrity-replays a completed DPCT bundle, selects one
exact detector plus an optional background detector/Z plane, creates one
immutable `tiled-dpct-numpy` or `tiled-qobt-numpy` job, and executes it in a
persistent spawned worker. The solver capability declares
`live_supported=false`. A four-pattern DPCT simulator profile supplies the
reference input used by the worker tests.

Selecting `backend="cupy"` with an explicit GPU ID instead creates
`tiled-dpct-cupy` or `tiled-qobt-cupy`. The supervisor admits an exclusive VRAM
lease before starting the worker. That worker owns the CUDA device context,
non-default stream, CuPy allocator, JIT cache, shape-dependent transfer
functions, per-tile transfers, and synchronization. Device placement, CUDA
runtime/driver versions, context startup, allocator reservation, and observed
allocator use are published in result metrics and provenance.

`OfflineProcessingRequest` can attach that request to the normal hardware-free
summary batch. Its immutable plan records a separate observation tuple per job:
the summary solvers receive all verified detector arrays, while the tiled
quantitative solver receives only the selected sample and optional background.
`process-headless` and the processing-only GUI both construct this typed plan
before execution. The GUI's copyable command serializes every optical,
regularization, background, and tiling value rather than relying on hidden UI
state. Tiled CUDA selection has its own GPU ID and whole-MiB VRAM reservation,
independent of the optional GPU mean-projection worker. Each requested CUDA
worker receives a distinct exclusive lease; selecting the same GPU for both is
rejected by normal resource admission.

Resolved geometry, timings, source checksums/configuration/calibration IDs,
model status, and donor identity are stored in the OME-Zarr `aht.provenance`
attribute. Its canonical SHA-256 is also stored in the immutable result
manifest. Cache replay verifies the array and the manifest-bound provenance;
mutating the provenance rejects the cached result.

## Characterization

The AHT tests cover invalid configurations, deterministic irregular-edge
placements, positive weights, one-tile equivalence, overlapped reconstruction
equivalence for a pointwise tile solver, full-Z/full-shot preservation,
multi-cycle background reduction, progress, cancellation, idempotent close,
shape rejection, uncovered/invalid output rejection, automatic geometry, the
explicit no-Z-tiling gate, stable GPU-OOM diagnostics, frozen donor numerical
outputs, background conventions, real overlapped DPCT execution, four-shot
bundle selection, spawned DPCT/qOBT workers, cache reuse, and provenance
tamper rejection. These tests establish donor equivalence for the small CPU
reference case. Explicit hardware tests additionally execute both DPCT and
qOBT workers on the installed Tesla P100 and Tesla P4 and compare their
published arrays to the CPU reference at `rtol=2e-5`, `atol=2e-6`. Production
quantitative accuracy still requires frozen instrument datasets and
calibration tolerances.

Run the explicit multi-GPU characterization with:

```console
$env:AHT_TEST_GPU_PROCESSING = "1"
$env:AHT_TEST_GPU_IDS = "0,1"
uv run pytest tests/test_quantitative_gpu_hardware.py -vv
```
