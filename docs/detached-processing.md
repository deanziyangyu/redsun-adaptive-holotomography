# Detached processing and offline replay

Phase 4 begins with a hardware-free processing boundary. It accepts only
committed references from a verified DPCT run bundle and never constructs a
detector, stage, MCU, IOC, or MMCore instance.

`process-headless` builds the packaged hardware-free RedSun application
profile and executes the detached batch through its processing presenter. The
worker module remains AHT-specific; RedSun owns its application composition
and lifecycle.

## Process topology

`ProcessingSupervisor` owns one persistent spawned process per solver ID. The
reference composition starts `mean-projection` and `quality-metrics` workers
and can append one tiled DPCT/qOBT worker and one nonlinear multi-layer
Born/multislice worker. It admits
their declared capabilities, submits all jobs before waiting, and collects each
outcome independently. A failure in one pipeline does not stop or invalidate
the other pipelines.

Each worker initializes its solver once and can process multiple jobs until it
is explicitly closed or restarted. CPU workers use Windows-safe multiprocessing
spawn. CuPy workers use a persistent length-framed Python subprocess because
the local Windows multiprocessing child could transfer arrays but blocked while
atomically writing its first JIT cache entry. This retains the useful
persistent-solver behavior from the pyhololab simple-acquisition reconstruction
manager while removing its serial solver loop and shared mutable image
ownership.

## Observation and message contract

`observations_from_dpct_bundle` first runs the durable DPCT integrity replay.
It then produces one immutable `Observation` per detector array, sorted by
detector identity. Every observation records the run, detector, channel,
sequence, configuration revision, replay key, checksum, shape, dtype, byte
order, and local Zarr URI.

Supervisor/worker queues carry schema-versioned MsgPack bytes. Jobs,
observations, status, capabilities, and results are encoded as metadata and
array references; image arrays are not nested into queue messages. The worker
resolves and revalidates each reference in its own address space before a
solver sees it.

## Reference solvers and products

The initial CPU-only allowlist contains:

- `mean-projection` 1.0.0: averages committed detector/channel inputs into a
  ZYX `float32` preview and publishes `product.ome.zarr#0` with OME-Zarr 0.5
  multiscale metadata;
- `quality-metrics` 1.0.0: computes channel/Z mean and standard deviation into
  a C,Z,metric `float32` namespaced Zarr v3 array.

With the `gpu-cupy` extra and an explicit `--gpu-id`,
`gpu-mean-projection` replaces the CPU projection pipeline. The supervisor
discovers NVIDIA inventory through `nvidia-smi` without creating a parent CUDA
context, admits an exclusive or explicitly shared VRAM lease against a 10%
margin, and passes only the placement to the child. The child owns its CuPy
context, non-default stream, JIT cache, and allocator for its full lifetime.
Admission executes a representative CUDA warmup before reporting ready.

The GPU subprocess uses the same schema-v1 MsgPack payloads over bounded
big-endian length-framed stdin/stdout. A file-backed cancellation flag remains
independent of the worker command loop. Its explicit cache directory is also
used for `TEMP`, `TMP`, and `TMPDIR`, so CuPy can atomically publish compiled
kernels in restricted deployments.

Each output directory is exclusive to one job. Publication writes the product,
reads it back, hashes the stored bytes, and atomically replaces
`result-manifest.json`. The manifest binds the result to the job, solver
version, configuration fingerprint, input checksums, timestamps, process ID,
metrics, and output reference.

On replay, the worker accepts a cached result only when its job identity,
configuration fingerprint, input checksums, product metadata, and product
checksum all match. Restarting a worker therefore preserves durable progress
without relying on process memory. A corrupt product fails that solver while
other pipelines continue.

## Cancellation and status

Cancellation is cooperative. A worker checks its process-owned cancellation
event during input loading and simulated work, publishes a failed status, and
does not publish a completion manifest. Status snapshots expose solver state,
heartbeat, input/output cursors, queue depth, worker PID, restart count,
cancellation request, and reserved CPU/GPU telemetry fields.

## Command-line replay

Create and process a hardware-free sample bundle with:

```console
uv run aht simulate-dpct --output RUN_DIRECTORY \
  --pattern-id 10 --pattern-id 11 --pattern-id 12 --pattern-id 13
uv run aht process-headless --input RUN_DIRECTORY --output PRODUCT_DIRECTORY
```

Running the second command again reports cached products after verifying them.
The command exits nonzero if either pipeline fails and reports failures by
solver ID.

Append a tiled quantitative CPU reference job by selecting a detector and
providing the instrument optics explicitly. For example:

```console
uv run aht process-headless --input RUN_DIRECTORY --output PRODUCT_DIRECTORY \
  --tiled-model dpct --tiled-detector dhm \
  --tiled-backend cupy --tiled-gpu-id 0 \
  --tiled-gpu-reservation-mib 256 \
  --wavelength-um 0.625 --numerical-aperture 0.8 \
  --pixel-size-um 0.108333 --pixel-size-z-um 1.0 \
  --source-azimuth-deg 0 90 180 270 \
  --tile-shape-yx 512 512 --tile-overlap-yx 96 96
```

The four required optical values have no CLI defaults. The generated command
includes every resolved scientific and tiling input, including defaults, so a
GUI preview is reproducible. Each ordinary summary job receives all verified
detector observations; the quantitative job receives only its chosen sample
detector and optional background detector. Omit the tiled backend/GPU options
to use the NumPy reference.

Append one nonlinear offline job with explicit volume geometry, optics,
illumination radius, source acquisition plane, normalization, optimizer, and
memory budget. For example:

```console
uv run aht process-headless --input RUN_DIRECTORY --output PRODUCT_DIRECTORY \
  --multilayer-model multi_born --multilayer-detector dhm \
  --multilayer-depth-layers 32 \
  --multilayer-voxel-size-zyx-um 0.25 0.108333 0.108333 \
  --multilayer-wavelength-um 0.625 --multilayer-na 0.8 \
  --multilayer-illumination-na 0.45 \
  --multilayer-max-iterations 5
```

The worker selects one Z plane from the committed CZYX observation, treats C
as illumination shots, and publishes a ZYX RI volume. Its preflight,
cancellation, CPU/CuPy placement, optimization histories, and donor provenance
are detailed in `multilayer-reconstruction.md`.

## Processing-only GUI

The optional Qt/Napari MVP resolves an `OfflineProcessingRequest` into the same
typed `ProcessingJob` objects before it displays or copies an equivalent
headless command. Its optional tiled quantitative group exposes the model,
sample/background selection, optics, regularization, and deterministic XY
tiling controls. A separate nonlinear group exposes multi-layer Born or the
replacement defocus-diverse multislice geometry, optics, source plane,
normalization, iterative controls, and its own CPU/GPU placement. Its Run action
executes that already-resolved plan through
the same supervisor and checksum-verifies published arrays before presenting
them as Napari image layers. The GUI composition imports no acquisition device
module and constructs no detector, stage, MCU, IOC, or MMCore object. See
`offline-processing-gui.md` for installation and the UI boundary.

## Bluesky processing flyers

`ProcessingFlyerDevice` is the acquisition-facing adapter for one detached
solver worker. `prepare` validates the job and starts the persistent worker;
`kickoff` opens committed-reference admission; `complete` closes input and
drains the batch without executing numerical kernels on the RunEngine event
loop; `collect` yields canonical progress and result events; ordinary
`stop`/`unstage` cancels and closes the worker idempotently.

The two-flyer replay composition exercises independent prepare, kickoff,
completion, collection, and abort cleanup in one real Bluesky run:

```console
uv run aht process-flyer-replay --input RUN_DIRECTORY \
  --output FLYER_PRODUCT_DIRECTORY
```

Both flyers share stable data-key schemas in `aht_processing_progress` and
`aht_processing_result`, with flyer and solver identities carried as values.
Result events contain only scalar provenance plus the durable output URI and
checksum. A `DocumentJournal` fsyncs every emitted run document to
`processing-documents.jsonl`. Processing failures are isolated and published
as failed progress events so one flyer does not prevent the other from draining
or device cleanup from running.

This implemented profile is run-coupled replay of a completed bundle. The
flyer already accepts committed observations incrementally after kickoff, but
the initial DPCT runner does not yet feed those references during physical
acquisition.

## Deliberately deferred

This slice does not yet provide a production-validated scientific tomography
plugin or a live acquisition observation feed. Tiled DPCT/qOBT and nonlinear
multi-layer Born/multislice are characterized offline references only. The
optional `redsun_aht.device.processing.ProcessingDeviceAdapter` remains a
future RedSun composition seam; numerical kernels and worker supervision stay under
`redsun_aht.processing`.

The deterministic core and donor-characterized NumPy Tikhonov adapters for
offline XY-tiled DPCT/qOBT now execute as spawned `ProcessingJob` workers from
the headless CLI and processing-only GUI. Their worker-owned CuPy variants are
available through both frontends and have passed CPU-equivalence tests on both
installed NVIDIA GPUs; production dataset/calibration gates remain in
progress. See
`xy-tiled-reconstruction.md` for the precise computational-tiling meaning,
donor provenance, implemented boundary, and characterization scope.
See `multilayer-reconstruction.md` for nonlinear model provenance, iterative
inputs, memory admission, physical GPU characterization, and remaining
scientific gates.

The OME image product follows the
[OME-Zarr 0.5 specification](https://ngff.openmicroscopy.org/0.5/). Storage is
implemented with the pinned Zarr v3 runtime documented in the
[Zarr storage guide](https://zarr.readthedocs.io/en/latest/user-guide/storage/).
The optional GPU runtime follows the official
[CuPy CUDA-12 installation guidance](https://docs.cupy.dev/en/stable/install.html).
