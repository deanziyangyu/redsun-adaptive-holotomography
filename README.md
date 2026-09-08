# redsun-adaptive-holotomography

Reusable acquisition and processing components for the Adaptive Multimodal
Holotomographic Microscope.

> [!IMPORTANT]
> Phases 1 through 3 are complete and Phase 4 processing is in progress. The
> repository provides public contracts, a RedSun 0.11 composition profile,
> immutable manifests, deterministic replay, durable calibration artifacts,
> hardware-free dual-camera and DPCT simulations, explicitly gated MMCore
> transports for PVCAM and FLIR, an MMCore shuttered-light backend with typed
> generic property handling, an MCU v1 protocol simulator and host adapter,
> an admitted ESP300 composite stage, durable OME-Zarr capture, and detached
> CPU/GPU reference solvers. Continuous camera acquisition,
> illumination service EPICS exposure,
> real MCU conformance, production-validated quantitative tomography and live-acquisition
> observation feeding are not operational yet. Optional persistent CuPy GPU
> reference workers, explicit
> NVIDIA resource admission, two-flyer Bluesky replay, local Tiled
> registration/query/replay, a hardware-free Napari processing MVP, and
> offline multi-layer Born/multislice CPU/CuPy reference workers are
> operational.

The distribution imports as `redsun_aht` and installs the `aht` command.

RedSun is the required application boundary for acquisition engines, devices,
profiles, presenters, views, and storage backends. The current implementation
only partially satisfies that requirement: several workflows still bypass the
corresponding RedSun interfaces. See
[`docs/redsun-native-architecture.md`](docs/redsun-native-architecture.md) for
the migration contract and [`docs/redsun-gap-report.md`](docs/redsun-gap-report.md)
for framework additions that require manual upstream review. Mimir and
pyhololab are behavior and operator-interface references, not parity targets.
AHT keeps distinct offline composition, detached processing, and customized
frontends while routing those features through RedSun.

## Development

The project targets Python 3.12+ and pins the first compatibility baseline to
RedSun 0.11.0.

```console
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy
```

When the Napari extra is installed in a restricted Windows environment, pytest
may auto-load Napari's theme fixture and fail before project tests run. See
[`docs/testing.md`](docs/testing.md) for the targeted plugin-disable and local
temporary-directory commands.

Run a hardware-free lifecycle smoke test with:

```console
uv run aht simulate --journal run-events.jsonl
```

The simulated command creates an immutable run manifest plus append-only event
and Bluesky-document journals, completes one run, and replays every artifact to
verify the terminal state. It constructs a device-free RedSun container with
an in-memory `BaseStorage`; it does not construct or contact hardware.

Exercise the two isolated mock camera services and their preview/acquisition
shared-memory rings with:

```console
uv run aht simulate-detectors
```

This command reserves unique simulation-only EPICS prefixes but starts no IOC,
loads no vendor SDK, and contacts no camera.

Exercise the Phase 3 scan/stage/illumination/dual-detector DPCT composition with:

```console
uv run aht simulate-dpct
```

The smoke recipe runs two simulated Z positions by two committed raw-8-bit MCU
patterns through RedSun-declared stage, illumination, and detector-group
facades, using a Bluesky generator executed by `redsun.engine.RunEngine`. Each
immutable shot record associates the settled framed pose, pattern ID, ordered
detector frames, exact run ID, and lossless acknowledgement. Cleanup stops detector
acquisition and stage motion, clears ordinary illumination output, disconnects
both detector services, and unlinks their shared memory even on failure. See
`docs/dpct-acquisition.md` for the ordering and scope contract.

Add `--output RUN_DIRECTORY` to persist the simulated run as a fresh verified
OME-Zarr 0.5/Zarr v3 bundle. Detector payloads are stored as independent CZYX
arrays and acknowledged only after write/readback checksum verification and an
fsynced frame-commit record. See `docs/dpct-storage.md` for the durable layout
and replay contract.

Replay a verified bundle through two independent, hardware-free processing
workers with:

```console
uv run aht process-headless --input RUN_DIRECTORY --output PRODUCT_DIRECTORY
```

The command builds the hardware-free `process-headless` RedSun container, whose
processing presenter starts persistent spawned workers for `mean-projection`
and `quality-metrics`. It submits both jobs before waiting and sends only
versioned MsgPack metadata and array references across process boundaries. The image-like
mean projection is published as OME-Zarr 0.5; the quality table uses a
namespaced Zarr v3 array. Atomic result manifests provide deterministic cache
and integrity replay. See `docs/detached-processing.md` for the worker and
result contracts and the remaining live-feed and scientific-solver work.

Install one Qt binding and launch the processing-only Napari application with:

```console
uv sync --extra pyqt --locked
uv run aht process-gui --input RUN_DIRECTORY --output PRODUCT_DIRECTORY
```

The form verifies the immutable input bundle, builds the same typed CPU or GPU
jobs used by the headless command, and only then renders a platform-quoted
command for review or clipboard use. Running the plan uses the same detached
workers and checksum-verifies each published array before adding it to Napari.
It does not construct acquisition hardware. See
`docs/offline-processing-gui.md` for the boundary and test contract.

Exercise the same CPU workers through real ophyd-async/Bluesky flyer lifecycle
and canonical processing streams with:

```console
uv run aht process-flyer-replay --input RUN_DIRECTORY \
  --output FLYER_PRODUCT_DIRECTORY
```

This stages two `ProcessingFlyerDevice` instances in one RunEngine run. Each
flyer prepares and owns one detached worker, accepts committed references after
kickoff, drains independently during completion, and collects lightweight
progress/result events. The result events contain product URIs and checksums,
not arrays. Every emitted start/descriptor/event-page/stop document is fsynced
to `processing-documents.jsonl` for catalog ingestion or deterministic replay.
The implemented profile replays a completed run; binding the same adapter to a
live acquisition observation feed remains pending.

Install the catalog runtime and register that completed run with an existing
Tiled server by adding its URI:

```console
uv sync --extra catalog-tiled --locked
uv run aht process-flyer-replay --input RUN_DIRECTORY \
  --output FLYER_PRODUCT_DIRECTORY --tiled-uri http://localhost:8000
```

The catalog stores the Bluesky processing run and externally registers the
verified raw and derived Zarr roots. Source-run metadata supports exact search;
the selected source URI is still passed through the checksum-verifying DPCT
replay reader. OME-Zarr bundles and the fsynced document journal remain the
durable records. The Tiled server must be allowed to read the source/product
storage roots and must configure the Bluesky JSON-sequence exporter for
document export. See `docs/tiled-catalog.md` for the server fragment, node
layout, duplicate policy, and replay boundary.

Install the optional Python 3.12+ CUDA-12 GPU runtime with:

```console
uv sync --extra gpu-cupy --locked
```

Then replace the CPU mean projection with an explicitly placed GPU worker:

```console
uv run aht process-headless --input RUN_DIRECTORY \
  --output PRODUCT_DIRECTORY --gpu-id 0 --gpu-reservation-mib 256
```

GPU discovery uses `nvidia-smi` without creating a CUDA context in the
supervisor. Admission applies the declared VRAM reservation, allowed/preferred
device ID, exclusivity, and a memory margin. The persistent subprocess owns its
CuPy context, non-default stream, JIT cache, and allocator; status/results report
device ID, reserved/observed memory, runtime/driver versions, startup, and kernel
timing. Both local Tesla P100 and Tesla P4 devices passed the explicit hardware
test. This reference projection validates supervision and placement, not the
scientific DPCT reconstruction model.

The Phase 2 control plane can run one SDK-free caproto IOC per simulated camera
service:

```console
uv run aht-camera-ioc --service-id fluorescence --prefix AHT:PVCAM:SIM:
uv run aht-camera-ioc --service-id dhm --prefix AHT:FLIR:SIM:
```

`CameraServiceDevice` is the typed ophyd-async application-side adapter. These
commands publish health, lifecycle commands, counters, and generation-aware
descriptor metadata.

Install the optional Python 3.12 Micro-Manager backend with:

```console
uv sync --all-groups --extra camera-mmcore --locked
```

That extra pins `pymmcore-plus==0.18.1` and `pymmcore==12.5.0.75.0`, whose
MMCore requires Device API 75 adapters. Real cameras use two independent
service processes. One service must receive a camera-only PVCAM configuration;
the other must receive a camera-only SpinnakerC configuration. Each service
constructs one non-singleton `CMMCorePlus`, validates its expected camera
identity, forces Micro-Manager auto-shutter off, and unloads all devices during
cleanup. Camera services do not load or command shutters, stages, or other
instrument devices.

The hardware backend is selected only by explicit `aht-camera-ioc` arguments.
For example, the process boundaries are:

```console
aht-camera-ioc --service-id fluorescence --prefix AHT:PVCAM: \
  --backend mmcore --mm-path <MM_DIR> --mm-config <PVCAM_CFG> \
  --adapter PVCAM --device-name Camera-1 --camera-label Camera-1

aht-camera-ioc --service-id dhm --prefix AHT:FLIR: \
  --backend mmcore --mm-path <MM_DIR> --mm-config <FLIR_CFG> \
  --adapter SpinnakerC --device-name "Blackfly S BFS-U3-20S4M" \
  --camera-label "Blackfly S BFS-U3-20S4M" \
  --serial-property "Serial Number" --expected-serial <SERIAL>
```

Machine-local configuration paths and serial numbers are not committed.

Install the optional ESP300 serial runtime with:

```console
uv sync --all-groups --extra stage-esp300 --locked
```

The stage slice retains the hardware-neutral scan and composite layers in
`redsun_aht.motion` and `redsun_aht.device.stage`. It provides exact-endpoint
linear points, equal-length list scans, N-dimensional grids, selected-axis
snaking, atomic multi-axis limit admission, explicit units/coordinate frame,
measured settling, and bounded ordinary-stop cleanup. Its Newport adapter uses
an already-open, explicitly owned serial port and derives axis units, encoder
resolution, software limits, and current position from read-only ESP300
queries before building the composite stage.

The physical COM1 read-only admission passed for one ESP300 3.08 controller
with two CMA-12CCCL axes and one CMA-12PP axis. All three reported millimeter
units, 0–12.5 mm software limits, and positions within those limits. No axis
was powered, homed, moved, or stopped during admission; machine serial numbers
remain outside Git. See `docs/esp300-stage.md` for donor provenance and the
command boundary.

For an MMCore backend, each triggered array is committed to the service-owned
lossless acquisition ring before `FRAMES_PUBLISHED` advances. The IOC publishes
the shared-memory name, run/frame IDs, generation, sequence, slot, shape, dtype,
byte order, nanosecond timestamp, and the two 32-character halves of its SHA-256
checksum. The designated consumer copies and verifies the frame, then writes
the exact sequence to `COMMAND:ACKNOWLEDGE`; a full ring rejects before the
camera is asked to expose. Disconnect unlinks the service-owned shared memory.
`--run-id` and `--acquisition-slots` make those per-process choices explicit.

This is a finite, per-frame lossless protocol, not continuous acquisition.
High-rate FCS and FLIR measurements demonstrate that live capture needs a
separate sequence-to-RAM capture owner, lossless chunked writer, and decimated
latest-wins preview lane. The proposed boundaries and measurable acceptance
gates are in [camera streaming and backpressure](docs/camera-streaming-backpressure.md).

The hardware test remains opt-in. It admits one exact real camera beside a
simulated peer and takes one unilluminated frame only when
`AHT_TEST_REAL_ACQUIRE=1` is explicitly set. The local PVCAM and SpinnakerC
profiles have both passed identity, checksum, acknowledgement, backpressure,
and cleanup gates; machine-specific evidence remains outside Git.

`aht camera-gui --prefix <CAMERA_IOC_PREFIX>` is an optional Qt client for one
already-running isolated camera service. It owns no MMCore/vendor SDK context;
one button asks the service to connect, arm, capture, checksum-verify, and
acknowledge a single frame before ordinary stop/disconnect, then displays the
detached copy. The UI may write no acquisition hardware configuration. Its
physical gate runs it offscreen beside an idle simulated peer, then persists the
detached frame as a distinct single-camera OME-Zarr bundle and runs the generic
mean-projection and quality-metrics workers. That is an acquisition-to-product
transport check, not a scientific DPCT/IDT reconstruction: the laser engine is
offline and no native-v1 illumination pattern cycle or calibrated stage pose
dataset is available.

Illumination follows the same isolation pattern. `MMCoreShutteredLightBackend`
owns one shutter-only Micro-Manager context, admits exactly one shutter device
with explicit label/adapter constraints, forces auto-shutter off, connects with
the shutter closed, and surfaces typed open/closed readback, optional
wavelength metadata, and an optional interlock property whose absence fails
admission. Every discovered device property is classified by one shared
generic rule set into boolean/enumerable/numeric/text kinds, and writes are
validated against that classification before reaching the core. Physical
light-engine admission, an illumination IOC, and the EPICS application device
remain pending; see `docs/mmcore-light.md` for the boundary and provenance.

Dual-camera registration uses one calibration plane per independent detector;
it never splits a detector image. The calibration recipe fits one Euclidean
OpenCV ECC transform from the moving detector into the reference-detector pixel
grid, optionally on downsampled planes, then applies that locked transform to
every later frame or stack. Detector IDs, direction, source/output shapes,
plane selection, downsample, interpolation, border policy, score, and matrix are
available as immutable artifact provenance. Calibration inputs must already
share a pixel grid; optical resampling or ROI selection is a separate explicit
calibration step. The array-only API can run in a normal processing worker or
behind the retained optional RedSun processing-device adapter.

The Phase 3 MCU host boundary uses a bounded CRC-16/CCITT-FALSE envelope,
COBS framing, correlated sequence numbers, typed errors, and idempotent retry.
It implements capabilities, transactional raw-8-bit pattern uploads,
commit-before-display, sequence definition/arming/start, `stop`, and `all_off`.
Semantic payloads and the deterministic firmware model live in `redsun_aht.mcu`;
the host and delimited serial adapters live in `redsun_aht.device.mcu`. The
serial adapter accepts an already-open pyserial-compatible port, so port
identity, baud rate, timeout, and ownership remain explicit deployment choices.
See `docs/mcu-protocol.md` for the v1 byte-level contract.

Exercise that same typed command set through a human-readable simulator console:

```console
uv run aht mcu-test --show-command-help
uv run aht mcu-test --led-count 4 \
  --command "get-capabilities" \
  --command "upload-pattern 17 0,32,64,255" \
  --command "show-pattern 17"
```

`--script COMMANDS.txt` reads the same grammar line by line, including blank
lines and `#` comments. The readable layer maps to `McuHostClient`; it does not
define an alternate firmware wire protocol or bypass capability, range,
transaction, retry, or typed endpoint-error handling. See
`docs/mcu-protocol.md` for the complete readable command table.

The currently connected Arduino donor firmware uses the restricted legacy
Micro-Manager two-digit command set and is not a conforming v1 endpoint. It has
not been contacted by this implementation. Watchdog, E-stop, safety-latch, and
safety-reset APIs are deliberately deferred from AHT; this package makes no
safety-subsystem claim. `stop` and `all_off` are deterministic operational
cleanup commands, not substitutes for an instrument safety system.

For the 61-LED simple panel, a strict, bounded native-v1 ATmega328P migration
is feasible only with fixed buffers and explicitly small map/sequence
capabilities. The 217-LED/motion target requires a higher-RAM controller; the
instrument decision records STM32F103C8 as the migration candidate. No Arduino
firmware has been flashed or contacted by AHT.

## Current package boundaries

- `domain`: immutable IDs, frames, poses, observations, solver state, processing
  results, calibration records, and run events.
- `protocols.py`: public detector, stage, motion-validation, processing, and
  remote-reconstruction extension seams.
- `device`: RedSun/ophyd-facing hardware adapters plus an optional processing
  device compatibility seam; concrete MCU host/serial adapters are in
  `device.mcu`, isolated Micro-Manager camera backends are in `device.camera`,
  the shuttered-light backend and shared property rules are in `device.light`,
  while shared MCU schemas/codecs remain in `mcu`.
- `motion`: hardware-neutral linear/list/grid/snake scan construction; concrete
  composite and ESP300 stage adapters remain under `device.stage`.
- `transport`: schema-versioned MsgPack service envelopes.
- `processing`: hardware-free replay, reference-only MsgPack messages,
  independently supervised worker processes, CPU/GPU quantitative reference
  kernels, the offline-only multi-layer Born numerical core, and the replacement
  defocus-diverse multislice/MSBP port with MATLAB dataset adapters.
- `storage`: immutable run manifests, append-only event journals, and durable
  OME-Zarr bundles.
- `catalog`: optional Tiled registration, source-run queries, external Zarr
  access, and verified source replay.
- `configurations`: inspectable hardware-free build/run factories.
- `providers.py` and `streams.py`: typed container keys and canonical event
  stream names.

See `docs/component-disposition.md` for the initial depend/upstream/port/rewrite
ledger.
