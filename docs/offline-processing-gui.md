# Offline processing GUI

The processing GUI is an optional Qt/Napari application for replaying a
completed, verified DPCT bundle. It is deliberately separate from acquisition:
launching it constructs no detector, stage, MCU, camera IOC, or MMCore process.

## Install and launch

Choose exactly one supported Qt binding:

```console
uv sync --extra pyqt --locked
# or
uv sync --extra pyside --locked
```

Create a hardware-free sample bundle when needed, then launch the application:

```console
uv run aht simulate-dpct --output RUN_DIRECTORY
uv run aht process-gui --input RUN_DIRECTORY --output PRODUCT_DIRECTORY
```

The paths are optional launch arguments and can instead be selected in the
form. The product directory may not equal, or be nested inside, the immutable
input run.

## Typed planning boundary

`OfflineProcessingRequest` normalizes and validates the input/output paths,
optional GPU placement, optional `OfflineTiledJobRequest`, and optional
`OfflineMultiLayerJobRequest`.
`resolve_offline_processing` integrity-replays the source bundle and creates
immutable `ProcessingJob` objects for the projection, quality, optional tiled
quantitative, and optional nonlinear multi-layer pipelines. The GUI preview is
rendered from that resolved plan; it is not parsed back into application state
and is never executed through a shell.

The copyable command is quoted for the active platform and is equivalent to:

```console
aht process-headless --input RUN_DIRECTORY --output PRODUCT_DIRECTORY
```

Selecting the GPU backend adds an explicit GPU ID and whole-MiB VRAM
reservation. Normal GPU discovery, admission, and persistent CuPy worker rules
still apply.

Enabling **Optional tiled quantitative job** adds explicit controls for the
DPCT/qOBT model, sample and optional background detector, optical constants,
illumination angles, background convention, regularization, and XY tiling. The
previewed command includes every value needed to reproduce the typed job. The
quantitative group can independently select its CuPy backend, GPU ID, and VRAM
reservation. This selection does not force the separate mean projection onto
CUDA; if both GPU workers are enabled, normal exclusive admission requires
different GPU IDs.

Enabling **Optional nonlinear multi-layer job** adds multi-layer Born or
multislice selection, one detector and acquisition Z plane, explicit output
depth/voxel geometry, optics, illumination NA, normalization, iterative
optimization, regularization, pupil recovery, CPU memory budget, and an
independent CuPy placement. Background normalization requires an explicit
second detector and background Z plane. The preview again contains every
resolved typed value.

## Execution and presentation

Run submits the resolved plan on the Qt global thread pool so the user
interface remains responsive. The presenter uses the same detached supervisor
as `process-headless`. For every successful product it resolves the local Zarr
URI, reads the published array, and verifies its SHA-256 checksum before adding
an image layer named `<source-run-id>:<solver-id>` to Napari. Stable output URIs
and per-solver failures remain visible in the text panel.

The base reference solvers are a three-dimensional mean-projection preview and
a small quality-metrics array. An enabled tiled quantitative job adds a ZYX
refractive-index product from the donor-characterized NumPy reference adapter.
An enabled nonlinear job adds a separately identified ZYX RI product and its
checksum-bound loss/restart/pupil provenance.
Napari image display is useful for workflow inspection, but it is not a
production scientific-validation claim.

## Verification boundary

Automated tests exercise typed planning, Windows and POSIX command quoting,
invalid placement/resource rejection, real detached CPU execution, checksum
verification, per-job observation selection, CLI dispatch, and the Qt widget
with an offscreen fake viewer.
Creating a real Napari canvas additionally requires a working OpenGL context;
headless Windows service sessions may not provide one even when the Qt widget
itself is fully functional.
