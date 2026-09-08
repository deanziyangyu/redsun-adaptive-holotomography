# Multi-layer reconstruction core

Status: implemented hardware-free Phase 4 reference pipeline. Numerical cores,
offline iterative jobs, worker-owned NumPy/CuPy execution, provenance, memory
preflight, cancellation, and CLI/GUI controls are operational.

## Provenance and scope

The multi-layer Born equations in `redsun_aht.processing.multilayer` were
adapted from the pyhololab `src/core/multilayer/models.py` donor at exact commit
`e8dd96bfa2d54784e00209769d594f305863b01a`, audited on 2026-08-20. The
`multislice` identity now refers exclusively to the newer MATLAB MSBP method
described below; the former pyhololab multislice implementation was removed.

The AHT-owned port currently provides:

- multi-layer Born and replacement multislice forward fields on explicit ZYX
  geometry;
- native-object reverse passes and multi-layer Born pupil derivatives;
- amplitude, intensity, and complex-field loss domains;
- ideal or supplied pupil handling, with pupil recovery limited to multi-layer
  Born;
- refractive-index/native-object conversion and physical-sign projection;
- deterministic ring illumination frequencies; and
- conservative cache and working-set estimates.

## Replacement defocus-diverse multislice port

The `multislice` model is a port of the newer MATLAB implementation in
`ut-cwo/Inverse-scattering-in-biological-samples-via-beam-propagation`. It
replaces the older multislice implementation and keeps the established public
model/worker identity. The supplied source files are dated 2025-08-16;
`CITATION.cff` records a 2026-01-31 release.

The port includes propagation-before-transmission slice ordering, truncated
FFT-grid illumination vectors, angle-dependent obliquity, volume-center and
image-plane propagation, amplitude or compensated-field losses, complex-field
digital refocusing at multiple focus offsets, zero-based skipped-shot handling,
seeded per-epoch angle shuffling, first-slice baseline subtraction,
non-negativity, FISTA/restart, relative-cost stopping, and the Rytov initializer.
The local Chambolle 3-D TV proximal operator replaces the MATLAB UNLocBoX
`prox_tv3d` dependency.

For CuPy execution, the persistent per-layer complex field cache is stored as
paired `float32` real and imaginary arrays. Complex arrays are assembled only
where CuPy FFT operations require them. This retains the MATLAB single-precision
storage contract without depending on a persistent `complex64` volume.

The input acquisition mode is explicit in the reconstruction contract. The
`aht_real_amplitude_focal_stack` mode describes the AHT instrument: real-valued
amplitudes are acquired at physical focal positions along the observation Z
axis, so numerical refocusing is disabled. The
`interferometric_complex_field` mode describes the supplied donor setup: both
field components are required and the complex field may be digitally refocused.
The canonical OME-Zarr schema records this mode and its component-array paths;
the HDF5 bridge selects the interferometric mode explicitly. Physical AHT focal
stack ingestion is the next migration phase and is not inferred from an
unlabelled array.

`redsun_aht.processing.msbp_matlab` reads both classic and v7.3 paper datasets.
It applies the MAT file's one-based ROI metadata and exposes `crop_shape_yx`,
`crop_start_yx`, and `max_shots` so a later validation pass can use a small,
aligned problem. It also extracts the supplied v7.3 `gpuArray` reconstruction,
removes MATLAB padding, and applies the same crop. Classic compressed MAT files
must be decompressed by SciPy before cropping; v7.3 inputs are sliced directly
through HDF5. Install the bridge with the `matlab-msbp` extra.

The supplied `LICENSE` is BSD 3-Clause, while its `CITATION.cff` says MIT. This
repository follows and redistributes the actual BSD license text under
`docs/third-party/inverse-scattering-msbp-BSD-3-Clause.txt`, and records both
values in result provenance so the discrepancy is visible.

An offline bundle selects `--multilayer-model multislice`. `msbp` remains an
input alias only in the low-level model factory; it is not a third public model
or worker identity. Exact experimental
illumination frequencies may be supplied by repeating
`--multilayer-illumination-fxy FX FY`; otherwise the existing deterministic ring
generator is used. `--multilayer-focus-offset-slices` accepts the MATLAB focus
offsets, and `--multilayer-skip-shots` uses Python zero-based indices. Nonzero
focus diversity is available only for the explicitly tagged
`interferometric_complex_field` mode. AHT focal-stack data instead uses its
physical CZYX focal-position axis and does not undergo digital refocusing.

It preserves every physical Z layer. `slice_binning_factor` is present in the
frozen configuration as an explicit future compatibility point, but values
other than one are rejected.

## Characterization

The initial NumPy characterization uses a frozen 3 x 8 x 10 ZYX object, a
0.625 um wavelength, 0.8 detection NA, 1.33 medium index, asymmetric XY voxel
size, nonzero defocus and padding, and four illuminations at 0.45 NA.

The multi-layer Born model matches its pinned pyhololab donor for complex
forward arrays, amplitude losses, native-object gradients, and pupil
derivatives. Repo-owned replacement-multislice fixtures retain representative
complex fields, statistics, loss, and update values rather than importing the
MATLAB donor at test time. Its directional-difference test explicitly accounts
for the MATLAB `BPM_update` convention: reported cost is the full squared norm,
while the stored update omits the obliquity multiplier.

This characterization establishes faithful equations and internally
consistent adjoints only. It does not establish experimental reconstruction
accuracy, convergence, identifiability, calibration sufficiency, or production
dataset compatibility.

## Offline iterative worker

The detached allowlist contains four explicit solver identities:

- `multilayer-multi_born-numpy`;
- `multilayer-multislice-numpy`;
- `multilayer-multi_born-cupy`; and
- `multilayer-multislice-cupy`.

The replacement keeps `multilayer-multislice-numpy` and
`multilayer-multislice-cupy`, now at solver version 2.0.0. There are no parallel
legacy multislice worker identities.

One selected committed detector observation must have CZYX axes. The job
selects one acquisition Z plane and interprets C as the illumination-shot
axis, producing one reconstructed ZYX refractive-index volume whose depth and
voxel size are explicit configuration inputs. Input normalization is also
explicit: `shot_mean`, `background`, or `none`. Background normalization
selects a second committed detector observation and one of its Z planes.

The iterative engine supports deterministic sequential or seeded random shot
ordering, gradient descent or FISTA, loss-increase restart, L2 and isotropic
3-D TV regularization and physical-sign projection. Optional global pupil
recovery remains available for multi-layer Born but is rejected for the MATLAB
multislice method because its donor does not implement that update. Cancellation
is checked at every iteration, illumination, and TV proximal iteration boundary.

The resolver computes a conservative working-set estimate before creating the
job. NumPy jobs must fit their explicit CPU memory budget. CuPy jobs must fit
their exclusive VRAM reservation, and the worker repeats the check against 80%
of free device memory after its CUDA context is created. The result manifest
and OME-Zarr attributes bind the source checksums, selected planes, complete
model/solver configuration, resolved illuminations, memory estimate, loss,
step and restart histories, pupil checksum, exact donor metadata, and output
checksum.

An explicit headless example is:

```console
aht process-headless --input RUN_DIRECTORY --output PRODUCT_DIRECTORY \
  --multilayer-model multislice --multilayer-detector dhm \
  --multilayer-depth-layers 32 \
  --multilayer-voxel-size-zyx-um 0.25 0.108333 0.108333 \
  --multilayer-wavelength-um 0.625 --multilayer-na 0.8 \
  --multilayer-illumination-na 0.45 \
  --multilayer-max-iterations 5
```

Selecting `--multilayer-backend cupy` additionally requires an explicit GPU
ID and whole-MiB VRAM reservation. The processing-only GUI exposes the same
typed job and renders all resolved values into its copyable command preview.

## Characterization boundary

The numerical core imports no device, acquisition, catalog, GUI, or worker
module. Its array namespace lets only the spawned worker own a CuPy context;
the supervisor performs NVIDIA discovery and lease admission without importing
CuPy.

Both nonlinear model identities passed detached CPU workflow tests. The
replacement multislice method was additionally exercised on the supplied
`040um_phantom_data.mat` with a 64 x 64 x 110 centered crop and 16 angles on the
Tesla P100. It stopped after 17 iterations, reduced loss from 8909.80 to
7730.65, produced finite non-negative delta RI, and used about 119 MB of the
CuPy pool. A second 32 x 32 x 110 run loaded all 500 angles, honored all 13
corrupted-angle exclusions, and stopped after 17 iterations in 454.9 seconds.
Its loss decreased from 69443.02 to 67029.44, its non-negative delta RI remained
finite with a 0.02623 maximum, and its CuPy pool used about 92 MB.

The supplied phantom reference file contains an all-zero `reconObj` and an
all-zero cost history, so it cannot serve as a quantitative ground-truth oracle.
The C. elegans FOV01 input was also exercised with all 250 angles, a centered
32 x 32 x 120 crop, all three focus planes, and three iterations. It completed
in 139.5 seconds, reduced loss from 50906.23 to 48886.68, and produced a finite
non-negative volume. Against the supplied cropped reference, this short run had
correlation 0.346 and RMSE 0.01035; it is not a substitute for the donor's full
38-iteration run. The current characterization establishes execution and short
run convergence on both real inputs, not production reconstruction accuracy.
The models remain offline-only and `live_supported` is false.
