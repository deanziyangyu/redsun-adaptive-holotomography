# Multi-layer reconstruction core

Status: implemented hardware-free Phase 4 reference pipeline. Numerical cores,
offline iterative jobs, worker-owned NumPy/CuPy execution, provenance, memory
preflight, cancellation, and CLI/GUI controls are operational.

## Provenance and scope

The backend-neutral forward/adjoint equations in
`redsun_aht.processing.multilayer` were adapted from the pyhololab
`src/core/multilayer/models.py` donor at exact commit
`e8dd96bfa2d54784e00209769d594f305863b01a`, audited on 2026-08-20. That
commit is explicitly a WIP checkpoint and contains no committed multi-layer
tests. It is behavioral and equation provenance, not production-validation
evidence.

The AHT-owned port currently provides:

- multi-layer Born and multislice forward fields on explicit ZYX geometry;
- exact reverse passes for native-object and pupil derivatives;
- amplitude, intensity, and complex-field loss domains;
- ideal or supplied pupil handling and projected pupil updates;
- refractive-index/native-object conversion and physical-sign projection;
- deterministic ring illumination frequencies; and
- conservative cache and working-set estimates.

It preserves every physical Z layer. `slice_binning_factor` is present in the
frozen configuration as an explicit future compatibility point, but values
other than one are rejected.

## Characterization

The initial NumPy characterization uses a frozen 3 x 8 x 10 ZYX object, a
0.625 um wavelength, 0.8 detection NA, 1.33 medium index, asymmetric XY voxel
size, nonzero defocus and padding, and four illuminations at 0.45 NA.

Both model implementations matched the pinned donor exactly for their complex
forward arrays, amplitude losses, native-object gradients, and pupil
derivatives. Repo-owned fixtures retain representative complex fields,
statistics, loss, and gradient values rather than importing the donor at test
time. Independent directional central differences agree with the implemented
adjoints to within 0.03% for multi-layer Born and 0.21% for multislice on that
fixture.

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

One selected committed detector observation must have CZYX axes. The job
selects one acquisition Z plane and interprets C as the illumination-shot
axis, producing one reconstructed ZYX refractive-index volume whose depth and
voxel size are explicit configuration inputs. Input normalization is also
explicit: `shot_mean`, `background`, or `none`. Background normalization
selects a second committed detector observation and one of its Z planes.

The iterative engine supports deterministic sequential or seeded random shot
ordering, gradient descent or FISTA, loss-increase restart, L2 and isotropic
3-D TV regularization, physical-sign projection, and optional global pupil
recovery. Cancellation is checked at every iteration, illumination, and TV
proximal iteration boundary.

The resolver computes a conservative working-set estimate before creating the
job. NumPy jobs must fit their explicit CPU memory budget. CuPy jobs must fit
their exclusive VRAM reservation, and the worker repeats the check against 80%
of free device memory after its CUDA context is created. The result manifest
and OME-Zarr attributes bind the source checksums, selected planes, complete
model/solver configuration, resolved illuminations, memory estimate, loss,
step and restart histories, recovered-pupil checksum, donor commit, and output
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

Both nonlinear models passed CPU-equivalence tests through real detached CuPy
workers on the installed Tesla P100 and Tesla P4 on 2026-08-20. This remains a
small synthetic workflow characterization. The models are offline-only,
`live_supported` is false, and no production instrument dataset, calibrated
illumination-frequency file, convergence study, or experimental ground truth
has yet passed.
