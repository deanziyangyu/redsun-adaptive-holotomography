---
name: offline-scientific-processing
description: Run a hardware-free AHT reconstruction or replay from recorded data and report exactly what the result establishes and what it does not.
---

# Offline scientific processing

## Scope

Reconstruction, replay and RI analysis with no session, device handle or network: a
verified DPCT bundle on disk, or a declared-axis TIFF plus a reconstruction
configuration. Read [AGENTS.md](../../AGENTS.md), [the
handbook](../../AGENTS.md) and [headless testing](../../docs/index.md#testing) first.

## Preconditions

- A `.venv` that imports `redsun_aht`; record interpreter and resolved versions.
- Identified inputs: the bundle or TIFF, its manifest or declared axes, and the
  reference/illumination/calibration the solver needs; digest anything untracked.
- For `--backend cupy`, a visible CUDA device and a repository-local cache
  ([runner](../../scripts/dsh/README.md)); NumPy implies no GPU.
- No acquisition session writing the same output directory.

## Procedure

Use the CLI, not an ad-hoc script:

```powershell
# detached solvers over a verified DPCT bundle
.\.venv\Scripts\python.exe -m redsun_aht process-headless --input <bundle> --output <dir>

# direct TIFF reconstruction, then GPU only when the device is identified and free
.\.venv\Scripts\python.exe -m redsun_aht reconstruct-tiff --solver msbp `
  --sample <sample.tif> --config <recon.yaml> --output <result.tif> --backend numpy

# replay a completed flyer run record, then analyze the RI product
.\.venv\Scripts\python.exe -m redsun_aht process-flyer-replay --input <run> --output <dir>
.\.venv\Scripts\python.exe -m redsun_aht analyze-ri --input <ri.tif> --output <report>
```

Add `--backend cupy --gpu-id <n> --gpu-reservation-mib 512` to `reconstruct-tiff` only
with a visible device. Omit `process-flyer-replay --tiled-uri` unless catalog
registration is in scope: it contacts a Tiled server, which an offline run must not do.

## Refusal conditions

Refuse, naming the failed precondition, rather than improvising:

- **No GPU** — `--backend cupy` with no device, driver or cache. Do not fall back
  to NumPy and report the run as a GPU result.
- **Missing calibration** — a required reference, illumination table, voxel size,
  wavelength or NA was not supplied. Do not substitute another dataset's value or
  an instrument default.
- **Partial or unverified dataset** — the manifest does not verify, a detector or
  shot is missing, or the TIFF axes are undeclared. Report partial with the
  missing identities; do not interpolate to completion.
- **Unidentified inputs** — a path with no provenance, or a digest that changed
  since it was described.

## Verification and evidence

- The command exits 0 **and** writes the declared products; a zero exit with no
  product is a failure.
- Re-running on the same inputs reproduces the product within the stated
  tolerance; say whether determinism was checked.
- The guard tests pass with their actual count, e.g.
  `pwsh -NoProfile -File scripts/dsh/run-tests.ps1 <test-module> -q`.
- Record: command and working directory; interpreter and resolved environment;
  solver and backend used; product names, shapes and dtypes with the
  configuration that produced them; the tolerance basis with its numbers.

## What this skill must not be used to claim

- That hardware, an acquisition plan or a live path works; offline replay
  exercises solvers, not devices.
- That a CPU result qualifies the GPU path, or a cropped run qualifies
  full-volume reconstruction quality.
- Scientific acceptance, publication validity or radiometric accuracy.
- That extraction promotes a method's evidence class.
