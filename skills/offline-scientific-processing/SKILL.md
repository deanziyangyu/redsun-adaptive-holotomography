---
name: offline-scientific-processing
description: Run a hardware-free AHT reconstruction or replay from recorded data and report exactly what the result establishes and what it does not.
---

# Offline scientific processing

## Scope

Offline DPCT or linear IDT reconstruction without an instrument session or device handle. Use a
declared-axis TIFF plus a reconstruction configuration. Read [AGENTS.md](../../AGENTS.md), [the
handbook](../../AGENTS.md) and [headless testing](../../docs/index.md#testing) first.

## Preconditions

- A `.venv` that imports `redsun_aht`; record interpreter and resolved versions.
- Identified inputs include the TIFF axes, channel order, optical settings and
  matching reference when background normalization is enabled.
- No acquisition session writing the same output directory.

## Procedure

Use the CLI, not an ad-hoc script:

```powershell
.\.venv\Scripts\python.exe -m redsun_aht reconstruct-tiff --solver dpct `
  --sample <sample.tif> --reference <background.tif> --config <recon.yaml> `
  --output <result.tif> --backend numpy
```

See the [offline DPCT documentation](../../docs/index.md#offline-dpct-reconstruction)
and [linear IDT documentation](../../docs/index.md#offline-linear-idt-reconstruction)
for NumPy and CuPy, ROI selection, reference handling, calibration and zero-pixel policy.

## Refusal conditions

Refuse, naming the failed precondition, rather than improvising:

- **Missing calibration** — a required reference, illumination table, voxel size,
  wavelength or NA was not supplied. Do not substitute another dataset's value or
  an instrument default.
- **Partial or unverified dataset** — an illumination channel is missing or
  the TIFF axes are undeclared. Report which inputs are missing.
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
