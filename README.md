# Redsun Adaptive Holotomography

Redsun Adaptive Holotomography (AHT) is the acquisition application for an adaptive
multimodal holotomographic microscope. It is built on
[Redsun](https://github.com/redsun-acquisition/redsun) 0.16 and serves every device over
PVAccess: there is no Channel Access client, server or dependency in this package.

The application is being rebuilt on the basis of the IX83 imaging bundle. The acquisition
core is new; the scientific reconstruction methods and the historical datasets are handled
separately and are not imported by the acquisition runtime.

## Status

What is implemented and exercised by the headless suite:

- editable Micro-Manager camera and stage services over PVA, with applied readbacks,
  a settings revision, setting-change and frame-latency records, and named axes with
  limits, precision and stop;
- a live stream routed through the run's own Event documents, so what a viewer shows is
  part of the sequence that can be traced, replayed and written;
- a tested finite-run storage component that writes one store per plane with a checksummed
  manifest and typed refusals for partial runs, missing frames and non-empty destinations;
- a four-pattern software DPCT path that takes fresh free-running frames after
  each LED pattern and writes a separate checksummed pixel store per detector;
- a MM-compatible Arduino illumination service over `oxiserial`, with an uncertain outcome
  reported rather than resent.

The acquisition page, detector selection controls, image view and DPCT plan entry are
present but have not had a host GUI acceptance run. The `AHT-MM-ASCII-91/1` sketch
supplies the two-digit command dialect without camera TTL. Optical pattern shape and
camera association require instrument qualification. The Arduino service requires
the sketch's `90` identity response before serving commands. Live camera capture
now writes a checksummed manifest through `RunFrameStore`; its run and detector
identity are supplied when assembling a receipt. The software DPCT plan binds
those identities when it opens its own store. No full hardware DPCT run has been
accepted yet.

## Install

```console
uv sync --extra all --locked
```

## Run

```console
pwsh -NoProfile -File scripts/run_session.ps1 -Check   # resolve and prove writable roots
pwsh -NoProfile -File scripts/run_session.ps1 --list   # sessions and services
pwsh -NoProfile -File scripts/run_session.ps1 service mmcore-camera -- --prefix AHT-CAM1:
pwsh -NoProfile -File scripts/run_session.ps1 session camera
```

`aht` is the direct entry point; the PowerShell wrapper resolves the repository, picks a
writable data root and a short temporary root for PVA, and exports the cache and loopback
settings a session needs.

## DPCT plan

`redsun_aht.plans.dpct_software.software_dpct_plan` is the session's DPCT action.
It selects `11`, `12`, `13`, `14` in order, reads a fresh frame from each free-running
camera after each pattern, and stores the pixels with their pattern identity. It
closes with `00`; a short run retains a partial manifest. The older
`redsun_aht.plans.dpct.dpct_plan` is an external-trigger contract and still refuses
the current camera.

## Development

```console
pwsh -NoProfile -File scripts/dsh/run-tests.ps1 tests -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src tests
```

The runner keeps every temporary tree inside the checkout, refuses to start under an
elevated token, and deletes the DACL-bearing directories after each run. Do not run the
suite elevated.

## Licence

Apache-2.0. Adapted source is attributed in `NOTICE` and in each ported module's docstring.
