# Redsun Adaptive Holotomography

Redsun Adaptive Holotomography (AHT) is the acquisition application for an adaptive
multimodal holotomographic microscope. It is built on
[Redsun](https://github.com/redsun-acquisition/redsun) and serves every device over
PVAccess: there is no Channel Access client, server or dependency in this package.

AHT includes acquisition plans, run-document routing, and frame storage.
Instrument profiles select the devices and services for each setup.

| Selector | Packaged profile | Device stack |
|---|---|---|
| `holospim` | [holospim.yaml](src/redsun_aht/profiles/holospim.yaml) | MMCore camera and survey stage, plus ESP32-S3 MCU for X28 fast Z, LED and rotation |
| `arh` | [arh_system.yaml](src/redsun_aht/profiles/arh_system.yaml) | ASI LX4000 XYZ, FLIR via Spinnaker, Lumencor Spectra via MMCore, and Arduino Due MCU |

Both profiles use the same AHT installation. Keep host-specific settings in
an overlay on each instrument machine.

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

The headless suite exercises synthetic and fake-device paths. Physical camera,
illumination, stage and trigger behavior require qualification for each setup.
See the [architecture and capability notes](docs/index.md).

## Install

```console
uv sync --locked
# For the camera GUI and MMCore support:
uv sync --locked --extra all
```

## Run

On Windows, use the [PowerShell launcher](scripts/README.md).

```powershell
pwsh -NoProfile -File scripts/run_session.ps1 -Check
pwsh -NoProfile -File scripts/run_session.ps1 --list
pwsh -NoProfile -File scripts/run_session.ps1 session profile arh --check
pwsh -NoProfile -File scripts/run_session.ps1 session profile holospim --check
```

The profile checks build mock devices and shut down without launching services.
Pass `--overlay <path>` to layer host settings over the packaged profile.
`--check` forces mock mode even if an overlay sets `mock: false`. The direct CLI
uses a separator before profile arguments, for example
`aht session profile -- arh --check`.

For host deployment, install the `camera-mmcore` extra and the matching
Micro-Manager adapters used by the profile. ARH also needs the vendor PySpin
wheel for its FLIR service. The `profile` command builds a headless device
session. The `camera` and `dpct` commands are separate GUI entry points.

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
