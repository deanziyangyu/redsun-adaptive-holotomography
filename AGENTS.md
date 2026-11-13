# Agent notes

This repository is a public acquisition package. Keep its documentation and test
recipes reproducible on a clean machine: no machine-specific paths, hardware
serials, incident logs or internal programme references.

## Read first

- [docs/testing.md](docs/testing.md) — suite layout and marker groups.
- [docs/examples.md](docs/examples.md) — executable hardware-free examples.
- [scripts/dsh/README.md](scripts/dsh/README.md) — the repository-relative runner.
- [skills/README.md](skills/README.md) — repository-local agent skills.

Prefer `pwsh -NoProfile -File scripts/dsh/run-tests.ps1 <pytest args> -q` over an
ad-hoc `pytest` line; it uses `.venv`, a unique `tmp/` run root and the
environment the recipes require.

**Headless only.** An agent session must never construct a Qt `QApplication` or a
napari viewer, and must never run the GUI test modules or `scripts/*qt*.py`. The
excluded files and the compliance check are in [docs/testing.md](docs/testing.md).

## Tests and temporary files

Run tests, workers and development tools from a normal, non-elevated session.
Keep temporary output under the repository's writable `tmp/` directory and use a
fresh pytest `--basetemp` for each run. Set `TEMP` and `TMP` to a writable
task-local location when a test needs a custom temp root, and never elevate a
test run or repair permissions as an agent.

## Hardware

- Documentation and tests never authorize connecting, moving, illuminating or
  flashing hardware.
- A read-only identity check is the maximum action without explicit maintainer
  authorization; motion, illumination and firmware changes need it in writing.
- Prefer simulation, then one real device alongside mocks. Characterize software
  timing before any hardware-triggered acquisition.

## Git

- Obtain manual approval before any remote push, branch creation or branch
  deletion.
- Never merge into `main` without manual approval for that specific integration.
- Keep experimental or stale documents out of a feature's staged changes unless
  their dependencies and relevance are verified.

## Architecture diagrams

After a tangible architectural change to component ownership, public interfaces,
lifecycle or data flow, update the repository's authoritative architecture
diagram in the same change, and keep adjacent prose consistent with the code.
Label incomplete paths explicitly rather than implying they work.

## Where the environment specifics live

Host recipes, environment-specific behaviour and incident records are maintained
outside this repository. Do not copy them here; reference the public contract instead, and
keep the run evidence with the offline record.
