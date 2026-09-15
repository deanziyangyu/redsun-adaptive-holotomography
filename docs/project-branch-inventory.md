# Project and branch inventory

Inventory date: 2026-09-15

This inventory records the repository cleanup performed after adopting RedSun
`0.13.0rc0`. A branch was removed only when it was merged, superseded by tested
code, or retained on an explicit donor remote. Canonical unmerged feature
branches were not modified.

## Active application

Repository: `deanziyangyu/redsun-adaptive-holotomography`

- `feat/base_devices` contained the active application history. Its older name
  and commits originated on RedSun `0.11.0`, but the final tree now pins
  `redsun[epics,zarr]==0.13.0rc0` and `ophyd-async==0.21.2`.
- Commit `4a007d4` adopts RedSun's service layer, ordinary ophyd-async EPICS
  devices, service/device-owned multidimensional storage, and standard Bluesky
  external assets.
- Commit `0ed6942` records the independent offline RI-analysis work rather than
  leaving it mixed into the RedSun migration worktree.
- The former `hw/aht-redsun-integration` branch ended at `3c6888b` and depended
  on an editable RedSun checkout in the `0.12` line. It was ancestry-merged with
  the `ours` strategy at `64fcbe4`, preserving its hardware-development record
  without restoring obsolete code, and its branch name was deleted.
- After this inventory is committed, `feat/base_devices` is to be
  fast-forwarded into `main` and both its local and remote branch names removed.

## RedSun framework checkout

Repository: canonical `redsun-acquisition/redsun`, with
`deanziyangyu/redsun` as the writable fork.

- Local and fork `main` were fast-forwarded from `aa4a587` (v0.12.2) to
  `0146e27` (the v0.13.0rc0 service release candidate).
- `feat/upstream` at `1a1879f` implemented the superseded EPICS
  service-device/shutdown design. Its local and fork branch names were deleted.
- `feat/pre-upstream` at `03009a5` stacked the superseded RedSun-owned
  multidimensional storage contract. Its local and fork branch names were
  deleted.
- The clean `redsun-hardware` worktree and its local
  `hw/aht-redsun-integration` branch were removed.
- Canonical `upstream/feat/experimental-di` remains available for emerging
  design reconnaissance and was not modified.

The deleted fork commits remain identifiable by the hashes above and are
normally recoverable from local reflogs until Git expires unreachable objects.
They should not be restored as application dependencies.

## RedSun Mimir reference checkout

Checkout directory: `redsun-pyhololab`

- Local `main` now tracks canonical `redsun-acquisition/redsun-mimir` at
  `75149be`.
- The local `refactor/pyhololab-integration-OA` branch was deleted because it
  targeted RedSun `0.11` and was explicitly archival.
- `donor/refactor/pyhololab-integration-OA` and
  `donor/refactor/upstream` remain on the stale personal fork solely as feature
  donors.
- Canonical `origin/feat/mmcore-upstream-devices` is unmerged emerging work and
  remains untouched. The already-merged motor-readback remote is also left to
  the canonical maintainers.

## Pyhololab reference checkout

Repository: `YipLab/pyhololab`

- Local `main` was fast-forwarded to `ec7285f`.
- Local `feat/3-live-recon` and the merged remote branches
  `1-basic-feature-bring-up`, `dev/sys/arh-development`,
  `feat/2-migration-to-qtpy-backend`, and `feat/3-live-recon` were deleted after
  ancestry checks proved all were contained by `origin/main`.
- The unrelated untracked `led_pos_ring3.h5` file was preserved.

## ARH parent repository

- `feat/experimental-spatial-solver` is active, tracks its remote exactly, and
  is not merged into `main`; it was retained.
- The parent worktree contains unrelated embedded-controller edits and nested
  repositories, so no parent commit or branch rewrite was performed.

## Dependency audit

The active AHT dependency specification and lock resolve RedSun
`0.13.0rc0`; there is no editable `../redsun` source override and no RedSun
`<0.13` constraint. Occurrences of version `0.11.0` in the lock belong to
unrelated packages such as `culsans` and `obstore`. Historical documents may
mention RedSun 0.11/0.12 when describing superseded designs, but those are not
runtime dependencies.
