# RedSun framework gap report

Status: the EPICS/device-lifecycle candidate is published for review on the
personal RedSun fork and tracked by canonical RedSun
[issue #107](https://github.com/redsun-acquisition/redsun/issues/107). No pull
request has been opened; the branch is waiting for owner review.

AHT compatibility baseline: RedSun 0.11.0. The upstream candidate is based on
canonical RedSun v0.12.2. Reinspect the current upstream revision before
implementing or filing each additional candidate.

## Branch reconnaissance

Branch inspection is a required planning step before each upstream issue or
implementation. The 2026-09-07 scan found:

- RedSun `main` at the v0.12.2 line still has no public EPICS service-device
  base and retains the sequential 2-D storage contract described below.
- RedSun `feat/experimental-di` contains a substantial experimental
  session/container, frontend, registry, wiring, layout, and headless
  composition redesign. Container, offline-builder, strict-build, and Qt-shell
  proposals must be reconciled with this branch before an upstream API is
  proposed.
- RedSun `feat/log-view` is narrowly focused on log presentation and does not
  resolve the AHT device or storage gaps.
- Canonical redsun-mimir `main` is the source reference (v0.4.1 at the time of
  inspection) and depends on the RedSun 0.12 line.
- Canonical redsun-mimir `feat/mmcore-upstream-devices` is an unmerged MMCore
  device expansion covering cameras, serial devices, shuttered/state devices,
  stages, filter-wheel presentation, providers, and wiring. It was three
  commits ahead and nineteen commits behind canonical `main` when inspected,
  so requirements should be harvested from it but integration assumptions
  must be rebased and retested.
- Canonical `feat/motor-readback-subscription` is already contained by `main`;
  it is useful history, not an emerging unmerged feature.
- The stale `deanziyangyu/redsun-pyhololab` fork and its
  `refactor/pyhololab-integration-OA` / `refactor/upstream` branches are donor
  material only. They are not the redsun-mimir source of truth.

Repeat this scan when work begins because remote branches are emerging design
inputs, not stable dependencies. Record relevant commits in the issue/PR
review packet.

## EPICS service device and device teardown

### Observed gap

RedSun accepts any ophyd-async `Device` as a declared device and publishes a
`HasAsyncShutdown` protocol, but it does not provide a public EPICS service
device base. `AppContainer.shutdown()` currently shuts down presenters but does
not invoke `HasAsyncShutdown` devices.

AHT consequently derives `CameraServiceDevice` directly from ophyd-async's
`EpicsDevice`, and its service stop/disconnect lifecycle is outside RedSun
container ownership.

### Proposed generic API

- Add a public `redsun.device.epics.EpicsServiceDevice` abstract class derived
  from ophyd-async `EpicsDevice` and RedSun's asynchronous shutdown contract.
- Let the base own the stable EPICS prefix/name contract while subclasses
  declare application-specific PVs and implement shutdown.
- Make container shutdown stop presenters first, then await shutdown of every
  device implementing `HasAsyncShutdown`; one failure must not skip remaining
  cleanup and must be reported to the caller/log.

### Acceptance criteria

- EPICS devices remain valid RedSun plugin devices and support ophyd mock
  connection.
- Container shutdown invokes each asynchronous device exactly once.
- Presenter shutdown precedes device shutdown.
- Partial failure still cleans the remaining devices.
- AHT can derive camera and future illumination adapters from the public base
  without a local compatibility abstraction.

### AHT impact

The dedicated AHT hardware branch (`hw/aht-redsun-integration`) consumes the
candidate explicitly from the sibling `redsun` checkout on `feat/pre-upstream`.
It routes the application-side lossless detector through the RedSun
`EpicsServiceDevice` and ophyd-async while retaining Caproto inside the IOC.
This is review evidence, not a production dependency: promotion still requires
an accepted upstream API and release or an explicit approved pin.

### Local upstream review candidate

A candidate is implemented in the sibling RedSun checkout on branch
`feat/upstream`, based on the inspected v0.12.2 canonical `main` line. The
checkout uses `deanziyangyu/redsun` as `origin` for review branches and
`redsun-acquisition/redsun` as `upstream`.
It contains:

- optional `redsun[epics]` / `ophyd-async[ca]` dependency metadata;
- `redsun.device.epics.EpicsServiceDevice` with RedSun's positional component
  identity and explicit EPICS service prefix;
- a container shutdown phase that stops presenters, awaits every structural
  `HasAsyncShutdown` device, continues after individual failures, and resets
  connection state; and
- focused EPICS and lifecycle tests.

The candidate is published at
[`deanziyangyu/redsun:feat/upstream`](https://github.com/deanziyangyu/redsun/tree/feat/upstream)
at commit `1a1879f`. Verification in the RedSun checkout: 562 tests passed,
statement coverage was 96%, Ruff and formatting checks passed, and Mypy passed
against both PyQt6 and PySide6. RedSun's hosted Codecov upload runs for `main`
or pull-request CI, so it will not run from the fork branch push alone.

Manual review should decide whether shutdown errors remain logged or are
aggregated for callers, whether the EPICS extra belongs in core, and how this
change should be rebased onto `feat/experimental-di`. AHT must not copy this
unmerged API locally.

## Multidimensional durable storage

### Observed gap

RedSun 0.11 `StreamSpec` describes sequential two-dimensional frames and
`OpenStore.write()` receives only a data key and array. `FrameSink.put()`
acknowledges queue admission, not durable persistence. The API cannot express
DPCT's detector/pattern/scan coordinates, per-frame provenance, readback
checksum, append-only commit record, or atomic complete/abort publication.

The bundled acquire-zarr backend writes time/y/x arrays and therefore cannot
preserve AHT's detector-separated CZYX contract without hidden conventions.

### Required generic capabilities

- Named multidimensional axes and deterministic frame placement.
- Per-frame metadata or an opaque write context that reaches the backend.
- An awaitable persisted-write receipt distinct from queue admission.
- Backend completion and abort hooks for transactional manifests/journals.
- Resource/datum metadata describing the real array path, index, shape, dtype,
  and chunk layout.

### Local upstream prototype

The RedSun `feat/pre-upstream` working branch at `03009a5` now contains an additive
`MultidimensionalOpenStore` protocol candidate. Its `IndexedWrite` carries a
named `FramePlacement`, immutable opaque provenance context, and source
checksum; `write_indexed()` returns a `PersistedWrite` receipt with the
backend-observed checksum and URI. The protocol also requires mutually
exclusive `complete()` and `abort()` terminal operations. It does not change
the sequential `OpenStore.write()` or require the existing Mimir storage
backends to migrate.

This is a deliberately narrow public-contract prototype, not an AHT cutover:
it does not yet include a Zarr implementation, coordinate-schema declaration,
event-model resource/datum representation, or replay proof. The existing
AHT DPCT Zarr store remains authoritative until a concrete backend satisfies
the acceptance criteria below. The prototype's focused and full RedSun test
suites pass; it requires owner review before promotion to `feat/upstream`, an
issue, or a pull request.

### Acceptance criteria

- Existing sequential frame users remain source-compatible or have a bounded
  migration path.
- A DPCT backend can implement all existing durability and replay tests without
  side channels in acquisition code.
- Detector acknowledgement occurs only after backend persistence and readback
  verification.
- Abort never publishes a completion manifest.

### AHT impact

DPCT Zarr remains on the current implementation until the upstream contract is
accepted. No new acquisition path may weaken the existing durability contract
to fit the current RedSun storage API.

### Issue-ready evidence

The current AHT regression contract provides the reproducer for a future
storage issue: one DPCT shot is
indexed by detector, scan position, illumination pattern, and channel; the
camera frame is acknowledged only after Zarr persistence and checksum
verification; completion publishes a manifest only after all expected shots
exist; failure leaves an inspectable incomplete bundle. Attempting to map this
onto RedSun 0.11/v0.12.2 loses named axes, write context, persisted receipts,
and complete/abort semantics. These points and the existing
`tests/test_dpct_zarr.py` success, tamper, incomplete, and collision tests
should be attached to the manually created RedSun issue.

## Conditional gaps to evaluate

- A generic offline lifecycle builder is needed only if headless
  `AppContainer` cannot own detached worker startup and shutdown cleanly.
- A public Qt main-window factory/layout hook is needed only if customized AHT
  `QtView` components cannot provide the intended frontend inside the existing
  `QtAppContainer` shell.
- A strict/fail-fast container build option should be considered for physical
  profiles because RedSun 0.11 logs device construction failures and continues.

For each confirmed gap, prepare an issue-ready reproducer, proposed public
contract, compatibility analysis, tests, and AHT impact. Decide issue and PR
grouping after branch inspection and obtain owner approval before publishing a
new issue or opening a pull request.
