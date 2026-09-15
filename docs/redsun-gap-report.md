# RedSun framework gap report

Status: RedSun `0.13.0rc0` resolves the EPICS service-lifecycle gap through a
container-owned service layer and ordinary ophyd-async devices. The earlier
EPICS/device-shutdown candidate on the personal fork is superseded and should
not be proposed for merge. AHT now validates the released-candidate design in
simulation and on the available FLIR hardware.

A historical issue/merge description remains in
[`redsun-feat-upstream-merge-note.md`](redsun-feat-upstream-merge-note.md), but
it no longer represents the recommended upstream change.

AHT validation baseline: RedSun `0.13.0rc0` with ophyd-async `0.21.2`.
Reinspect the current upstream revision and emerging branches before
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

### Resolution in 0.13.0rc0

RedSun now declares services separately from devices, launches services before
building dependent devices, injects the configured service prefix, and closes
launched subprocesses during container teardown. AHT therefore uses
ophyd-async's EPICS device directly and does not require a RedSun EPICS
subclass, `service_prefix` property, or device-level `shutdown()` protocol.

The service owns MMCore and the vendor SDK. The ophyd-async device is the
application abstraction over its EPICS interface. Hardware-specific standby
or resume signals remain optional IOC/device features rather than a generic
EPICS lifecycle contract.

Validation covers a RedSun-launched simulated camera IOC and the physical FLIR
camera. In both cases the service/device composition produced external assets
and container shutdown stopped the launched service. The AHT implementation
also supports exact retained-frame acknowledgement only after durable write
and checksum verification.

### Superseded local upstream review candidate

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

The candidate remains published for history at
[`deanziyangyu/redsun:feat/upstream`](https://github.com/deanziyangyu/redsun/tree/feat/upstream)
at commit `1a1879f`. Verification in the RedSun checkout: 562 tests passed,
statement coverage was 96%, Ruff and formatting checks passed, and Mypy passed
against both PyQt6 and PySide6. RedSun's hosted Codecov upload runs for `main`
or pull-request CI, so it will not run from the fork branch push alone.

Do not merge or copy this candidate into AHT. RedSun `0.13.0rc0` addresses the
underlying need with the maintainer's service model and intentionally leaves
device-level shutdown out.

## Multidimensional durable storage

### Historical limitation and validation result

RedSun 0.11 `StreamSpec` describes sequential two-dimensional frames and
`OpenStore.write()` receives only a data key and array. `FrameSink.put()`
acknowledges queue admission, not durable persistence. The API cannot express
DPCT's detector/pattern/scan coordinates, per-frame provenance, readback
checksum, append-only commit record, or atomic complete/abort publication.

The bundled acquire-zarr backend writes time/y/x arrays and therefore cannot
preserve AHT's detector-separated CZYX contract without hidden conventions.

RedSun `0.13.0rc0` now supplies the missing service lifecycle and composes
ordinary ophyd-async devices over launched or attached services. The storage
shim remains, but the relevant open question is now whether a service-owned
writer can bypass it and emit standard Bluesky external-asset documents
through the RedSun RunEngine without loss. That path now passes in
hardware-free, RedSun-launched-service, and FLIR-only hardware tests. A RedSun
multidimensional storage API gap is not established.

### AHT service/device capabilities

- Named multidimensional axes and deterministic frame placement.
- Per-frame metadata or an opaque write context that reaches the backend.
- An awaitable persisted-write receipt distinct from queue admission.
- Backend completion and abort hooks for transactional manifests/journals.
- Resource/datum metadata describing the real array path, index, shape, dtype,
  and chunk layout.

These are requirements for AHT's service/device-owned writer, not proposed
RedSun public classes unless a framework-level reproducer proves otherwise.

An initial protocol prototype is available on
`deanziyangyu/redsun:feat/pre-upstream` at commit `03009a5`, but maintainer
feedback superseded its RedSun-owned storage direction. The current pre-issue
validation draft is maintained in
[`redsun-multidimensional-storage-issue-note.md`](redsun-multidimensional-storage-issue-note.md).
It records the service/device-owned design and the standard Bluesky
external-asset path that must be validated before filing an issue. The
older [`redsun-multidimensional-storage-pr-note.md`](redsun-multidimensional-storage-pr-note.md)
is retained as design history only.

### Validation evidence

- A service-backed ophyd-async device can participate without registering a
  RedSun frame sink or storage backend.
- Standard Resource/Datum documents pass through the RedSun RunEngine with
  explicit non-sequential logical placement and checksum-resolvable assets.
- A DPCT service/device writer can implement all existing durability and replay
  tests without RedSun-specific frame-ingress APIs.
- Detector acknowledgement occurs only after backend persistence and readback
  verification.
- Abort never publishes a completion manifest.
- Non-linear placement and actual motor readbacks remain associated with the
  correct external frame.
- Generic `bluesky-tiled-plugins.TiledWriter` creates the run and external
  nodes but cannot directly read the DPCT array: its legacy normalizer reduces
  Datum placement to sequential StreamDatum ranges and its generic Zarr
  consolidator registers the group URI with an event-stacked shape. This is an
  AHT/Tiled adapter gap, not a RedSun composition or storage gap.

### AHT impact

DPCT Zarr remains device/service-owned. No new acquisition path may weaken the
existing durability contract to fit the current RedSun storage shim. The internal
`DimensionSpec`/asset-spec, indexed-write, placement, receipt, completion, and
abort concepts may be reused in AHT without making them RedSun APIs.

### Disposition

The current AHT regression contract supplies the application-side acceptance
case: one DPCT shot is
indexed by detector, scan position, illumination pattern, and channel; the
camera frame is acknowledged only after Zarr persistence and checksum
verification; completion publishes a manifest only after all expected shots
exist; failure leaves an inspectable incomplete bundle. Attempting to map this
onto RedSun 0.11/v0.12.2 loses named axes, write context, persisted receipts,
and complete/abort semantics. The `0.13.0rc0` reproducer now proves the
service-owned ophyd-async/Bluesky path. Do not file a RedSun multidimensional
feature issue. Implement a format-aware AHT/Tiled registration adapter, and
only consider a separate `bluesky-tiled-plugins` proposal if the resulting
group-component Zarr support and placement model are broadly reusable.

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
