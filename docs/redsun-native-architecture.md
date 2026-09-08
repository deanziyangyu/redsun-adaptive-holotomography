# RedSun-native application architecture

## Requirement

`redsun-adaptive-holotomography` is a RedSun showcase application. Application
execution must enter through RedSun's public interfaces and schemas:

- `redsun.engine.RunEngine` owns Bluesky execution;
- RedSun profiles and containers own component construction;
- acquisition hardware and services are RedSun-declared devices;
- presenters and virtual signals/slots own application routing;
- interactive widgets are RedSun `QtView` components; and
- persistent acquisition backends implement RedSun storage contracts.

Mimir and pyhololab are references for behavior and operator workflow. Feature
parity with either application is not a requirement.

## Migration state

| Responsibility | Required boundary | State after this implementation |
|---|---|---|
| Engine | `redsun.engine.RunEngine` | DPCT and processing-flyer execution use the RedSun engine; architecture tests reject direct Bluesky engine imports |
| Profiles | RedSun AppConfig/YAML and plugin manifest | Packaged RedSun YAML profiles replace the parallel `ApplicationProfile` model |
| Devices | RedSun-declared ophyd devices | DPCT stage, illumination, detector groups, and the online camera service are RedSun-declared facades |
| EPICS | Public RedSun EPICS abstract device | The AHT hardware branch consumes RedSun `feat/pre-upstream`'s `EpicsServiceDevice`; production cutover remains subject to upstream review/release |
| DPCT plan | Bluesky generator executed by RedSun | The public simulation path runs `dpct_plan` through RedSun; the old async runner remains only as a compatibility surface for existing low-level tests/callers |
| Views | AHT-customized RedSun `QtView` components | Camera, finite multishot, and processing widgets are `QtView` components; processing retains its AHT Napari layout |
| Offline processing | Hardware-free RedSun container | CLI and GUI processing composition now build RedSun containers around the detached processing presenter |
| Storage | `StorageIO`, `OpenStore`, `BaseStorage`, and external-resource documents | DPCT and camera Zarr stores are constructed and called directly |

## Project-specific composition

Three areas intentionally differ from Mimir while remaining RedSun-native:

1. The offline application is an AHT hardware-free RedSun container.
2. Detached numerical workers remain in `redsun_aht.processing` and are exposed
   as RedSun-composed processing flyers.
3. GUI layouts and Napari workflows are AHT-owned `QtView` implementations,
   informed by pyhololab's operator experience.

## Migration rules

- Existing CLI names remain stable while their implementation is cut over one
  workflow at a time.
- Bluesky plan stubs are allowed inside plans; direct construction of a
  Bluesky `RunEngine` outside RedSun is not.
- Caproto remains valid inside isolated IOCs. Application-side EPICS adapters
  must use the RedSun device abstraction.
- Qt widgets may be composed inside a `QtView`; standalone application launch
  paths are retired after equivalent RedSun profiles are accepted.
- Numerical kernels remain framework-independent. Their application lifecycle
  is owned by RedSun presenters, devices, plans, and containers.
- A storage migration is accepted only if it preserves multidimensional DPCT
  coordinates, durable write acknowledgement, checksums, append-only commit
  records, completion/abort semantics, and deterministic replay.
- If RedSun lacks a required generic abstraction, record and inspect the gap,
  implement it upstream after manual approval, and wait for merge before the
  AHT cutover. Do not add a project-local shadow framework API.

## Implementation plan

1. Inspect RedSun and redsun-mimir local and remote branches before scoping
   each migration slice, and record emerging APIs that could supersede a local
   proposal.
2. Replace parallel AHT profile models and direct Bluesky engine construction
   with RedSun YAML/AppContainer profiles and `redsun.engine.RunEngine`.
3. Route DPCT stage, illumination, and detector execution through
   RedSun-declared ophyd facades and a Bluesky generator plan.
4. Convert camera and offline-processing frontends to AHT-customized RedSun
   `QtView` composition without adopting Mimir's application container.
5. Keep the EPICS base/device-shutdown upstream change reviewable; validate the
   AHT camera adapter on its dedicated hardware branch before an accepted
   RedSun release is promoted to production profiles.
6. Prototype the durable multidimensional storage requirements upstream and
   prepare issue-ready compatibility evidence. Preserve AHT's current durable
   DPCT store until RedSun can represent its semantics without loss.
7. Enforce the architecture with import/structure tests, profile composition
   tests, plan success/failure tests, GUI construction tests, and storage
   durability/replay tests.

## Testing unmerged RedSun changes on local hardware

The throwaway integration branch belongs in AHT, not RedSun. The current branch
is `hw/aht-redsun-integration`; it resolves its editable `../redsun` path
dependency to `feat/pre-upstream` at `1a1879f`. `feat/upstream` remains the
review-ready RedSun snapshot, while `feat/pre-upstream` is the RedSun working
line. Record both repository commits in every hardware test report.

Hardware-only tests must remain explicitly selected, preserve normal operator
interlocks, and never run as part of the default suite. Put reusable contract
tests in RedSun and AHT-specific integration or device-admission tests in AHT.

Do not accumulate unrelated experiments on `feat/upstream`. Prefer one branch
or stacked worktree per upstream proposal, rebase it as the upstream API moves,
and delete it after merge or rejection. Promote the AHT hardware branch's
candidate dependency into production profiles only after the RedSun API is
accepted and released, or after an explicit approved pin.
