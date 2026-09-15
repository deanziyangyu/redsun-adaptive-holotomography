# RedSun-native application architecture

## Requirement

`redsun-adaptive-holotomography` is a RedSun showcase application. Application
execution must enter through RedSun's public interfaces and schemas:

- `redsun.engine.RunEngine` owns Bluesky execution;
- RedSun profiles and containers own component construction;
- acquisition services and devices are declared through RedSun profiles;
- presenters and virtual signals/slots own application routing;
- interactive widgets are RedSun `QtView` components; and
- persistent acquisition backends are owned by the declared service/device and
  publish standard Bluesky external-asset documents through RedSun execution.

Mimir and pyhololab are references for behavior and operator workflow. Feature
parity with either application is not a requirement.

## Migration state

| Responsibility | Required boundary | State after this implementation |
|---|---|---|
| Engine | `redsun.engine.RunEngine` | DPCT and processing-flyer execution use the RedSun engine; architecture tests reject direct Bluesky engine imports |
| Profiles | RedSun AppConfig/YAML and plugin manifest | Packaged RedSun YAML profiles replace the parallel `ApplicationProfile` model |
| Devices | RedSun-declared ophyd devices | DPCT stage, illumination, and detector groups are RedSun-declared facades; the camera group composes an ordinary ophyd-async EPICS service device |
| EPICS services | RedSun service declaration plus ordinary ophyd-async device | RedSun `0.13.0rc0` launches the camera IOC before device construction, injects its prefix, and stops it with container teardown; no AHT-specific RedSun EPICS base or device `shutdown()` is required |
| DPCT plan | Bluesky generator executed by RedSun | The public simulation path runs `dpct_plan` through RedSun; the old async runner remains only as a compatibility surface for existing low-level tests/callers |
| Views | AHT-customized RedSun `QtView` components | Camera, finite multishot, and processing widgets are `QtView` components; processing retains its AHT Napari layout |
| Offline processing | Hardware-free RedSun container | CLI and GUI processing composition now build RedSun containers around the detached processing presenter |
| Storage | Service/device-owned durable writer plus standard external-asset documents | The DPCT writer owns indexed OME-Zarr persistence, receipts, checksums, and completion; the detector emits Resource/Datum documents through RedSun. A format-aware AHT/Tiled adapter remains pending for direct live-run array access |

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
- Caproto remains isolated inside RedSun-declared IOCs. Application-side EPICS
  devices use ophyd-async directly and are composed by RedSun's service layer.
- Qt widgets may be composed inside a `QtView`; standalone application launch
  paths are retired after equivalent RedSun profiles are accepted.
- Numerical kernels remain framework-independent. Their application lifecycle
  is owned by RedSun presenters, devices, plans, and containers.
- A storage migration is accepted only if it preserves multidimensional DPCT
  coordinates, durable write acknowledgement, checksums, append-only commit
  records, completion/abort semantics, and deterministic replay.
- If RedSun lacks a required generic abstraction, record and inspect the gap,
  implement it upstream after manual approval, and wait for merge before the
  AHT cutover. Format-specific service/device writers and catalog adapters may
  remain application-owned; do not turn them into a shadow RedSun framework.

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
5. Completed: adopt RedSun `0.13.0rc0` service declarations and compose the
   existing ophyd-async EPICS camera without a custom RedSun subclass.
6. Completed: prove service/device-owned multidimensional persistence and
   external Resource/Datum emission in simulation, a RedSun-launched service,
   and FLIR-only hardware. Do not file the superseded RedSun storage proposal.
7. Implement the remaining format-aware AHT/Tiled adapter that resolves each
   detector data key to its array inside the DPCT OME-Zarr group while
   retaining logical placement and measured readbacks.
8. Enforce the architecture with import/structure tests, profile composition
   tests, plan success/failure tests, GUI construction tests, and storage
   durability/replay tests.

## Testing unmerged RedSun changes on local hardware

Short-lived branches tracked to `deanziyangyu/redsun` are appropriate for
hardware characterization before a RedSun change is merged, provided they are
treated as experiments rather than AHT dependencies. Use a dedicated RedSun
worktree and virtual environment for each hardware experiment, branch from the
exact upstream commit under review, and track the branch to the personal fork
(`origin`), never canonical `upstream`.

Keep AHT's normal lock file pinned to a released RedSun version. In the
hardware worktree only, install the candidate RedSun checkout as an editable or
path dependency and record both repository commit IDs in the test report.
Hardware-only tests must remain explicitly selected, preserve normal operator
interlocks, and never run as part of the default suite. Put reusable contract
tests in RedSun and AHT-specific integration or device-admission tests in AHT.

Do not accumulate unrelated experiments on `feat/upstream`. Prefer one branch
or stacked worktree per upstream proposal, rebase it as the upstream API moves,
and delete it after merge or rejection. Production profiles and the committed
AHT lock must not depend on those branches; cut over only after the accepted
RedSun API is released or intentionally pinned to an approved commit.
