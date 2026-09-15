# RedSun `feat/upstream` merge note

This document is a paste-ready replacement body for the existing RedSun issue
and a prospective pull-request description. It is documentation only: it does
not open or modify an issue or pull request.

## Suggested title

Add an EPICS service-device base and container-managed asynchronous device
shutdown

## Proposed issue / pull-request body

### Summary

Merge [`deanziyangyu/redsun:feat/upstream`](https://github.com/deanziyangyu/redsun/tree/feat/upstream)
at `1a1879f` into RedSun. The branch is based on canonical RedSun v0.12.2
(`aa4a587`) and adds two related framework capabilities:

1. a public `redsun.device.epics.EpicsServiceDevice` base for typed
   ophyd-async clients of independently running EPICS services; and
2. container-managed asynchronous device shutdown after presenters stop and
   before hook/component teardown.

The change is deliberately generic. It contains no AHT camera PV schema,
Caproto IOC, Micro-Manager integration, application profile, or GUI code.

### Problem

RedSun can declare any ophyd-async `Device`, and already exposes the structural
`HasAsyncShutdown` protocol, but it does not currently provide a public base
for an EPICS-backed service device. Applications must therefore derive such
devices directly from ophyd-async's `EpicsDevice`, bypassing a RedSun-owned
lifecycle boundary.

There is also a lifecycle gap in `AppContainer.shutdown()`: presenters are
stopped, but built devices implementing `HasAsyncShutdown` are not awaited.
An application can consequently release its container while a remote service,
subscription, acquisition task, or process-owned hardware resource remains
active.

This is visible in `redsun-adaptive-holotomography`, where each camera SDK and
Micro-Manager context runs in an isolated Caproto IOC process. The application
side needs a typed ophyd-async device for the IOC control plane, and container
shutdown must call the device's safe stop/disconnect sequence. That requirement
is not AHT-specific; it applies to any independently running EPICS service
managed through a RedSun application.

### Proposed API

The branch adds the optional `redsun[epics]` dependency group, backed by
`ophyd-async[ca]>=0.21.2`, and the following public abstraction:

```python
from abc import ABC, abstractmethod

from ophyd_async.epics.core import EpicsDevice


class EpicsServiceDevice(EpicsDevice, ABC):
    def __init__(
        self,
        name: str,
        /,
        *,
        prefix: str,
        with_pvi: bool = False,
    ) -> None: ...

    @property
    def service_prefix(self) -> str: ...

    @abstractmethod
    async def shutdown(self) -> None: ...
```

The positional-only component name matches RedSun's component-construction
convention. The EPICS prefix is explicit, retained unchanged as
`service_prefix`, and rejected when empty. Subclasses declare their own PVs and
own the service-specific safe shutdown procedure.

`AppContainer.shutdown()` becomes:

1. disconnect virtual wiring;
2. shut down presenters, preventing new device commands;
3. await built `HasAsyncShutdown` devices in reverse build order;
4. shut down hooks;
5. release components; and
6. destroy remaining objects.

One device failure is logged and does not prevent cleanup of the remaining
devices. Releasing components also resets the container's device-connection
state so a later rebuild does not inherit stale connection bookkeeping.

### Why the EPICS base and container change belong together

An abstract EPICS device without container ownership leaves safe teardown to
each application launcher. Container shutdown without a clear service-device
contract still forces applications to invent inconsistent wrappers. Together,
the changes establish one RedSun-owned boundary while leaving all PV schemas
and service implementations in their appropriate packages.

### Compatibility

- Existing `Device`, `OpenStore`, presenter, view, and profile APIs are
  unchanged.
- EPICS dependencies remain optional; non-EPICS installations do not acquire
  Channel Access dependencies.
- Existing devices without `HasAsyncShutdown` retain their current teardown
  behavior.
- Existing devices that already implement `HasAsyncShutdown` will now be
  called by `AppContainer.shutdown()`. This is an intentional lifecycle
  behavior change and should be called out in release notes.
- Presenter shutdown still occurs before hook and component teardown.
- The implementation uses RedSun's existing synchronous-to-asynchronous
  bridge rather than introducing a second event-loop owner.

### Validation

At `feat/upstream@1a1879f`:

- 562 RedSun tests passed;
- statement coverage was 96%;
- Ruff and formatting checks passed; and
- Mypy passed with both PyQt6 and PySide6 environments.

The 96% result is local coverage evidence. RedSun's hosted Codecov upload is
configured for canonical `main` or pull-request CI, so the fork branch push by
itself does not produce a hosted Codecov result; that check should be confirmed
on the eventual pull request.

Focused tests verify that:

- the component name and EPICS prefix are preserved;
- an empty prefix is rejected;
- `EpicsServiceDevice` structurally satisfies `HasAsyncShutdown`;
- presenters stop before asynchronous devices;
- every qualifying device is attempted even if one shutdown raises;
- a device-shutdown error is logged; and
- container connection state is reset after release.

The candidate was also exercised from an AHT hardware-integration worktree
using an editable checkout of this exact RedSun branch. The application-side
camera control plane was routed through an `EpicsServiceDevice` subclass and
ophyd-async while Caproto and each vendor SDK remained isolated in the IOC.

Hardware checks completed for both admitted cameras:

- FLIR/SpinnakerC: identity admission and one unilluminated frame, including
  descriptor SHA-256 verification, exact acknowledgement, shared-memory
  unlink, and disconnect;
- PVCAM: the same admission, integrity, acknowledgement, cleanup, and
  disconnect checks; and
- simultaneous camera-service operation with no cross-service control-plane
  interference.

The repeated 60 Hz IOC live-drain characterization reported zero MMCore buffer
overruns and no IOC errors:

| Case | Drain rate after `LIVE_STATE=running` | Earlier recorded rate |
|---|---:|---:|
| PVCAM single | 726 fps | 715 fps |
| FLIR single | 315 fps | 295 fps |
| PVCAM + FLIR | 718 fps / 308 fps | 714 fps / 294 fps |

These measurements are control-plane/camera-drain checks, not GUI transfer or
durable-storage benchmarks. They indicate that adopting the RedSun/ophyd-async
device boundary did not regress the isolated Caproto acquisition path.

### Non-goals

- Defining application-specific PVs or a universal camera schema.
- Moving Caproto, Micro-Manager, vendor SDKs, or hardware threads into RedSun.
- Treating Mimir feature parity as a requirement.
- Adding a generic offline application container or changing Qt layout APIs.
- Adding multidimensional transactional storage. The separate prototype on
  `feat/pre-upstream@03009a5` requires its own design review and must not be
  included in this merge.

### Review decisions requested

1. Should device-shutdown failures remain logged after best-effort cleanup, as
   implemented, or should the container aggregate and raise them after every
   device has been attempted?
2. Is reverse build order the desired default for asynchronous device
   teardown?
3. Should `redsun[epics]` remain a separate optional extra, and is
   `ophyd-async[ca]>=0.21.2` the appropriate lower bound?
4. Is an abstract `shutdown()` the right contract, or should RedSun provide a
   narrower default implementation for service devices?
5. Should this lifecycle phase be ported unchanged into
   `feat/experimental-di`, or represented there as a session/container hook?

### Acceptance checklist

- [ ] Public import path and constructor convention are accepted.
- [ ] Optional dependency placement and lower bound are accepted.
- [ ] Presenter-before-device shutdown ordering is accepted.
- [ ] Failure reporting semantics are accepted.
- [ ] Reverse device teardown order is accepted.
- [ ] Rebuild/connection-state reset behavior is accepted.
- [ ] The interaction with `feat/experimental-di` has an explicit follow-up.
- [ ] Release notes identify the new shutdown call for existing
      `HasAsyncShutdown` devices.

### Candidate scope

The branch changes six files: dependency metadata and lock data, the new EPICS
base, `AppContainer` shutdown ordering/state reset, and focused device/container
tests. Please review and merge `feat/upstream@1a1879f` independently of later
storage experiments.
