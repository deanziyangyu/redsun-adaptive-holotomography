# Initial component disposition ledger

This ledger records the first Phase 1 decisions. A donor path is evidence of
behavior, not a source tree to copy wholesale.

| Capability | Source | Disposition | Phase 1 treatment |
|---|---|---|---|
| Dependency injection and application runtime | RedSun 0.11.0 public API | `depend` / `adapt` | RedSun is the mandatory application boundary; migrate custom execution paths that currently bypass its engine, device, profile, presenter, view, or storage interfaces |
| Canonical document stream pattern | RedSun/Mimir public pattern | `depend` | Declare AHT-owned stream constants; no donor implementation copied |
| Storage ownership | `redsun.storage.BaseStorage` | `depend` | The hardware-free profile owns one injected `BaseStorage` through an AHT-owned public-`StorageIO` memory adapter; detector-owned stores arrive with the first detector slice |
| MMCore camera and stage foundations | Current RedSun/Mimir public API | `depend` / `upstream` | Isolated SDK-free and MMCore camera services plus typed ophyd-async adapter implemented; stage work remains in Phase 3 |
| Serial, shutter, filter-wheel, Hamamatsu, and ASI adapters | `redsun-pyhololab` donor | `rewrite` / `upstream` | ESP300 transport/composition rewritten against the current AHT contract with donor characterization. The shuttered-light slice is rewritten as a process-local admitted backend with a shared generic property-introspection layer; serial, state/filter-wheel, Hamamatsu, and ASI adapters remain pending. Do not copy RedSun 0.10 lifecycle or property helpers |
| Median/background behavior | Current document-routed median presenter | `depend` / `port` | Temporal median/divide behavior is preserved in durable immutable calibration artifacts; document-router integration remains in progress |
| Mimir composition root and UC2/Youseetoo profiles | `redsun-pyhololab` donor | `reference` / `retire` | Keep donor modules out of AHT and use them only as behavior/UI references; Mimir feature parity is not an AHT requirement |
| Run lifecycle and recovery | ARH restructuring contracts | `rewrite` | Implement an AHT-owned deterministic reducer and append-only journal |
| Robotic validation and remote SLAR integration | Future robotic repository | `defer` | Expose protocols only; no dependency, implementation, or socket use |
| MCU host integration | AHT semantic protocol plus RedSun/ophyd adapter | `rewrite` | Native v1 framing, transactional patterns/sequences, deterministic simulator and transport-facing adapter implemented; real serial conformance remains pending compatible firmware. Safety APIs are deferred outside this AHT implementation |
| Processing device compatibility | RedSun flyer/device composition | `adapt` | `device.processing.ProcessingFlyerDevice` now implements standard ophyd-async/Bluesky lifecycle and external-reference collection over detached workers; the structural `ProcessingDeviceAdapter` seam remains available for later RedSun composition |
| Persistent reconstruction manager behavior | `pyhololab/src/simple_acquisition/recon_manager.py` | `rewrite` | Preserve one initialized solver per persistent worker and buffered offline inputs; replace the donor's serial shared-array loop with reference-only MsgPack queues and independent concurrent workers |
| GPU execution and telemetry | pyhololab CuPy/Torch utilities and detached-processing plan | `rewrite` | CuPy CUDA-12 reference workers own their subprocess context, stream, cache, allocator, and telemetry; quantitative DPCT/qOBT reference adapters are characterized on the installed P100 and P4 but still require production instrument datasets |
| XY-tiled DPCT/qOBT | `pyhololab` commit `405e53b` | `port` / `adapt` | Deterministic full-Z/full-shot XY planning, padding, raised-cosine blending, cancellation, timing, telemetry, validation, donor-equivalent NumPy Tikhonov adapters, spawned offline jobs, checksum-bound provenance, CLI/GUI CPU and GPU controls, and worker-owned CuPy adapters characterized on P100/P4 are implemented under `processing`; stage-mosaic terminology is rejected |
| Nonlinear multi-layer IDT | `pyhololab` commit `e8dd96bfa2d54784e00209769d594f305863b01a` | `port` / `adapt` | Backend-neutral multi-layer Born/multislice forward-adjoint equations, deterministic iterative optimization, regularization, cancellation, memory preflight, checksum-bound provenance, CLI/GUI controls, and separate NumPy/CuPy workers characterized on P100/P4 are implemented under `processing`; the donor is WIP and has no committed multi-layer tests, and production instrument datasets remain pending |

The first reused public pattern is the RedSun typed-provider/canonical-stream
boundary. The AHT declarations are project-owned names and types, while
container/runtime behavior remains a pinned dependency rather than vendored
framework code.

## RedSun-native application status

This repository is intended to exercise RedSun as the application framework,
not merely carry it as a dependency. RedSun is currently used for the
dependency/plugin boundary, typed storage/provider injection, one simulation
lifecycle presenter, and the standard Bluesky lifecycle of
`ProcessingFlyerDevice`. DPCT acquisition, camera control, profile selection,
the processing GUI, and most stage/light/MCU orchestration still bypass the
corresponding RedSun abstractions. The current
[RedSun manifest](../src/redsun_aht/redsun.yaml) therefore declares no devices
or views and only the simulation lifecycle presenter.

The required migration is:

1. **RedSun-owned profiles and simulation.** Make RedSun configuration the
   source of truth and add hardware-free headless and Qt profiles using the
   same devices and plans as the physical profile.
2. **RedSun acquisition runtime.** Expose the existing DPCT ordering and
   cleanup contracts through Bluesky plans and RedSun's `RunEngine`, `Action`,
   continuous-plan, plan-spec, and plan-widget APIs. Preserve AHT's immutable
   shot records and protocol validation.
3. **Device and view facades.** Wrap stage, MCU/illumination, and detector
   services in RedSun/ophyd-compatible devices, then replace or embed the
   corresponding custom controls with `QtView` components, including a Napari
   image view and storage view.
4. **Document routing and storage integration.** Route live-view, median, and
   processing observations through canonical Bluesky streams and document
   presenters. Implement DPCT persistence through RedSun storage abstractions
   while retaining CZYX layout, manifests, checksums, durable acknowledgement,
   and provenance.
5. **Instrument profiles and acceptance tests.** Add explicit AHT simulation
   and hardware configuration profiles, test all declared ports and lifecycle
   transitions, and verify that hardware-free startup never imports or contacts
   vendor services.

Mimir and pyhololab are behavioral and operator-interface references, not code
to copy wholesale or feature targets. AHT pins RedSun 0.11.0; missing generic
framework abstractions are reported for manual upstream review rather than
silently replaced with project-local framework layers. See
[`redsun-native-architecture.md`](redsun-native-architecture.md) and
[`redsun-gap-report.md`](redsun-gap-report.md).
