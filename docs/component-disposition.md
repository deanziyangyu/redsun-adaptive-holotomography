# Initial component disposition ledger

This ledger records the first Phase 1 decisions. A donor path is evidence of
behavior, not a source tree to copy wholesale.

| Capability | Source | Disposition | Phase 1 treatment |
|---|---|---|---|
| Dependency injection and application runtime | RedSun 0.11.0 public API | `depend` | Pin the released API; declare typed provider keys locally |
| Canonical document stream pattern | RedSun/Mimir public pattern | `depend` | Declare AHT-owned stream constants; no donor implementation copied |
| Storage ownership | `redsun.storage.BaseStorage` | `depend` | The hardware-free profile owns one injected `BaseStorage` through an AHT-owned public-`StorageIO` memory adapter; detector-owned stores arrive with the first detector slice |
| MMCore camera and stage foundations | Current RedSun/Mimir public API | `depend` / `upstream` | Isolated SDK-free and MMCore camera services plus typed ophyd-async adapter implemented; stage work remains in Phase 3 |
| Serial, shutter, filter-wheel, Hamamatsu, and ASI adapters | `redsun-pyhololab` donor | `rewrite` / `upstream` | ESP300 transport/composition rewritten against the current AHT contract with donor characterization. The shuttered-light slice is rewritten as a process-local admitted backend with a shared generic property-introspection layer; serial, state/filter-wheel, Hamamatsu, and ASI adapters remain pending. Do not copy RedSun 0.10 lifecycle or property helpers |
| Median/background behavior | Current document-routed median presenter | `depend` / `port` | Temporal median/divide behavior is preserved in durable immutable calibration artifacts; document-router integration remains in progress |
| Mimir composition root and UC2/Youseetoo profiles | `redsun-pyhololab` donor | `retire` | Never import into AHT |
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
