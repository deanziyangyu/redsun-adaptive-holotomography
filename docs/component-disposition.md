# Initial component disposition ledger

This ledger records the first Phase 1 decisions. A donor path is evidence of
behavior, not a source tree to copy wholesale.

| Capability | Source | Disposition | Phase 1 treatment |
|---|---|---|---|
| Dependency injection and application runtime | RedSun 0.11.0 public API | `depend` | Pin the released API; declare typed provider keys locally |
| Canonical document stream pattern | RedSun/Mimir public pattern | `depend` | Declare AHT-owned stream constants; no donor implementation copied |
| Storage ownership | `redsun.storage.BaseStorage` | `depend` | Reserve the public ownership boundary; concrete storage arrives with the first detector slice |
| MMCore camera and stage foundations | Current RedSun/Mimir public API | `depend` / `upstream` | Deferred until the detector vertical slice |
| Serial, shutter, filter-wheel, Hamamatsu, and ASI adapters | `redsun-pyhololab` donor | `rewrite` / `upstream` | Do not copy RedSun 0.10 lifecycle or property helpers |
| Median/background behavior | Current document-routed median presenter | `depend` / `port` | Preserve behavior in future immutable calibration artifacts |
| Mimir composition root and UC2/Youseetoo profiles | `redsun-pyhololab` donor | `retire` | Never import into AHT |
| Run lifecycle and recovery | ARH restructuring contracts | `rewrite` | Implement an AHT-owned deterministic reducer and append-only journal |
| Robotic validation and remote SLAR integration | Future robotic repository | `defer` | Expose protocols only; no dependency, implementation, or socket use |

The first reused public pattern is the RedSun typed-provider/canonical-stream
boundary. The AHT declarations are project-owned names and types, while
container/runtime behavior remains a pinned dependency rather than vendored
framework code.
