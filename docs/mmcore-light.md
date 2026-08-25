# Micro-Manager shuttered light boundary

Status: implemented as a process-local hardware-free characterized backend.
Physical shutter/laser-engine admission, an illumination service IOC, and the
application-side EPICS device remain pending.

## Authorities and provenance

The behavioral donor is
`redsun-pyhololab/src/redsun_mimir/device/mmcore/_shuttered.py` (Lumencor
Spectra six-channel shuttered engine) in the archival RedSun 0.10-era tree. It
is behavior and equation provenance only: AHT does not copy its singleton
`Core.instance()` ownership, in-process device loading from the application
composition root, deprecated `mm_property_signal(..., datatype=...)` property
helpers, hard-coded channel/preset tables, or GUI-facing preset application.

The AHT-owned rewrite instead follows the isolated camera-service pattern:
exactly one process-local `CMMCorePlus` context per service, admission of
exactly one shutter device, explicit identity constraints, safe defaults, and
typed generic property handling.

## Implemented backend

`redsun_aht.device.light.MMCoreShutteredLightBackend` provides:

- one profile-driven `CMMCorePlus` context loaded from an explicit
  system-configuration file and Micro-Manager path;
- admission of exactly one loaded device that must be a shutter whose label
  matches `expected_label`, with optional adapter/device-name constraints;
- forced auto-shutter off before and during ownership, so no camera-side
  logic can toggle illumination;
- admission into the safe state: the shutter is explicitly closed immediately
  after validation;
- typed open/closed readback through `state()` and exactly one commanded
  transition through `set_shutter()`;
- optional wavelength metadata (`wavelength_nm`) carried on the admitted
  identity; and
- optional interlock surfacing: when `interlock_property` names a device
  property that is missing at admission, connection fails; otherwise
  `interlock_state()` reports its coerced readback.

Disconnect is idempotent and always attempts, in order: auto-shutter off,
shutter closed, and `unloadAllDevices`; per-action failures are collected into
an exception group so unload is never skipped by an earlier failure.

## Generic property introspection

`redsun_aht.device.light.properties` classifies every discovered device
property with deterministic precedence:

1. allowed values exactly `("0", "1")` classify as boolean;
2. any other non-empty allowed-value set classifies as enumerable;
3. declared limits or a float-parseable value classify as numeric; and
4. everything else classifies as text.

Writes are validated against the classified kind before reaching the core:
booleans accept `bool`, `0`, `1`, `"0"`, or `"1"`; enumerables require exact
membership; numerics are parsed and checked against inclusive limits; text
passes through as-is; read-only properties are rejected before formatting.

This layer is deliberately shared and dependency-free so later serial,
state/filter-wheel, Hamamatsu, and ASI adapters can reuse it instead of the
donor's removed `datatype=`/`readonly=` helper signature.

## Characterization boundary

All characterization runs against fake cores implementing the narrow
`ShutterCoreLike` protocol surface. No Lumencor or other physical light engine,
serial bus, or real `pymmcore-plus` runtime has been contacted by this slice.
The backend imports `pymmcore_plus` only inside the default core factory, and
only after both configuration paths have been admitted.

Still pending for this capability:

- physical read-only admission evidence for one real shuttered engine;
- an illumination service IOC exposing health, commands, readback, and the
  interlock signal over EPICS; and
- the ophyd-async application-side device and recipe integration.
