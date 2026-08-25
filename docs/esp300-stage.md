# Newport ESP300 stage boundary

Status: simulated composite and scan layers implemented; read-only physical
admission passed; physical motor-power, home, and motion gates remain pending.

## Authorities and provenance

The behavioral donor is
`pyhololab/src/simple_acquisition/newport_esp_stage.py`. AHT preserves its
observed 1-based controller axes, 0-based/public XYZ mapping concept, shared
serial lock, 20 ms motion polling convention, absolute/relative position
commands, position caching concept, unit/encoder queries, software-limit
queries, and ordinary axis stop behavior. It does not copy the donor's global
GUI state, background polling thread, blanket ten-attempt retries, suppressed
exceptions, implicit motor-power changes, or destructor-driven hardware I/O.

The byte-level command source is the supplied Newport
`ESP300-User-Manual.pdf`. The admitted serial profile is 19200 baud, eight data
bits, no parity, one stop bit, RTS/CTS hardware flow control, and a bounded
one-second read/write timeout.

## AHT composition

- `redsun_aht.motion` constructs immutable linear, list, grid, and snake scan
  points without constructing hardware.
- `CompositeStageDevice` validates every axis and limit before issuing the
  first move, records explicit units and coordinate frame in `PoseSample`,
  requires motion-done plus position tolerance, and issues ordinary stop to
  every backend after partial-start failure or settle timeout.
- `Esp300Controller` serializes ASCII access to one already-open port. The
  caller owns serial construction and closure.
- `Esp300AxisBackend` maps absolute move, position, motion-done, and ordinary
  stop onto the composite backend boundary.
- `build_esp300_composite_stage` derives the public axis specs from the
  admitted controller snapshot rather than hard-coding travel or resolution.

## Command mapping

| AHT operation | ESP300 wire command |
|---|---|
| Controller identity | `VE?` |
| Stage model and serial | `xxID?` |
| Displacement unit | `xxSN?` |
| Encoder resolution | `xxSU?` |
| Left/right software limits | `xxSL?`, `xxSR?` |
| Actual position | `xxTP?` |
| Absolute move | `xxPAnn` |
| Motion complete | `xxMD?` (`1` is complete) |
| Ordinary axis stop | `xxST` |

The donor consistently appends `?` for all queries, including `TP`; the
controller accepted that form during admission. Commands are ASCII and
carriage-return terminated. Responses are bounded, non-empty ASCII lines.

## Physical admission evidence

On 2026-08-20, the explicit COM1 read-only gate admitted an ESP300 Version 3.08
controller with three attached CMA-12-family axes. All axes reported
millimeters, positive encoder resolution, 0–12.5 mm software limits, and a
current position within those limits. Exact stage serial numbers are retained
only in the machine-local session record.

Admission sent only `VE?`, `ID?`, `SN?`, `SU?`, `SL?`, `SR?`, and `TP?` queries.
It sent no `MO`, `MF`, `OR`, `PA`, `PR`, `ST`, or controller-abort command.
