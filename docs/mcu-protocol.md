# AHT MCU host protocol v1

Status: implemented and covered by host-side conformance tests; not yet tested
against physical native-v1 firmware.

Each decoded packet is:

| Field | Encoding |
|---|---|
| Protocol version | `u8`, currently `1` |
| Message type | `u8` |
| Flags | `u8` bit field |
| Sequence | `u16-le` |
| Payload length | `u16-le`, maximum 4096 |
| Payload | Command-specific bytes |
| Checksum | CRC-16/CCITT-FALSE `u16-le` over all preceding decoded bytes |

The complete decoded packet is COBS encoded and terminated with `0x00`.
CRC-16/CCITT-FALSE uses polynomial `0x1021`, initial value `0xffff`, no reflected
input/output, and no final XOR. The standard `123456789` check value is `0x29b1`.

## Message and flag values

| Message | Value |
|---|---:|
| `GET_CAPABILITIES` | `0x01` |
| `HEARTBEAT` | `0x02` |
| `ALL_OFF` | `0x10` |
| `BEGIN_PATTERN` | `0x11` |
| `WRITE_PATTERN_CHUNK` | `0x12` |
| `COMMIT_PATTERN` | `0x13` |
| `SHOW_PATTERN` | `0x14` |
| `DEFINE_SEQUENCE` | `0x15` |
| `ARM_SEQUENCE` | `0x16` |
| `START_SEQUENCE` | `0x17` |
| `STOP` | `0x18` |
| `ACK` | `0x70` |
| `ERROR` | `0x71` |
| `TELEMETRY` | `0x72` |

Flags are `RESPONSE_REQUIRED=0x01`, `RETRY=0x02`, and `UNSOLICITED=0x04`.
Unknown flags and message values are rejected.

Every response repeats its request sequence. An ACK begins with the original
request message value followed by an optional response body. An error begins
with the original request value and a `u16-le` typed error code, followed by one
`u8`-length-prefixed UTF-8 detail string. A retried request uses the same
sequence, message type, and payload with `RETRY` set. Firmware returns the
cached response and must not repeat display, trigger, or sequence-start effects.
Reuse of a sequence for different content returns `SEQUENCE_CONFLICT`.

## Transaction payloads

All integers are little-endian.

- `BEGIN_PATTERN`: pattern ID `u16`, panel revision `u16`, encoding `u8`, LED
  count `u16`, and expected pattern CRC-32 `u32`.
- `WRITE_PATTERN_CHUNK`: pattern ID `u16`, offset `u16`, count `u16`, then raw
  values. Raw unsigned 8-bit intensity is mandatory encoding value `1`.
- `COMMIT_PATTERN` and `SHOW_PATTERN`: identifier `u16`.
- `DEFINE_SEQUENCE`: sequence ID `u16`, step count `u16`, then fixed steps of
  pattern ID `u16`, dwell microseconds `u32`, and trigger boolean `u8`.
- `ARM_SEQUENCE`: sequence ID `u16`.
- `START_SEQUENCE`, `STOP`, and `ALL_OFF`: empty payload.

Pattern writes modify staging storage only. Commit requires complete byte
coverage, the declared panel revision and LED count, capacity compliance, and
the expected CRC-32 before atomically publishing the pattern. Sequence
definitions may reference only committed patterns.

## Human-readable test command set

The native-v1 simulator exposes the existing typed host operations as readable
lines through `aht mcu-test`. This is a test console, not a second firmware
protocol. Every line is parsed into a typed semantic command and executed by
`McuHostClient`, which still creates the binary envelopes documented above.

| Readable command | Existing host operation |
|---|---|
| `get-capabilities` | `McuHostClient.get_capabilities()` |
| `upload-pattern ID VALUE ...` | transactional `upload_pattern(ID, bytes)` |
| `show-pattern ID` | `show_pattern(ID)` |
| `define-sequence ID PATTERN:DWELL[:TRIGGER] ...` | `define_sequence(ID, steps)` |
| `arm-sequence ID` | `arm_sequence(ID)` |
| `start-sequence` | `start_sequence()` |
| `stop` | `stop()` |
| `all-off` | `all_off()` |

Integers accept decimal or `0x` notation. Pattern bytes can be separated by
spaces or commas. Sequence dwell values are microseconds. The optional trigger
token accepts `true`/`false`, `on`/`off`, `yes`/`no`, or `1`/`0`, and defaults
to true. Blank lines and text after `#` are ignored.

Run an ordered command list directly:

```console
uv run aht mcu-test --led-count 4 --panel-revision 3 \
  --command "get-capabilities" \
  --command "upload-pattern 17 0,32,64,255" \
  --command "show-pattern 17" \
  --command "define-sequence 2 17:250:on" \
  --command "arm-sequence 2" \
  --command "start-sequence" \
  --command "stop" \
  --command "all-off"
```

Or put the same lines in a UTF-8 file and use `--script COMMANDS.txt`. Each
successful operation prints a stable `OK ...` line. Parser and typed endpoint
failures print `ERROR line N: ...` and stop the script with a nonzero exit code.
The console composition intentionally targets `SimulatedMcu`; it must not be
pointed at the connected legacy two-digit firmware.

## Firmware-target decision

The connected 16 MHz/2 kB-SRAM ATmega328P Nano remains a legacy compatibility
endpoint. The active hardware variant is the 91-LED TTL panel; the 217-LED
source is deprecated. A fixed-buffer native-v1 migration is feasible for the
91-LED illumination-only profile and must advertise its narrow map/sequence
capacity rather than the simulator's capacity. Future motion or substantially
larger dynamic pattern budgets favor the higher-RAM STM32F103C8 target. Neither
target has been flashed or contacted by this package; firmware and
electrical-profile evidence remain in the instrument repository.

## Scope boundary

This AHT implementation does not define or claim watchdog, E-stop, safety-latch,
or safety-reset behavior. Those capabilities are deferred to later instrument
and firmware integration. `STOP` and `ALL_OFF` provide deterministic normal
operation cleanup only. The connected legacy two-digit Micro-Manager firmware
is not wire-compatible with v1 and must not be passed to this native adapter.
