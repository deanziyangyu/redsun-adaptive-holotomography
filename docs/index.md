# Architecture

The acquisition application serves every device over PVAccess and composes a session from
Redsun declarations. This page describes the modules, the data flow they implement, and
what is deliberately not built yet.

## Layout

| Package | Role |
|---|---|
| `contracts/` | Frozen value types and protocols: camera settings and descriptions, frame and run identity, plan requests and outcomes, asset references and plane receipts. Importing it pulls no bluesky, ophyd, fastcs, p4p, Qt or redsun. |
| `device/mmcore/` | `MMCamera` and `MMStage` over the camera and stage services, plus the settings adapter (`describe_settings`, `apply_settings`, `capture_frame`, `stop`) and the position adapter (`axes`, `move`, `read_position`). |
| `device/arduino/` | The MM-compatible LED command encoding, the `oxiserial` owner, the PVA device and the `contracts.Illumination` adapter. |
| `services/` | FastCS controllers including `mmcore_camera`, `spinnaker_camera`, `mmcore_stage`, and `arduino_led`; each owns its hardware connection. |
| `plans/` | Live streaming, four-pattern DPCT, time-lapse and coordinated MCU acquisition plans. |
| `presenter/` | `DetectorPresenter` routes run documents to the viewer and offers the live-stream plan; `AcquisitionPresenter` offers the acquisition actions. |
| `view/` | Detector settings and ROI, the acquisition page, the preview layer adapter and its driver. |
| `storage/` | The finite-run zarr writer, its manifest and the plane-receipt layer. |
| `configurations/` | The lazy CLI registry, the camera and DPCT sessions, the wiring helpers and the profiles. |
| `common/` | Stream names shared by plans and presenters. |

## Data flow

```
FastCS service (owns the hardware, publishes PVA attributes)
        │
        ▼
ophyd-async device (signals: buffer, frame_count, captured, state, settings)
        │
        ├── plan (Bluesky messages) ──► RunEngine ──► documents
        │                                             │
        │                                             ├── Events (preview, per detector)
        │                                             └── stream_resource / stream_datum
        ├── live capture ──► camera service RunFrameStore
        │                    (frames, manifest, optional timing sidecar)
        └── software DPCT ──► fresh frame ──► RunFrameStore
                                               (pixels, manifest, checksums)
```

`RunFrameStore` writes a manifest and per-frame checksums for both paths.
The live camera service opens an unbound store; its caller supplies run,
detector and expected-frame identity when assembling a receipt. Software DPCT
binds those identities when it opens its store.

The live stream is not a side channel: each camera buffer and its setting-change signals are
monitored **inside** the run, so the RunEngine emits the preview Events as part of the
document sequence. Each detector therefore gets its own `<detector>_live_view` stream,
because `event_model` refuses two descriptors that share a stream name with different data
keys.

## Experimental Xbox survey input

`operator_input.xinput` reads Windows XInput and maps a held A button, left stick and D-pad to normalized survey intent; B, release and disconnect request a stop. The stateful adapter requires A and B released, the left stick in its deadzone, and the Z D-pad released before re-arming after startup, disconnect or B. A separate in-process rehearsal maps the intent to bounded synthetic composite-axis moves and tests a simulated reservation; neither component schedules polling, commands physical axes, enforces cross-process leases, or guarantees a physical stop. The in-process pure mapper and injected polling function permit hardware-free tests.

```
XInput state -> re-arm and deadman mapping -> simulated survey -> synthetic axes
```

## Experimental simulated stack PVA service

`services.simulated_stack` exposes the in-process finite-stack model through an experimental PVA JSON request/result pair. A typed `PvaSimulatedStack` client submits one request and validates simulated trigger records returned by the service. A separate synthetic survey write is refused while the in-process composite is reserved by a stack; it is not queued. A run-ID-scoped simulated abort publishes a partial result and releases the reservation; the experimental plan attempts it on timeout, without a guarantee against late starts. It is a hardware-free rehearsal, not the physical MCU protocol or an authentication service; a new configured transaction may reuse an ID. It is available from an experimental headless Redsun presenter; a co-located fake camera counts synthetic edges independently, a simulated composite stage holds the fast-Z state, and an optional frame store writes fabricated pixels. No physical camera is acquired.

```
Headless Redsun plan -> typed PVA client -> fake MCU -> simulated result
         |
         +-> optional synthetic frame store
```

## Experimental unified MCU serial owner

The unified MCU service also exposes the open-loop ULN2003 rotation axis through
`PvaRotationMotor`, registered as `unified-rotation` and declared in both PVA
time-lapse sessions. The `rotation` axis is a standard movable in assumed
degrees. The default follows EMMsh roll/pitch: 2,048 steps per revolution and
no extra gear factor, with direction reversed; all three assumptions are configurable. Position is
the firmware's software counter, not an encoder reading. Zeroing sets that
counter without homing. Disabling the coils removes holding torque. The
rotation client shares the serial owner with LED and Z, and reserves it during
each move; other clients cannot operate it during that reservation. The motor
interface is available for composition. The separate `rotation` Qt session
uses Redsun's built-in Positioner presenter and motor widget, including its
degree jog, go-to and stop controls. Run `aht session rotation -- --port PORT`
to launch the sole unified MCU service, or omit `--port` to connect to an
already running service at `TLMCU:`. The physical presenter/PVA path has been
exercised; the Qt window and measured angular calibration remain unverified.

```
Redsun rotation motor -> rotation PVA client -> unified MCU service -> ULN2003
                                                    ├-> LED
                                                    └-> Z controller
```

The experimental `device.esp300_readonly` component serializes read-only
controller identity and three-axis snapshots on one caller-owned serial port,
converts the supported millimetre profile to micrometres, and quarantines an
uncertain response stream. `services.esp300_readonly` publishes an explicitly
refreshed snapshot and host-monotonic query bracket over PVA, draining an
in-flight read before port closure. It does not expose motion or a lease;
controller-reported positions are not simultaneous or independent measurements.
`device.pva_esp300_readonly` validates that snapshot through a typed Redsun
device without presenting a movable-stage interface. The timestamp clock belongs
to the service host and is not automatically comparable across machines.

The experimental headless `controller_evidence_session` composes this read-only
coarse observation and the unified fast-Z PVA client as separate roles. Its plan
records controller reports only and requires host-asserted readiness; it is not a
camera acquisition, a cross-device lease or a verified physical stop.
An explicit-port factory can declare both services as session-owned processes;
idle launch and shutdown have been rehearsed. Active-motion shutdown remains
unqualified and must not be treated as an emergency stop. For a managed service
shutdown, the unified service now cancels and waits briefly for the active stack
request before closing the port; this only attempts abort and cannot establish
delivery or a physical stop.

A synthetic PVA stage adapter routes coordinated manual moves and a reserved
stack through the same fake service. Its cooperative service reservation
rejects unowned manual and stack requests while held; clients must explicitly
acquire it as well as the local coordinator token. This is simulation only.

An experimental in-process `CompositeMotionCoordinator` maps named axes to
already-owned stage objects. Manual jog and a token-reserved acquisition can
share it, with whole-request target prevalidation and sequential backend moves.
An error after submission or a stop request quarantines the instance. It has
synthetic tests only: it does not enforce ownership against clients that bypass
it, is not a service-side or cross-process lease, and stop acknowledgements do
not establish a physical stop. It is not yet bound to a production session.

```
manual intent ----+-> local composite coordinator -> fake PVA stage service
acquisition token-+        -> fake service reservation (synthetic loopback)
```

The separate synthetic stack service accepts bounded XYZ jog increments. The
headless simulated jog presenter runs the XInput bridge and sends manual requests
through the same typed PVA client; its synthetic position snapshot is published at startup and after manual,
jog and stack-step operations. It is not an independent physical readback. The service arbitrates requests against its in-process stack reservation.
The caller starts and stops at most one presenter polling loop; each new loop requires
neutral input before re-arming. The polling loop samples input while one write is pending, but cannot cancel or
verify the outcome of an already submitted write. This is a simulation, with no
physical jog, watchdog, verified stop, GUI view or cross-process authorization.

```
XInput -> presenter polling loop --+
manual -> same presenter ----------+-> typed PVA client -> simulated service -> axes
                                                          -> stack reservation
```

`device.unified_mcu` is a direct, experimental Python owner for the unified
illumination and X-stack firmware dialect. It opens one explicit serial endpoint,
checks the firmware identity, serializes ordinary illumination and stack
operations, and can issue a run-scoped abort while a stack is active. It checks
controller configuration, state and ordered software-reported pulse attempts;
those reports do not prove an electrical edge or a camera exposure. An uncertain
  outcome inhibits later operations. It does not provide a cross-process lease,
  and it must not share a port with the separate
MM-compatible LED service. Camera arming, frame reconciliation, physical stop
qualification and restart-safe recovery are not implemented.

```
experimental Redsun client -> PVA controller -> unified serial owner -> firmware
           |                         |                 |
           +-> partial result        +-> abort        +-> reported pulse attempts
```

The cooperative reservation holder can also issue numeric illumination commands
through a token-scoped PVA attribute while idle, including blanking between
blocks. Unreserved clients retain the original attribute and cannot use it
while a reservation is held. This reservation is not a security boundary, and a
failed command leaves the client uncertain. A separate selectable gate can
represent a mechanical shutter or a light-source enable; concrete device
bindings are still required by the service-owned time-lapse gate callbacks.
The current 0.16 protocol permits transaction IDs to be reused after a fresh
configuration and arm. A repeated start without fresh admission is rejected.
The reservation holder allocates local command tags for a run; they are not
durable acquisition identities. Earlier protocols are archived, not supported
by the current host.
The headless session can bind a stationary LED channel directly to its camera
and MCU clients. It configures exposure, color and brightness and avoids
rewriting an unchanged LED channel within one reservation; its logical gate is
a no-op. A new reservation invalidates that local cache.

For a separate software-only UniMMCore rehearsal, a registered synthetic stage
defers its sequence until a custom MDA engine's post-camera-start hook. This is
tested through an MDA runner with a synthetic camera and MCU model, using both
ordinary Z events and an explicitly sequenced event. Its fake camera has its own
counter. A dropped-trigger test shows the MDA runner assigns subsequent frames
by order, which can misidentify a plane; separate reconciliation rejects that
incomplete capture. An experimental bounded in-memory quarantine holds copied
frames without MDA plane labels until reconciliation. It is used in the test,
not yet in a production MDA sink. An experimental sequence-only UniMMCore PVA
adapter and custom engine now exercise the loopback service with fake serial
and camera. The caller must hold the cooperative MCU reservation before releasing
the armed sequence; the PVA client rechecks it at request admission. This does
not reserve the camera or survive a service restart. Manual moves and physical
position readback are unsupported; abort,
camera readiness and physical hardware are not qualified. The custom engine
waits a bounded interval for MCU completion before normal MDA teardown; timeout
or cancellation requests an abort, without proving a physical stop.
The stage accepts the same `trigger_mask` as the Redsun stack definition; fake
MDA/PVA tests cover the default and mask 6, not physical camera triggering.

The experimental `services.unified_mcu` controller requires an explicit port and
PVA prefix, identifies and blanks at startup, serializes LED and stack requests,
and publishes the observed event prefix after completion or failure. Readiness is
an explicit host assertion, not a sensed camera signal. PVA clients have no
authorization isolation, and process shutdown and abort do not prove a motor
stop. This surface has fake-serial tests including a loopback PVA client; it is
not a qualified acquisition service or a substitute for independent safety
and recovery work.

An opt-in cooperative MCU reservation blocks competing stack and manual-output
requests while held. It does not authenticate clients, reserve the camera,
survive a service restart, expire on disconnect, or prove that motion stopped.
Abort remains available without the token. Uncertain serial outcomes prevent
normal reservation release, requiring reconciliation.

The same owner exposes automatic camera pulse enable/disable and one-shot pulse
requests for outputs 1–3. Status records command acknowledgements, not measured
edges. A known disabled setting blocks stack admission; initial state is unknown
until explicitly set. The firmware's latched gate is not exposed by this surface.
Stack definitions accept `trigger_mask` (default 1); bits 1, 2 and 4 select
outputs 1, 2 and 3 in the current firmware. The PVA service accepts requests with
or without this field; the typed client sends it explicitly.
Physical timing, output voltage, inter-output skew and camera response are not
qualified.

An experimental async stack orchestration arms a caller-supplied bounded camera
adapter before submitting the stack, then collects independent camera counters
and reconciles them. It requests abort on post-submission failure and closes the
camera adapter; neither action proves a physical stop. It has synthetic-camera
and loopback PVA tests, but no real camera adapter or acquisition session yet.

The Micro-Manager camera service also offers `--defer-initial-frame`, an
experimental opt-in that sizes a monochrome buffer and an initially empty
capture from camera-reported geometry and pixel type without snapping at
startup. Prepare capture before starting acquisition; this mode rejects
unsupported layouts and refuses the normal per-frame snap fallback if a
sequence cannot start. This does not establish camera readiness or qualify
external triggering. In-process fake tests and a direct-service FLIR probe with
trigger mode off exercised deferred startup and capture. A supervised FLIR PVA
rehearsal also prepared and stored two free-running frames; external-trigger
behavior and a camera-ready handshake remain unqualified.
The camera's `sequence_started` PVA status reflects successful return from the
MMCore sequence-start call and clears when stop begins. It is published on the
service scan cadence and is not a hardware-ready or exposure signal.
The experimental one-use `PvaCaptureEvidence` adapter prepares a bounded
capture and waits for that status, returning only a store URI and service
frame count marked unassociated. The caller owns trigger configuration and
cleanup; this adapter does not establish a camera counter or a resource lease.
In deferred mode, setting and property writes are refused while a capture or
acquisition is active, and accepted property or pixel-type changes do not snap.
Capture layout is refreshed while idle; this check is not a cross-client lease.
The adapter has also been exercised with a physical FLIR in Line0 mode without
commanded pulses: collection timed out with zero saved frames and settings were
restored. This does not establish electrical trigger response or correlation.
The experimental `unassociated_stack` workflow combines this capture interface
with one MCU stack, returning the service count and controller report separately
without frame-to-plane associations. A loopback PVA test uses fake serial and
event-generated synthetic camera frames; ownership, external trigger setup,
and illumination readiness remain caller assertions.
It acquires the cooperative MCU reservation before camera preparation and requires
it again at stack request admission. It releases the reservation only after a
complete count and successful cleanup, and retains it on failure when still held.
The reservation does not protect the camera or survive a service restart.
The corresponding headless `unassociated_stack_plan` records the requested stack
and store in Bluesky run metadata and emits separate evidence fields. The
experimental `UnassociatedStackSession` exposes it for one camera and one MCU
service; a fake loopback session has passed, but physical acquisition is not
qualified.
`BoundedFrameBatch` copies frames under explicit limits and permits one flush
attempt. The experimental buffered camera-service path uses it; buffered frames
are not durable until the downstream store has closed successfully.
On failure the plan emits a diagnostic event with best-effort separate reports
and read errors before marking the run failed. It does not infer plane mapping
or a verified physical stop from those reports.

The experimental typed PVA client permits transaction tags to be reused after a
fresh configuration and arm on firmware 0.16. A cancellation or uncertain
outcome quarantines the instance against further submissions; the client does
not automatically replay an uncertain request. Its cooperative reservation does
not survive service restart. Explicit abort and progress reads remain available;
neither verifies a physical stop.

The independent `device.stack_reconciliation` function can compare complete
controller reports with a separately supplied camera-counter sequence and an
explicit baseline and exclusivity assertion. A matching sequence yields only
order-inferred associations; rejected sequences remain unassociated. It is not
yet connected to a real camera or acquisition plan, and a matching count cannot
prove trigger-to-exposure causality.

## Stage readback provenance

The MMCore stage service publishes a readback-provenance signal consumed by `MMStage.read_position()`. `DemoCamera` is labelled simulated and cannot be overridden to measured; other adapters default to unknown. A reviewed instrument profile may explicitly set `--readback-provenance measured` when its controller readback is verified, or `simulated` for another simulator. Unknown provenance returns an unavailable stage reading rather than measured positions. This change requires the updated service and client together: an older service lacks the new signal. Controller-readback provenance alone does not establish independent mechanical measurement or calibration.

## Experimental simulated stack model

`simulation.mcu_stack` is an in-process, hardware-free model for exploring bounded axes, exclusive ownership, a single-start finite stack and trigger/frame identity reconciliation. A headless Bluesky rehearsal emits simulated trigger events and, after successful reconciliation, simulated plane events. It includes a headless Redsun session and simulated plan presenter, but no PVA service, firmware integration, real camera, or physical timing model. An optional existing frame store can persist small synthetic pixels for rehearsal; this does not qualify physical acquisition.

```
Headless Redsun session -> simulated MCU stack -> simulated trigger identities
                                                 |
                                          fake camera identities
                                                 |
                                      exact frame reconciliation
                                                 |
                                   Bluesky simulated plane events
```

## Illumination and software frame capture

The bundle declares a 91-LED v1 profile (`panel91`, revision 1). Its Arduino service
implements two ASCII digits terminated with CRLF at 9600 baud, 8N1:

| Command | Pattern | Meaning |
|---|---|---|
| `00` | — | blank every output, no camera trigger |
| `01` | — | clear the shutter-closed flag, no camera trigger |
| `11`..`14` | 1..4 | the four semicircles (`group1`..`group4` of the firmware) |
| `15` | 5 | centre LED |

`10` (array off) is available in the MM sketch but omitted from the service surface.
The LED index tables live in firmware; the host records the panel profile, pattern id
and command bytes. The `AHT-MM-ASCII-91/1` sketch accepts this dialect and emits no
camera TTL. An instrument deployment must verify its firmware identity and optical
patterns. The Arduino service checks the sketch's `90` identity response at startup
and when replacing the port. The default DPCT plan reads a fresh free-running frame after each pattern
write and does not infer a hardware trigger from the serial echo.

## The DPCT sequence

`redsun_aht.plans.dpct_software.software_dpct_plan` is the session's DPCT action.
It requires a camera that advertises free-running frames, supplies a fresh YX
readout, and names a local store. The sequence is:

1. opens one run carrying the run identity, the detector list, the axes and the pattern
   sequence with the panel profile;
2. stages the detectors, the stage and the illumination service;
3. writes `00`, then `01`;
4. for each of `11`, `12`, `13`, `14`: writes the pattern once, waits for a fresh
   frame per detector, appends its pixels and pattern identity to `RunFrameStore`,
   then emits an Event with the frame identity and command receipt;
5. closes each pixel store, writes `00`, unstages in reverse order and closes the run.

A timeout, a fault or an uncertain command closes the run as `abort` and returns a
`PlanOutcome` whose `partial` names the ordinals and patterns that were not produced.
The older `dpct_plan` remains an external-trigger contract and still rejects the
current `MMCamera`. Serial echo and a fresh free-running frame do not prove optical
pattern correctness or exposure timing; those require instrument qualification.

## Storage

One store per plane: a zarr v3 array at `<store>/<data-key>` grown one `(1, H, W)` chunk per
frame, an atomic manifest with shape, dtype, chunking, planned and stored frame counts,
per-frame checksums with optional setting revision and pattern id, the run and detector identity and
the setting timeline, and a `<store>/<data-key>.timing.json` sidecar for each timed capture.
The camera service's revision marks are sampled when it appends a frame; they
do not establish the configuration at exposure start. Software DPCT leaves the
per-frame setting revision unknown because its buffer and revision are read separately.
The writer emits manifest schema v2 and the receipt reader accepts v1 and v2.

Completeness is planned against stored frames. `close()` always writes the manifest, so an
interrupted run is a closed, refused short store rather than an open directory, and the
receipt layer names the condition: missing frames, extra frames, partial or overfull run,
checksum mismatch, mixed schema, or a store that already holds a run.

Software DPCT writes into `RunFrameStore` directly. Live capture constructs it
in the camera service. The live plan does not yet publish a receipt document;
receipt assembly needs the run and detector identity from the caller.

## Testing

`pwsh -NoProfile -File scripts/dsh/run-tests.ps1 tests -q` runs the headless suite with a
run-scoped temporary root. The runner refuses to start under an elevated token, because
pytest creates its temporary trees with a restrictive DACL that a normal session then cannot
delete, and it removes those trees after every run.

The suite is headless: no test constructs a `QApplication` or a napari viewer. GUI behaviour
is qualified on the host, and hardware behaviour only under the instrument gate below.

## Hardware gate

Nothing in this repository qualifies the physical instruments. Before the DPCT path is used
on hardware, an operator must: confirm the camera's actual external-trigger mode and line
(the packaged profiles are simulation profiles), confirm the LED panel revision against the
firmware actually flashed, and verify that `00` blanks the outputs. Motion, illumination and
firmware changes need explicit authorisation.

The default camera frame store records the finite frame count requested when a
capture opens in its manifest; that count remains fixed for the active capture.
A partial closed store can therefore be distinguished by comparing its frame
count with `planned_frames`. Unbounded captures have no planned count, and
custom stores must implement the optional count-binding capability to record it.
A complete count alone does not establish trigger correctness or durability.

An experimental `SoftwareSnapCamera` uses a distinct `snap_request` PVA
attribute to request one idle software snap from a deferred-frame camera
service. It is separate from `MMCamera.capture_frame`, which waits for a new
free-running frame. The request is not retried; a lost response is uncertain.
The returned count and buffer are separate reads, so the caller must have
exclusive camera ownership. This does not implement a camera lease or hardware
triggering.

```
SoftwareSnapCamera -> one PVA snap request -> idle camera service -> MMCore snap
MMCamera           -> wait for new frame  -> free-running camera stream
```

The unified MCU serial owner can also request a serialized read-only X-family
aggregate state, preserving the raw reply and controller-reported target,
position, speed and status. The separate `PvaXState` client requests a correlated
state through that same serial owner; it is not an independent mechanical
measurement or a movable-stage adapter.

An independent experimental `PvaSteppedAxis` provides a bounded host-stepped
PVA axis; `SteppedPvaStage` is its optional non-sequenced UniMMCore adapter.
Each move is one bounded single-plane MCU stack, including a pulse
at the selected output, followed by controller-state verification. The caller
must hold a cooperative MCU reservation and assert camera readiness; the client
rechecks its reservation at request admission. Camera exclusivity remains the
caller's responsibility, and stop/abort is unverified. Its
position is converted controller readback, not independent mechanical metrology.
It has fake-runner and fake-PVA coverage and has not been qualified for physical
MDA.

```
UniMMCore software MDA -> stepped PVA stage -> one-plane MCU request per move
                      -> separate software camera snap per plane
```

The experimental `software_stepped_plan` prevalidates a finite list, then uses
the same stepped adapter and one idle software snap per plane in a headless
Bluesky run. The caller retains the MCU reservation and asserts camera ownership; emitted
position and frame summaries are associated by host order, without
exposure-time proof. An optional create-only store records pixels and planned
frame count; partial captures retain a partial manifest. The plan consumes the independent PVA axis
directly; it is not yet a general Redsun motor device.

The experimental `UniMMCoreStepSink` writes finite, Z-only MDA frames through
the same create-only writer as the Redsun software-step plan. It checks host
indices in order and latches callback failures; those indices are not measured
exposure positions. A partial store records the planned and saved counts.

The experimental `Esp300UniStage` adapts an injected, already-admitted
single-owner ESP300 Z transport to a non-sequenced UniMMCore focus stage. It
checks configured micrometre bounds, converts to millimetres, and latches
uncertain motion and stop requests. It does not open the controller, establish exclusivity or verify a physical stop.
The separate experimental `Esp300ZSerial` implements bounded PA/MD/TP over a
caller-owned endpoint. Its admission factory takes a read-only snapshot on that
endpoint, and each move checks idle and matching target/position before PA.
It does not enable or home the motor or establish endpoint exclusivity. A fake transport
and camera exercise actual UniMMCore software MDA; physical ESP300 integration
remains unqualified.

```
UniMMCore software MDA -> bounded focus adapter -> injected ESP300 transport
                                                 -> bounded serial transport (fake tested only)
```

The experimental direct `UnifiedMcu.software_shots` workflow targets firmware
0.4: it configures and admits a bounded MCU sequence, waits for each matching
shot-ready report, and awaits a caller-supplied software camera snap before the
next shot. An uncertain result attempts abort and quarantines the serial owner.
The callback and camera exclusivity are caller responsibilities; this direct
workflow is not yet exposed by the PVA service or qualified on physical devices.

`acquire_software_shots` connects the direct owner to a separately controlled
software-snap camera and caller-provided frame sink. It records ordered shot
metadata and host receipt times after the camera-service transfer; these are
not sensor exposure timestamps. Fake serial and loopback fake camera tests cover
completion and partial sink failure. Camera exclusivity remains caller-managed.

The experimental `run_software_shot_mda` composes one finite software MDA run
with the exclusive direct MCU shot owner. A runner worker waits for each
controller-ready report, snaps one frame, calls a synchronous sink and only
then acknowledges that shot to the owner. MDA indices represent host-order
associations, not measured exposure positions. This is an experimental direct
integration with synthetic tests; camera ownership remains caller-managed, and
it is not a PVA service or a hardware-triggered acquisition.

For caller-managed open illumination, the direct bridge can select
`auto_shutter=False`: it requires the core's auto-shutter to be disabled and a
constant pattern per plane, and uses one existing MCU shot per plane while the
runner consumes all frames before advancing. The caller owns the illumination
lifecycle. This experimental path has synthetic and fake-sketch coverage; it is
not yet connected to the acquisition plans. With `auto_shutter=True`, the
bridge requires an enabled core auto-shutter and configured shutter device,
and uses per-frame handshakes while the core manages shutter cycling. This has
synthetic and demo-device coverage, not physical shutter validation.

```
MCU serial owner -> ordered shot-ready handshake -> MDA runner -> camera snap
                  <- frame sink completed         <- synchronous frame sink
```

The experimental callback-driven `timelapse_plan` provides a Bluesky
position/time/Z/channel/pattern loop, per-snap or acquisition shutter gating,
and host start timestamps. Its device operations and frame storage are caller
callbacks. It is a synthetic-tested orchestration slice, not yet connected to
the MCU bridge, full donor manifest, survey/revisit chain, or presenter. An
experimental `TimeLapseFrameSink` can persist ordered frames with the existing
create-only writer and a host-time/index sidecar; its completion flag covers
the recorded frame count, not the full instrument run.

```
time-lapse plan -> caller motion/capture/shutter callbacks -> caller-owned devices
                -> requested indices and host timestamps -> Bluesky events
```

An experimental `reviewed_revisit_timelapse_chain` passes the ordered targets
returned by an injected revisit into a separate injected time-lapse run, with
an optional XY anchor return after success. It has synthetic composition
coverage; production review artifacts, stage bindings and presenter registration
are not connected.

```
caller-owned revisit -> returned target IDs/XY -> separate time-lapse plan
                                                   -> caller-owned frame store
```

The experimental `timelapse_mcu_plan` composes position/time/channel blocks
with direct software camera snaps, the MCU shot owner and time-lapse sink; it
does not use the pymmcore-plus MDA engine. Its synthetic coverage includes both stage bindings and partial
storage on sink failure. It checks camera exposure readback and can enforce a
host frame-start period across blocks and a per-position timepoint-start floor.
Time/position order can be selected, with a global timepoint floor for
time-outer order; a missed floor may continue or abort. These ordering and
deadline options have synthetic coverage.
Block events are emitted after completion; the caller owns the sink
and device lifecycle. Presenter and chain integration, and physical validation
remain incomplete. The experimental time-lapse adapter has also been rehearsed
with physical devices; this is not acceptance of all settings or timing modes.
Firmware capability and the caller's selected hold policy determine whether
the MCU LED remains on across blocks.
Experimental per-channel pattern/exposure recipes and caller-supplied native
asynchronous illumination gates have synthetic coverage. A gate can represent
a source enable or a mechanical shutter and must raise if its operation fails;
no particular physical gate backend or per-frame MCU LED gating is qualified.

`acquire_pva_internal_planes` is a separate experimental service-owned path:
the MCU's host-ready event gates each bounded, internally triggered camera
service capture, and the service stores each plane before the next shot. It
avoids opening a second camera core. Exclusive camera ownership is currently
caller-asserted; the per-plane association is inferred from host order. It has
fake PVA tests and a limited physical rehearsal, but is not yet wired into the
time-lapse presenter or the reviewed-revisit chain.

The experimental `pva_timelapse_plan` composes that path into position,
timepoint and channel blocks using the same service owners. It reads the camera
exposure after channel configuration, publishes host-associated plane events,
and takes caller-supplied acquisition gate and external-stage callbacks. Store
URIs are unique across the run; partial completed-plane events survive a later
capture failure. Its cleanup closes the supplied gate and requests an MCU blank
while the reservation remains certain. Its trigger configuration and camera
exclusivity are caller assertions. This plan has fake-device and loopback tests;
presenter registration, physical gate bindings and a physical plan run remain
incomplete. It keeps the acquisition gate open across consecutive blocks of the
same channel, closes it before changing channels, and allows the configuration
callback to select the next gate. A fake reviewed-revisit chain carries returned
target IDs and order into run metadata. Timepoint lateness is host-measured and
the caller can continue or abort on a missed deadline.
An optional per-plane host timing callback reports readiness, external-stage
completion, camera-start acknowledgement, store completion and camera-stop
acknowledgement; these observations are not sensor-exposure timestamps.
When a channel supplies changing per-frame LED patterns, the same plan sends a
software SHOT and requests a camera-service snap per frame, storing each plane
in the service and emitting host-associated pattern events after the block returns
or fails, including acknowledged shots in an incomplete plane. Constant-pattern
frames continue to use the finite burst path. An optional `journal_path` writes
a create-only host-local JSONL run and event record, including plane-store URIs;
it flushes each line but does not fsync or provide crash-proof per-shot logging.
The camera store itself does not contain these host-inferred pattern associations. A
camera frame publication lock serializes snap and live-update acknowledgements.
The camera controller also has an opt-in finite buffered capture mode: it starts
one bounded internal sequence per capture, waits briefly for a full sequence to
stop naturally, and checks overflow and extra frames. An interrupted sequence is
explicitly stopped. The default continuous acquisition path is unchanged. The
finite mode has fake and limited physical coverage; its trigger configuration
and exclusive camera ownership remain caller responsibilities.
The camera service accepts `--finite-capture` together with
`--buffered-capture --defer-initial-frame`. It uses continuous acquisition for
live view when no capture was prepared before acquisition starts; a capture
opened after a live sequence has already started stays continuous until restart.
The time-lapse end state is selectable as blank or center; the LED gate may be
a no-op. Center uses the legacy pattern command, which can generate an
automatic camera pulse when enabled. Acquisition failures request blanking.
The experimental `auto_shutter=True` policy requires caller-supplied asynchronous
per-shot open and close operations and routes even constant patterns through
software snaps; a no-op LED gate is valid when no physical gate is required.
The per-shot close is attempted after snap failure. The callbacks are not a
physical shutter binding and their electrical or optical behavior is unverified.

An experimental `PvaTimeLapsePresenter` exposes this plan through `plan_map`
when explicitly given service owners and instrument callbacks. A fake-service
PVA loopback exercises that entry. It is not yet registered by the production
session; the original entry leaves reservation and trigger configuration to its caller.
Its managed entry acquires the MCU reservation, allocates run IDs and releases a
certain reservation after the plan; an uncertain reservation remains for
reconciliation. Camera exclusivity and trigger configuration remain asserted by
the caller, and concrete service/gate bindings are still supplied externally.
An experimental headless `PvaTimeLapseSession` injects one camera and one MCU
PVA client into that presenter; it offers the managed plan after explicit
instrument callback configuration. A human-readable IX83-style confirmation
label and the MCU command IDs are included in the Bluesky run metadata; the
Bluesky UID remains the unique run identity. This session does not yet replace
the application's production session or provide a camera-service lease.

```
MCU service -> plane ready -> camera service -> bounded plane store
            <- host confirms after camera stop and store completion
experimental presenter -> Bluesky time-lapse -> service captures -> events / stores
                   -> caller-supplied gate and position/channel bindings
                   -> optional per-shot async gate around camera snaps (experimental)
```

The experimental `acquire_automatic_shots` arms an injected triggered camera,
concurrently drains its independently numbered frames and runs one direct MCU
shared-config automatic stack. It checks the final MCU software count, camera
baseline and sequence continuity before providing order-inferred plane/pattern
labels; mismatched frames remain unassociated. Host receipt times are supplied
by the camera adapter, not sensor timestamps. This is synthetic-test coverage
only: no physical triggered-camera adapter, wiring verification or stop guarantee
is provided.

```
Triggered camera arm -> concurrent frame drain -------------------> reconciliation
MCU serial owner     -> one automatic shared-config execution ---> final count
```

The experimental `UniMMCoreTriggeredCamera` adapter provides a finite camera
sequence for the automatic capture workflow. The caller configures and verifies
its external-trigger property and supplies a validated metadata counter source
and sequence-specific baseline. A serialized worker drains and copies frames,
records host receipt times, detects buffer overflow and requests sequence stop.
Tests use fake cores; the adapter has also been exercised with a real camera in
internal-trigger diagnostic mode. Neither case validates external trigger wiring
or proves that metadata counters detect every missed exposure.

```
UniMMCore finite sequence -> serialized frame drain -> independent metadata counter
                                                -> automatic capture reconciliation
```

The experimental unified PVA service also supports a reservation-token-scoped
software-shot request: it publishes each MCU ready event and waits for matching
host confirmation before advancing. Its typed client supports bound and unbound
MCU stage modes; unbound nominal positions are not physical stage readback. A
separate MDA bridge can call an external per-plane hook for another stage. The
handshake and service are fake/loopback tested; external leases, physical
readiness and optical settling are not established by these reports.

```
MDA camera -> PVA shot client -> reserved MCU serial owner -> SHOT readiness
           -> optional external per-plane stage hook          -> optional MCU X move
```

### Direct Spinnaker camera

The experimental `spinnaker-camera` service provides a direct PySpin camera
alongside the MMCore service. Install the matching vendor wheel separately in
the Python environment; PySpin is imported when the service opens the camera and
is not a project dependency. Select a camera by serial number and use the
`SpinnakerCamera` PVA client or `PvaSpinnakerTimeLapseSession` for the shared
capture and plan interfaces. A camera can be owned by only one service at a time.

The default transport format selects `Mono12p` or `Mono12Packed` when available;
10-bit packed and other supported formats can be selected explicitly. PySpin's
image processor converts frames to owned `uint16` Mono16 arrays before the SDK
buffer is released. Packing reduces transport bandwidth; saved arrays are not
packed. Capture metadata records the source and output formats, native frame ID
and camera timestamp. Camera timestamps are not synchronized with the MCU.
Finite captures are subject to both the frame and byte buffer limits, based on
the converted array size. Incomplete frames, observed frame-ID gaps and available
stream error counters fail capture rather than silently reporting success.
Unbounded live recordings stream directly to the store; finite captures use
the bounded memory buffer. A free-running camera may continue producing frames
after a finite recording closes, while triggered captures check for extra frames.

For per-pattern software triggering, a continuous camera sequence stays open for
each plane. Each MCU shot is followed by one camera software-trigger request,
which acknowledges after the camera service has received the frame. The
automatic external-trigger stack instead arms one continuous camera capture for
the full stack. Both paths have been exercised with a physical FLIR and MCU;
other plans and camera models still require their own integration validation.

`pva_gated_stack` is the host-paced alternative: it enables `c1`, arms a single
external-trigger sequence for the whole stack, then sends one `SHOT` per pattern
and waits for its frame before advancing. Firmware 0.17 pulses the camera after
each `SHOT` pattern is rendered. `c0` suppresses these pulses for the existing
software-snap path. Legacy pattern commands and the autonomous `SHOTAUTO` path
remain available. Stored frames are associated with pattern and plane by their
matched order and counts, not by an independently measured exposure timestamp.

For the existing camera or DPCT session, a custom profile can retain the
`camera1` service name and prefix while replacing its service entry with:

```yaml
camera1:
  plugin_name: redsun-aht
  plugin_id: spinnaker-camera
  prefix: "AHT-CAM1:"
  args:
    serial: "CAMERA_SERIAL"
    pixel-format: auto
    buffer-limit-mib: 512
```

Use this entry under `services`, then pass the profile to the corresponding
`aht session` command. The regular camera device uses the common PVA surface;
the typed `SpinnakerCamera` additionally exposes the software-trigger command.
Available GenICam properties include `TriggerDelay` (in microseconds),
`LineSelector` and `LineInverter`, as well as the trigger mode and source;
the camera determines which of them are available and writable.
An interrupted or unacknowledged software trigger faults and stops acquisition,
because the exposure outcome cannot safely be inferred or repeated.
During live viewing the service pauses and resumes the SDK sequence to apply
settings, without issuing an extra exposure; a changed image shape is published
with the next frame. Settings that could change the capture are refused while a
store is open.
