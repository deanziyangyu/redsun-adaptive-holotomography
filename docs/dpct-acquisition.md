# DPCT acquisition orchestration

Status: complete hardware-free orchestration slice; powered stage motion,
native-v1 MCU hardware, illuminated camera timing, and scientific
reconstruction gates remain pending.

## Ordering

`DpctRunner` executes the scan-position product as the outer loop and committed
illumination patterns as the inner loop. For each position it:

1. validates and commands the composite stage;
2. waits for motion-complete and measured tolerance agreement;
3. selects one committed MCU pattern;
4. triggers every configured detector concurrently;
5. reads the ordered lossless frame metadata;
6. verifies run and detector identity;
7. acknowledges each exact frame sequence; and
8. appends one immutable `DpctShotRecord`.

The resulting default logical grouping is position/pattern/detector with each
frame retaining its independent detector channel. This maps to the planned
`Z,C,Y,X` array grouping without collapsing detector, modality, view, or time
coordinates into hidden dimensions.

## Traceability and cleanup

Every shot records its global shot index, source scan index, pattern ID,
measured `PoseSample`, and ordered detector frames. Frames must carry the exact
recipe run ID and detector ordering or the run fails before acknowledgement.

Normal completion, validation failure, detector failure, and cancellation all
enter the same bounded cleanup chain: ordinary detector stop, MCU `ALL_OFF`,
composite-stage stop, and detector disconnect/shared-memory unlink. `ALL_OFF`
and stage stop are normal-operation cleanup only; this AHT implementation does
not claim a safety subsystem.

## Hardware-free profile

`aht simulate-dpct` constructs one simulated Z axis, a native-v1 simulated MCU
with two committed four-address patterns, and two isolated mock detector
services. It executes four shots (two positions by two patterns) and releases
all shared memory. It opens no serial port, IOC, MMCore context, vendor SDK,
camera, or physical stage.
