# One-frame camera acquisition GUI

Status: implemented service client. Physical PVCAM and FLIR one-frame gates
passed on 2026-08-23; the live-view path is sequence-backed and has separate
camera-only timing evidence. It is not a durable streaming-acquisition UI.

`aht camera-gui --prefix PREFIX:` connects only to an already-running
`aht-camera-ioc` service. The GUI contains no Micro-Manager, vendor SDK, stage,
illumination, or MCU construction. Its **Capture one frame** action preserves
the broad-compatibility validation path and follows this exact sequence:

1. Request service connect and verify the admitted camera reaches `ready`.
2. Arm and trigger exactly one service-owned lossless shared-memory frame.
3. Copy the declared slot, recompute SHA-256, and acknowledge its exact
   sequence number.
4. Request ordinary stop and disconnect; then render only the detached copy.

The optional hardware test starts that service with a camera-only MMCore
configuration beside an idle simulated peer. It runs the Qt widget with an
offscreen platform, asserts the preview is present, verifies the peer published
no frame, and confirms the real service disconnects. PVCAM uses `SerialNumber`;
SpinnakerC uses `Serial Number` in the current Device API 75 installation.

The test writes the copied frame to `CameraCaptureZarrStore`, a separate
single-camera OME-Zarr layout carrying CZYX `(1, 1, Y, X)` data and independent
source/stored checksums. The generic offline mean-projection and quality-metrics
workers verify and consume that bundle. This is deliberately not labeled DPCT,
qOBT, or IDT reconstruction: an unilluminated one-frame camera test has no
native-v1 pattern sequence, illuminated sample, calibration, or measured pose
series required by those scientific models.

## Sequence-backed live view

Repeatedly invoking the single-frame action would serialize camera trigger,
shared-memory copy, checksum, EPICS metadata, and acknowledgement work,
thereby throttling acquisition. The **Start live view** action instead starts
one continuous MMCore sequence. The service drains MMCore's own circular buffer
at a maximum 60 Hz, publishes only the newest image as an EPICS byte waveform,
and records preview skips separately from buffer overruns. The GUI polls that
latest-frame PV and never issues `TRIGGER` during live view.

`snapImage()` is intentionally retained for **Capture one frame**, validation,
and adapters that do not yet support sequence acquisition. Finite list scans
use `startSequenceAcquisition(frame_count, interval_ms, stop_on_overflow)` at
the backend boundary; the live viewer uses
`startContinuousSequenceAcquisition(interval_ms)`. Neither preview mode is a
durable streaming/storage contract. See [camera streaming and backpressure](camera-streaming-backpressure.md).

For the 64x64 PVCAM FCS live profile, start the camera-only IOC from the
admitted `pvcam_base.cfg` with `--pvcam-fcs-profile`; it explicitly applies the
validated ROI, readout, clearing, trigger, and streaming settings after MMCore
loads the base configuration. Leave `--live-publish-hz` at its 60 Hz default:
the 2026-08-25 comparison found 10 Hz slightly slower (3.125 s versus 3.006 s
for 2,000 frames) without reducing camera-buffer overruns, which were zero in
both cases.
