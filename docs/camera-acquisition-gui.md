# One-frame camera acquisition GUI

Status: implemented service client; physical PVCAM and FLIR one-frame gates
passed on 2026-08-23. It is not a live continuous-acquisition application.

`aht camera-gui --prefix PREFIX:` connects only to an already-running
`aht-camera-ioc` service. The GUI contains no Micro-Manager, vendor SDK, stage,
illumination, or MCU construction. Its single capture action follows this exact
sequence:

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

## High-rate non-goal

This widget is not a live viewer. Repeatedly invoking its single-frame action
would serialize camera trigger, shared-memory copy, checksum, EPICS metadata,
and acknowledgement work, thereby throttling acquisition. A future GUI uses a
separate latest-wins preview ring with an explicit decimation ratio (for
example 1:10) and independently reported preview loss. It must never delay the
lossless storage lane. See [camera streaming and backpressure](camera-streaming-backpressure.md).
