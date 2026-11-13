# AHT architecture

AHT composes instruments from Redsun profiles. A FastCS service owns each
hardware connection and exposes device controls and readbacks over PVAccess.
Ophyd-async clients expose those signals to Bluesky plans and presenters.

## Instrument profiles

| Selector | Devices |
|---|---|
| `holospim` | MMCore camera and survey stage, plus the ESP32-S3 MCU for X28 fast Z, LED illumination and rotation |
| `arh` | ASI LX4000 XYZ, FLIR via Spinnaker, Lumencor Spectra via MMCore, and Arduino Due via unified EMMSH |

The [README](../README.md#run) shows how to check either profile in mock mode
and layer in host settings. Both profiles build headless device sessions.
The camera and DPCT GUI sessions have separate entry points.

## Acquisition and storage

```text
FastCS service → ophyd-async device → Bluesky plan → RunEngine documents
                                                        ├─ preview events
                                                        └─ stream asset references
```

| Source area | Role |
|---|---|
| `configurations/`, `profiles/` | CLI dispatch, service selection and session wiring |
| `services/`, `device/`, `robotic/` | Service owners, PVA clients and instrument-specific adapters |
| `contracts/`, `plans/` | Acquisition requests, frame and run identity, and plan orchestration |
| `presenter/`, `view/` | Run-document routing, detector controls and image display |
| `storage/` | Zarr frame storage, manifests and plane receipts |

Live preview follows per-detector Events from the run. The camera service can
write frames directly, and software DPCT writes frames by illumination pattern.
`RunFrameStore` records frame checksums and preserves partial captures. Frame
counts and host-ordered associations do not establish exposure timing.

The experimental stack and time-lapse paths coordinate camera capture,
illumination and motion through the selected service owners. The Due service
exposes commanded rotary counts, stop and LED blanking. Calibrate physical
axes, shutter behavior and trigger timing on each setup before use.

## Testing

Use the [Windows headless runner](../scripts/dsh/README.md) for pytest. The suite
uses synthetic inputs, fake backends and loopback PVAccess. Keep GUI windows
and physical devices out of headless checks.
