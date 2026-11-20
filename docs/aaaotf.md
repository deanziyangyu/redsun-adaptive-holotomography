# HoloSPIM AAAOTF laser shutter

The HoloSPIM profile binds its `laser` device to the persistent AA Opto MDSnC
service on loopback PVAccess. The service holds one PyMMCore AAAOTF adapter
connection. A Redsun session leases the writer and submits named profiles
through the RunEngine. The service remains available across session exits.
The MCU continues to run LED pattern and trigger plans.

Use the AHT environment with the `camera-mmcore` extra. Supply a dedicated
Micro-Manager configuration with only the AAAOTF and its serial adapter, a
matching MMCore adapter directory, and a local calibrated profile. Resolve the
serial endpoint from the current Windows device assignment at commissioning.
Keep the resolved configuration local.

```powershell
aht service aaaotf -- --config $AotfConfig --adapter-path $MmAdapters --profile $LaserProfiles --core-log $AotfLog
aht session holospim --profile $HoloSpimProfile
```

Run the service in its own console or supervisor. Use a commissioned live
HoloSPIM profile for the desktop command. The packaged profile starts in mock
mode. The service's default PVA prefix is
`HOLOSPIM-AAAOTF:`. The HoloSPIM `laser_service` entry supplies the prefix and
loopback address without a process module. HoloSPIM host profiles inherit this
attachment unless they explicitly supply a laser binding.

The profile JSON uses schema `redsun.aaaotf-profiles/v1`. It contains
`device_label`, a `slots` map keyed by string channel numbers, and a
`profiles` map whose values are lists of integer channels. Each configured
slot has `frequency_mhz`, `maximum_intensity` (the adapter value), and
`power_percent`. Define `all_off` as an empty list. At startup the service
verifies all eight slots OFF before applying the configured RF settings.
For a shutter-only run, use `preserve_rf_slots` in place of `slots`. List
each eligible channel as an integer. The service uses its existing frequency
and power and applies only the profile's ON/OFF state.

The HoloSPIM **Hardware** dock provides **All Off**, **488 nm**, and **561 nm**
presets and a separate **All Off** button. It displays the active profile,
slot states, current channel, RF properties and service fault. Controls are
admitted when the session resolves one attached AAAOTF device. RF and preset
buttons require a healthy service status. Manual requests wait for an idle
acquisition RunEngine and respect its device locks.

The RF form accepts channel 1–8, frequency 50–200 MHz, maximum intensity
0–2200 in adapter units, and power 0–100%. **Apply RF settings with slots off**
closes all slots before changing those values. Readbacks appear separately
from the editable requested values. Select a commissioned wavelength preset
to enable its slot after editing RF settings.

The same preset action is available as `holospim_laser_profile` in the
Acquisition panel, with RF edits under `holospim_laser_settings`. The wavelength
presets close the shutter, apply and verify the RF settings, then turn
slot 1 ON. Both presets use 100% power. The 488 nm preset uses 112.28 MHz
and maximum intensity 1750. The 561 nm preset uses 92.63 MHz and maximum
intensity 1900. **All Off** closes every slot. The service and serial
connection stay open while you switch presets.

Bluesky plans can also use `abs_set(laser, profile, wait=True)` with the
service's shutter profile names.

The status signal includes all properties reported by the loaded AAAOTF
adapter and lists those writable while connected. Send a property mapping
through the same leased device to set the channel and RF properties at run
time. For example, a plan can do the following:

```python
yield from abs_set(laser, "all_off", wait=True)
yield from abs_set(laser, {
    "Channel": 1,
    "Frequency (MHz)": 92.63,
    "Maximum intensity (dB)": 1900,
    "Power (% of max)": 100,
}, wait=True)
```

`State` accepts 0 or 1. Channel and RF changes require all slots OFF.
A batch applies channel and RF settings before its optional State write.
The service keeps the serial link open. Name, Description and Port are
visible in status. Port is a pre-initialization adapter property.
RF property values reflect MMCore's adapter readback for its current
selection. Optical power and firmware state require separate measurement.

Session release or writer lease expiry requests all slots OFF. A USB loss
requires operator recovery and a fresh all-off service startup. Treat a failed transition as an
unknown outcome until checked. Shut down the independent process with Ctrl+C
when taking the instrument out of service.
