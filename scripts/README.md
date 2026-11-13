# Launchers

The application is PVA-only and is normally started through the PowerShell
wrapper, which prepares the host paths before Python imports anything. The
`aht` command itself stays import-light: it does not import napari, Qt or
`pymmcore_plus` until a session or service actually runs.

## `scripts/run_session.ps1` — the host entry point

Run it from anywhere; it resolves the repository from `$PSScriptRoot` and uses
`.venv\Scripts\python.exe`.

```powershell
.\scripts\run_session.ps1 -Check                      # resolve and probe every path
.\scripts\run_session.ps1                             # list sessions and services
.\scripts\run_session.ps1 session <name>              # run one session
.\scripts\run_session.ps1 service <name>              # run one PVA service
.\scripts\run_session.ps1 session <name> --prefix AHT:CAM:PVCAM:
.\scripts\run_session.ps1 -DataRoot E:\redsun-aht-data session <name>
```

Everything after `Command` is forwarded. Do **not** insert the `--` separator
that `aht` accepts: PowerShell rejects a bare `--` in a script invocation.
Use `-Check` rather than `--check` for the same reason; a lone bare `--check`
is still recognised.

The wrapper exports:

| Variable | Value |
|---|---|
| `REDSUN_DATA_ROOT` | resolved data root |
| `REDSUN_NAPARI_CACHE` | `<repository>\.napari_cache` |
| `REDSUN_AHT_REPO` | repository root |
| `PYMM_LOG_FILE` | `<data root>\pymmcore-plus.log` |
| `TEMP`, `TMP` | `<data root>\tmp` (the launcher then moves them to a short PVA root) |
| `CUPY_CACHE_DIR`, `CUDA_CACHE_PATH` | `<data root>\tmp\cupy`, `<data root>\tmp\cuda` |
| `PYTHONPATH` | `<repository>\src` |
| `EPICS_PVA_ADDR_LIST` | `127.0.0.1` |
| `EPICS_PVA_AUTO_ADDR_LIST` | `NO` |

The data root defaults to `R:\redsun_aht_data`. If that drive refuses writes,
the wrapper falls back to `<repository>\.redsun_data` and prints the fallback.
An explicit `-DataRoot` or inherited `REDSUN_DATA_ROOT` is never replaced: it is
refused with its exact path when unwritable.

## `scripts/launch_app.py` — the bootstrap

Call it directly only for diagnostics; the wrapper does this for you.

```powershell
.\.venv\Scripts\python.exe scripts\launch_app.py --check
.\.venv\Scripts\python.exe scripts\launch_app.py session <name> --prefix AHT:CAM:PVCAM:
```

It redirects napari's cache and Redsun's data root, ensures a short writable
`TEMP`/`TMP` for `p4p`/libevent, then calls `redsun_aht.__main__.main`. The
redirect is not a global `platformdirs` patch: the Micro-Manager camera service
needs the real local app data directory to find its device-interface bundle.

## `aht` — the command line

`aht` is the installed entry point (`redsun_aht.__main__:main`). It can run
without the wrapper for inspection, but a real session needs the environment
the wrapper prepares.

```powershell
aht --check
aht --list
aht session <name> -- <arguments>
aht service <name> -- <arguments>
```

`--list` prints the configured sessions and services. Until the coordinator
adds `registry()` to `redsun_aht.configurations`, both sections print
`no sessions registered yet` / `no services registered yet`, and dispatch is
unavailable. `aht --check` probes the repository, data root, temporary root,
napari cache and Redsun data root.

### Registry contract

`aht` reads one function from `redsun_aht.configurations`:

```python
def registry() -> tuple[Mapping[str, Entry], Mapping[str, Entry]]:
    """Return (sessions, services), each keyed by the CLI name."""
```

`Entry` is `Callable[[Sequence[str]], int | None]`: it receives the arguments
after `--`, runs the session or service, and returns its exit status or
`None`. The package must keep Qt and napari out of its import path so `--list`
stays headless.

## When `--check` fails

`-Check` runs the launcher first and then reports; the row that says
`NOT WRITABLE` is the one to fix.

- **`data root ... NOT WRITABLE`** — pass `-DataRoot` pointing at a writable
  volume, or repair the ACL on the exact path in the message. The default
  `R:\redsun_aht_data` falls back to the checkout, so a failure here usually
  means the fallback is also unwritable.
- **`redsun data ... NOT WRITABLE`** — you ran `aht --check` directly. On this
  host `%LOCALAPPDATA%\redsun` refuses writes; run
  `.\scripts\run_session.ps1 -Check`, which relocates Redsun's root before the
  check.
- **`temp root ... NO short writable temporary root found`** or
  **`pva ... event base FAILS`** — `p4p`/libevent needs a writable `TEMP` no
  longer than about 90 characters. Set `TMP`/`TEMP` to such a path, or let the
  wrapper use `<data root>\tmp` and the launcher's short fallback.
- **`napari cache ... NOT WRITABLE`** — delete the directory or set
  `REDSUN_NAPARI_CACHE` to a writable path. napari rebuilds its theme SVGs, so
  an empty directory is fine.
- **`aht: error: unknown session ...`** — run `--list`; the registry belongs to
  the coordinator and may still be empty.
