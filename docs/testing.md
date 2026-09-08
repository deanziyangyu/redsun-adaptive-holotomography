# Test environment notes

## Semantic test groups

The suite remains in one `tests/` directory because a test may exercise more
than one boundary. `tests/conftest.py` applies stable pytest markers from a
central file-to-group map, avoiding duplicated or path-coupled tests.

| Group | Purpose | Selection |
|---|---|---|
| `core` | Pure domain, manifest, transport, and buffer contracts | `-m core` |
| `service` | Device adapters, simulated services, isolated IOCs, and stage/MCU behavior | `-m service` |
| `processing` | Detached workers, reconstruction, storage, and GPU orchestration | `-m processing` |
| `redsun` | RedSun-owned profiles, containers, plans, presenters, and `QtView` composition | `-m redsun` |
| `gui` | AHT-customized Qt/Napari user-interface construction | `-m gui` |
| `integration` | Cross-boundary CLI, profile, storage, catalog, and architecture checks | `-m integration` |
| `hardware` | Explicit physical-camera, stage, or GPU admission tests | `-m hardware` |
| `catalog` | Optional local Tiled catalog runtime | `-m catalog` |

For the normal software-only gate, exclude optional physical and catalog
dependencies:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest -m "not hardware and not catalog" -p no:napari -p no:napari-plugin-engine --basetemp=.pytest-tmp
```

The hardware group remains opt-in. Its tests require their existing explicit
environment variables and must run only from an operator-approved instrument
session. The isolated integration branch is AHT's
`hw/aht-redsun-integration`; its `uv` source resolves the adjacent RedSun
checkout on `feat/pre-upstream` as an editable dependency. RedSun's
`feat/upstream` branch remains review-ready and must not receive AHT hardware
work.

Use the narrowest hardware selector for the admitted resource:

| Resource | Explicit gate | Selector |
|---|---|---|
| GPU worker characterization | `AHT_TEST_GPU_PROCESSING=1` and `AHT_TEST_GPU_ID` or `AHT_TEST_GPU_IDS` | `-m hardware tests/test_gpu_processing_hardware.py tests/test_multilayer_gpu_hardware.py tests/test_quantitative_gpu_hardware.py` |
| ESP300 identity admission | `AHT_TEST_REAL_ESP300=1` plus the documented ESP300 identity variables | `-m hardware tests/test_esp300_stage.py` |
| Real MMCore camera admission | the required `AHT_TEST_REAL_*` camera variables | `-m hardware tests/test_camera_ioc.py` |
| GUI-mediated camera frame | the same camera variables plus `AHT_TEST_REAL_GUI_ACQUIRE=1` | `-m hardware tests/test_camera_acquisition_gui.py` |

Do not set a broad hardware environment and run the whole repository. Confirm
the intended device, interlocks, and one narrow test target first. The current
hardware selector collected ten tests and skipped all ten when no admission
gate was enabled.

The assertion-only `test_processing_device_compatibility_seam_is_public` was
removed as stale: `tests/test_processing_flyer.py` exercises the actual
`ProcessingFlyerDevice` against the same protocol contract.

### Caproto / ophyd-async process boundary

Channel Access routing is process-initialization state for aioca. Set
`EPICS_CA_ADDR_LIST` and `EPICS_CA_AUTO_ADDR_LIST=NO` before importing
ophyd-async or creating its EPICS devices. The full DemoCamera integration
test deliberately starts the client in a fresh subprocess after allocating the
Caproto IOC port; mutating those variables after aioca is imported can route a
test to a stale IOC and produce a misleading `CANothing` write failure.

## Napari pytest fixture and Windows cache permissions

Installing the GUI extra registers two global pytest plugins: `napari` and
`napari-plugin-engine`. Pytest auto-loads them even for tests that do not use
Napari. Napari's autouse `_clean_themes` fixture imports the theme registry,
which generates colorized SVGs under a path similar to:

```text
%LOCALAPPDATA%\napari\Cache\<environment-id>\_themes\dark
```

In a sandboxed Windows session that path may be unreadable or unwritable. The
result is a `PermissionError` during test setup, before AHT test code runs.
Concurrent or previously interrupted test processes can produce a similar
failure if the theme cache is locked. Pytest's default `%TEMP%\pytest-of-...`
directory may independently be unavailable in the same environment.

### Preferred repository test command

Disable only the unrelated Napari plugins, keep the other installed pytest
plugins, and place temporary data in the checkout:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest -p no:napari -p no:napari-plugin-engine --basetemp=.pytest-tmp
```

The offscreen Qt platform is sufficient for AHT's widget-construction tests;
tests requiring a real Napari canvas/OpenGL context should run in a desktop or
GPU-enabled CI session instead.

### Fully isolated plugin loading

For diagnostics, prevent every third-party pytest plugin from auto-loading:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:QT_QPA_PLATFORM = "offscreen"
uv run pytest --basetemp=.pytest-tmp
```

If a selected test needs a third-party plugin, load only that plugin with
`-p <module>`. This is less convenient than disabling the two Napari plugins,
but it makes plugin coupling explicit.

### Other valid approaches

- Run the suite in an environment containing the core/dev dependencies but
  not the Napari GUI extra, and run GUI tests in a separate job.
- In a non-sandboxed session, grant the test account access to Napari's cache
  and pytest's temporary root. Do not remove a cache directory while another
  Napari process is using it.
- Use a top-level checkout-relative `--basetemp` path. A nested path requires
  its parent directory to exist before pytest starts.

Changing `LOCALAPPDATA` in the process is not a reliable workaround on
Windows: the `appdirs` implementation used by Napari may resolve the shell
known-folder path instead of consulting that environment variable. AHT should
not patch Napari's private fixture or cache functions in `conftest.py`.
