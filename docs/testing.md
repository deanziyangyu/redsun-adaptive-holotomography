# Test environment notes

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
