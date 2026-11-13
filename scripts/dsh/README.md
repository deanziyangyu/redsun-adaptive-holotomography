# DSH test runner

Repository-relative Windows test runner. It contains no machine-specific path: the
repository root comes from `$PSScriptRoot`, so a clone anywhere works, from any current
directory. Failure modes and verification steps are in
[headless testing](../../docs/index.md#testing).

## Usage

```powershell
# one module
pwsh -NoProfile -File scripts/dsh/run-tests.ps1 tests/test_contracts_frames.py -q

# a keyword selection, or a marker group
pwsh -NoProfile -File scripts/dsh/run-tests.ps1 tests -q -k contracts
pwsh -NoProfile -File scripts/dsh/run-tests.ps1 -m integration tests

# keep the heavy temp trees for a post-mortem
pwsh -NoProfile -File scripts/dsh/run-tests.ps1 tests -q -KeepArtifacts

# the default regression suite
pwsh -NoProfile -File scripts/dsh/run-tests.ps1 -q
```

Every argument is passed to pytest unchanged. The runner always supplies `--basetemp`; a
second one on the command line wins, leaving the run-root basetemp unused.

## What it does

1. Resolves the repository from `$PSScriptRoot`; requires
   `.venv\Scripts\python.exe` and names `uv sync --all-groups` if it is missing.
2. **Refuses to run under an elevated token.** pytest creates its temporary
   trees with a restrictive DACL, and an elevated process makes them owned by
   the administrator SID, so the invoking user can neither read nor delete them.
   `-AllowElevated` overrides the guard; do that only when you accept that the
   run root may need elevation to remove.
3. Prunes run roots older than seven days, best effort.
4. Creates `tmp/dsh-test-runs/<yyyyMMddTHHmmss>-<id>/` with `temp\`, `cache\`,
   `localappdata\`.
5. Exports, for that process tree only: `TEMP`/`TMP` → `temp`;
   `WIN_PD_OVERRIDE_LOCAL_APPDATA` → `localappdata`; `NAPARI_TEST_CACHE` →
   `localappdata\napari-cache`; `PYMM_LOG_FILE` → `<run root>\pymmcore-plus.log`;
   `CUPY_CACHE_DIR` and `CUDA_CACHE_PATH` → `cache`; `PYTHONUTF8=1`; and
   `scripts\dsh` prepended to `PYTHONPATH`. It also limits OpenBLAS, OpenMP and
   MKL to one thread per spawned process to avoid numerical-worker resource
   exhaustion in combined service selections.
6. Runs `python -m pytest --basetemp <run root>\basetemp <args>` from the
   repository root, teeing output to `<run root>\pytest.log`.
7. Deletes `basetemp\`, `temp\`, `cache\` and `localappdata\` when the run ends,
   keeping only the small log. `-KeepArtifacts` skips this step.
8. Prints the run root, the log path and the exit code, and exits with it.

[sitecustomize.py](sitecustomize.py) is why the `PYTHONPATH` entry matters: it forces
`os.mkdir` to `0o777`, so pytest's `mode=0o700` run directories do not become unreadable
DACLs. [tests/conftest.py](../../tests/conftest.py) repeats the patch, because a direct
`python -m pytest` does not go through the runner.

## Limits

It does not install or repair the environment, does not elevate, does not modify tracked
files (output stays under the git-ignored `tmp/`), and does not enable hardware — a
hardware test still needs its own gate and authorization.

## Artifacts

| Path | Contents | Kept |
|---|---|---|
| `<run root>\pytest.log` | Full output with the exit code | Always |
| `<run root>\temp\`, `cache\`, `localappdata\` | Run-scoped scratch, CuPy/CUDA cache, napari cache | Only with `-KeepArtifacts` |
| `<run root>\basetemp\` | pytest `tmp_path` tree | Only with `-KeepArtifacts` |

Keep the log when the run is evidence; do not treat basetemp contents as evidence.

## Related

[headless testing](../../docs/index.md#testing) ·
[headless testing](../../docs/index.md#testing)
