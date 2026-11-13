# Repository-relative pytest runner for DSH sessions.
#
# All arguments are passed through to pytest unchanged, for example:
#   pwsh -NoProfile -File scripts/dsh/run-tests.ps1 tests/test_contracts_*.py -q
#
# The script resolves the repository from its own location, uses the repository
# .venv, writes every artifact under tmp/dsh-test-runs/<stamp>-<id>/, and never
# edits tracked files. See scripts/dsh/README.md.
#
# Temporary-file policy (mirrors the IX83 bundle's runner):
#   * refuse to run under an elevated token, because artifacts created by an
#     admin SID cannot be deleted by the invoking user afterwards;
#   * create directories with a permissive mode through scripts/dsh/sitecustomize.py,
#     because pytest 9 asks for mode 0o700 and on Windows that becomes a DACL the
#     next process cannot traverse or delete;
#   * point TEMP, TMP and --basetemp inside the run root, then delete the heavy
#     directories when the run ends, keeping only the small pytest log;
#   * prune run roots older than seven days at startup.

# Arguments are taken from the automatic ``$args`` array rather than a param
# block: a declared parameter named ``-m`` or ``-k`` would bind pytest's own
# options, and ``[CmdletBinding()]`` would bind ``-v`` to ``-Verbose``. Only the
# two switches this runner owns are intercepted; everything else, including
# ``-k``, ``-m``, ``-x`` and ``-v``, is forwarded verbatim.
$keepArtifacts = $false
$allowElevated = $false
$pytestArgs = @()
foreach ($argument in $args) {
    switch -CaseSensitive ($argument) {
        '-KeepArtifacts' { $keepArtifacts = $true; continue }
        '-AllowElevated' { $allowElevated = $true; continue }
        default { $pytestArgs += $argument }
    }
}

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Repository environment missing: $python. Create it with 'uv sync --all-groups' in $repoRoot."
}

function Test-IsElevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if ((Test-IsElevated) -and -not $allowElevated) {
    throw @'
Refusing to run the suite under an elevated token.

pytest creates its temporary trees with a restrictive DACL. When an elevated
process creates them, the resulting directories are owned by the administrator
SID and a normal session can neither read nor delete them, which leaves an
undeletable tmp/ tree behind.

Run the suite from an ordinary user shell. If you must override this, pass
-AllowElevated and accept that the run root may need elevation to remove.
'@
}

$runsRoot = Join-Path $repoRoot 'tmp\dsh-test-runs'
New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null

# Best effort: drop run roots from earlier weeks. Failures are ignored, because a
# stale ACL-protected tree is exactly what this runner tries to stop producing.
$cutoff = (Get-Date).AddDays(-7)
Get-ChildItem -LiteralPath $runsRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt $cutoff } |
    ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$runId = [guid]::NewGuid().ToString('N').Substring(0, 8)
$runRoot = Join-Path $runsRoot ("{0}-{1}" -f $stamp, $runId)
$temp = Join-Path $runRoot 'temp'
$cache = Join-Path $runRoot 'cache'
$localAppData = Join-Path $runRoot 'localappdata'
$basetemp = Join-Path $runRoot 'basetemp'
$logPath = Join-Path $runRoot 'pytest.log'

foreach ($dir in @($temp, $cache, $localAppData)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# Run-scoped temporary and cache roots. pytest clears --basetemp, so that path
# is deliberately left for pytest to create.
$env:TEMP = $temp
$env:TMP = $temp
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = $localAppData
$env:NAPARI_TEST_CACHE = Join-Path $localAppData 'napari-cache'
$env:PYMM_LOG_FILE = Join-Path $runRoot 'pymmcore-plus.log'
$env:CUPY_CACHE_DIR = $cache
$env:CUDA_CACHE_PATH = $cache
$env:PYTHONUTF8 = '1'
# Spawned numerical workers otherwise multiply BLAS threads per service and can
# exhaust memory before reaching the assertion under combined-suite load.
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# Put the mkdir shim ahead of any existing PYTHONPATH entry.
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    $PSScriptRoot + [IO.Path]::PathSeparator + $env:PYTHONPATH
} else {
    $PSScriptRoot
}

Write-Host "run root: $runRoot"
Write-Host "basetemp: $basetemp"

$code = 1
Push-Location $repoRoot
try {
    & $python -m pytest "--basetemp=$basetemp" @pytestArgs 2>&1 | Tee-Object -FilePath $logPath
    $code = $LASTEXITCODE
}
finally {
    Pop-Location

    if (-not $keepArtifacts) {
        # Remove the trees that carry per-directory DACLs. The log stays: it is a
        # few kilobytes and is the only part of the run worth keeping.
        foreach ($dir in @($basetemp, $temp, $cache, $localAppData)) {
            if (Test-Path -LiteralPath $dir) {
                Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        Get-ChildItem -LiteralPath $runRoot -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne 'pytest.log' } |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "log: $logPath"
Write-Host "exit code: $code"
exit $code
