<#
.SYNOPSIS
Start an AHT session or service from this checkout on the instrument host.

.DESCRIPTION
Adapted from the IX83 imaging bundle's `scripts/run_session.ps1`
(`redsun-searching-is-imaging`, revision `3a3bf06`, Apache-2.0) and the
archived AHT wrapper.

Resolves the repository from this script's own location, prepares the paths a
Redsun/AHT process needs on Windows, then hands the command to
`scripts/launch_app.py`. That launcher relocates napari's cache, relocates
Redsun's data root and assures a short PVA temporary root before importing
`redsun_aht`.

What the script sets, and why:

- `REDSUN_NAPARI_CACHE` keeps napari's theme cache inside the checkout; napari
  rebuilds theme SVGs on first import and dies where the cache refuses writes.
- `REDSUN_DATA_ROOT` is the session data root (logs, settings, stores). The
  default is `R:\redsun_aht_data`. When that drive refuses writes the script
  falls back to `.redsun_data` in this checkout and says so. An explicit
  `-DataRoot` or an inherited `REDSUN_DATA_ROOT` is never silently replaced: an
  unwritable one is refused with its exact path.
- `PYMM_LOG_FILE` keeps pymmcore_plus's import-time log inside the data root.
- `TEMP`/`TMP` and the CuPy/CUDA caches live under the data root; the launcher
  then moves `TEMP`/`TMP` to a short writable root for `p4p`/libevent.
- `PYTHONPATH` puts this checkout's `src` ahead of site-packages, so managed
  services started with `-m` import the checkout rather than a stale install.
- The loopback `EPICS_PVA_*` variables let a service and a client on one host
  find each other.

The wrapper forwards everything after `Command` to the application. Unlike the
`aht` command line it does not need the `--` separator - PowerShell rejects a
bare `--` in a script invocation - so write
`run_session.ps1 session <name> --prefix AHT:...` rather than inserting `--`.

.PARAMETER Command
The `aht` subcommand: `session`, `service`, `--list` or `--check`. Defaults to
`--list`, because the session registry is owned by the coordinator and a bare
launch should not guess a session name.

.PARAMETER DataRoot
Where Redsun keeps logs, settings, session storage and the caches. Defaults to
`REDSUN_DATA_ROOT` when the caller already exported it, otherwise to
`R:\redsun_aht_data`. The script refuses, naming the exact path, when an
explicit root is not writable; the default may fall back to the checkout.

.PARAMETER Check
Resolve and probe every launcher path, print the command that would run, and
exit without starting anything. This is a switch because PowerShell binds a
bare `--check` inconsistently for `-File` invocations; `-Check` is unambiguous.

.PARAMETER Arguments
Everything else is passed through to the application.

.EXAMPLE
.\scripts\run_session.ps1 -Check

.EXAMPLE
.\scripts\run_session.ps1

Lists the configured sessions and services ("no sessions registered yet" until
the registry lands).

.EXAMPLE
.\scripts\run_session.ps1 session camera-gui --prefix AHT:CAM:PVCAM:

.EXAMPLE
.\scripts\run_session.ps1 service camera

.EXAMPLE
.\scripts\run_session.ps1 -DataRoot E:\redsun-aht-data session live
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0)]
    [string] $Command = "--list",

    [string] $DataRoot,

    [switch] $Check,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments = @()
)

$ErrorActionPreference = "Stop"

# A bare `--check` reaches the remainder list rather than the switch, so accept
# it when it is the only argument and no command was named explicitly.
if (-not $Check -and $Command -eq "--list" -and $Arguments.Count -eq 1 -and $Arguments[0] -eq "--check") {
    $Check = $true
    $Arguments = @()
}

$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "no interpreter at $python - create the checkout's environment first"
}

# Probe with the interpreter, not with PowerShell: the launcher itself runs
# under Python, so interpreter-specific access rules (sandboxing, a launcher
# shim) decide whether the data root is usable, not the wrapper's own token.
function Test-DataRootWritable {
    param([string] $Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        try {
            New-Item -ItemType Directory -Force -Path $Path -ErrorAction Stop | Out-Null
        }
        catch {
            return $false
        }
    }
    $probe = Join-Path $Path (".aht-write-probe-" + [guid]::NewGuid().ToString("N"))
    & $python -c "import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text('probe', encoding='utf-8'); p.unlink()" $probe 2>$null
    return $LASTEXITCODE -eq 0
}

$explicit = $false
if ($DataRoot) {
    $requested = $DataRoot
    $explicit = $true
}
elseif ($env:REDSUN_DATA_ROOT -and $env:REDSUN_DATA_ROOT.Trim()) {
    $requested = $env:REDSUN_DATA_ROOT
    $explicit = $true
}
else {
    $requested = "R:\redsun_aht_data"
}

$data = $null
try {
    $data = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($requested)
}
catch {
    if ($explicit) {
        throw "data root '$requested' is not a usable path: $($_.Exception.Message)"
    }
}

if (-not $data -or -not (Test-DataRootWritable $data)) {
    if ($explicit) {
        $shown = if ($data) { $data } else { $requested }
        throw "data root '$shown' is not writable - repair its permissions or pass -DataRoot with a writable volume"
    }
    $defaultRoot = if ($data) { $data } else { $requested }
    $data = Join-Path $repo ".redsun_data"
    if (-not (Test-DataRootWritable $data)) {
        throw "data root '$data' is not writable (default '$defaultRoot' is also not writable) - repair its permissions or pass -DataRoot with a writable volume"
    }
    Write-Host "fallback      $defaultRoot is not writable; using repository fallback"
}

$env:REDSUN_DATA_ROOT = $data
$env:REDSUN_NAPARI_CACHE = Join-Path $repo ".napari_cache"
$env:REDSUN_AHT_REPO = $repo
$env:PYMM_LOG_FILE = Join-Path $data "pymmcore-plus.log"
$env:PYTHONUTF8 = "1"

$scratch = Join-Path $data "tmp"
foreach ($directory in $scratch, (Join-Path $scratch "cupy"), (Join-Path $scratch "cuda")) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}
$env:TEMP = $scratch
$env:TMP = $scratch
$env:CUPY_CACHE_DIR = Join-Path $scratch "cupy"
$env:CUDA_CACHE_PATH = Join-Path $scratch "cuda"

$sourceTree = Join-Path $repo "src"
$env:PYTHONPATH = $sourceTree

# AHT services bind loopback; the launcher's clients must search loopback.
$env:EPICS_PVA_ADDR_LIST = "127.0.0.1"
$env:EPICS_PVA_AUTO_ADDR_LIST = "NO"

Push-Location $repo
try {
    if ($Check) {
        Write-Host "command       $Command"
        & $python (Join-Path $PSScriptRoot "launch_app.py") --check
    }
    else {
        & $python (Join-Path $PSScriptRoot "launch_app.py") $Command @Arguments
    }
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
