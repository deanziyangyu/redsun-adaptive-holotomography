# Run PVA tests with a working native runtime.
#
# The uv-managed venv interpreter cannot create libevent's event base on this
# host (access denied), while the installed base CPython 3.12 can - with the
# venv's site-packages on PYTHONPATH and a SHORT TEMP root, because a long temp
# path makes libevent fail with "Temp file name is too long". This is the recipe
# recorded by the IX83 bundle and reproduced here on 2026-11-02 (112 passed,
# zero skips across the four PVA suites; parity check exit 0).
#
# This script is host-only. Do not run it inside a restricted agent session that
# forbids hardware/PVA use without the session's authorization.

param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs,
    [string]$BasePython = $(if ($env:AHT_BASE_PYTHON) { $env:AHT_BASE_PYTHON } else { 'C:\Users\yipgroup\AppData\Local\Programs\Python\Python312\python.exe' }),
    [string]$ShortTemp = $(if ($env:AHT_PVA_TMP) { $env:AHT_PVA_TMP } else { 'C:\pvatmp' })
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not (Test-Path $BasePython)) { throw "base Python not found: $BasePython" }
if ($ShortTemp.Length -gt 40) {
    throw "PVA temp root must be short (<= 40 chars) or libevent fails: $ShortTemp"
}
New-Item -ItemType Directory -Force -Path $ShortTemp | Out-Null

$env:PYTHONPATH = "$repo\src;$repo\.venv\Lib\site-packages"
$env:TEMP = $ShortTemp
$env:TMP = $ShortTemp
$env:EPICS_PVA_ADDR_LIST = '127.0.0.1'
$env:EPICS_PVA_AUTO_ADDR_LIST = 'NO'
# The base interpreter can have an older Qt binding than the project venv.
# Prefer the venv's Qt 6 binding unless the caller already selected one.
if (-not $env:QT_API) {
    if (Test-Path "$repo\.venv\Lib\site-packages\PyQt6\QtCore.pyd") {
        $env:QT_API = 'pyqt6'
    }
    elseif (Test-Path "$repo\.venv\Lib\site-packages\PySide6\QtCore.pyd") {
        $env:QT_API = 'pyside6'
    }
}

if (-not $PytestArgs -or $PytestArgs.Count -eq 0) {
    $PytestArgs = @(
        'tests\test_camera_pva_service.py',
        'tests\test_detector_pva_service.py',
        'tests\test_mcu_pva_service.py',
        'tests\test_stage_pva_service.py',
        '-q', '-rs'
    )
}

Write-Host "base python : $BasePython"
Write-Host "pva temp    : $ShortTemp"
Write-Host "loopback    : $env:EPICS_PVA_ADDR_LIST"
Push-Location $repo
try {
    & $BasePython -m pytest @PytestArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
