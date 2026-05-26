[CmdletBinding()]
param(
    [int]$Frames = 30,
    [int]$Warmup = 3,
    [int]$Width = 960,
    [int]$Height = 540,
    [double]$Quality = 0.65,
    [string]$Model = "",
    [string]$Device = "",
    [int]$ImgSize = 0,
    [double]$Conf = -1,
    [switch]$Half
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
}
elseif (Test-Path $BundledPython) {
    $PythonExe = $BundledPython
}
else {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

$ArgsList = @(
    "scripts\bench_detector.py",
    "--frames", [string]$Frames,
    "--warmup", [string]$Warmup,
    "--width", [string]$Width,
    "--height", [string]$Height,
    "--quality", [string]$Quality
)

if ($Model) {
    $ArgsList += @("--model", $Model)
}
if ($Device) {
    $ArgsList += @("--device", $Device)
}
if ($ImgSize -gt 0) {
    $ArgsList += @("--img-size", [string]$ImgSize)
}
if ($Conf -ge 0) {
    $ArgsList += @("--conf", [string]$Conf)
}
if ($Half) {
    $ArgsList += "--half"
}

Push-Location $Root
try {
    $env:YOLO_CONFIG_DIR = Join-Path $Root ".ultralytics"
    New-Item -ItemType Directory -Force $env:YOLO_CONFIG_DIR | Out-Null
    & $PythonExe @ArgsList
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
