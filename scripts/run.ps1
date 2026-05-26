[CmdletBinding()]
param(
    [string]$HostName = "0.0.0.0",
    [int]$Port = 8766,
    [switch]$Reload
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

$env:HOST = $HostName
$env:PORT = [string]$Port
$env:YOLO_CONFIG_DIR = Join-Path $Root ".ultralytics"
New-Item -ItemType Directory -Force $env:YOLO_CONFIG_DIR | Out-Null

$UvicornArgs = @("-m", "uvicorn", "app.main:app", "--host", $HostName, "--port", [string]$Port)
if ($Reload) {
    $UvicornArgs += "--reload"
}

Push-Location $Root
try {
    & $PythonExe @UvicornArgs
}
finally {
    Pop-Location
}
