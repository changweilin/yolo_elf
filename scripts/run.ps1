[CmdletBinding()]
param(
    [string]$HostName = "0.0.0.0",
    [int]$Port = 8766,
    [ValidateSet("fast", "accurate")]
    [string]$DetectMode,
    [string]$FastModel,
    [string]$AccurateModel,
    # Comma-separated open-vocabulary prompts for YOLO-World / YOLOE models.
    [string]$Classes,
    [double]$ConfThresh,
    [int]$ImgSize,
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

# Detector overrides: only set the env var when the matching parameter is passed,
# so omitted parameters keep app/config.py defaults (or values already exported).
if ($PSBoundParameters.ContainsKey("DetectMode"))    { $env:DETECT_MODE = $DetectMode }
if ($PSBoundParameters.ContainsKey("FastModel"))     { $env:YOLO_MODEL = $FastModel }
if ($PSBoundParameters.ContainsKey("AccurateModel")) { $env:YOLO_MODEL_ACCURATE = $AccurateModel }
if ($PSBoundParameters.ContainsKey("Classes"))       { $env:YOLO_CLASSES = $Classes }
if ($PSBoundParameters.ContainsKey("ConfThresh"))    { $env:CONF_THRESH = [string]$ConfThresh }
if ($PSBoundParameters.ContainsKey("ImgSize"))       { $env:IMG_SIZE = [string]$ImgSize }

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
