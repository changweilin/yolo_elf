[CmdletBinding()]
param()

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

Push-Location $Root
try {
    $env:YOLO_CONFIG_DIR = Join-Path $Root ".ultralytics"
    New-Item -ItemType Directory -Force $env:YOLO_CONFIG_DIR | Out-Null
    & $PythonExe -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $PythonExe -m py_compile "scripts\bench_detector.py"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $Node = Get-Command node -ErrorAction Stop
    $JavaScriptFiles = @(
        "static\phone.js",
        "static\viewer.js",
        "static\theme.js",
        "scripts\build-static.mjs",
        "scripts\start-server.mjs"
    )
    foreach ($File in $JavaScriptFiles) {
        & $Node.Source --check $File
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}
