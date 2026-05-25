[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$Cuda,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Resolve-Python {
    if ($Python) {
        return (Resolve-Path $Python).Path
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"),
        "python",
        "py"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -like "*.exe" -and (Test-Path $candidate)) {
            return $candidate
        }
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    throw "Python was not found. Install Python 3.10+ or pass -Python C:\path\to\python.exe."
}

$PythonExe = Resolve-Python
Write-Host "Using Python: $PythonExe"

Push-Location $Root
try {
    $env:YOLO_CONFIG_DIR = Join-Path $Root ".ultralytics"
    New-Item -ItemType Directory -Force $env:YOLO_CONFIG_DIR | Out-Null
    Invoke-Native $PythonExe -m venv .venv
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    Invoke-Native $VenvPython -m pip install --upgrade pip setuptools wheel

    if ($Cuda) {
        Write-Host "Installing CUDA PyTorch wheels from $TorchIndexUrl"
        Invoke-Native $VenvPython -m pip install torch torchvision torchaudio --index-url $TorchIndexUrl
    }

    Invoke-Native $VenvPython -m pip install -r requirements.txt
    Invoke-Native $VenvPython -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    Invoke-Native $VenvPython -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
}
finally {
    Pop-Location
}
