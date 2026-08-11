[CmdletBinding()]
param(
    [ValidateSet("", "--lint-only", "--no-lint")]
    [string]$Mode = ""
)

# Thin shim: the checks themselves live in scripts/check.mjs so CI can run the
# identical set on Linux. Kept because `npm test` and the docs point here.
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $Root
try {
    $Node = Get-Command node -ErrorAction Stop
    if ($Mode) {
        & $Node.Source "scripts\check.mjs" $Mode
    }
    else {
        & $Node.Source "scripts\check.mjs"
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
