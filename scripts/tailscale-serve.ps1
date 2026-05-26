[CmdletBinding()]
param(
    [int]$Port = 8766
)

$ErrorActionPreference = "Stop"
tailscale serve --bg --https=$Port $Port
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
tailscale serve status
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
