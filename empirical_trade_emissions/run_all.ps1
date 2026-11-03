param(
    [switch]$SkipTradeRebuild
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$ScriptRoot = Join-Path $ProjectRoot 'scripts'

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked -Program 'python' -Arguments @(
    (Join-Path $ScriptRoot '01_build_emissions.py'), '--workspace', $WorkspaceRoot
)
Invoke-Checked -Program 'python' -Arguments @(
    (Join-Path $ScriptRoot '02_build_classifications.py'), '--workspace', $WorkspaceRoot
)
if (-not $SkipTradeRebuild) {
    Invoke-Checked -Program 'python' -Arguments @(
        (Join-Path $ScriptRoot '03_build_trade.py'), '--workspace', $WorkspaceRoot
    )
}
Invoke-Checked -Program 'python' -Arguments @(
    (Join-Path $ScriptRoot '04_build_panels.py'), '--workspace', $WorkspaceRoot
)
Invoke-Checked -Program 'Rscript' -Arguments @(
    (Join-Path $ScriptRoot '05_estimate_models.R'), $WorkspaceRoot
)
Invoke-Checked -Program 'Rscript' -Arguments @(
    (Join-Path $ScriptRoot '05b_bias_correction.R'), $WorkspaceRoot
)
Invoke-Checked -Program 'python' -Arguments @(
    (Join-Path $ScriptRoot '06_make_report.py'), '--workspace', $WorkspaceRoot
)
Invoke-Checked -Program 'python' -Arguments @(
    (Join-Path $ScriptRoot '07_verify_delivery.py'), '--workspace', $WorkspaceRoot
)

Write-Host "Complete: $ProjectRoot\results\empirical_report_zh.md"
