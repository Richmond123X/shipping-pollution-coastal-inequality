[CmdletBinding()]
param(
    [ValidateSet("all", "prepare", "ais", "post", "check")]
    [string]$Stage = "all",
    [ValidateRange(1, 16)]
    [int]$Cores = 4
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ais = "F:\SQL_AIS_SingleShip"
$python = "C:\Users\86176\AppData\Local\Programs\Python\Python311\python.exe"
$sampleIntervalSec = 0
$prefetchBatchSize = 0
$shardSize = 500
$runLogs = Join-Path $root "logs\integrated_2024"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python was not found: $python"
}
if (-not (Test-Path -LiteralPath $ais -PathType Container)) {
    throw "AIS directory was not found: $ais"
}

$missing = @()
foreach ($month in 1..12) {
    $db = Join-Path $ais ("ais_2024{0:D2}.db" -f $month)
    if (-not (Test-Path -LiteralPath $db -PathType Leaf)) {
        $missing += $db
    }
}
if ($missing.Count -gt 0) {
    throw "The 2024 AIS database set is incomplete:`n$($missing -join "`n")"
}

New-Item -ItemType Directory -Force -Path $runLogs | Out-Null

function Invoke-PolicyStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Script,
        [string[]]$StepArgs = @()
    )
    $scriptPath = Join-Path $root $Script
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
        throw "Script was not found: $scriptPath"
    }
    $logPath = Join-Path $runLogs ("{0}.log" -f $Name)
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host "[$Name] $Script $($StepArgs -join ' ')" -ForegroundColor Cyan
    Write-Host "Log: $logPath" -ForegroundColor DarkGray
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    # Windows PowerShell can promote a native program's stderr records to
    # terminating errors when the global preference is Stop.  Keep the full
    # traceback flowing through Tee, then handle the actual process exit code.
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $python -u $scriptPath @StepArgs 2>&1 | Tee-Object -FilePath $logPath
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    if ($code -ne 0) {
        throw "Step failed: $Name (exit code $code). See $logPath"
    }
}

Write-Host "V12 policy simulation: direct 2024 AIS mode" -ForegroundColor Green
Write-Host "AIS: $ais"
Write-Host "Stage: $Stage; cores: $Cores"
Write-Host "Trajectory resolution: full-frequency (no temporal sampling)"
Write-Host "AIS I/O: original worker-per-ship database reads (no prefetch)"
Write-Host "Each ship: one continuous annual calculation, then monthly output"
Write-Host "Scheduling: independent annual shards; continue after a shard failure"
Write-Host "Checkpoint unit: one completed $shardSize-ship shard; an interrupted partial shard restarts"
Write-Host "The same command can be rerun safely after an interruption."

if ($Stage -eq "check") {
    Write-Host "Preflight check: PASS (Python, AIS directory, and 12 monthly databases)." -ForegroundColor Green
    exit 0
}

if ($Stage -in @("all", "prepare")) {
    Invoke-PolicyStep "01_gt_crosswalk_2024" "02_build_container_gt_crosswalk.py"
    Invoke-PolicyStep "02_input_validation_pre" "00_validate_policy_inputs.py"

    $summaryPath = Join-Path $root "tables\policy_input_readiness_summary.json"
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    if ([int]$summary.critical_blockers_excluding_grid_net_effect -gt 0) {
        throw "Input validation has blocking items. See tables\policy_input_readiness.csv"
    }

}

if ($Stage -in @("all", "ais")) {
    Invoke-PolicyStep "03_direct_ais_annual_shards_2024" "03_run_policy_ais_shards.py" @(
        "--n-cores", [string]$Cores,
        "--shard-size", [string]$shardSize,
        "--prefetch-batch-size", [string]$prefetchBatchSize,
        "--sample-interval-sec", [string]$sampleIntervalSec
    )
}

if ($Stage -in @("all", "post")) {
    $runStatePath = Join-Path $root "intermediate\ais_policy_2024_direct\direct_run_state.json"
    if (-not (Test-Path -LiteralPath $runStatePath -PathType Leaf)) {
        throw "The 2024 direct AIS run is not complete: $runStatePath"
    }
    $runState = Get-Content -LiteralPath $runStatePath -Raw | ConvertFrom-Json
    if ($runState.status -ne "complete") {
        throw "The 2024 direct AIS status is not complete: $($runState.status)"
    }

    Invoke-PolicyStep "05_clustering_audit_tables" "00c_build_clustering_supplement_tables.py"
    Invoke-PolicyStep "06_merge_2024" "04_merge_policy_ais_shards.py"
    Invoke-PolicyStep "07_scenarios_s1_s7" "05_compute_policy_scenarios.py"
    Invoke-PolicyStep "08_table7" "05a_build_table7_diagnostics.py"
    Invoke-PolicyStep "09_redesigned_figures" "06b_make_policy_redesign_panels.py"
    Invoke-PolicyStep "10_overlap_figures" "06a_make_overlap_figures.py"
    Invoke-PolicyStep "11_input_validation_final" "00_validate_policy_inputs.py"
    Invoke-PolicyStep "12_formal_verification" "08_verify_formal_outputs.py"
}

Write-Host ""
Write-Host "Run complete. Formal root: $root" -ForegroundColor Green
Write-Host "Tables: $root\tables"
Write-Host "Figures: $root\figures"
Write-Host "Logs: $runLogs"
