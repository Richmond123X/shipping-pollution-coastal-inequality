$ErrorActionPreference = "Stop"
Set-Location "E:\transfer justice"
$env:PYTHONUTF8 = "1"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Finalize clustered AIS 202512 (target-only integrity check)" -ForegroundColor Cyan
Write-Host "The source database remains read-only."
Write-Host "============================================================" -ForegroundColor Cyan

python "emission_network\finalize_clustered_build.py" `
  --month "202512" `
  --source "D:\SQL_AIS_SingleShip\ais_202512.db" `
  --building "F:\SQL_AIS_SingleShip_2025_clustered\ais_202512.building.db" `
  --final "F:\SQL_AIS_SingleShip_2025_clustered\ais_202512.db" `
  --log "F:\SQL_AIS_SingleShip_2025_clustered\cluster_rebuild.log" `
  --sample-count 3

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host "202512 finalized successfully." -ForegroundColor Green
} else {
    Write-Host "202512 finalization failed with exit code $exitCode." -ForegroundColor Red
}
Write-Host "This window will remain open for inspection."
