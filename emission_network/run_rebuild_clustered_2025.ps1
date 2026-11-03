$ErrorActionPreference = "Stop"
Set-Location "E:\transfer justice"
$env:PYTHONUTF8 = "1"

$months = (1..12 | ForEach-Object { "2025{0:D2}" -f $_ }) -join ","

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "2025 AIS clustered rebuild (PRIMARY KEY WITHOUT ROWID)" -ForegroundColor Cyan
Write-Host "Source : D:\SQL_AIS_SingleShip (READ ONLY)"
Write-Host "Output : F:\SQL_AIS_SingleShip_2025_clustered"
Write-Host "Temp   : E:\sqlite_temp_clustered_2025"
Write-Host "Months : $months"
Write-Host "Progress is also logged to output\cluster_rebuild.log"
Write-Host "============================================================" -ForegroundColor Cyan

python "emission_network\rebuild_clustered_ais.py" `
  --source-dir "D:\SQL_AIS_SingleShip" `
  --output-dir "F:\SQL_AIS_SingleShip_2025_clustered" `
  --temp-dir "E:\sqlite_temp_clustered_2025" `
  --months $months `
  --sample-count 3

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host "All requested months completed successfully." -ForegroundColor Green
} else {
    Write-Host "Rebuild stopped with exit code $exitCode. Check cluster_rebuild.log." -ForegroundColor Red
}
Write-Host "This window will remain open for inspection."
