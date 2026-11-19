$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment not found. Run .\setup_windows.ps1 first."
}
if (-not (Test-Path ".\outputs\features\splits\train.csv")) {
    throw "Part 3 feature splits not found. Run .\scripts\run_feature_engineering.py first."
}

Write-Host "Checking optional forecasting frameworks..." -ForegroundColor Cyan
& $Python .\scripts\check_forecasting_environment.py
if ($LASTEXITCODE -ne 0) { throw "Environment check failed." }

Write-Host "Training LightGBM on the Part 3 chronological splits..." -ForegroundColor Cyan
& $Python .\scripts\run_forecasting.py --models lightgbm
if ($LASTEXITCODE -ne 0) { throw "LightGBM forecasting run failed. See outputs\reports\part4_model_status.json." }

Write-Host "Running project test suite..." -ForegroundColor Cyan
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Project tests failed." }

Write-Host "Part 4 run completed. Review outputs\reports\part4_lightgbm_metrics.json." -ForegroundColor Green
