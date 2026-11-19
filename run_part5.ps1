$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Project virtual environment not found. Run .\setup_windows.ps1 first."
}
& $Python .\scripts\run_evaluation.py --window-days 7 --n-jobs 4
if ($LASTEXITCODE -ne 0) { throw "Part 5 evaluation failed with exit code $LASTEXITCODE" }
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Automated tests failed with exit code $LASTEXITCODE" }
Write-Host "Part 5 completed. See outputs\reports\part5_evaluation_report.html" -ForegroundColor Green
