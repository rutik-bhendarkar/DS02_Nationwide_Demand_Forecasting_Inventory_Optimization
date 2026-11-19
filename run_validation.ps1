$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run .\setup_windows.ps1 first."
}
& $Python .\scripts\validate_data.py --report .\reports\part1_validation_report.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
