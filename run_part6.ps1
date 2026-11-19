$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Virtual environment not found. Run .\setup_windows.ps1 first." }
Write-Host "DS-02 Part 6 - Inventory Optimization" -ForegroundColor Cyan
& $Python -m pip install -r .\requirements-optimization.txt
if ($LASTEXITCODE -ne 0) { throw "Optimization dependency installation failed." }
& $Python .\scripts\run_inventory_optimization.py --solver auto
if ($LASTEXITCODE -ne 0) { throw "Inventory optimization failed." }
Write-Host "Part 6 outputs created under outputs\inventory" -ForegroundColor Green
