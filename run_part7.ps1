$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Virtual environment not found. Run .\setup_windows.ps1 first." }
Write-Host "DS-02 Part 7 - Interactive Plotly Dash" -ForegroundColor Cyan
& $Python -m pip install -r .\requirements-dashboard.txt
if ($LASTEXITCODE -ne 0) { throw "Dashboard dependency installation failed. Check internet access and Python version." }
& $Python .\scripts\run_dashboard.py --host 127.0.0.1 --port 8050
if ($LASTEXITCODE -ne 0) { throw "Dashboard process exited with an error." }
