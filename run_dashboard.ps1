$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run .\setup_windows.ps1 -Profile Integrated first." }
& $Python -c "import dash, plotly; print('Dash', dash.__version__, '| Plotly', plotly.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dashboard dependencies..." -ForegroundColor Yellow
    & $Python -m pip install -r .\requirements-dashboard.txt
    if ($LASTEXITCODE -ne 0) { throw "Dashboard dependencies could not be installed." }
}
& $Python .\scripts\smoke_test_dashboard.py
if ($LASTEXITCODE -ne 0) { throw "Dashboard HTTP smoke test failed; review the error above." }
& $Python .\scripts\run_dashboard.py --host 127.0.0.1 --port 8050
if ($LASTEXITCODE -ne 0) { throw "Dashboard failed to start." }
