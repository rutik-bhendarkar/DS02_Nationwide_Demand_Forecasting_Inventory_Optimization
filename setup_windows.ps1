param(
    [ValidateSet("Core", "Integrated")]
    [string]$Profile = "Integrated",
    [switch]$SkipUpgradePip
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "DS-02 — Nationwide Demand Forecasting & Inventory Optimization" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
if (-not (Test-Path ".\pyproject.toml")) {
    throw "Run setup_windows.ps1 from the extracted DS-02 project root."
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    $created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @("3.12", "3.11", "3.13")) {
            & py "-$version" --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                & py "-$version" -m venv .venv
                if ($LASTEXITCODE -eq 0) { $created = $true; break }
            }
        }
    }
    if (-not $created) {
        Write-Host "Python launcher 3.12/3.11/3.13 not found; using active python." -ForegroundColor Yellow
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Could not create .venv. Install 64-bit Python 3.12 and retry." }
    }
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python --version
if ($LASTEXITCODE -ne 0) { throw "Project Python environment is not executable." }
if (-not $SkipUpgradePip) {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed. Check network access." }
}

& $Python -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "Core project installation failed." }

if ($Profile -eq "Integrated") {
    Write-Host "Installing integrated pipeline, dashboard and test dependencies..." -ForegroundColor Cyan
    & $Python -m pip install -r .\requirements-integration.txt
    if ($LASTEXITCODE -ne 0) { throw "Integrated dependencies failed. Check Python version/network, then rerun setup." }
}

Write-Host "Setup completed. Next run: .\run_all.ps1" -ForegroundColor Green
if ($Profile -eq "Core") {
    Write-Host "Core-only profile selected; install requirements-integration.txt for Parts 2-7." -ForegroundColor Yellow
}
