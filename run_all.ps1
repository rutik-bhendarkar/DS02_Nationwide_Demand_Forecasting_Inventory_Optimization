param(
    [switch]$SkipSetup,
    [switch]$SkipTests,
    [switch]$SkipBacktest,
    [switch]$StartDashboard,
    [ValidateSet("all", "part1", "part2", "part3", "part4", "part5", "part6", "part7")]
    [string]$FromStage = "all"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not $SkipSetup) {
    # Re-run setup against an existing .venv too, so required integrated dependencies are present.
    & .\setup_windows.ps1 -Profile Integrated
    if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
        throw "Environment setup did not create .venv."
    }
}
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Project environment missing. Run .\setup_windows.ps1 -Profile Integrated first."
}

$PipelineArgs = @(".\scripts\run_pipeline.py", "--from-stage", $FromStage)
if ($SkipTests) { $PipelineArgs += "--skip-tests" }
if ($SkipBacktest) { $PipelineArgs += "--skip-backtest" }
& $Python @PipelineArgs
if ($LASTEXITCODE -ne 0) { throw "Integrated pipeline failed. Review the stage output above." }

if ($StartDashboard) {
    Write-Host "Starting DS-02 Dash application at http://127.0.0.1:8050" -ForegroundColor Green
    & .\run_dashboard.ps1
} else {
    Write-Host "All requested pipeline stages and release checks completed." -ForegroundColor Green
    Write-Host "Launch dashboard with: .\run_dashboard.ps1" -ForegroundColor Cyan
}
