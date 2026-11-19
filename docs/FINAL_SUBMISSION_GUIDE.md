# DS-02 Final Submission Guide

## Project
**Nationwide Demand Forecasting and Inventory Optimization**  
Nexvia Technologies Internship · Data Science Track · DS-02

## What the integrated system does
1. Ingests and validates daily store/SKU sales data.
2. Explores demand trends, seasonality, regional/store/product performance, missingness and outliers.
3. Creates calendar, lag, rolling and commercial features with chronological train/validation/test splits.
4. Trains the default LightGBM global demand model; Prophet, Darts and PyTorch Forecasting are optional adapters requiring their separate dependencies.
5. Evaluates forecasts against a 7-day seasonal-naive baseline and runs rolling-origin evaluation.
6. Computes safety stock, reorder points, order-up-to targets and constrained purchase order recommendations.
7. Presents demand, forecasts and inventory recommendations in a Plotly Dash dashboard.
8. Runs the end-to-end workflow, automated tests, release readiness checks and submission documentation.

## One-time Windows setup
Use 64-bit Python 3.12 where possible (the package metadata supports Python 3.10–3.13). In PowerShell:

```powershell
cd "D:\Internship\DS By Nexvia\DS02_Nationwide_Demand_Forecasting_Inventory_Optimization_FULL_INTEGRATED"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1 -Profile Integrated
```

The integrated profile installs core dependencies, LightGBM, SciPy, Plotly, Dash and test tools. Prophet, Darts, PyTorch Forecasting and OR-Tools are optional model/solver stacks. Install them separately using the relevant `requirements-*.txt` file after checking the local Python/platform compatibility. For a PyTorch/CUDA setup, use the official PyTorch selector first.

## Run the full integrated pipeline

```powershell
.\run_all.ps1
```

This runs Part 1 validation → Part 2 EDA → Part 3 features → Part 4 LightGBM forecasting → Part 5 model evaluation/backtest → Part 6 inventory optimization → Part 7 static dashboard preview → automated tests → release readiness report.

For a faster run that skips rolling-origin backtesting:

```powershell
.\run_all.ps1 -SkipBacktest
```

## Launch the interactive dashboard

After the pipeline succeeds:

```powershell
.\run_dashboard.ps1
```

Open `http://127.0.0.1:8050`. Press `Ctrl+C` in the terminal to stop it.

## Run checks separately

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe .\scripts\verify_release.py --require-dashboard
```

Release readiness JSON is written to `outputs/final/release_readiness.json`.

## Important data and model caveats
- The bundled nationwide retail CSV is synthetic development data. Replace it with authorized, validated company data before any business claim.
- The default forecasting model is LightGBM. Alternative model adapters are not evidence of training; check `outputs/reports/part4_model_status.json` for per-model status.
- Model selection uses validation metrics. The test partition is held out for final reporting.
- Inventory settings (lead time, service level, costs, supplier MOQ/pack, budget and capacity) are scenario inputs. Validate them against actual ERP/procurement records before use.
- Observed sales can be censored by stockouts and may not equal unconstrained demand.
- The inventory optimizer uses OR-Tools when available and an explicitly recorded SciPy MILP fallback otherwise.
- Do not claim cost savings, inventory reductions or forecast performance on production data based on the synthetic sample.

## Suggested demo sequence
1. Show source validation and provenance.
2. Show EDA demand trend, regional and SKU comparisons.
3. Explain lag/rolling features and chronological data splits.
4. Show LightGBM validation/test metrics and seasonal-naive comparison.
5. Show safety stock, reorder point and constrained order recommendation assumptions.
6. Change dashboard region/store/SKU/date filters and show that charts/KPIs update.
7. Open `outputs/final/release_readiness.json` and explain optional dependency statuses.

## Submission contents
Submit source code, project documentation, tests, the final release readiness report, and a concise demo. Exclude `.venv`, secrets, temporary caches and unnecessary generated intermediate files unless the evaluator explicitly requests them.

## Verify the dashboard before demonstration

After setup installs Dash, run the app wiring/HTTP smoke test:

```powershell
.\\.venv\\Scripts\\python.exe .\\scripts\\smoke_test_dashboard.py
```

Then launch the real local server:

```powershell
.\\run_dashboard.ps1
```

The helper runs the smoke test before starting Dash. A passing test-client check verifies the Dash Flask route responds, but you should still open the browser and interact with filters/charts before recording your internship demo.
