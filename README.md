# DS-02 — Nationwide Demand Forecasting and Inventory Optimization

**Organization:** Nexvia Technologies Internship  
**Track:** Data Science  
**Project code:** DS-02  
**Current delivery:** Parts 1–8 integrated — Data & Environment Setup; EDA; Feature Engineering; Forecasting; Evaluation; Inventory Optimization; Interactive Dashboard; Final Integration & Submission

This project is designed to forecast product demand across multiple stores and support inventory replenishment decisions that reduce stockouts and excess inventory.

> **Data integrity notice:** The bundled CSV is generated synthetic test data. It is not real Nexvia or retailer data, is not a market benchmark, and must not be used to claim model performance or business savings.

## Eight-part implementation plan

1. **Data & Environment Setup** — project structure, dependencies, data schema, ingestion, validation, sample data and tests.
2. **Exploratory Data Analysis** — data quality, demand trends, seasonality, store/product/region analysis and outliers.
3. **Feature Engineering** — calendar, exact calendar lags, past-only rolling statistics, known-future commercial variables, stable identifier codes, leakage-aware modeling data, and chronological train/validation/test partitions.
4. **Forecasting Models** — Prophet, LightGBM, Darts and PyTorch Forecasting where appropriate.
5. **Model Evaluation & Comparison** — time-based backtesting, MAE, RMSE, WAPE, bias and model selection.
6. **Inventory Optimization** — safety stock, reorder points, lead time, service levels and OR-Tools.
7. **Interactive Dashboard** — Plotly Dash views for nationwide demand and inventory decisions.
8. **Integration & Submission** — end-to-end pipeline, tests, documentation, demo and final packaging.

See [ROADMAP_8_PARTS.md](ROADMAP_8_PARTS.md) for milestone exit criteria.

## Project structure

See [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md). The code uses a `src/` package layout. Core ingestion and validation are in `src/demand_inventory/data/`.

## Windows setup (PowerShell)

Recommended interpreter for this project: **64-bit Python 3.12**. Python 3.10–3.13 is allowed by the project metadata, but optional forecasting packages can have their own platform constraints.

```powershell
cd "D:\Internship\DS By Nexvia\DS02_Nationwide_Demand_Forecasting_Inventory_Optimization_FULL_INTEGRATED"

# Create/reuse project-local .venv and install integrated pipeline + dashboard dependencies
.\setup_windows.ps1 -Profile Integrated

# Activate the environment (optional if using the explicit python path below)
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

The setup script creates `.venv` only if it does not already exist. It installs the integrated runtime, LightGBM, SciPy, Plotly, Dash and test dependencies. It does not install the optional heavy Prophet/Darts/PyTorch Forecasting stacks automatically. It does not touch DS-01 or any other project.

## Validate the bundled sample dataset

From the project root, after setup:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_data.py
```

Or use the installed package CLI:

```powershell
.\.venv\Scripts\python.exe -m demand_inventory.cli validate --input .\data\sample\nationwide_retail_daily_sales_synthetic.csv
```

Optional JSON report:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_data.py --report .\reports\part1_validation_report.json
```

## Run automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Regenerate the synthetic sample

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_sample_data.py --days 180 --seed 42
```

Do not regenerate or replace an actual business dataset with synthetic data. The generator is for development only.

## Dependency strategy

Install the core environment first. Heavy model packages are optional and are separated by phase:

| Requirement file / extra | Intended phase |
|---|---|
| `requirements.txt` | Part 1 core data work |
| `requirements-integration.txt` | Integrated Parts 1–8 default run (LightGBM, SciPy, Plotly, Dash, tests) |
| `requirements-dev.txt` / `.[dev]` | Tests and linting |
| `requirements-eda.txt` / `.[eda]` | Part 2 interactive EDA HTML output (Plotly) |
| `requirements-prophet-lightgbm.txt` / `.[prophet_lightgbm]` | Part 4 |
| `requirements-darts.txt` / `.[darts]` | Part 4 |
| `requirements-pytorch-forecasting.txt` / `.[torch_forecasting]` | Part 4 |
| `requirements-optimization.txt` / `.[optimization]` | Part 6 |
| `requirements-dashboard.txt` / `.[dashboard]` | Part 7 |

Install only the package group needed for the current milestone. PyTorch should be installed using the official selector for the machine's CPU/CUDA configuration before PyTorch Forecasting. See the official [PyTorch Forecasting installation guide](https://pytorch-forecasting.readthedocs.io/en/stable/installation.html). For Windows OR-Tools, verify current wheel/platform prerequisites in the [Google OR-Tools installation guide](https://developers.google.com/optimization/install) before installing. LightGBM's official installation notes recommend a 64-bit environment and note a possible Visual C++ runtime requirement on Windows: [LightGBM installation](https://www.lightgbm.net/lightgbm-python-installation-guide).

## Input schema

The required columns are `date`, `store_id`, `sku_id`, `units_sold`, and `unit_price`. One row represents one store-SKU-date. The business key is `(date, store_id, sku_id)`. See [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for definitions and real-world ingestion cautions.

## Run Part 2 — Exploratory Data Analysis

After completing the Windows setup, run the reproducible EDA pipeline:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_eda.py
```

You can analyze a different validated CSV and choose another output folder:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_eda.py `
  --input .\data\raw\your_sales_file.csv `
  --output-dir .\outputs\eda
```

The source dataset must satisfy the Part 1 input contract. The standard Windows setup installs the `eda` extra (Plotly) so that the interactive HTML dashboard can be generated. EDA outputs are written to `outputs/eda/`; the machine-readable summary and human-readable findings are written to `reports/part2_eda_summary.json` and `reports/part2_eda_findings.md`.

### Part 2 output inventory

The pipeline produces daily/monthly/weekday demand, region/state/city/store/category/SKU/product summaries, month-of-year patterns, descriptive promotion/holiday comparisons, closing-stock pressure, missingness, store-SKU date coverage, IQR outlier flags, numeric correlations, and a self-contained Plotly HTML dashboard when Plotly is installed.

`average_daily_units` means the average of the dimension's **daily total** (for example, all SKU demand summed per region per date). `average_units_per_record` is separately reported as the average of an individual store-SKU-date row. These metrics must not be confused.

### Part 2 interpretation limits

- The bundled dataset is synthetic development data, not real retailer or Nexvia data.
- The EDA is descriptive. Promotion/holiday comparisons are not causal estimates.
- Sales may be censored when stock is unavailable; observed sales do not necessarily equal unconstrained demand.
- IQR outliers are review flags, not automatic data errors. No outliers are removed by the EDA pipeline.
- No forecasting models are trained and no forecast accuracy is claimed in Part 2.

## Run Part 3 — Feature Engineering

After completing setup, run the reusable feature pipeline:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_feature_engineering.py
```

Custom data and options:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_feature_engineering.py `
  --input .\data\raw\your_sales_file.csv `
  --output-dir .\outputs\features `
  --lags 1 7 14 28 `
  --rolling-windows 7 14 28 `
  --min-history-days 28
```

Part 3 writes `demand_features.csv`, `modeling_dataset.csv`, `feature_metadata.json`, and `feature_engineering_summary.json` under `outputs/features/`, plus chronological partitions under `outputs/features/splits/`. All rows for a target date stay in the same partition. See `docs/PART_3_COMPLETION.md` for feature definitions, leakage controls, assumptions, verification, and limitations.

## Milestone status

- **Part 1 — Complete:** reproducible project scaffold, data ingestion/validation, synthetic test data, and tests.
- **Part 2 — Complete:** reproducible EDA tables, findings, HTML dashboard, and dedicated tests.
- **Part 3 — Complete:** leakage-aware calendar/lag/rolling/commercial features, modeling-ready dataset, feature metadata, CLI, and tests.
- **Part 4 — Core implementation complete; optional frameworks pending runtime validation:** LightGBM trained and evaluated; Prophet, Darts and PyTorch Forecasting adapters are implemented but were not trained in the build runtime because those optional packages were unavailable.
- **Part 5 — Complete for currently trained models:** validation-first ranking, test reporting, rolling-origin backtesting and segment diagnostics.
- **Part 6 — Core implementation complete:** safety stock, reorder point, EOQ, MOQ/pack constraints and budget/capacity-constrained procurement optimization. OR-Tools CBC is implemented as preferred backend; the packaged validation run used the explicitly recorded SciPy MILP fallback because OR-Tools was not installed in the build runtime.
- **Part 7 — Implementation complete; local Dash runtime install required:** interactive Plotly Dash dashboard, shared filters, KPI cards, forecast/inventory views and recommendation table. The build environment was offline and Dash was not installed, so the app server could not be launched here; dashboard data services, Plotly figures, source compilation and the rest of the test suite were verified.
- **Part 8 — Integrated delivery complete:** unified pipeline runner, release-readiness checks, Windows setup/launch scripts, CI workflow, final handover and submission guide. Target-Windows live Dash launch and optional framework installation remain environment-specific verification items.

See `docs/PART_2_COMPLETION.md`, `docs/PART_3_COMPLETION.md`, and `docs/PART_4_COMPLETION.md` for milestone scope, findings, commands, and acceptance checks.

## Part 4 — Forecasting models

Part 4 adds a common, leakage-aware forecasting workflow over the Part 3 chronological
train/validation/test splits. The default command trains the global LightGBM demand regressor:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_forecasting.py --models lightgbm
```

Optional adapters are also included for Prophet (per store-SKU series), Darts
(ExponentialSmoothing per store-SKU series), and PyTorch Forecasting's Temporal Fusion
Transformer. These packages are deliberately optional because their installation footprints
and platform constraints differ. Check availability with:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_forecasting_environment.py
```

Install optional stacks individually:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-prophet-lightgbm.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements-darts.txt
# Install the compatible CPU/CUDA PyTorch build first, then:
.\.venv\Scripts\python.exe -m pip install -r .\requirements-pytorch-forecasting.txt
```

To attempt all four model adapters, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_forecasting.py --models all
```

A framework that is not installed is marked `not_installed`; no placeholder forecast or metric
is produced for it. Training failures are recorded explicitly in
`outputs/reports/part4_model_status.json`.

### Part 4 outputs

- `models/trained/lightgbm_demand_regressor.joblib`
- `outputs/forecasts/validation_lightgbm_predictions.csv`
- `outputs/forecasts/test_lightgbm_predictions.csv`
- `outputs/reports/part4_lightgbm_metrics.json`
- `outputs/reports/lightgbm_feature_importance.csv`
- `outputs/reports/part4_model_status.json`
- `outputs/reports/part4_model_comparison.csv`
- `outputs/reports/part4_forecasting_report.html`

Metrics include MAE, RMSE, WAPE, sMAPE, bias and R-squared where defined. The final test
partition is not used for fitting or early stopping. Forecasts in this milestone use a rolling
one-step-ahead protocol: actual preceding-day demand is available in Part 3 lag features at each
daily origin. This is not a fixed-origin 23-day open-loop forecast.

The bundled dataset is synthetic development data. Its results validate the software pipeline
only and must not be represented as retailer/business performance.


## Part 5 — Model Evaluation & Comparison

Run the validation-first model comparison and expanding-window rolling-origin LightGBM backtest after completing Parts 3 and 4:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_evaluation.py --window-days 7 --n-jobs 4
.\.venv\Scripts\python.exe -m pytest -q
```

Or run the PowerShell wrapper:

```powershell
.\run_part5.ps1
```

Part 5 selects the model using validation WAPE only. The chronological test set is reserved for final reporting. It compares compatible model prediction files against a 7-day seasonal-naive baseline and produces regional/store/SKU diagnostics, demand-tier analysis, daily errors, largest misses, an HTML report, and a rolling-origin backtest. Rolling-origin folds refit at 7-date windows and make sequential one-step-ahead predictions using previously observed lag values; this is not a fixed-origin multi-step forecast. See `docs/PART_5_COMPLETION.md` for the verified sample results and interpretation limits.


## Part 6 — Inventory Optimization Engine

Run the inventory policy and constrained replenishment optimizer after Parts 1–5:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_inventory_optimization.py --solver auto
```

Or use the Windows wrapper (installs the optional OR-Tools/SciPy optimization dependencies first):

```powershell
.\run_part6.ps1
```

To run explicitly with OR-Tools CBC after installing it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-optimization.txt
.\.venv\Scripts\python.exe .\scripts\run_inventory_optimization.py --solver ortools
```

To run with SciPy MILP explicitly:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_inventory_optimization.py --solver scipy
```

Part 6 creates one policy row per store/SKU using the trailing observed daily-demand mean and standard deviation, latest observed closing stock, lead time, service-level z-score and periodic-review period. It computes safety stock, reorder point, order-up-to level, days of supply and EOQ. The integer optimizer selects pack quantities subject to a total INR budget, aggregate unit capacity, per-line MOQ, pack multiple and order-need bounds. OR-Tools CBC is preferred; SciPy MILP is used and logged as a fallback if OR-Tools is unavailable.

Outputs are written to `outputs/inventory/`: `inventory_policy.csv`, `optimized_purchase_orders.csv`, `supplier_summary.csv`, `inventory_optimization_summary.json`, and `inventory_optimization_report.html`. See `docs/PART_6_COMPLETION.md` for formulas, run instructions and assumptions.

Important: the included dataset is synthetic. The sample does not include current open purchase orders, backorders, supplier contracts, actual ordering costs or holding-cost rates. Defaults in `config/inventory_policy.yaml` are illustrative scenario assumptions, not verified business inputs. The demand estimator uses trailing observed history and does not pretend that Part 5 test predictions are future forecasts. Validate all inputs before operational decisions.


## Part 7 — Interactive Plotly Dash dashboard

Install the optional dashboard stack (Dash and Plotly) from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-dashboard.txt
```

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_dashboard.py --host 127.0.0.1 --port 8050
```

Or use the Windows helper, which installs the dashboard dependencies and starts the app:

```powershell
.\run_part7.ps1
```

Open `http://127.0.0.1:8050`. The dashboard contains shared Region, State, Store, Category, SKU and demand-date filters; observed demand and held-out LightGBM prediction comparison; regional and SKU demand; inventory risk; on-hand versus reorder-point positioning; supplier order value; and a paginated-by-display (top 30) selected recommendation table. Inventory metrics represent the Part 6 latest-stock snapshot; the date range filters historical demand and forecasts but does not rewrite the inventory snapshot.

The dashboard reads generated Part 1–6 project outputs and does not fabricate forecasts or orders. Run `run_part4.ps1` / `run_part6.ps1` first if the corresponding prediction/inventory outputs are absent. For a visual preview that does not require the Dash server, run `scripts\build_dashboard_preview.py`; it writes `outputs\dashboard\part7_dashboard_preview.html`.

### Part 7 runtime validation note

The dashboard source and data/figure layers were compiled and tested in the build environment. Dash could not be installed in that environment because external package downloads were unavailable, so the live Dash HTTP server was **not** launched during packaging. On Windows, `run_part7.ps1` installs `requirements-dashboard.txt` and starts the application. Dash's official installation guide documents `pip install dash`: https://dash.plotly.com/installation.


## Part 8 — Run the integrated project

The final integration runner executes Parts 1–7 in order, runs the automated test suite, and writes a release-readiness report:

```powershell
.\run_all.ps1
```

For a shorter run that skips the rolling-origin refit stage:

```powershell
.\run_all.ps1 -SkipBacktest
```

To resume from a stage (earlier-stage output files must already exist):

```powershell
.\.venv\Scripts\python.exe .\scripts\run_pipeline.py --from-stage part4
```

Launch the live Dash dashboard separately after the data pipeline has generated forecasts and inventory outputs:

```powershell
.\run_dashboard.ps1
```

Then open `http://127.0.0.1:8050`. Run the final release check explicitly with:

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_release.py --require-dashboard
```

The readiness report is saved to `outputs/final/release_readiness.json`. A successful file-level release check does not imply that optional Prophet, Darts or PyTorch Forecasting models were trained. Check `outputs/reports/part4_model_status.json`; each model's actual status is recorded there. The bundled dataset remains synthetic development data.
