# DS-02 Project Handover

## Architecture at a glance
- `src/demand_inventory/data/`: ingestion, schema and validation.
- `src/demand_inventory/analysis/`: EDA and descriptive outputs.
- `src/demand_inventory/features/`: leakage-aware time-series feature engineering.
- `src/demand_inventory/forecasting/`: common model interface, LightGBM and optional adapters.
- `src/demand_inventory/evaluation/`: metric comparison and rolling-origin backtesting.
- `src/demand_inventory/inventory/`: inventory policy calculations and constrained optimization.
- `src/demand_inventory/dashboard/`: Dash data service, chart figures and application callbacks.
- `scripts/`: reproducible stage CLIs and release verification.
- `tests/`: unit and integration tests.

## Main commands
- Setup: `./setup_windows.ps1 -Profile Integrated`
- Full pipeline: `./run_all.ps1`
- Dashboard: `./run_dashboard.ps1`
- Test suite: `./.venv/Scripts/python.exe -m pytest -q`
- Release check: `./.venv/Scripts/python.exe ./scripts/verify_release.py --require-dashboard`

## Main generated outputs
- `reports/part1_validation_report.json`
- `reports/part2_eda_summary.json` and EDA CSVs/HTML under `outputs/eda/`
- Feature tables and chronological splits under `outputs/features/`
- Models, predictions and metrics under `models/` and `outputs/forecasts/`, `outputs/reports/`
- Inventory policies and purchase-order recommendations under `outputs/inventory/`
- Dash preview under `outputs/dashboard/`; live app at `127.0.0.1:8050`
- Final readiness report under `outputs/final/release_readiness.json`

## Known limitations to disclose
1. Bundled data is synthetic.
2. Prophet, Darts and PyTorch Forecasting have adapters but must be separately installed and trained; inspect model status JSON.
3. If OR-Tools is not installed, the inventory optimizer can use SciPy MILP; report its backend metadata.
4. Actual ERP stock positions, inbound purchase orders, supplier constraints, holding costs and shortage costs are not provided by the synthetic sample and must be replaced with authorized values.
5. A local Dash server must be launched on the target machine after dependency setup; static HTML preview is not a substitute for validating the live web app.
