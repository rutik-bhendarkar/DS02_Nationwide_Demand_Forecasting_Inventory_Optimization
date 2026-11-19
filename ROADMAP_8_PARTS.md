# DS-02 — Eight-part implementation roadmap

| Part | Scope | Exit criteria |
|---|---|---|
| 1. Data & Environment Setup | Project scaffold, Python environment, dependency groups, input schema, sample data, ingestion, validation, tests, documentation | **COMPLETE** — setup documented; sample CSV loads; schema validation passes; tests pass |
| 2. EDA | Data profiling, trend/seasonality, store/SKU/region/category analysis, missingness and outlier diagnostics | **COMPLETE** — reproducible CSV tables, JSON summary, Markdown findings, Plotly HTML, and automated tests |
| 3. Feature Engineering | Calendar, lag, rolling, promotion, holiday, price, store and SKU features; time-based splits | **COMPLETE** — reusable leakage-aware target-date features, metadata, modeling frame, chronological date splits, CLI, and tests |
| 4. Forecasting Models | Prophet, LightGBM, Darts, PyTorch Forecasting implementations where data volume and compute justify them | **PARTIAL** — LightGBM trained/evaluated; optional Prophet, Darts and TFT adapters implemented but require dependency installation and target-environment validation |
| 5. Evaluation & Model Selection | Rolling-origin backtesting; MAE, RMSE, WAPE, sMAPE, bias, R²; region/store/SKU/demand-tier diagnostics; validation-based model ranking | **COMPLETE for trained LightGBM + seasonal-naive baseline** — validation-first selection, final test reporting, four expanding windows, error analysis and Plotly report. Other frameworks are included only after they produce compatible prediction files |
| 6. Inventory Optimization | Demand/lead-time uncertainty, safety stock, reorder points, service levels, MOQ/pack constraints, budget/capacity-constrained integer procurement | **CORE COMPLETE** — policy and order recommendations generated; OR-Tools CBC backend implemented but this build was validated with SciPy MILP fallback because OR-Tools was unavailable in the build runtime |
| 7. Interactive Dashboard | Plotly Dash UI, nationwide filters, forecast and inventory views, recommendation table | **IMPLEMENTATION COMPLETE; live server launch pending Dash installation on target Windows machine** |
| 8. Integration & Submission | End-to-end workflow, tests, packaging, README, architecture, demo, GitHub preparation | **INTEGRATED DELIVERY COMPLETE** — pipeline runner, release verifier, Windows helpers, CI workflow, handover and submission guide; live Dash launch still requires target-machine runtime verification |

## Implementation policy

- Use supplied business data when available and approved. The bundled synthetic dataset is only for development and test plumbing.
- Never invent model accuracy or business savings. Metrics must be computed from time-ordered holdout periods.
- Do not randomly split time-series rows. Use chronological or rolling-origin validation in later parts.
- Keep optional heavy dependencies separated so the base environment stays easier to install on Windows.
- Every milestone must have tests and a completion note before the next milestone begins.

## Part 4 — Forecasting Models (core implemented; optional adapters pending runtime validation)

- Global LightGBM regression model trained on Part 3's 44 leakage-aware features.
- Validation-based early stopping and untouched chronological test evaluation.
- Seasonal-naive 7-day lag baseline for context.
- Per-series Prophet adapter, per-series Darts ExponentialSmoothing adapter, and
  PyTorch Forecasting Temporal Fusion Transformer adapter.
- Explicit framework availability detection; unavailable stacks are reported as
  `not_installed`, never given fabricated metrics.
- Common metric calculation and prediction export for validation and test partitions.

Part 5 will handle broader backtesting, error segmentation, model comparison, and
selection criteria. The current LightGBM result is not an all-model winner claim.


## Part 5 — Evaluation & Model Selection (completed for current trained models)

- Model comparisons require exact date/store/SKU key and actual-target matches against the canonical Part 3 split.
- Select using validation WAPE only; use RMSE and MAE as tie-breakers. Test metrics are final holdout reporting, not selection criteria.
- Compare trained models with the 7-day seasonal-naive baseline.
- Generate segment-level test metrics, daily metrics, demand-tier metrics and largest-error diagnostics.
- Run expanding-window rolling-origin LightGBM evaluation with a refit every seven test dates and rolling one-step-ahead prediction semantics.
- Full results and the exact forecast protocol are in `docs/PART_5_COMPLETION.md`.


## Part 6 — Inventory Optimization (core implemented)

- Build one inventory policy per store/SKU from trailing observed daily-demand statistics and the latest observed closing-stock snapshot.
- Calculate safety stock, reorder point, periodic-review order-up-to level, days of supply and EOQ.
- Use explicit scenario defaults for missing open PO/backorder and supplier MOQ/pack data; supplier constraints can be supplied in a CSV.
- Optimize integer order pack counts subject to budget, aggregate capacity, MOQ, order multiple and policy order-need bounds.
- OR-Tools CBC is the preferred backend; SciPy MILP fallback is explicit and recorded.
- Generate inventory policy, optimized order lines, supplier summary, JSON metadata and Plotly HTML report.
- Full formulas, limitations and run evidence are documented in `docs/PART_6_COMPLETION.md`.


## Part 7 — Interactive dashboard (implementation complete; runtime install pending)

- Responsive Plotly Dash dashboard with shared region/state/store/category/SKU and demand-date filters.
- KPI cards for demand, stores/SKUs, held-out forecast WAPE, reorder candidates, order lines, planned order value and on-hand stock.
- Interactive observed-demand / saved LightGBM forecast view; regional and SKU demand charts; inventory risk; stock-position scatter; supplier order-value chart; top selected replenishment table.
- Snapshot inventory data is deliberately not changed by the historical demand date filter.
- Core data, filtering, KPI and Plotly figure layers have automated tests. Dash server install/launch remains to be verified on the target Windows machine because the build environment had no external package download access.
- Details and Windows run instructions are in `docs/PART_7_COMPLETION.md`.


## Part 8 — Final integration and submission (integrated delivery)

- `run_all.ps1` and `scripts/run_pipeline.py` orchestrate Parts 1–7 in dependency order.
- `scripts/verify_release.py` validates expected source/output artifacts, model-selection metadata, synthetic provenance and Dash dependency availability.
- `setup_windows.ps1 -Profile Integrated` provisions the integrated runtime and reuses an existing `.venv`.
- `run_dashboard.ps1` installs the Dash runtime if needed and starts the interactive application at `http://127.0.0.1:8050`.
- GitHub Actions workflow runs tests and Python compilation.
- Final submission guide and project handover document operational steps, limitations and a suggested demo sequence.
- Live Dash server startup and optional Prophet/Darts/TFT/OR-Tools installation must be verified on the target Windows machine; they are not falsely marked as executed in the offline build environment.
