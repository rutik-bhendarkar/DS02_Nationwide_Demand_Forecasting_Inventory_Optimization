# Part 8 — Final Integration & Submission

## Delivered
- Unified PowerShell environment setup with Core and Integrated profiles.
- `scripts/run_pipeline.py` orchestrating Parts 1–7 in dependency order.
- `scripts/verify_release.py` checking required source and generated artifacts, model/evaluation metadata, provenance and Dash dependency availability.
- `run_all.ps1` for full workflow, tests and release check.
- `run_dashboard.ps1` for a separate interactive Dash launch.
- Integrated dependency manifest, final submission guide, handover documentation and CI workflow.

## Workflow order
1. Part 1 data validation.
2. Part 2 EDA.
3. Part 3 feature engineering and chronological splits.
4. Part 4 LightGBM training.
5. Part 5 evaluation and rolling-origin backtest.
6. Part 6 inventory policy and optimization.
7. Part 7 dashboard HTML preview generation.
8. Full pytest suite and release-readiness verification.

## Verification status
The final archive is verified for ZIP integrity. The project test suite is run as part of Part 8 packaging; exact counts and release readiness details are included in `outputs/final/release_readiness.json` and in the handover note. The build environment did not include Dash, so the live Dash server must be launched on Windows after `setup_windows.ps1 -Profile Integrated` installs the dashboard dependency. Optional Prophet, Darts, PyTorch Forecasting and OR-Tools availability is tracked separately and must not be claimed as trained/used unless per-run metadata confirms it.

## Data honesty
Bundled data is synthetic. No production retail or financial benefit claims are made. Inventory and supplier settings remain scenario assumptions until replaced with authorized ERP/procurement data.

## Executed integration evidence (build environment)
- Full `scripts/run_pipeline.py` run completed without stage errors.
- Part 1 validation: 38,880 rows, 21 columns, zero missing values and zero duplicate business keys in bundled sample.
- Part 3: 44 features; chronological train/validation/test partitions; 32,832 modeling rows.
- Part 4: LightGBM trained; test WAPE 13.399% on synthetic sample.
- Part 5: four rolling-origin folds completed; overall LightGBM WAPE 13.314% versus seasonal-naive baseline WAPE 19.187% on synthetic sample.
- Part 6: 216 inventory policy rows and 70 reorder candidates; this build used SciPy MILP because OR-Tools was unavailable.
- Part 7: six Plotly chart figures and static HTML preview generated. Dash is not installed in the build environment; `scripts/smoke_test_dashboard.py` is provided to perform a Flask test-client HTTP check after installing the integrated dependencies on Windows.
- Automated test suite: 65 passed.
- Release verifier: 18 required artifact/model/provenance checks passed; the Dash runtime dependency check is recorded as an environment warning (not available in the offline build runtime).

The metrics above are computed on synthetic development data and are not business performance claims.
