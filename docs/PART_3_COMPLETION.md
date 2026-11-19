# Part 3 Completion — Feature Engineering

**Project:** DS-02 — Nationwide Demand Forecasting and Inventory Optimization  
**Organization:** Nexvia Technologies Internship  
**Milestone:** 3 of 8  
**Status:** Complete for the bundled synthetic development dataset

## Scope delivered

The feature-engineering pipeline converts validated daily store-SKU sales rows into a target-date feature panel. Each row represents the demand to predict for one store, one SKU, and one target date. It also creates chronological train/validation/test partitions by date.

### Feature groups

1. **Calendar:** year, quarter, month, ISO week, day of month/year/week, weekend, month-start/month-end flags, cyclical month/weekday encodings, and series age.
2. **Identifiers:** deterministic integer codes for store, SKU, region, state, city, category, and supplier when present. The mappings are saved in metadata; they must be reused unchanged at inference time, and unseen identifiers require an explicit unknown-category policy in later model-serving work.
3. **Known-future commercial covariates:** target-date unit price, discount percentage, promotion flag, holiday flag, price after discount, and promotion-discount interaction when source columns exist.
4. **Exact calendar lags:** units sold 1, 7, 14, and 28 calendar days before the target date.
5. **Past-only rolling summaries:** rolling mean, sample standard deviation, sum, and observation count for the preceding 7, 14, and 28 calendar days.

## Leakage safeguards

- Lag features are joined by exact calendar date. A missing previous-day observation remains missing; the pipeline does not silently use the previous available row as a one-day lag.
- Rolling windows use `closed='left'`, excluding the target date.
- The same-day observed `units_sold` is stored as `target_units_sold` and is never included in the declared model feature list.
- Same-day opening/replenishment/closing stock and net sales value are excluded from model features because these fields can be contemporaneous with or downstream of realized sales.
- Same-target-date price, discount, promotion and holiday variables are labelled known-future covariates. They are valid only when a future pricing/promotion/calendar plan is actually available at prediction time.
- No missing historical demand is imputed or replaced with zero.
- The prepared modeling frame retains rows only when all configured direct lags exist and the selected historical rolling window contains the required number of prior observations.
- The chronological split sorts unique target dates and assigns complete dates to train, validation, or test. It never randomly shuffles store-SKU rows.

## Files added

- `src/demand_inventory/features/engineering.py` — reusable feature construction and modeling-frame preparation.
- `scripts/run_feature_engineering.py` — command-line pipeline runner.
- `tests/test_feature_engineering.py` — leakage, date alignment, schema, determinism and pipeline tests.
- `outputs/features/demand_features.csv` — full feature panel including warm-up rows.
- `outputs/features/modeling_dataset.csv` — rows with required historical context.
- `outputs/features/feature_metadata.json` — feature lists, mappings, assumptions and leakage controls.
- `outputs/features/feature_engineering_summary.json` — machine-readable run summary.
- `outputs/features/splits/train.csv`, `validation.csv`, `test.csv` — chronological partitions by unique target date.

## Reproduce

```powershell
.\.venv\Scripts\python.exe .\scripts\run_feature_engineering.py
.\.venv\Scripts\python.exe -m pytest -q
```

Custom input and output:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_feature_engineering.py `
  --input .\data\raw\your_sales_file.csv `
  --output-dir .\outputs\features `
  --lags 1 7 14 28 `
  --rolling-windows 7 14 28 `
  --min-history-days 28
```

## Verification on the bundled synthetic sample

- Input rows: 38,880
- Store count: 18
- SKU count: 12
- Date range: 2025-01-01 through 2025-06-29
- Declared feature columns: 44
- Full feature panel rows: 38,880
- Modeling rows after 28-day warm-up/history requirement: 32,832
- Warm-up rows excluded: 6,048
- Missing values across declared features in the modeling frame: 0
- Duplicate `(date, store_id, sku_id)` keys in the modeling frame: 0
- Chronological split: 70% train / 15% validation / 15% test by unique target date (no random row shuffle)
- All rows for the same target date are kept in one split
- Automated test suite at completion: 27 passed

All results above describe the bundled synthetic development dataset only. They are not retailer benchmarks, forecasts, or claims of business impact.

## Limitations and next step

This pipeline establishes one-step-ahead target-date features. Same-day promotion, price and holiday values must be supplied from a known future plan for future dates. Multi-step forecasting requires a strategy such as recursive prediction or horizon-specific direct models; this milestone does not train models or claim forecast accuracy.

**Next:** Part 4 — forecasting model implementations behind a common train/predict interface.
