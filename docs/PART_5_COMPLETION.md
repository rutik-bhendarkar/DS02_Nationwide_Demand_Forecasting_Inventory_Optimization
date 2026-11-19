# Part 5 — Model Evaluation & Comparison

**Status:** Completed for the trained LightGBM model and the seasonal-naive baseline.

**Data source:** Bundled synthetic development dataset. These results are not actual retailer performance.

## Evaluation protocol

- Compare model prediction files against the canonical Part 3 validation and test splits using `date`, `store_id`, and `sku_id` keys.
- Exclude a prediction file if its identifiers or actual targets do not exactly match the canonical split.
- Compare only models that have compatible predictions for both validation and test.
- Select a model using **validation WAPE** (lower is better), with RMSE and MAE as tie-breakers.
- Report test performance as a final chronological holdout estimate; do not use test metrics to select the model.
- Include a 7-day seasonal-naive forecast (`units_sold_lag_7d`) as the baseline.
- Report MAE, RMSE, WAPE, sMAPE, signed bias and R². WAPE and bias are undefined when total actual demand is zero; the metric helper returns `null` for those cases.

## Validation and final test results

| Model | Split | MAE | RMSE | WAPE | sMAPE | Bias | R² |
|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM | Validation | 3.031 | 4.130 | 13.123% | 13.465% | -1.365% | 0.862 |
| Seasonal naive (7-day lag) | Validation | 4.333 | 5.912 | 18.760% | 19.129% | -0.514% | 0.717 |
| LightGBM | Test | 3.175 | 4.403 | 13.399% | 13.521% | -2.077% | 0.850 |
| Seasonal naive (7-day lag) | Test | 4.547 | 6.308 | 19.187% | 19.442% | -0.238% | 0.692 |

LightGBM validation WAPE is 30.05% lower than the baseline, and test WAPE is 30.17% lower than the baseline (relative reductions, not percentage-point differences).

**Model-selection conclusion:** LightGBM is the selected model among the currently trained and compatible candidates. It is not an all-framework winner: Prophet, Darts and PyTorch Forecasting were not trained in the Part 4 environment and therefore are not included as model competitors.

## Expanding-window rolling-origin backtest

The test dates are evaluated in four sequential windows: 7, 7, 7 and 2 calendar dates. The model is refitted at the start of each window using only observations dated before that window. After a window completes, its observed targets are added to the training history for the next refit.

Within each window, forecasts are **rolling one-step-ahead** predictions. Part 3 lag features for a target date use previously observed demand. This is not a fixed-origin 7-day multi-step forecast.

| Fold | Forecast window | Training rows | LightGBM WAPE | Baseline WAPE |
|---|---|---:|---:|---:|
| 1 | 2025-06-07 to 2025-06-13 | 27,864 | 13.050% | 19.225% |
| 2 | 2025-06-14 to 2025-06-20 | 29,376 | 13.010% | 18.495% |
| 3 | 2025-06-21 to 2025-06-27 | 30,888 | 13.701% | 19.426% |
| 4 | 2025-06-28 to 2025-06-29 | 32,400 | 13.868% | 20.472% |
| **All folds** | **23 test dates** | — | **13.314%** | **19.187%** |

The rolling-origin LightGBM WAPE is 30.61% lower than the 7-day seasonal-naive baseline across the same 4,968 test observations.

## Additional diagnostics generated

- Model comparison on validation and test.
- Validation-based model ranking and selection flag.
- Test metrics by region, state, store, category and SKU.
- Test metrics by retrospective low-, medium- and high-demand terciles.
- Daily test metrics for each model.
- Largest 100 absolute errors for the validation-selected model.
- Four-fold rolling-origin predictions, fold metrics and aggregate metrics.
- Interactive Plotly HTML report.

## Output inventory

- `outputs/reports/part5_model_comparison.csv`
- `outputs/reports/part5_model_ranking.csv`
- `outputs/reports/part5_segment_performance.csv`
- `outputs/reports/part5_demand_tier_performance.csv`
- `outputs/reports/part5_daily_metrics.csv`
- `outputs/reports/part5_largest_forecast_errors.csv`
- `outputs/reports/part5_rolling_origin_fold_metrics.csv`
- `outputs/reports/part5_rolling_origin_metrics.json`
- `outputs/reports/part5_evaluation_summary.json`
- `outputs/reports/part5_evaluation_report.html`
- `outputs/forecasts/part5_rolling_origin_lightgbm_predictions.csv`

## Reproduction

From the project root with the `.venv` created by `setup_windows.ps1`:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_evaluation.py --window-days 7 --n-jobs 4
.\.venv\Scripts\python.exe -m pytest -q
```

Alternatively:

```powershell
.\run_part5.ps1
```

The pipeline requires the Part 3 feature split files and a trained Part 4 LightGBM model/prediction files. Install the Part 4 LightGBM optional dependencies before running on a new environment.

## Verification

- Python compilation check: passed for `src`, `scripts` and `tests`.
- Automated test suite: **40 passed**.
- Model comparison script: completed successfully.
- Rolling-origin backtest: **4 folds**, all 23 test dates evaluated.
- Report inventory: all paths listed in `outputs/reports/part5_evaluation_summary.json` exist.
- Archive integrity: checked after packaging.

`ruff` was not installed in the build environment, so linting was not performed. The completion claim is based on compilation, pytest, the actual model-comparison run, and the actual LightGBM rolling-origin refits.
