# Part 4 Completion — Forecasting Models

## Scope delivered

Part 4 adds a shared forecasting layer on top of Part 3's 44 leakage-aware features and
chronological train/validation/test files.

Implemented adapters:

1. **LightGBM** — global panel `LGBMRegressor`, trained and evaluated in this build.
2. **Prophet** — one model per store-SKU series, with optional approved known-future regressors.
3. **Darts** — one ExponentialSmoothing model per store-SKU series.
4. **PyTorch Forecasting** — Temporal Fusion Transformer for rolling one-step-ahead panel prediction.

Adapters for Prophet, Darts and PyTorch Forecasting are optional and use lazy imports. Their
availability is recorded by `scripts/check_forecasting_environment.py`. If a framework is not
installed, the runner marks it `not_installed` and does not create dummy metrics or forecasts.

## Reproducible training protocol

- Fit on the Part 3 training split only.
- Use validation for LightGBM early stopping / model diagnostics.
- Keep the chronological test split untouched for final evaluation.
- Forecast protocol is rolling one-step-ahead. The features for each target date include only
  prior observed demand and covariates that are assumed known from the planning calendar/catalog.
- Forecast values are clipped at zero because unit demand cannot be negative.
- Compare model results on the same target rows and forecast protocol before selecting a winner.

## LightGBM result (bundled synthetic development dataset)

The current execution trained `LGBMRegressor` with 44 features. Early stopping selected iteration
165 from a maximum of 500 estimators. The model was fitted using 22,896 training rows and used
4,968 validation rows for early stopping. Final test evaluation used 4,968 untouched test rows.

| Test metric | 7-day seasonal-naive baseline | LightGBM |
|---|---:|---:|
| MAE | 4.5467 | 3.1751 |
| RMSE | 6.3083 | 4.4027 |
| WAPE | 19.1871% | 13.3989% |
| sMAPE | 19.4418% | 13.5209% |
| Bias | -0.2378% | -2.0768% |
| R² | 0.6921 | 0.8500 |

The LightGBM WAPE is approximately 30.17% lower than the 7-day lag baseline on this synthetic
test partition. This is a development-pipeline result, not a claim about actual retailer data.

## Current framework execution status in the build environment

| Framework | Adapter | Installed here | Trained here |
|---|---|---:|---:|
| LightGBM | LGBMRegressor | Yes | Yes |
| Prophet | Prophet per store-SKU | No | No |
| Darts | ExponentialSmoothing per store-SKU | No | No |
| PyTorch Forecasting | Temporal Fusion Transformer | No | No |

## Outputs

- `models/trained/lightgbm_demand_regressor.joblib`
- `outputs/forecasts/validation_lightgbm_predictions.csv`
- `outputs/forecasts/test_lightgbm_predictions.csv`
- `outputs/reports/part4_lightgbm_metrics.json`
- `outputs/reports/part4_model_status.json`
- `outputs/reports/part4_completion_summary.json`
- `outputs/reports/lightgbm_feature_importance.csv`
- `outputs/reports/part4_model_comparison.csv`
- `outputs/reports/part4_forecasting_report.html`

## Run commands

```powershell
.\.venv\Scripts\python.exe .\scripts\check_forecasting_environment.py
.\.venv\Scripts\python.exe .\scripts\run_forecasting.py --models lightgbm
.\.venv\Scripts\python.exe -m pytest -q
```

Optional model installation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-prophet-lightgbm.txt
.\.venv\Scripts\python.exe -m pip install -r .\requirements-darts.txt
# Install a compatible PyTorch CPU/CUDA build first, then:
.\.venv\Scripts\python.exe -m pip install -r .\requirements-pytorch-forecasting.txt
.\.venv\Scripts\python.exe .\scripts\run_forecasting.py --models all
```

## Limitations and next step

- Prophet, Darts and TFT adapters have not been exercised in this runtime because their optional
  dependencies are absent. Install and test them in the target Windows environment before claiming
  those algorithms are trained or validated.
- The sample dataset is explicitly synthetic and relatively short (180 calendar days). Yearly
  seasonal behaviour and true national retail performance cannot be established from it.
- The current protocol is one-step rolling-origin prediction, not a fixed-origin 23-day horizon.
- Part 5 should add expanding-window backtesting, per-region/store/SKU error slices, forecast
  interval evaluation where available, and model-selection criteria on consistent dates.

## Official API references used for the optional adapters

- Prophet Python quick start (`ds` and `y` input frame): https://meta.github.io/prophet/docs/quick_start.html
- LightGBM Python API (`LGBMRegressor`): https://lightgbm.readthedocs.io/en/stable/Python-API.html
- Darts installation and package migration notes: https://github.com/unit8co/darts/blob/master/INSTALL.md
- PyTorch Forecasting Temporal Fusion Transformer: https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
