# Model Card — LightGBM Demand Regressor

## Intended purpose

Development baseline for next-day retail unit-demand prediction at store-SKU level. The model
supports experimentation and the later inventory optimization milestone. It is not approved for
production purchasing decisions.

## Data and provenance

- Source: bundled DS-02 synthetic development dataset.
- Target: `target_units_sold`.
- Train rows: 22,896.
- Validation rows: 4,968.
- Test rows: 4,968.
- Model features: 44 Part 3 features.
- Partitioning: chronological by unique target date; dates do not overlap.

## Model

- Estimator: LightGBM `LGBMRegressor`.
- Objective: `regression_l1`.
- Learning rate: 0.035.
- Maximum estimators in run: 500.
- Best iteration: 165 (validation early stopping).
- Random state: 42.
- Categorical code columns: passed as pandas categorical features.

## Test performance on synthetic sample

- MAE: 3.1751 units.
- RMSE: 4.4027 units.
- WAPE: 13.3989%.
- sMAPE: 13.5209%.
- Bias: -2.0768%.
- R²: 0.8500.

7-day lag baseline WAPE: 19.1871%. The test sample result is not a production performance
estimate and should not be generalized to real retailers.

## Leakage and evaluation notes

- Model fitting uses only the training partition.
- Validation is used for early stopping.
- Test is untouched during fitting and early stopping.
- Forecasts use rolling one-step-ahead lag features; preceding observed daily demand is available
  for each target date.
- Same-target-date pricing/promotion/holiday features are valid only when available from an
  approved future planning source at prediction time.
- Inventory outcomes such as closing stock and same-day net sales are excluded from model inputs.

## Known limitations

- Synthetic data only; no real retailer pilot or prospective test has been performed.
- 180-day sample does not support robust yearly seasonality assessment.
- No uncertainty intervals are currently reported for LightGBM.
- Feature importance indicates model reliance, not causation.
- Retraining policy, drift monitoring and production data contracts remain future work.
