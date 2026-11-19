# Part 2 completion record — Exploratory Data Analysis

## Scope delivered

- Reproducible EDA entry point: `scripts/run_eda.py`.
- Reusable analysis implementation: `src/demand_inventory/analysis/eda.py`.
- Daily and monthly demand trends; weekday and month-of-year patterns.
- Region, state, city, store, category, SKU and product performance tables.
- Descriptive promotion and holiday comparisons.
- Closing-stock pressure summary, without claiming that zero stock proves lost sales.
- Missingness profiling and store-SKU calendar coverage diagnostics.
- Store-SKU-specific 1.5 × IQR demand outlier flags; no observations are automatically removed.
- Numeric correlation table.
- Self-contained interactive Plotly HTML dashboard (when Plotly is installed).
- Machine-readable JSON summary and Markdown findings.
- Automated tests for output generation, correct daily aggregation semantics, non-mutation, invalid inputs and `top_n` validation.

## Metric correctness correction

Dimension-level tables now distinguish:

- `average_units_per_record`: average units for an individual store-SKU-date row.
- `average_daily_units`: average of the dimension's total units summed per calendar date.
- `median_daily_units`: median of the dimension's total units summed per calendar date.

This prevents regional/store/category/SKU daily demand from being mislabeled as a row-level average.

## Bundled synthetic dataset results

- Rows: 38,880 across 21 columns.
- Date coverage: 2025-01-01 to 2025-06-29 (180 days).
- Stores: 18; SKUs: 12; regions: 6.
- Observed units sold: 882,002.
- Mean daily nationwide observed sales: 4,900.01 units.
- Missing cells: 0; duplicate date-store-SKU keys: 0.
- Store-SKU series: 216; series with missing days: 0.
- IQR-flagged observations: 590 (1.5175%). These are review flags, not proven errors.

All figures above describe only the bundled synthetic development dataset. They are not real business results, sales forecasts, or performance claims.

## Run commands (PowerShell)

The standard Windows setup installs `.[dev,eda]` so Plotly is available for the self-contained HTML report. For an existing environment, install the EDA extra once:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[eda]"
```

Then run:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_eda.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Acceptance status

- EDA script executes successfully against the bundled sample.
- CSV tables, JSON summary, Markdown report and interactive HTML dashboard are generated.
- Automated test suite: **14 passed** at milestone packaging time.
- No forecasting models are trained in this milestone.

## Interpretation limitations

- The input is synthetic development data.
- Observed sales can be censored by inventory availability.
- Promotion and holiday comparisons are descriptive, not causal.
- Outlier flags must be reviewed using stock, promotion, holiday and source-system context before any treatment.
- Forecast feature creation, time-ordered validation and model training are reserved for later milestones.
