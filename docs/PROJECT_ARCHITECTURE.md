# Project architecture

```text
DS02_Nationwide_Demand_Forecasting_Inventory_Optimization/
├── config/                    # YAML configuration
├── data/
│   ├── raw/                   # Original source extracts (not committed)
│   ├── processed/             # Cleaned/feature-ready data (not committed)
│   └── sample/                # Clearly labelled synthetic development data
├── docs/                      # Data dictionary, roadmap, completion notes
├── outputs/
│   ├── eda/                   # Part 2 EDA tables and HTML dashboard
│   ├── features/              # Part 3 feature and modeling datasets
│   ├── forecasts/             # Saved train/validation/test predictions
│   ├── inventory/             # Part 6 policy and optimized order outputs
│   ├── dashboard/             # Part 7 static interactive chart preview
│   └── reports/               # Generated model/evaluation reports
├── scripts/                   # Data, EDA, features, forecasting, evaluation, inventory, dashboard
├── assets/                    # Dash dashboard styles
├── src/demand_inventory/
│   ├── analysis/              # Part 2 exploratory analysis
│   ├── data/                  # Schema, ingestion, validation
│   ├── features/              # Part 3 leakage-aware feature engineering
│   ├── forecasting/           # Part 4 model adapters and prediction outputs
│   ├── evaluation/            # Part 5 validation/backtesting/metrics
│   ├── inventory/             # Part 6 policy and procurement optimization
│   ├── dashboard/             # Part 7 Dash data services, figures and app
│   ├── utils/                 # Shared utilities
│   ├── config.py              # YAML configuration loader
│   └── cli.py                 # Validation CLI
├── tests/                     # Automated Part 1–7 tests
├── pyproject.toml
├── requirements*.txt
├── setup_windows.ps1
└── README.md
```

## Data flow through Part 3

1. CSV source → Pandas ingestion → schema/type/range/uniqueness checks → validation report.
2. Validated data → descriptive EDA tables and figures → findings and quality summary.
3. Validated data → target-date calendar and known-future covariates + exact historical lags + past-only rolling statistics → full feature panel → warm-up filtered modeling dataset and feature metadata.

Parts 4–7 add model training, evaluation, inventory policies/optimization and the interactive Dash dashboard. See each `docs/PART_N_COMPLETION.md` for the exact scope, tests, runtime requirements and limitations.
