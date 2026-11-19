# Part 1 completion record — Data & Environment Setup

## Scope delivered

- Python package scaffold using a `src/` layout.
- Core dependency manifest plus optional dependency groups for dashboard, Prophet/LightGBM, Darts, PyTorch Forecasting, and OR-Tools.
- YAML configuration and `.env.example` template.
- Canonical daily store-SKU schema and data dictionary.
- CSV ingestion and validation for required columns, dates, numeric values, non-negative values, discount range, and duplicate business keys.
- Deterministic synthetic sample data for 18 stores, 12 SKUs, and 180 days (38,880 rows).
- Windows setup and validation scripts.
- Automated tests for valid data, missing columns, negative units, discount range, invalid dates, duplicate keys, and bundled sample data.

## Explicit limitations

Part 1 does not yet implement EDA, forecasting models, model evaluation, inventory optimization, or a dashboard. Those are assigned to Parts 2–7. The bundled dataset is synthetic and is not a real business dataset or evidence of model performance.

## Acceptance commands

```powershell
# from the project root
.\setup_windows.ps1
.\.venv\Scripts\python.exe .\scripts\validate_data.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Expected sample dimensions

- Rows: 38,880
- Stores: 18
- SKUs: 12
- Regions: 6
- Daily observations per store-SKU series: 180
- Date span: 2025-01-01 to 2025-06-29
