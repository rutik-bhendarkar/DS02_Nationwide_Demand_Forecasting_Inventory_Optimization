# Part 7 — Interactive Plotly Dash Dashboard

## Delivered

- Responsive Plotly Dash application with a dedicated dashboard layout and stylesheet.
- Shared Region, State, Store, Category, SKU and demand-date filters.
- KPI cards for observed demand, average daily demand, stores/SKUs, saved test forecast WAPE, reorder candidates, selected order lines, planned order value and on-hand stock.
- Observed demand against saved held-out LightGBM forecasts.
- Region demand and top SKU demand charts.
- Inventory risk distribution and on-hand versus reorder-point scatter plot.
- Supplier purchase-order value chart and selected replenishment recommendation table.
- Reset filters action and clear data provenance/assumption notices.
- Data-service layer does not import Dash; tests can exercise filters and calculations without web dependencies.
- Static Plotly preview generated at `outputs/dashboard/part7_dashboard_preview.html`.

## Data behavior and honesty controls

- The dashboard reads Part 1 sample sales, Part 4 saved test forecasts and Part 6 optimized inventory records. It does not train models or fabricate predictions/orders at page load.
- Forecast WAPE is calculated from saved held-out predictions only.
- Inventory policy and order views are a latest-observed stock snapshot. The historical date filter intentionally filters sales/forecast history, not the snapshot inventory position.
- If the forecast/inventory outputs are absent, the affected chart/table shows an unavailable state; no dummy values are created.
- The sample source is synthetic development data. Its KPIs and recommendations are demonstrations of the pipeline, not actual retail performance or operational purchase advice.

## Windows installation and launch (PowerShell)

```powershell
cd "D:\Internship\DS By Nexvia\DS02_Nationwide_Demand_Forecasting_Inventory_Optimization_FULL_INTEGRATED"
.\.venv\Scripts\python.exe -m pip install -r .\requirements-dashboard.txt
.\.venv\Scripts\python.exe .\scripts\run_dashboard.py --host 127.0.0.1 --port 8050
```

Or run `.\run_part7.ps1` from the project root. Open `http://127.0.0.1:8050` and stop the server with `Ctrl+C`.

If data outputs are missing, first run the relevant Part 4 and Part 6 pipelines. To generate a non-server preview, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_dashboard_preview.py
```

## Verification performed during packaging

- Python compilation: passed.
- Full automated project tests: 62 passed.
- Dashboard service tests: loading actual generated inputs, options, dimension filters, date filtering semantics, KPI consistency, empty-filter handling and Plotly figure construction.
- Static Plotly dashboard preview: generated successfully.
- ZIP integrity: verified after packaging.

## Runtime limitation

Dash was not installed in the isolated build environment and external package downloads were unavailable. Therefore the live Dash HTTP server was not launched here. The dashboard dependency group is declared in `requirements-dashboard.txt` (`dash>=3.0,<5.0`, `plotly>=5.22`) and `pyproject.toml`. The target Windows environment must install that optional group and launch the app to complete browser-level acceptance testing.
