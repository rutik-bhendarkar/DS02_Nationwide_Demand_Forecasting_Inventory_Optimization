# Part 6 — Inventory Optimization Engine

## Scope implemented

- Per-store/SKU trailing demand estimates (mean and sample standard deviation over a configurable calendar window).
- Safety stock, lead-time demand, reorder point, periodic-review order-up-to level, days of supply and EOQ.
- Latest observed closing-stock snapshot per store/SKU.
- Open purchase orders and backorders are incorporated when those columns exist; otherwise explicit YAML defaults are used.
- Replenishment quantity rounded to configurable MOQ and order multiple.
- Constrained integer procurement allocation under a total INR budget and aggregate unit capacity.
- OR-Tools CBC backend preferred; SciPy MILP is an explicit fallback if OR-Tools is not installed.
- Policy CSV, optimized order CSV, supplier summary, JSON summary and Plotly HTML report.

## Core formulas

- Safety stock = `z × daily-demand standard deviation × sqrt(lead time)`.
- Reorder point = `mean daily demand × lead time + safety stock`.
- Order-up-to level = `mean daily demand × (lead time + review period) + z × daily-demand standard deviation × sqrt(lead time + review period)`.
- EOQ = `sqrt(2 × annual demand × ordering cost / annual holding cost per unit)`.

The optimizer maximizes priority-score-weighted order units subject to budget, capacity, suggested-need, MOQ and pack-multiple constraints. It is a procurement allocation model, not a claim of globally optimal supply-chain performance beyond those specified constraints.

## Data and assumption limitations

The included sample dataset is synthetic. Latest observed closing stock is used as the snapshot stock level. The sample does not provide open purchase orders, backorders, supplier MOQ, supplier pack multiples, ordering cost or holding-cost rates. Therefore the YAML file contains clearly labelled scenario defaults for those variables. Replace these with verified procurement/finance data before operational use.

The demand parameters use trailing observed daily demand, not historical Part 5 test predictions as if they were future forecasts. This prevents treating holdout backtest rows as a forward forecast.

## Solver execution status

OR-Tools is included as the preferred optional dependency in `requirements-optimization.txt`. When unavailable, the script uses SciPy MILP and records the actual backend in `inventory_optimization_summary.json`. The build environment used for this package did not have OR-Tools and could not download packages, so the integrated run is validated with SciPy MILP. Install OR-Tools on the target machine and run with `--solver ortools` to verify the CBC backend there.

## Run

```powershell
.\run_part6.ps1
```

or

```powershell
.\.venv\Scripts\python.exe .\scripts\run_inventory_optimization.py --solver auto
```

Outputs are written to `outputs/inventory/`.
