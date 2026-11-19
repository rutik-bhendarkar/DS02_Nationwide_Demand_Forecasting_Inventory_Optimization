"""Run Part 6 inventory policy and constrained procurement optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_inventory.inventory.optimization import optimize_replenishment  # noqa: E402
from demand_inventory.inventory.policy import (  # noqa: E402
    build_inventory_policy,
    load_inventory_config,
)


def _load_sales(path: Path, constraints_path: Path | None) -> pd.DataFrame:
    sales = pd.read_csv(path, parse_dates=["date"])
    if constraints_path:
        constraints = pd.read_csv(constraints_path)
        required = {"supplier_id", "minimum_order_qty", "order_multiple"}
        missing = sorted(required - set(constraints.columns))
        if missing:
            raise ValueError(f"Supplier constraints file is missing columns: {missing}")
        if constraints["supplier_id"].duplicated().any():
            raise ValueError("Supplier constraints must contain one row per supplier_id")
        sales = sales.drop(columns=["minimum_order_qty", "order_multiple"], errors="ignore")
        sales = sales.merge(constraints, on="supplier_id", how="left", validate="many_to_one")
    return sales


def _write_supplier_summary(orders: pd.DataFrame, path: Path) -> None:
    summary = (
        orders.groupby("supplier_id", dropna=False)
        .agg(
            store_sku_lines=("sku_id", "size"),
            selected_order_lines=("optimization_selected", "sum"),
            ordered_units=("optimized_order_qty_units", "sum"),
            order_value_inr=("optimized_order_cost_inr", "sum"),
            suggested_units=("suggested_order_qty_units", "sum"),
            unfunded_units=("unfunded_suggested_qty_units", "sum"),
        )
        .reset_index()
        .sort_values("order_value_inr", ascending=False)
    )
    summary.to_csv(path, index=False)


def _create_html_report(policy: pd.DataFrame, orders: pd.DataFrame, summary: dict, path: Path) -> None:
    try:
        import plotly.express as px
        import plotly.io as pio
    except ImportError as exc:
        raise RuntimeError("Plotly is required for the Part 6 HTML report. Install requirements-eda.txt") from exc

    selected = orders.loc[orders["optimization_selected"]].copy()
    charts: list[str] = []
    if not policy.empty:
        fig = px.scatter(
            policy,
            x="mean_daily_demand_units",
            y="on_hand_units",
            color="risk_level",
            size="suggested_order_qty_units",
            hover_data=["store_id", "sku_id", "reorder_point_units", "days_of_supply"],
            title="Inventory position versus trailing daily demand",
            labels={"mean_daily_demand_units": "Mean daily demand (units)", "on_hand_units": "Latest observed on-hand (units)"},
        )
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))
    if not selected.empty:
        top = selected.nlargest(20, "optimized_order_cost_inr")
        fig = px.bar(
            top,
            x="optimized_order_cost_inr",
            y="store_id",
            color="region",
            hover_data=["sku_id", "optimized_order_qty_units", "supplier_id"],
            orientation="h",
            title="Top selected replenishment lines by order value",
            labels={"optimized_order_cost_inr": "Order value (INR)", "store_id": "Store"},
        )
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    kpis = [
        ("Store/SKU policies", f"{len(policy):,}"),
        ("Reorder candidates", f"{int(policy['reorder_required'].sum()):,}"),
        ("Selected order lines", f"{summary['order_lines_selected']:,}"),
        ("Order quantity", f"{summary['total_order_units']:,} units"),
        ("Optimized spend", f"₹{summary['optimized_spend_inr']:,.2f}"),
        ("Budget remaining", f"₹{summary['budget_remaining_inr']:,.2f}"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in kpis
    )
    table_cols = [
        "store_id", "sku_id", "supplier_id", "risk_level", "on_hand_units",
        "reorder_point_units", "suggested_order_qty_units", "optimized_order_qty_units",
        "optimized_order_cost_inr", "unfunded_suggested_qty_units",
    ]
    table = orders.loc[orders["optimization_selected"], table_cols].head(50).to_html(index=False, classes="data", float_format=lambda x: f"{x:,.2f}")
    assumption = (
        "Synthetic development data is not real retailer information. Default MOQ, pack multiple, "
        "ordering cost, holding rate, budget, and capacity are scenario assumptions unless overridden "
        "by actual supplier and finance inputs. Demand parameters use trailing observed daily demand, "
        "not a future LightGBM forecast. On-hand is the latest observed closing stock; missing open "
        "PO and backorder data default to configured values."
    )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Part 6 Inventory Optimization</title>"
        "<style>body{font-family:Arial,sans-serif;background:#f5f7fb;color:#182235;margin:28px}"
        ".kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}"
        ".kpi{background:white;padding:16px;border-radius:12px;box-shadow:0 2px 10px #0001}.label{color:#65738a;font-size:12px}"
        ".value{font-size:22px;font-weight:700;margin-top:8px}.panel{background:white;padding:16px;border-radius:12px;margin:16px 0}"
        ".data{border-collapse:collapse;width:100%;font-size:12px}.data th,.data td{padding:8px;border-bottom:1px solid #e5e9f0;text-align:right}"
        ".data th{background:#edf2f8}.note{background:#fff8e6;padding:14px;border-left:4px solid #e0a52b}"
        "</style></head><body><h1>Nationwide Inventory Optimization</h1>"
        "<p>Part 6 decision-support report — review assumptions before operational use.</p>"
        f"<section class='kpis'>{kpi_html}</section>"
        + "".join(f"<section class='panel'>{chart}</section>" for chart in charts)
        + f"<section class='panel'><h2>Selected purchase order recommendations</h2>{table}</section>"
        + f"<section class='note'><b>Model and scenario limitations:</b> {assumption}</section>"
        + f"<section class='panel'><h2>Solver metadata</h2><pre>{json.dumps(summary, indent=2)}</pre></section>"
        + "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Part 6 inventory policies and optimize orders.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/sample/nationwide_retail_daily_sales_synthetic.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config/inventory_policy.yaml")
    parser.add_argument("--supplier-constraints", type=Path, default=None)
    parser.add_argument("--budget-inr", type=float, default=None)
    parser.add_argument("--capacity-units", type=int, default=None)
    parser.add_argument("--solver", choices=["auto", "ortools", "scipy"], default="auto")
    args = parser.parse_args()

    if not args.input.is_absolute():
        args.input = ROOT / args.input
    if not args.config.is_absolute():
        args.config = ROOT / args.config
    if args.supplier_constraints is not None and not args.supplier_constraints.is_absolute():
        args.supplier_constraints = ROOT / args.supplier_constraints
    settings = load_inventory_config(args.config)
    sales = _load_sales(args.input, args.supplier_constraints)
    policy = build_inventory_policy(sales, settings)
    budget = float(args.budget_inr if args.budget_inr is not None else settings.get("budget_inr", 500_000))
    capacity = int(args.capacity_units if args.capacity_units is not None else settings.get("capacity_units", 5_000))
    orders, summary = optimize_replenishment(policy, budget, capacity, args.solver)

    out = ROOT / "outputs/inventory"
    out.mkdir(parents=True, exist_ok=True)
    policy.to_csv(out / "inventory_policy.csv", index=False)
    orders.to_csv(out / "optimized_purchase_orders.csv", index=False)
    _write_supplier_summary(orders, out / "supplier_summary.csv")
    summary.update({
        "source_data": str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else args.input.name,
        "data_label": "SYNTHETIC_TEST_DATA_NOT_REAL_BUSINESS_DATA" if "synthetic" in args.input.name.lower() else "USER_SUPPLIED_DATA",
        "policy_rows": int(len(policy)),
        "reorder_candidates": int(policy["reorder_required"].sum()),
        "reorder_policy_method": "periodic_review_order_up_to_level",
        "safety_stock_formula": "z * daily_demand_std * sqrt(lead_time_days)",
        "reorder_point_formula": "mean_daily_demand * lead_time_days + safety_stock",
        "order_up_to_formula": "mean_daily_demand * (lead_time_days + review_period_days) + z * std_daily_demand * sqrt(lead_time_days + review_period_days)",
        "eoq_formula": "sqrt(2 * annual_demand * ordering_cost / annual_holding_cost_per_unit)",
        "scenario_assumptions": settings,
        "operational_warning": "Validate on-hand, open POs, supplier MOQ/pack constraints, ordering cost, holding rate, budget and capacity against real systems before operational use.",
    })
    (out / "inventory_optimization_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _create_html_report(policy, orders, summary, out / "inventory_optimization_report.html")
    print("Part 6 inventory optimization completed")
    print(json.dumps({k: summary[k] for k in ["solver_backend", "solver_status", "policy_rows", "reorder_candidates", "order_lines_selected", "total_order_units", "optimized_spend_inr", "budget_remaining_inr", "unfunded_suggested_units"]}, indent=2))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
