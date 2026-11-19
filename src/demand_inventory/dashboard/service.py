"""Data loading, filtering and KPI logic for the Part 7 dashboard.

This module deliberately has no Dash dependency, which makes the business
logic independently testable and lets users validate their data before
installing the optional web dashboard stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FILES = {
    "sales": Path("data/sample/nationwide_retail_daily_sales_synthetic.csv"),
    "forecasts": Path("outputs/forecasts/test_lightgbm_predictions.csv"),
    "inventory": Path("outputs/inventory/optimized_purchase_orders.csv"),
    "inventory_summary": Path("outputs/inventory/inventory_optimization_summary.json"),
    "supplier_summary": Path("outputs/inventory/supplier_summary.csv"),
}


def _read_csv(path: Path, *, parse_dates: list[str] | None = None, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required dashboard input is missing: {path}. Run the relevant DS-02 pipeline first."
            )
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates or [])


def load_dashboard_data(
    project_root: str | Path,
    *,
    sales_path: str | Path | None = None,
    forecasts_path: str | Path | None = None,
    inventory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load dashboard inputs from generated project outputs.

    The date filter applies to the sales and forecast views. Inventory is a
    point-in-time snapshot and is filtered by geography/store/product only.
    """
    root = Path(project_root).resolve()
    paths = {
        "sales": Path(sales_path) if sales_path else root / DEFAULT_FILES["sales"],
        "forecasts": Path(forecasts_path) if forecasts_path else root / DEFAULT_FILES["forecasts"],
        "inventory": Path(inventory_path) if inventory_path else root / DEFAULT_FILES["inventory"],
    }
    paths = {key: value if value.is_absolute() else root / value for key, value in paths.items()}

    sales = _read_csv(paths["sales"], parse_dates=["date"])
    forecasts = _read_csv(paths["forecasts"], parse_dates=["date"], required=False)
    inventory = _read_csv(paths["inventory"], parse_dates=["snapshot_date"], required=False)

    sales_required = {"date", "store_id", "sku_id", "units_sold"}
    if missing := sorted(sales_required - set(sales.columns)):
        raise ValueError(f"Sales dashboard input is missing columns: {missing}")
    sales["units_sold"] = pd.to_numeric(sales["units_sold"], errors="coerce")
    if sales["units_sold"].isna().any() or (sales["units_sold"] < 0).any():
        raise ValueError("Sales dashboard input has invalid units_sold values")

    if not forecasts.empty:
        forecast_required = {"date", "store_id", "sku_id", "actual_units_sold", "forecast_units_sold"}
        if missing := sorted(forecast_required - set(forecasts.columns)):
            raise ValueError(f"Forecast output is missing columns: {missing}")
        forecasts["actual_units_sold"] = pd.to_numeric(forecasts["actual_units_sold"], errors="coerce")
        forecasts["forecast_units_sold"] = pd.to_numeric(forecasts["forecast_units_sold"], errors="coerce")
        forecasts = forecasts.dropna(subset=["date", "actual_units_sold", "forecast_units_sold"])

    if not inventory.empty:
        inventory_required = {"store_id", "sku_id", "risk_level", "on_hand_units", "reorder_point_units"}
        if missing := sorted(inventory_required - set(inventory.columns)):
            raise ValueError(f"Inventory output is missing columns: {missing}")

    summary_path = root / DEFAULT_FILES["inventory_summary"]
    inventory_summary: dict[str, Any] = {}
    if summary_path.exists():
        inventory_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    supplier_summary = _read_csv(root / DEFAULT_FILES["supplier_summary"], required=False)

    synthetic = "synthetic" in paths["sales"].name.lower()
    return {
        "sales": sales,
        "forecasts": forecasts,
        "inventory": inventory,
        "inventory_summary": inventory_summary,
        "supplier_summary": supplier_summary,
        "data_label": "SYNTHETIC DEVELOPMENT DATA" if synthetic else "USER-SUPPLIED DATA",
        "source_file": paths["sales"].name,
        "project_root": root,
    }


def _filter_dimension(frame: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    for column, value in filters.items():
        if value not in (None, "") and column in result.columns:
            result = result.loc[result[column].astype(str) == str(value)]
    return result


def apply_dashboard_filters(
    data: dict[str, Any],
    *,
    region: str | None = None,
    state: str | None = None,
    store_id: str | None = None,
    category: str | None = None,
    sku_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Apply consistent geographic/product filters across dashboard views."""
    dimensions = {
        "region": region,
        "state": state,
        "store_id": store_id,
        "category": category,
        "sku_id": sku_id,
    }
    sales = _filter_dimension(data["sales"], dimensions)
    forecasts = _filter_dimension(data["forecasts"], dimensions)
    inventory = _filter_dimension(data["inventory"], dimensions)
    supplier_summary = data.get("supplier_summary", pd.DataFrame()).copy()

    if start_date and not sales.empty:
        sales = sales.loc[sales["date"] >= pd.Timestamp(start_date)]
        if not forecasts.empty:
            forecasts = forecasts.loc[forecasts["date"] >= pd.Timestamp(start_date)]
    if end_date and not sales.empty:
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        sales = sales.loc[sales["date"] <= end]
        if not forecasts.empty:
            forecasts = forecasts.loc[forecasts["date"] <= end]

    if not inventory.empty:
        # Supplier-level summary is recomputed from the filtered inventory
        # recommendations so its values follow the active dashboard filters.
        order_qty = pd.to_numeric(inventory.get("optimized_order_qty_units", 0), errors="coerce")
        if not isinstance(order_qty, pd.Series):
            order_qty = pd.Series(order_qty, index=inventory.index)
        cost = pd.to_numeric(inventory.get("optimized_order_cost_inr", 0), errors="coerce")
        if not isinstance(cost, pd.Series):
            cost = pd.Series(cost, index=inventory.index)
        inventory = inventory.copy()
        inventory["_order_qty"] = order_qty.fillna(0)
        inventory["_order_cost"] = cost.fillna(0)
        supplier_summary = (
            inventory.groupby("supplier_id", dropna=False)
            .agg(order_lines=("_order_qty", lambda x: int((x > 0).sum())),
                 ordered_units=("_order_qty", "sum"), order_value_inr=("_order_cost", "sum"))
            .reset_index()
            .sort_values("order_value_inr", ascending=False)
        )

    return {
        **data,
        "sales_filtered": sales,
        "forecasts_filtered": forecasts,
        "inventory_filtered": inventory,
        "supplier_summary_filtered": supplier_summary,
    }


def get_filter_options(data: dict[str, Any]) -> dict[str, list[str]]:
    sales = data["sales"]
    inventory = data["inventory"]
    columns = {
        "region": "region",
        "state": "state",
        "store_id": "store_id",
        "category": "category",
        "sku_id": "sku_id",
    }
    options: dict[str, list[str]] = {}
    for key, col in columns.items():
        values: set[str] = set()
        for frame in (sales, inventory):
            if col in frame.columns:
                values.update(frame[col].dropna().astype(str).unique().tolist())
        options[key] = sorted(values)
    return options


def summarize_kpis(filtered: dict[str, Any]) -> dict[str, float | int]:
    sales = filtered["sales_filtered"]
    forecast = filtered["forecasts_filtered"]
    inventory = filtered["inventory_filtered"]
    total_units = float(sales["units_sold"].sum()) if not sales.empty else 0.0
    n_dates = int(sales["date"].nunique()) if not sales.empty else 0
    reorder_count = int(inventory.get("reorder_required", pd.Series(dtype=bool)).astype(bool).sum()) if not inventory.empty else 0
    order_qty = pd.to_numeric(inventory.get("optimized_order_qty_units", pd.Series(dtype=float)), errors="coerce").fillna(0)
    order_cost = pd.to_numeric(inventory.get("optimized_order_cost_inr", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return {
        "total_units": total_units,
        "average_daily_units": total_units / n_dates if n_dates else 0.0,
        "stores": int(sales["store_id"].nunique()) if not sales.empty else 0,
        "skus": int(sales["sku_id"].nunique()) if not sales.empty else 0,
        "forecast_rows": int(len(forecast)),
        "forecast_wape_pct": _wape(forecast["actual_units_sold"], forecast["forecast_units_sold"]) if not forecast.empty else float("nan"),
        "inventory_policies": int(len(inventory)),
        "reorder_candidates": reorder_count,
        "selected_order_lines": int((order_qty > 0).sum()),
        "ordered_units": float(order_qty.sum()),
        "planned_order_value_inr": float(order_cost.sum()),
        "on_hand_units": float(pd.to_numeric(inventory.get("on_hand_units", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
    }


def _wape(actual: pd.Series, predicted: pd.Series) -> float:
    denominator = float(pd.to_numeric(actual, errors="coerce").abs().sum())
    if denominator == 0:
        return float("nan")
    return float((pd.to_numeric(actual, errors="coerce") - pd.to_numeric(predicted, errors="coerce")).abs().sum() / denominator * 100)
