"""Transparent inventory policy calculations from observed daily sales.

The module estimates demand from a trailing historical window. It does not
pretend that historical backtest predictions are future forecasts. All default
commercial constraints (MOQ, order multiple, order cost, holding rate) are
scenario assumptions unless supplied by the user's own procurement data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

KEYS = ["store_id", "sku_id"]
REQUIRED_COLUMNS = {
    "date", "store_id", "sku_id", "units_sold", "closing_stock_units",
    "lead_time_days", "unit_cost", "supplier_id",
}


@dataclass(frozen=True)
class InventoryConfig:
    lookback_window_days: int = 28
    review_period_days: int = 7
    z_score: float = 1.645
    service_level: float = 0.95
    default_open_purchase_orders_units: float = 0.0
    default_backorders_units: float = 0.0
    ordering_cost_per_order_inr: float = 500.0
    annual_holding_cost_rate: float = 0.25
    default_minimum_order_qty: int = 12
    default_order_multiple: int = 6
    budget_inr: float = 500_000.0
    capacity_units: int = 5_000


def load_inventory_config(path: str | Path) -> dict[str, Any]:
    """Load inventory scenario settings from YAML and validate key values."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    settings = config.get("inventory", config)
    required_positive = [
        "lookback_window_days", "review_period_days", "z_score",
        "ordering_cost_per_order_inr", "annual_holding_cost_rate",
        "default_minimum_order_qty", "default_order_multiple",
        "budget_inr", "capacity_units",
    ]
    for key in required_positive:
        if key in settings and float(settings[key]) <= 0:
            raise ValueError(f"inventory.{key} must be greater than zero")
    service_level = float(settings.get("service_level", 0.95))
    if not 0 < service_level < 1:
        raise ValueError("inventory.service_level must be between 0 and 1")
    return settings


def _validate_sales_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Sales data is missing required inventory columns: {missing}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("Sales data contains invalid dates")
    for column in ["units_sold", "closing_stock_units", "lead_time_days", "unit_cost"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any() or not np.isfinite(data[column]).all():
            raise ValueError(f"Sales data contains invalid numeric values in {column}")
    if (data["units_sold"] < 0).any():
        raise ValueError("units_sold cannot contain negative values")
    if (data["closing_stock_units"] < 0).any():
        raise ValueError("closing_stock_units cannot contain negative values")
    if (data["lead_time_days"] <= 0).any():
        raise ValueError("lead_time_days must be greater than zero")
    if (data["unit_cost"] <= 0).any():
        raise ValueError("unit_cost must be greater than zero")
    if data.duplicated(["date", *KEYS]).any():
        raise ValueError("Sales data contains duplicate date/store/SKU records")
    return data.sort_values([*KEYS, "date"]).reset_index(drop=True)


def _round_up_multiple(value: float, multiple: int) -> int:
    if value <= 0:
        return 0
    return int(np.ceil(value / multiple) * multiple)


def build_inventory_policy(
    sales: pd.DataFrame,
    config: dict[str, Any] | InventoryConfig | None = None,
) -> pd.DataFrame:
    """Create one inventory policy row per store/SKU.

    Demand statistics use a trailing calendar window ending on each series'
    latest observation. Missing dates within that window are treated as zero
    demand, assuming the source is intended to be a complete daily panel.
    Inventory position includes open POs/backorders only when the source has
    those columns; otherwise explicit YAML defaults are used.
    """
    data = _validate_sales_frame(sales)
    if isinstance(config, InventoryConfig):
        settings = config.__dict__.copy()
    else:
        settings = dict(config or {})

    cfg = InventoryConfig(**{
        field: settings.get(field, getattr(InventoryConfig(), field))
        for field in InventoryConfig.__dataclass_fields__
    })
    if cfg.lookback_window_days <= 0 or cfg.review_period_days <= 0:
        raise ValueError("Lookback and review period must be positive")
    if cfg.z_score <= 0 or not 0 < cfg.service_level < 1:
        raise ValueError("z_score must be positive and service_level must be between 0 and 1")
    if cfg.annual_holding_cost_rate <= 0 or cfg.ordering_cost_per_order_inr <= 0:
        raise ValueError("Ordering cost and annual holding rate must be positive")
    if cfg.default_minimum_order_qty <= 0 or cfg.default_order_multiple <= 0:
        raise ValueError("MOQ and order multiple must be positive")

    rows: list[dict[str, Any]] = []
    has_open_po = "open_purchase_orders_units" in data.columns
    has_backorders = "backorders_units" in data.columns
    has_moq = "minimum_order_qty" in data.columns
    has_multiple = "order_multiple" in data.columns

    for (store_id, sku_id), group in data.groupby(KEYS, sort=True, observed=True):
        group = group.sort_values("date")
        latest = group.iloc[-1]
        snapshot_date = pd.Timestamp(latest["date"]).normalize()
        start_date = snapshot_date - pd.Timedelta(days=cfg.lookback_window_days - 1)
        history = group.loc[group["date"].between(start_date, snapshot_date)]
        demand_series = history.groupby(history["date"].dt.normalize())["units_sold"].sum()
        calendar = pd.date_range(start_date, snapshot_date, freq="D")
        demand = demand_series.reindex(calendar, fill_value=0.0).astype(float)

        mean_daily = float(demand.mean())
        std_daily = float(demand.std(ddof=1)) if len(demand) > 1 else 0.0
        std_daily = 0.0 if not np.isfinite(std_daily) else std_daily
        lead_time = max(1, int(np.ceil(float(latest["lead_time_days"]))))
        unit_cost = float(latest["unit_cost"])
        on_hand = float(latest["closing_stock_units"])
        open_po = float(latest["open_purchase_orders_units"]) if has_open_po else cfg.default_open_purchase_orders_units
        backorders = float(latest["backorders_units"]) if has_backorders else cfg.default_backorders_units
        if open_po < 0 or backorders < 0:
            raise ValueError("Open purchase orders and backorders cannot be negative")
        inventory_position = max(0.0, on_hand + open_po - backorders)

        safety_stock = cfg.z_score * std_daily * np.sqrt(lead_time)
        lead_time_demand = mean_daily * lead_time
        reorder_point = lead_time_demand + safety_stock
        protection_period = lead_time + cfg.review_period_days
        order_up_to = (
            mean_daily * protection_period
            + cfg.z_score * std_daily * np.sqrt(protection_period)
        )
        days_of_supply = inventory_position / mean_daily if mean_daily > 0 else (float("inf") if inventory_position > 0 else 0.0)
        reorder_required = bool(inventory_position <= reorder_point)
        moq = int(latest["minimum_order_qty"]) if has_moq and pd.notna(latest["minimum_order_qty"]) else cfg.default_minimum_order_qty
        multiple = int(latest["order_multiple"]) if has_multiple and pd.notna(latest["order_multiple"]) else cfg.default_order_multiple
        if moq <= 0 or multiple <= 0:
            raise ValueError("minimum_order_qty and order_multiple must be positive")
        raw_need = max(0.0, order_up_to - inventory_position) if reorder_required else 0.0
        suggested_order = _round_up_multiple(raw_need, multiple)
        if reorder_required and suggested_order > 0:
            suggested_order = max(suggested_order, _round_up_multiple(moq, multiple))

        annual_demand = mean_daily * 365.0
        annual_holding_cost_per_unit = unit_cost * cfg.annual_holding_cost_rate
        eoq = float(np.sqrt((2.0 * annual_demand * cfg.ordering_cost_per_order_inr) / annual_holding_cost_per_unit)) if annual_demand > 0 else 0.0
        days_cover_gap = max(0.0, lead_time - (days_of_supply if np.isfinite(days_of_supply) else lead_time * 2)) / lead_time
        reorder_gap = max(0.0, reorder_point - inventory_position) / max(reorder_point, 1.0)
        urgency = 1.0 + 2.0 * min(1.0, days_cover_gap) + min(1.0, reorder_gap)
        risk = "Critical" if days_of_supply < lead_time else "High" if reorder_required else "Normal"

        rows.append({
            "snapshot_date": snapshot_date.date().isoformat(),
            "store_id": store_id,
            "sku_id": sku_id,
            "product_name": latest.get("product_name", sku_id),
            "category": latest.get("category", "Unknown"),
            "region": latest.get("region", "Unknown"),
            "state": latest.get("state", "Unknown"),
            "supplier_id": latest.get("supplier_id", "Unknown"),
            "demand_method": f"trailing_{cfg.lookback_window_days}d_observed_mean_std",
            "lookback_window_days": cfg.lookback_window_days,
            "mean_daily_demand_units": mean_daily,
            "std_daily_demand_units": std_daily,
            "lead_time_days": lead_time,
            "review_period_days": cfg.review_period_days,
            "service_level": cfg.service_level,
            "z_score": cfg.z_score,
            "safety_stock_units": safety_stock,
            "lead_time_demand_units": lead_time_demand,
            "reorder_point_units": reorder_point,
            "order_up_to_level_units": order_up_to,
            "on_hand_units": on_hand,
            "open_purchase_orders_units": open_po,
            "backorders_units": backorders,
            "inventory_position_units": inventory_position,
            "days_of_supply": days_of_supply,
            "unit_cost_inr": unit_cost,
            "annual_demand_units": annual_demand,
            "eoq_units": eoq,
            "minimum_order_qty_units": moq,
            "order_multiple_units": multiple,
            "reorder_required": reorder_required,
            "risk_level": risk,
            "raw_order_need_units": raw_need,
            "suggested_order_qty_units": suggested_order,
            "suggested_order_cost_inr": suggested_order * unit_cost,
            "priority_score": urgency,
            "assumptions_used": "MOQ/order multiple and cost parameters are configurable scenario defaults unless provided by source data",
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No store/SKU inventory policies could be generated")
    return result.sort_values(["risk_level", "region", "store_id", "sku_id"]).reset_index(drop=True)
