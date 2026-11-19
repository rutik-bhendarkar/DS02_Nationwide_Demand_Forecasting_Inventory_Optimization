from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_inventory.inventory.optimization import optimize_replenishment
from demand_inventory.inventory.policy import build_inventory_policy


@pytest.fixture
def small_sales() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    return pd.DataFrame({
        "date": list(dates) * 2,
        "store_id": ["S1"] * 35 + ["S2"] * 35,
        "sku_id": ["P1"] * 70,
        "units_sold": [10 + (i % 3) for i in range(35)] + [20 + (i % 4) for i in range(35)],
        "closing_stock_units": [15] * 35 + [300] * 35,
        "lead_time_days": [5] * 70,
        "unit_cost": [100.0] * 70,
        "supplier_id": ["SUP1"] * 70,
        "region": ["North"] * 35 + ["South"] * 35,
        "state": ["Delhi"] * 35 + ["Karnataka"] * 35,
        "product_name": ["Test Product"] * 70,
        "category": ["Staples"] * 70,
    })


def test_policy_has_one_row_per_store_sku_and_latest_snapshot(small_sales):
    policy = build_inventory_policy(small_sales, {"lookback_window_days": 28, "review_period_days": 7})
    assert len(policy) == 2
    assert set(policy["snapshot_date"]) == {"2025-02-04"}
    assert set(policy["store_id"]) == {"S1", "S2"}
    assert set(policy["on_hand_units"]) == {15.0, 300.0}


def test_safety_stock_and_reorder_point_formulas(small_sales):
    cfg = {"lookback_window_days": 28, "review_period_days": 7, "z_score": 1.645}
    policy = build_inventory_policy(small_sales, cfg)
    row = policy.loc[policy["store_id"] == "S1"].iloc[0]
    expected_ss = 1.645 * row["std_daily_demand_units"] * np.sqrt(5)
    expected_rop = row["mean_daily_demand_units"] * 5 + expected_ss
    assert row["safety_stock_units"] == pytest.approx(expected_ss)
    assert row["reorder_point_units"] == pytest.approx(expected_rop)


def test_order_qty_respects_reorder_flag_moq_and_pack_multiple(small_sales):
    policy = build_inventory_policy(small_sales, {"default_minimum_order_qty": 12, "default_order_multiple": 6})
    row = policy.loc[policy["store_id"] == "S1"].iloc[0]
    assert bool(row["reorder_required"])
    assert row["suggested_order_qty_units"] >= 12
    assert row["suggested_order_qty_units"] % 6 == 0
    high_stock = policy.loc[policy["store_id"] == "S2"].iloc[0]
    assert not bool(high_stock["reorder_required"])
    assert high_stock["suggested_order_qty_units"] == 0


def test_missing_calendar_dates_are_filled_with_zero_demand(small_sales):
    gap = small_sales.loc[~((small_sales["store_id"] == "S1") & (small_sales["date"] == pd.Timestamp("2025-01-30")))]
    policy = build_inventory_policy(gap, {"lookback_window_days": 28})
    row = policy.loc[policy["store_id"] == "S1"].iloc[0]
    assert row["mean_daily_demand_units"] < small_sales.loc[small_sales["store_id"] == "S1", "units_sold"].tail(28).mean()


def test_policy_rejects_duplicate_business_keys(small_sales):
    duplicated = pd.concat([small_sales, small_sales.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_inventory_policy(duplicated)


def test_policy_rejects_negative_demand(small_sales):
    invalid = small_sales.copy()
    invalid.loc[0, "units_sold"] = -1
    with pytest.raises(ValueError, match="negative"):
        build_inventory_policy(invalid)


def test_scipy_optimizer_respects_budget_capacity_moq_and_multiples():
    policy = pd.DataFrame([
        {"store_id": "S1", "sku_id": "A", "supplier_id": "SUP1", "suggested_order_qty_units": 60,
         "minimum_order_qty_units": 12, "order_multiple_units": 6, "unit_cost_inr": 100.0,
         "priority_score": 3.0, "reorder_required": True},
        {"store_id": "S2", "sku_id": "B", "supplier_id": "SUP2", "suggested_order_qty_units": 48,
         "minimum_order_qty_units": 12, "order_multiple_units": 6, "unit_cost_inr": 80.0,
         "priority_score": 1.0, "reorder_required": True},
    ])
    orders, summary = optimize_replenishment(policy, budget_inr=2_400, capacity_units=30, backend="scipy")
    assert orders["optimized_order_cost_inr"].sum() <= 2_400 + 1e-6
    assert orders["optimized_order_qty_units"].sum() <= 30
    for row in orders.loc[orders["optimization_selected"]].itertuples():
        assert row.optimized_order_qty_units >= row.minimum_order_qty_units
        assert row.optimized_order_qty_units % row.order_multiple_units == 0
    assert summary["solver_backend"] == "scipy"


def test_optimizer_prefers_higher_priority_when_budget_is_limited():
    policy = pd.DataFrame([
        {"store_id": "S1", "sku_id": "A", "supplier_id": "SUP1", "suggested_order_qty_units": 24,
         "minimum_order_qty_units": 6, "order_multiple_units": 6, "unit_cost_inr": 10.0,
         "priority_score": 5.0, "reorder_required": True},
        {"store_id": "S2", "sku_id": "B", "supplier_id": "SUP2", "suggested_order_qty_units": 24,
         "minimum_order_qty_units": 6, "order_multiple_units": 6, "unit_cost_inr": 10.0,
         "priority_score": 1.0, "reorder_required": True},
    ])
    orders, _ = optimize_replenishment(policy, budget_inr=120, capacity_units=12, backend="scipy")
    selected = orders.loc[orders["optimized_order_qty_units"] > 0, "sku_id"].tolist()
    assert selected == ["A"]


def test_optimizer_rejects_nonpositive_budget(small_sales):
    policy = build_inventory_policy(small_sales)
    with pytest.raises(ValueError, match="budget_inr"):
        optimize_replenishment(policy, budget_inr=0, backend="scipy")


def test_optimizer_rejects_invalid_backend(small_sales):
    policy = build_inventory_policy(small_sales)
    with pytest.raises(ValueError, match="backend"):
        optimize_replenishment(policy, backend="unknown")


def test_optimizer_empty_candidates_returns_zero_orders():
    policy = pd.DataFrame([
        {"store_id": "S1", "sku_id": "A", "supplier_id": "SUP1", "suggested_order_qty_units": 0,
         "minimum_order_qty_units": 12, "order_multiple_units": 6, "unit_cost_inr": 100.0,
         "priority_score": 1.0, "reorder_required": False},
    ])
    orders, summary = optimize_replenishment(policy, backend="scipy")
    assert orders["optimized_order_qty_units"].sum() == 0
    assert summary["order_lines_selected"] == 0


def test_eoq_calculation_uses_explicit_ordering_and_holding_costs(small_sales):
    settings = {
        "lookback_window_days": 28,
        "ordering_cost_per_order_inr": 500.0,
        "annual_holding_cost_rate": 0.25,
    }
    policy = build_inventory_policy(small_sales, settings)
    row = policy.loc[policy["store_id"] == "S1"].iloc[0]
    annual_holding = row["unit_cost_inr"] * 0.25
    expected = np.sqrt(2 * row["annual_demand_units"] * 500.0 / annual_holding)
    assert row["eoq_units"] == pytest.approx(expected)


def test_source_supplier_moq_and_pack_multiple_override_defaults(small_sales):
    sales = small_sales.copy()
    sales["minimum_order_qty"] = 30
    sales["order_multiple"] = 10
    policy = build_inventory_policy(
        sales,
        {"default_minimum_order_qty": 12, "default_order_multiple": 6},
    )
    row = policy.loc[policy["store_id"] == "S1"].iloc[0]
    assert row["minimum_order_qty_units"] == 30
    assert row["order_multiple_units"] == 10
    assert row["suggested_order_qty_units"] % 10 == 0
    if row["suggested_order_qty_units"] > 0:
        assert row["suggested_order_qty_units"] >= 30


def test_auto_solver_uses_scipy_fallback_when_ortools_missing(small_sales):
    pytest.importorskip("scipy")
    policy = build_inventory_policy(small_sales)
    orders, summary = optimize_replenishment(policy, backend="auto")
    assert summary["solver_backend"] in {"ortools", "scipy_fallback_ortools_unavailable"}
    assert (orders["optimized_order_qty_units"] >= 0).all()
    assert summary["optimized_spend_inr"] <= summary["budget_inr"] + 1e-6
