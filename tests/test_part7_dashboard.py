from __future__ import annotations

import importlib.util

import pandas as pd
import plotly.graph_objects as go
import pytest

from demand_inventory.dashboard.figures import (
    demand_forecast_figure,
    inventory_risk_figure,
    regional_demand_figure,
    sku_demand_figure,
    stock_position_figure,
    supplier_orders_figure,
)
from demand_inventory.dashboard.service import (
    apply_dashboard_filters,
    get_filter_options,
    load_dashboard_data,
    summarize_kpis,
)


def _load(project_root):
    return load_dashboard_data(project_root)


def test_dashboard_loads_real_project_outputs(project_root):
    data = _load(project_root)
    assert len(data["sales"]) == 38_880
    assert not data["forecasts"].empty
    assert len(data["inventory"]) == 216
    assert data["data_label"] == "SYNTHETIC DEVELOPMENT DATA"


def test_filter_options_are_sorted_and_present(project_root):
    options = get_filter_options(_load(project_root))
    assert options["region"] == sorted(options["region"])
    assert len(options["region"]) == 6
    assert len(options["store_id"]) == 18
    assert len(options["sku_id"]) == 12


def test_global_filters_apply_to_sales_forecasts_and_inventory(project_root):
    data = _load(project_root)
    filtered = apply_dashboard_filters(data, region="North", store_id="ST001", category="Staples", sku_id="SKU001")
    for key in ("sales_filtered", "forecasts_filtered", "inventory_filtered"):
        frame = filtered[key]
        assert set(frame["store_id"].astype(str)) <= {"ST001"}
        if "region" in frame:
            assert set(frame["region"].astype(str)) <= {"North"}
        if "category" in frame:
            assert set(frame["category"].astype(str)) <= {"Staples"}
        if "sku_id" in frame:
            assert set(frame["sku_id"].astype(str)) <= {"SKU001"}


def test_date_filter_limits_demand_and_forecast_but_not_snapshot_inventory(project_root):
    data = _load(project_root)
    filtered = apply_dashboard_filters(data, start_date="2025-06-01", end_date="2025-06-10")
    assert filtered["sales_filtered"]["date"].min() >= pd.Timestamp("2025-06-01")
    assert filtered["sales_filtered"]["date"].max() <= pd.Timestamp("2025-06-10")
    if not filtered["forecasts_filtered"].empty:
        assert filtered["forecasts_filtered"]["date"].min() >= pd.Timestamp("2025-06-01")
        assert filtered["forecasts_filtered"]["date"].max() <= pd.Timestamp("2025-06-10")
    assert len(filtered["inventory_filtered"]) == len(data["inventory"])


def test_kpi_calculations_match_filtered_source(project_root):
    data = _load(project_root)
    filtered = apply_dashboard_filters(data, region="North")
    kpis = summarize_kpis(filtered)
    sales = filtered["sales_filtered"]
    inv = filtered["inventory_filtered"]
    assert kpis["total_units"] == pytest.approx(sales["units_sold"].sum())
    assert kpis["stores"] == sales["store_id"].nunique()
    assert kpis["skus"] == sales["sku_id"].nunique()
    assert kpis["inventory_policies"] == len(inv)
    assert kpis["planned_order_value_inr"] == pytest.approx(inv["optimized_order_cost_inr"].sum())


def test_empty_filter_results_are_handled(project_root):
    filtered = apply_dashboard_filters(_load(project_root), region="Not a real region")
    assert filtered["sales_filtered"].empty
    kpis = summarize_kpis(filtered)
    assert kpis["total_units"] == 0
    assert kpis["stores"] == 0


def test_plotly_figures_are_created_from_filtered_inputs(project_root):
    filtered = apply_dashboard_filters(_load(project_root))
    figures = [
        demand_forecast_figure(filtered["sales_filtered"], filtered["forecasts_filtered"]),
        regional_demand_figure(filtered["sales_filtered"]),
        sku_demand_figure(filtered["sales_filtered"]),
        inventory_risk_figure(filtered["inventory_filtered"]),
        stock_position_figure(filtered["inventory_filtered"]),
        supplier_orders_figure(filtered["inventory_filtered"]),
    ]
    assert all(isinstance(fig, go.Figure) for fig in figures)
    assert all(len(fig.data) > 0 for fig in figures)


def test_dashboard_module_gives_clear_optional_dependency_message(project_root):
    from demand_inventory.dashboard.app import create_app

    if importlib.util.find_spec("dash") is None:
        with pytest.raises(RuntimeError, match="Plotly Dash is not installed"):
            create_app(project_root)
    else:
        app = create_app(project_root)
        assert app.title.startswith("Nationwide Demand")
        assert len(app.callback_map) >= 2
