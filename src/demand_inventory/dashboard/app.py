"""Part 7 interactive Plotly Dash application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from demand_inventory.dashboard.figures import (
    demand_forecast_figure,
    inventory_risk_figure,
    regional_demand_figure,
    sku_demand_figure,
    stock_position_figure,
    supplier_orders_figure,
)
from demand_inventory.dashboard.service import apply_dashboard_filters, get_filter_options, load_dashboard_data, summarize_kpis

ROOT = Path(__file__).resolve().parents[3]


def _money(value: float) -> str:
    return f"₹{value:,.0f}"


def _kpi_cards(kpis: dict[str, Any], data_label: str):
    from dash import html

    items = [
        ("Observed demand", f"{kpis['total_units']:,.0f}", "units in selected period"),
        ("Average daily demand", f"{kpis['average_daily_units']:,.1f}", "units per calendar day"),
        ("Stores / SKUs", f"{kpis['stores']} / {kpis['skus']}", "distinct entities in scope"),
        ("Test forecast WAPE", f"{kpis['forecast_wape_pct']:.2f}%" if pd.notna(kpis["forecast_wape_pct"]) else "Not available", "held-out LightGBM predictions"),
        ("Reorder candidates", f"{kpis['reorder_candidates']:,}", "store-SKU inventory policies"),
        ("Selected order lines", f"{kpis['selected_order_lines']:,}", f"{kpis['ordered_units']:,.0f} units planned"),
        ("Planned order value", _money(kpis["planned_order_value_inr"]), "scenario recommendation, not actual spend"),
        ("On-hand stock", f"{kpis['on_hand_units']:,.0f}", "latest inventory snapshot"),
    ]
    return [
        html.Div([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"), html.Div(note, className="kpi-note")], className="kpi-card")
        for label, value, note in items
    ]


def _recommendation_table(inventory: pd.DataFrame):
    from dash import html

    columns = [
        ("Region", "region"), ("Store", "store_id"), ("SKU", "sku_id"),
        ("Product", "product_name"), ("Risk", "risk_level"),
        ("On hand", "on_hand_units"), ("Reorder point", "reorder_point_units"),
        ("Suggested qty", "suggested_order_qty_units"), ("Optimized qty", "optimized_order_qty_units"),
        ("Order value", "optimized_order_cost_inr"),
    ]
    if inventory.empty:
        return html.Div("No inventory records match the selected filters. Run Part 6 to generate inventory recommendations.", className="empty-state")
    selected = inventory.loc[pd.to_numeric(inventory["optimized_order_qty_units"], errors="coerce").fillna(0) > 0].copy()
    if selected.empty:
        return html.Div("No purchase orders were selected under the current optimization constraints.", className="empty-state")
    selected = selected.sort_values(["priority_score", "optimized_order_cost_inr"], ascending=False).head(30)
    head = html.Thead(html.Tr([html.Th(label) for label, _ in columns]))
    rows = []
    for _, row in selected.iterrows():
        cells = []
        for label, col in columns:
            value = row.get(col, "—")
            if pd.isna(value):
                value = "—"
            elif col == "optimized_order_cost_inr":
                value = _money(float(value))
            elif col in {"on_hand_units", "reorder_point_units", "suggested_order_qty_units", "optimized_order_qty_units"}:
                value = f"{float(value):,.0f}"
            cells.append(html.Td(str(value)))
        rows.append(html.Tr(cells))
    return html.Div(html.Table([head, html.Tbody(rows)], className="recommendation-table"), className="table-scroll")


def create_app(
    project_root: str | Path = ROOT,
    *,
    sales_path: str | Path | None = None,
    forecasts_path: str | Path | None = None,
    inventory_path: str | Path | None = None,
):
    """Create the Dash app. Import Dash lazily so core data tests need no Dash install."""
    try:
        from dash import Dash, Input, Output, dcc, html
    except ImportError as exc:
        raise RuntimeError(
            "Plotly Dash is not installed. From the project root run: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements-dashboard.txt"
        ) from exc

    root = Path(project_root).resolve()
    data = load_dashboard_data(
        root, sales_path=sales_path, forecasts_path=forecasts_path, inventory_path=inventory_path
    )
    options = get_filter_options(data)
    sales = data["sales"]
    min_date = sales["date"].min().date().isoformat()
    max_date = sales["date"].max().date().isoformat()
    data_label = data["data_label"]

    def dropdown(identifier: str, label: str):
        return html.Div([
            html.Label(label, htmlFor=identifier),
            dcc.Dropdown(id=identifier, options=[{"label": v, "value": v} for v in options[identifier]],
                         value=None, placeholder=f"All {label.lower()}", clearable=True, searchable=True),
        ], className="filter-control")

    app = Dash(
        __name__,
        title="Nationwide Demand & Inventory Intelligence | DS-02",
        assets_folder=str(root / "assets"),
        suppress_callback_exceptions=False,
        update_title="Updating dashboard…",
    )

    app.layout = html.Div([
        html.Header([
            html.Div([
                html.Div("DS-02  /  NEXVIA TECHNOLOGIES", className="eyebrow"),
                html.H1("Nationwide Demand & Inventory Intelligence"),
                html.P("Demand visibility, forecast evaluation and replenishment decision support", className="subtitle"),
            ]),
            html.Div([html.Span("●", className="status-dot"), html.Span("DEVELOPMENT DATA" if "SYNTHETIC" in data_label else "USER DATA", className="data-pill")], className="header-status"),
        ], className="topbar"),
        html.Div([
            html.Aside([
                html.Div("GLOBAL FILTERS", className="section-eyebrow"),
                dropdown("region", "Region"),
                dropdown("state", "State"),
                dropdown("store_id", "Store"),
                dropdown("category", "Category"),
                dropdown("sku_id", "SKU"),
                html.Div([html.Label("Demand date range"), dcc.DatePickerRange(
                    id="date-range", min_date_allowed=min_date, max_date_allowed=max_date,
                    start_date=min_date, end_date=max_date, display_format="DD MMM YYYY",
                    minimum_nights=0, className="date-range")], className="filter-control"),
                html.Button("Reset filters", id="reset-filters", n_clicks=0, className="reset-button"),
                html.Div([html.Div("DATA PROVENANCE", className="section-eyebrow"),
                          html.P(f"Source: {data['source_file']}", className="source-text"),
                          html.P("Inventory metrics reflect the latest stock snapshot. Demand-date filters do not rewrite that snapshot.", className="source-text")], className="provenance-card"),
            ], className="sidebar"),
            html.Main([
                html.Div([html.Div("EXECUTIVE OVERVIEW", className="section-eyebrow"), html.Div(id="active-filter-note", className="filter-note")], className="content-heading"),
                html.Div(id="kpi-grid", className="kpi-grid"),
                html.Div([html.Section([dcc.Graph(id="demand-forecast-chart", config={"displayModeBar": True, "responsive": True})], className="panel panel-wide"),
                          html.Section([dcc.Graph(id="regional-chart", config={"displayModeBar": False, "responsive": True})], className="panel"),
                          html.Section([dcc.Graph(id="sku-chart", config={"displayModeBar": False, "responsive": True})], className="panel")], className="chart-grid chart-grid-main"),
                html.Div([html.Section([dcc.Graph(id="inventory-risk-chart", config={"displayModeBar": False, "responsive": True})], className="panel"),
                          html.Section([dcc.Graph(id="stock-position-chart", config={"displayModeBar": True, "responsive": True})], className="panel panel-wide")], className="chart-grid"),
                html.Div([html.Section([dcc.Graph(id="supplier-chart", config={"displayModeBar": False, "responsive": True})], className="panel panel-wide"),
                          html.Section([html.Div([html.H3("Replenishment recommendations"), html.Span("TOP 30 SELECTED LINES", className="table-tag")], className="panel-heading"), html.Div(id="recommendations-table")], className="panel table-panel")], className="chart-grid chart-grid-lower"),
                html.Div([html.Span(f"Data label: {data_label}"), html.Span("Forecast metrics refer only to saved held-out test predictions."), html.Span("Inventory optimizer results are scenario recommendations; validate assumptions before procurement.")], className="disclaimer-bar"),
            ], className="main-content"),
        ], className="dashboard-shell"),
        html.Footer("DS-02 · Nationwide Demand Forecasting & Inventory Optimization · Nexvia Technologies Internship", className="footer"),
    ], className="app-root")

    @app.callback(
        Output("region", "value"), Output("state", "value"), Output("store_id", "value"),
        Output("category", "value"), Output("sku_id", "value"),
        Output("date-range", "start_date"), Output("date-range", "end_date"),
        Input("reset-filters", "n_clicks"), prevent_initial_call=True,
    )
    def reset_filters(_clicks: int):
        return None, None, None, None, None, min_date, max_date

    @app.callback(
        Output("kpi-grid", "children"),
        Output("demand-forecast-chart", "figure"),
        Output("regional-chart", "figure"),
        Output("sku-chart", "figure"),
        Output("inventory-risk-chart", "figure"),
        Output("stock-position-chart", "figure"),
        Output("supplier-chart", "figure"),
        Output("recommendations-table", "children"),
        Output("active-filter-note", "children"),
        Input("region", "value"), Input("state", "value"), Input("store_id", "value"),
        Input("category", "value"), Input("sku_id", "value"),
        Input("date-range", "start_date"), Input("date-range", "end_date"),
    )
    def update_dashboard(region, state, store_id, category, sku_id, start_date, end_date):
        filtered = apply_dashboard_filters(
            data, region=region, state=state, store_id=store_id, category=category,
            sku_id=sku_id, start_date=start_date, end_date=end_date,
        )
        kpis = summarize_kpis(filtered)
        active = [f"{name}: {value}" for name, value in {
            "Region": region, "State": state, "Store": store_id,
            "Category": category, "SKU": sku_id,
        }.items() if value]
        note = " · ".join(active) if active else "All regions, stores and products"
        note += f" · {start_date or min_date} to {end_date or max_date}"
        return (
            _kpi_cards(kpis, data_label),
            demand_forecast_figure(filtered["sales_filtered"], filtered["forecasts_filtered"]),
            regional_demand_figure(filtered["sales_filtered"]),
            sku_demand_figure(filtered["sales_filtered"]),
            inventory_risk_figure(filtered["inventory_filtered"]),
            stock_position_figure(filtered["inventory_filtered"]),
            supplier_orders_figure(filtered["inventory_filtered"]),
            _recommendation_table(filtered["inventory_filtered"]),
            note,
        )

    return app
