"""Plotly figures for Part 7. Figures only reflect supplied filtered data."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#4F8EF7", "#14B8A6", "#A78BFA", "#F59E0B", "#F87171", "#38BDF8"]
LAYOUT = {
    "template": "plotly_white",
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#27364B"},
    "margin": {"l": 42, "r": 22, "t": 52, "b": 42},
    "legend": {"orientation": "h", "y": 1.12, "x": 0},
}


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font={"size": 14, "color": "#718096"})
    fig.update_layout(**LAYOUT, xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def demand_forecast_figure(sales: pd.DataFrame, forecasts: pd.DataFrame) -> go.Figure:
    if sales.empty:
        return empty_figure("No demand observations match these filters.")
    daily = sales.groupby("date", as_index=False)["units_sold"].sum().sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["units_sold"], mode="lines", name="Observed demand",
                             line={"color": PALETTE[0], "width": 2.2}))
    if not forecasts.empty:
        pred = forecasts.groupby("date", as_index=False).agg(
            actual_units_sold=("actual_units_sold", "sum"),
            forecast_units_sold=("forecast_units_sold", "sum"),
        ).sort_values("date")
        fig.add_trace(go.Scatter(x=pred["date"], y=pred["forecast_units_sold"], mode="lines+markers",
                                 name="LightGBM test forecast", line={"color": PALETTE[1], "width": 2.5, "dash": "dash"},
                                 marker={"size": 5}))
    fig.update_layout(title="Demand trend and held-out forecast comparison", xaxis_title="Date",
                      yaxis_title="Units sold", hovermode="x unified", **LAYOUT)
    return fig


def regional_demand_figure(sales: pd.DataFrame) -> go.Figure:
    if sales.empty or "region" not in sales:
        return empty_figure("Regional demand is unavailable for this filter.")
    summary = sales.groupby("region", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=True)
    fig = px.bar(summary, x="units_sold", y="region", orientation="h", text_auto=".2s",
                 color="region", color_discrete_sequence=PALETTE, title="Demand by region")
    fig.update_layout(showlegend=False, xaxis_title="Units sold", yaxis_title="", **LAYOUT)
    return fig


def sku_demand_figure(sales: pd.DataFrame) -> go.Figure:
    if sales.empty:
        return empty_figure("No SKU demand matches these filters.")
    label = "product_name" if "product_name" in sales.columns else "sku_id"
    summary = sales.groupby(["sku_id", label], as_index=False)["units_sold"].sum().nlargest(12, "units_sold")
    fig = px.bar(summary.sort_values("units_sold"), x="units_sold", y=label, orientation="h",
                 color="units_sold", color_continuous_scale="Blues", title="Top products by observed demand",
                 hover_data=["sku_id"])
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Units sold", yaxis_title="", **LAYOUT)
    return fig


def inventory_risk_figure(inventory: pd.DataFrame) -> go.Figure:
    if inventory.empty or "risk_level" not in inventory:
        return empty_figure("Inventory policy output is not available. Run Part 6 first.")
    summary = inventory.groupby("risk_level", as_index=False).size().rename(columns={"size": "store_sku_count"})
    fig = px.pie(summary, names="risk_level", values="store_sku_count", hole=0.62,
                 color="risk_level", color_discrete_map={"Critical": "#DC4C64", "High": "#F59E0B", "Medium": "#4F8EF7", "Low": "#14B8A6"},
                 title="Store-SKU inventory risk")
    fig.update_layout(**LAYOUT)
    return fig


def stock_position_figure(inventory: pd.DataFrame) -> go.Figure:
    required = {"on_hand_units", "reorder_point_units", "risk_level"}
    if inventory.empty or not required.issubset(inventory.columns):
        return empty_figure("Inventory position data is unavailable.")
    hover = [c for c in ["store_id", "sku_id", "product_name", "days_of_supply", "suggested_order_qty_units"] if c in inventory]
    fig = px.scatter(inventory, x="reorder_point_units", y="on_hand_units", color="risk_level",
                     size="suggested_order_qty_units" if "suggested_order_qty_units" in inventory else None,
                     hover_data=hover, color_discrete_map={"Critical": "#DC4C64", "High": "#F59E0B", "Medium": "#4F8EF7", "Low": "#14B8A6"},
                     title="On-hand stock versus reorder point")
    upper = max(float(inventory["on_hand_units"].max()), float(inventory["reorder_point_units"].max()), 1)
    fig.add_trace(go.Scatter(x=[0, upper], y=[0, upper], mode="lines", name="At reorder point",
                             line={"color": "#94A3B8", "dash": "dot"}))
    fig.update_layout(xaxis_title="Reorder point (units)", yaxis_title="On-hand stock (units)", **LAYOUT)
    return fig


def supplier_orders_figure(inventory: pd.DataFrame) -> go.Figure:
    needed = {"supplier_id", "optimized_order_cost_inr"}
    if inventory.empty or not needed.issubset(inventory.columns):
        return empty_figure("Purchase-order recommendations are unavailable. Run Part 6 first.")
    summary = inventory.groupby("supplier_id", as_index=False).agg(
        order_value_inr=("optimized_order_cost_inr", "sum"),
        order_units=("optimized_order_qty_units", "sum"),
    ).sort_values("order_value_inr", ascending=False)
    summary = summary.loc[summary["order_value_inr"] > 0]
    if summary.empty:
        return empty_figure("No optimized purchase orders are selected under the current scenario.")
    fig = px.bar(summary, x="supplier_id", y="order_value_inr", color="order_value_inr",
                 color_continuous_scale="Tealgrn", hover_data=["order_units"], title="Selected purchase-order value by supplier")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Supplier", yaxis_title="Order value (INR)", **LAYOUT)
    return fig


def forecast_analytics_figure(sales: pd.DataFrame, forecasts: pd.DataFrame) -> go.Figure:
    """Create forecast analytics charts showing model performance and error analysis.
    
    This is Phase 2 - Advanced Forecasting Analytics.
    """
    if sales.empty:
        return empty_figure("No demand observations match these filters.")
    if forecasts.empty:
        return empty_figure("No forecast data available.")
    
    # Aggregate forecasts and sales by date for overall comparison
    daily_sales = sales.groupby("date", as_index=False)["units_sold"].sum().sort_values("date")
    pred = forecasts.groupby("date", as_index=False).agg(
        actual_units_sold=("actual_units_sold", "sum"),
        forecast_units_sold=("forecast_units_sold", "sum"),
    ).sort_values("date")
    
    # Calculate overall metrics
    from demand_inventory.forecasting.metrics import calculate_regression_metrics
    
    if not pred.empty and not pred["actual_units_sold"].isna().all() and not pred["forecast_units_sold"].isna().all():
        metrics = calculate_regression_metrics(
            pred["actual_units_sold"], pred["forecast_units_sold"]
        )
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        wape = metrics["wape_pct"]
        smape = metrics["smape_pct"]
        r2 = metrics["r2"]
    else:
        mae = rmse = wape = smape = r2 = 0.0
    
    # Actual vs Predicted Demand chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_sales["date"],
        y=daily_sales["units_sold"],
        mode="lines",
        name="Actual demand",
        line={"color": PALETTE[0], "width": 2.2}
    ))
    if not pred.empty:
        fig.add_trace(go.Scatter(
            x=pred["date"],
            y=pred["forecast_units_sold"],
            mode="lines+markers",
            name="LightGBM forecast",
            line={"color": PALETTE[1], "width": 2.5, "dash": "dash"},
            marker={"size": 5}
        ))
    
    # Add metric annotations as layout shapes or labels
    fig.update_layout(
        title="Actual vs Predicted Demand",
        xaxis_title="Date",
        yaxis_title="Units sold",
        hovermode="x unified",
        **LAYOUT,
        annotations=[
            dict(
                x=0.02, y=0.98, xref="paper", yref="paper",
                showarrow=False,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                text=f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | WAPE: {wape:.1f}% | sMAPE: {smape:.1f}% | R²: {r2:.2f}",
            )
        ],
    )
    
    return fig
