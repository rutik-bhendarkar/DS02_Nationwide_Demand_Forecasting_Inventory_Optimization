"""Build a static interactive HTML preview of the Part 7 Dash charts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import plotly.io as pio
from demand_inventory.dashboard.figures import (
    demand_forecast_figure, inventory_risk_figure, regional_demand_figure,
    sku_demand_figure, stock_position_figure, supplier_orders_figure,
)
from demand_inventory.dashboard.service import apply_dashboard_filters, load_dashboard_data, summarize_kpis


def main() -> int:
    data = load_dashboard_data(ROOT)
    filtered = apply_dashboard_filters(data)
    kpis = summarize_kpis(filtered)
    figures = [
        demand_forecast_figure(filtered["sales_filtered"], filtered["forecasts_filtered"]),
        regional_demand_figure(filtered["sales_filtered"]),
        sku_demand_figure(filtered["sales_filtered"]),
        inventory_risk_figure(filtered["inventory_filtered"]),
        stock_position_figure(filtered["inventory_filtered"]),
        supplier_orders_figure(filtered["inventory_filtered"]),
    ]
    labels = ["Demand and Forecast", "Regional Demand", "Top Products", "Inventory Risk", "Stock Position", "Supplier Orders"]
    sections = []
    for i, (label, fig) in enumerate(zip(labels, figures, strict=True)):
        sections.append(f"<section><h2>{label}</h2>{pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if i == 0 else False)}</section>")
    kpi_cards = "".join(
        f"<div class='kpi'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in [
            ("Observed units", f"{kpis['total_units']:,.0f}"),
            ("Stores", str(kpis["stores"])), ("SKUs", str(kpis["skus"])),
            ("Reorder candidates", str(kpis["reorder_candidates"])),
            ("Selected order lines", str(kpis["selected_order_lines"])),
            ("Planned order value", f"₹{kpis['planned_order_value_inr']:,.0f}"),
        ]
    )
    html = """<!doctype html><html><head><meta charset='utf-8'><title>DS-02 Part 7 Dashboard Preview</title>
    <style>body{margin:28px;background:#f4f7fb;color:#17243a;font-family:Segoe UI,Arial}h1{margin-bottom:4px}.muted{color:#6b7a90}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0}.kpi,section{background:#fff;border:1px solid #e6ebf2;border-radius:14px;padding:16px;margin-bottom:15px}
    .kpi span{display:block;font-size:11px;color:#6b7a90}.kpi strong{display:block;font-size:23px;margin-top:8px}.note{background:#fff8e8;padding:14px;border-left:4px solid #e6a83b}
    </style></head><body><h1>Nationwide Demand & Inventory Intelligence</h1>
    <p class='muted'>Static interactive chart preview of the Plotly Dash application. Live dashboard filters and callbacks require Dash.</p>
    <div class='note'>Synthetic development dataset. Inventory outputs are scenario recommendations, not verified procurement decisions.</div>
    <div class='grid'>""" + kpi_cards + "</div>" + "".join(sections) + "</body></html>"
    out = ROOT / "outputs/dashboard/part7_dashboard_preview.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard preview created: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
