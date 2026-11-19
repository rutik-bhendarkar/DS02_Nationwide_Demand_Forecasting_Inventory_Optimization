"""Generate reproducible Part 4 comparison and forecast visual reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_part4_reports(root: str | Path, model_results: dict[str, Any]) -> dict[str, str]:
    root = Path(root)
    report_dir = root / "outputs/reports"
    forecast_dir = root / "outputs/forecasts"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_key, result in model_results.items():
        if not isinstance(result, dict) or result.get("status") != "trained":
            continue
        split_metrics = result.get("splits", {})
        test_metrics = split_metrics.get("test", {})
        rows.append({"model": result.get("model", model_key), "split": "test", **test_metrics})
        baseline = result.get("baseline_7_day_lag_test")
        if baseline:
            rows.append({"model": "Seasonal naive (7-day lag)", "split": "test", **baseline})

    comparison_path = report_dir / "part4_model_comparison.csv"
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.drop_duplicates(subset=["model", "split"], keep="last")
        comparison.to_csv(comparison_path, index=False)

    html_paths: dict[str, str] = {}
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        figures = []
        prediction_path = forecast_dir / "test_lightgbm_predictions.csv"
        if prediction_path.exists():
            predictions = pd.read_csv(prediction_path, parse_dates=["date"])
            daily = predictions.groupby("date", as_index=False).agg(
                actual_units_sold=("actual_units_sold", "sum"),
                forecast_units_sold=("forecast_units_sold", "sum"),
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["actual_units_sold"], mode="lines+markers",
                name="Actual demand",
            ))
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["forecast_units_sold"], mode="lines+markers",
                name="LightGBM forecast",
            ))
            fig.update_layout(
                title="Part 4 — Daily Nationwide Test Demand: Actual vs LightGBM",
                xaxis_title="Date", yaxis_title="Units sold", template="plotly_white",
                hovermode="x unified",
            )
            figures.append(fig)

        importance_path = report_dir / "lightgbm_feature_importance.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path).head(15).sort_values("importance_gain")
            fig = go.Figure(go.Bar(
                x=importance["importance_gain"], y=importance["feature"], orientation="h",
                name="Gain importance",
            ))
            fig.update_layout(
                title="Top 15 LightGBM Features by Gain",
                xaxis_title="Gain importance", yaxis_title="Feature", template="plotly_white",
            )
            figures.append(fig)

        if not comparison.empty:
            metrics_to_plot = [column for column in ("mae", "rmse", "wape_pct") if column in comparison]
            for metric in metrics_to_plot:
                fig = go.Figure(go.Bar(
                    x=comparison["model"], y=comparison[metric], text=comparison[metric].round(3),
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"Test-set model comparison — {metric}",
                    xaxis_title="Model", yaxis_title=metric, template="plotly_white",
                )
                figures.append(fig)

        if figures:
            dashboard_path = report_dir / "part4_forecasting_report.html"
            blocks = [figure.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                      for i, figure in enumerate(figures)]
            header = """<!doctype html><html><head><meta charset='utf-8'>
            <title>DS-02 Part 4 Forecasting Report</title>
            <style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}
            h1{margin-bottom:6px}.note{color:#475569;margin-bottom:24px}</style>
            </head><body><h1>DS-02 — Part 4 Forecasting Report</h1>
            <p class='note'>Development evaluation on the bundled synthetic dataset; not actual retailer performance.</p>"""
            dashboard_path.write_text(header + "\n".join(blocks) + "</body></html>", encoding="utf-8")
            html_paths["forecast_report_html"] = dashboard_path.relative_to(root).as_posix()
    except ImportError:
        html_paths["forecast_report_html"] = "not_created_plotly_not_installed"

    if comparison_path.exists():
        html_paths["model_comparison_csv"] = comparison_path.relative_to(root).as_posix()
    return html_paths


def build_completion_summary(root: str | Path, status: dict[str, Any]) -> Path:
    root = Path(root)
    report_dir = root / "outputs/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "part": "Part 4 — Forecasting Models",
        "data_source": "Bundled synthetic development dataset; not real retailer data.",
        "models": status.get("models", {}),
        "availability": status.get("availability", {}),
        "forecast_protocol": status.get("forecast_protocol"),
        "interpretation": (
            "Only models with status='trained' have measured metrics. Models marked not_installed "
            "were not trained and have no performance metrics. Compare models on identical test "
            "rows and forecast protocols before selecting a winner."
        ),
    }
    path = report_dir / "part4_completion_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path
