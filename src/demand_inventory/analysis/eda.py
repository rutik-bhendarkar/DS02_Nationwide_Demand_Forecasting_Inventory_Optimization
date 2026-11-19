"""Reproducible exploratory analysis for daily store-SKU sales data.

The EDA layer is deliberately descriptive: it does not train forecasting models,
fill missing demand, infer lost sales, or claim business impact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_inventory.data.validation import validate_sales_frame

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _safe_float(value: Any) -> float | None:
    """Convert a scalar to a JSON-safe float."""
    if pd.isna(value):
        return None
    return float(value)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy without mutating the caller's DataFrame."""
    required = {"date", "store_id", "sku_id", "units_sold", "unit_price"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"EDA requires these columns: {', '.join(missing)}")

    data = frame.copy(deep=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce", format="mixed")
    data["units_sold"] = pd.to_numeric(data["units_sold"], errors="coerce")
    data["unit_price"] = pd.to_numeric(data["unit_price"], errors="coerce")

    if "net_sales_value" in data.columns:
        data["analysis_sales_value"] = pd.to_numeric(
            data["net_sales_value"], errors="coerce"
        )
        data["sales_value_basis"] = "provided net_sales_value column"
    else:
        discount = (
            pd.to_numeric(data["discount_pct"], errors="coerce").fillna(0)
            if "discount_pct" in data.columns
            else 0
        )
        data["analysis_sales_value"] = (
            data["units_sold"] * data["unit_price"] * (1 - discount)
        )
        data["sales_value_basis"] = "estimated from units_sold, unit_price and discount_pct"
    return data


def _aggregate(frame: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """Aggregate totals and true daily demand statistics for each dimension.

    ``average_units_per_record`` describes one store-SKU-date row. In contrast,
    ``average_daily_units`` and ``median_daily_units`` first sum all rows for a
    dimension on each calendar date, then summarize those daily totals. This
    distinction matters for regions, stores, categories, SKUs and months.
    """
    grouped = frame.groupby(dimensions, dropna=False, observed=True)
    result = grouped.agg(
        records=("units_sold", "size"),
        total_units_sold=("units_sold", "sum"),
        average_units_per_record=("units_sold", "mean"),
        median_units_per_record=("units_sold", "median"),
        total_sales_value=("analysis_sales_value", "sum"),
        average_unit_price=("unit_price", "mean"),
    ).reset_index()

    daily_frame = frame.assign(_analysis_date=frame["date"].dt.normalize())
    daily_by_dimension = (
        daily_frame.groupby([*dimensions, "_analysis_date"], dropna=False, observed=True)
        .agg(_daily_units=("units_sold", "sum"))
        .reset_index()
    )
    daily_stats = (
        daily_by_dimension.groupby(dimensions, dropna=False, observed=True)
        .agg(
            average_daily_units=("_daily_units", "mean"),
            median_daily_units=("_daily_units", "median"),
            observed_days=("_analysis_date", "nunique"),
        )
        .reset_index()
    )
    result = result.merge(daily_stats, on=dimensions, how="left", validate="one_to_one")
    for column in (
        "total_sales_value", "average_units_per_record", "median_units_per_record",
        "average_daily_units", "median_daily_units", "average_unit_price",
    ):
        result[column] = result[column].round(3 if column != "total_sales_value" else 2)
    return result.sort_values("total_units_sold", ascending=False).reset_index(drop=True)


def _outlier_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flag high/low demand observations using 1.5-IQR fences per store-SKU series."""
    group_columns = ["store_id", "sku_id"]
    records: list[pd.DataFrame] = []
    groups_analyzed = 0

    for (store_id, sku_id), group in frame.groupby(group_columns, dropna=False, observed=True):
        values = group["units_sold"].dropna()
        if len(values) < 4:
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        groups_analyzed += 1
        flagged = group.loc[
            group["units_sold"].lt(lower) | group["units_sold"].gt(upper),
            ["date", "store_id", "sku_id", "units_sold"],
        ].copy()
        if not flagged.empty:
            flagged["q1"] = q1
            flagged["q3"] = q3
            flagged["iqr_lower_fence"] = lower
            flagged["iqr_upper_fence"] = upper
            flagged["outlier_direction"] = np.where(
                flagged["units_sold"] < lower, "low", "high"
            )
            records.append(flagged)

    columns = [
        "date", "store_id", "sku_id", "units_sold", "q1", "q3",
        "iqr_lower_fence", "iqr_upper_fence", "outlier_direction",
    ]
    outliers = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=columns)
    if not outliers.empty:
        outliers["date"] = pd.to_datetime(outliers["date"]).dt.strftime("%Y-%m-%d")
        outliers = outliers.sort_values("units_sold", ascending=False).reset_index(drop=True)

    stats = {
        "method": "1.5 × IQR fences calculated separately for each store_id + sku_id series",
        "series_analyzed": groups_analyzed,
        "outlier_observations": int(len(outliers)),
        "outlier_observation_pct": round(len(outliers) / max(len(frame), 1) * 100, 4),
        "interpretation": (
            "Statistical flags for review, not proof of data errors. Promotions, holidays, "
            "stock availability and local events may explain unusual demand."
        ),
    }
    return outliers, stats


def _coverage_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure observed date coverage against each series' own min/max date range."""
    rows: list[dict[str, Any]] = []
    for (store_id, sku_id), group in frame.groupby(["store_id", "sku_id"], observed=True):
        dates = group["date"].dropna().dt.normalize()
        if dates.empty:
            continue
        min_date, max_date = dates.min(), dates.max()
        expected = int((max_date - min_date).days + 1)
        observed = int(dates.nunique())
        rows.append({
            "store_id": store_id,
            "sku_id": sku_id,
            "first_date": min_date.strftime("%Y-%m-%d"),
            "last_date": max_date.strftime("%Y-%m-%d"),
            "expected_calendar_days": expected,
            "observed_days": observed,
            "missing_calendar_days": max(expected - observed, 0),
            "coverage_pct": round(observed / expected * 100, 3) if expected else None,
        })
    return pd.DataFrame(rows)


def _plot_dashboard(tables: dict[str, pd.DataFrame], destination: Path) -> bool:
    """Write a self-contained interactive Plotly EDA dashboard when Plotly is installed."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False

    daily = tables["daily_demand"]
    region = tables.get("region_performance", pd.DataFrame())
    weekday = tables["weekday_demand"]
    # Keep the public output key consistent with the rest of the project:
    # the source column is sku_id, while the report table is sku_performance.
    sku = tables["sku_performance"].head(10)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Daily units sold",
            "Units sold by region",
            "Average demand by weekday",
            "Top 10 SKUs by units sold",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["total_units_sold"], mode="lines",
            name="Daily units", line={"width": 2},
        ), row=1, col=1,
    )
    if not region.empty:
        fig.add_trace(
            go.Bar(
                x=region["region"], y=region["total_units_sold"],
                name="Region demand", showlegend=False,
            ), row=1, col=2,
        )
    fig.add_trace(
        go.Bar(
            x=weekday["weekday"], y=weekday["average_units_sold"],
            name="Weekday average", showlegend=False,
        ), row=2, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=sku["sku_id"], y=sku["total_units_sold"],
            name="SKU demand", showlegend=False,
        ), row=2, col=2,
    )
    fig.update_layout(
        title="DS-02 | Nationwide Demand Exploratory Analysis",
        template="plotly_white",
        height=850,
        hovermode="x unified",
        margin={"l": 50, "r": 30, "t": 90, "b": 65},
    )
    fig.update_xaxes(tickangle=-25)
    fig.write_html(destination, include_plotlyjs=True, full_html=True)
    return True


def _build_findings(summary: dict[str, Any]) -> str:
    """Create a human-readable report using only calculated dataset values."""
    date_range = summary["date_range"]
    lines = [
        "# Part 2 — Exploratory Data Analysis Findings",
        "",
        f"> **Data source:** {summary['data_source_note']}",
        "> This report is descriptive. It does not evaluate forecasting accuracy or prove causal effects.",
        "",
        "## Dataset overview",
        "",
        f"- Rows: **{summary['row_count']:,}**",
        f"- Columns: **{summary['column_count']:,}**",
        f"- Date coverage: **{date_range['start']} to {date_range['end']}** ({date_range['calendar_days']} calendar days)",
        f"- Stores: **{summary['cardinality'].get('store_id', 0):,}**",
        f"- SKUs: **{summary['cardinality'].get('sku_id', 0):,}**",
        f"- Regions: **{summary['cardinality'].get('region', 'not supplied')}**",
        "",
        "## Demand and sales",
        "",
        f"- Total observed units sold: **{summary['demand']['total_units_sold']:,.0f}**",
        f"- Average units per row: **{summary['demand']['mean_units_per_record']:,.2f}**",
        f"- Median units per row: **{summary['demand']['median_units_per_record']:,.2f}**",
        f"- Daily demand average: **{summary['demand']['average_daily_units']:,.2f} units**",
        f"- Sales value basis: **{summary['sales_value_basis']}**",
        f"- Total analysis sales value: **{summary['demand']['total_sales_value']:,.2f}**",
        "",
        "## Data quality and coverage",
        "",
        f"- Missing cells: **{summary['quality']['missing_cells']:,}**",
        f"- Duplicate date-store-SKU keys: **{summary['quality']['duplicate_business_keys']:,}**",
        f"- Store-SKU series analyzed for coverage: **{summary['coverage']['series_count']:,}**",
        f"- Series with at least one missing calendar day: **{summary['coverage']['series_with_gaps']:,}**",
        "",
        "## Outlier review",
        "",
        f"- Flagged observations: **{summary['outliers']['outlier_observations']:,}** ({summary['outliers']['outlier_observation_pct']:.4f}% of rows)",
        f"- Method: {summary['outliers']['method']}.",
        f"- Interpretation: {summary['outliers']['interpretation']}",
        "",
        "## Interpretation cautions",
        "",
        "- This dataset is a development sample; results are not evidence about actual nationwide retail demand.",
        "- Observed sales may be lower than true demand when inventory is unavailable (censored demand).",
        "- Calendar patterns are descriptive only. Promotion and holiday relationships are not causal estimates.",
        "- Outlier flags should be reviewed with promotion, holiday, stock and source-system context before any removal.",
        "- Forecast model training and time-based model evaluation are reserved for Parts 4 and 5.",
        "",
        "## Generated outputs",
        "",
        "See the `outputs/eda/` folder for CSV analysis tables and the interactive HTML dashboard, and `reports/part2_eda_summary.json` for the machine-readable summary.",
    ]
    return "\n".join(lines) + "\n"


def run_eda(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    top_n: int = 10,
    data_source_note: str = "User-supplied dataset; source not independently verified.",
) -> dict[str, Any]:
    """Generate EDA tables, JSON summary, markdown findings, and Plotly HTML.

    Parameters
    ----------
    frame:
        Input daily store-SKU observations. The input is never mutated.
    output_dir:
        Directory receiving CSV tables and the interactive HTML report.
    top_n:
        Number of top stores/SKUs/categories included in summary highlights.
    data_source_note:
        Explicit source caveat written into the report.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    data = _prepare_frame(frame)
    if data["date"].isna().any():
        raise ValueError("EDA cannot run with invalid dates; validate/fix the input first.")
    if data["units_sold"].isna().any():
        raise ValueError("EDA cannot run with non-numeric or missing units_sold values.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports_dir = output.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    daily = (
        data.groupby(data["date"].dt.normalize(), observed=True)
        .agg(
            total_units_sold=("units_sold", "sum"),
            total_sales_value=("analysis_sales_value", "sum"),
            active_store_count=("store_id", "nunique"),
            active_sku_count=("sku_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"date": "date"})
    )
    daily["total_sales_value"] = daily["total_sales_value"].round(2)
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    monthly_source = data.assign(month=data["date"].dt.to_period("M").astype(str))
    monthly = _aggregate(monthly_source, ["month"])

    weekday_source = data.assign(
        weekday=data["date"].dt.day_name(),
        weekday_number=data["date"].dt.dayofweek,
    )
    weekday = (
        weekday_source.groupby(["weekday_number", "weekday"], observed=True)
        .agg(
            records=("units_sold", "size"),
            total_units_sold=("units_sold", "sum"),
            average_units_sold=("units_sold", "mean"),
        )
        .reset_index()
        .sort_values("weekday_number")
    )
    weekday["average_units_sold"] = weekday["average_units_sold"].round(3)

    tables: dict[str, pd.DataFrame] = {
        "daily_demand": daily,
        "monthly_demand": monthly,
        "weekday_demand": weekday,
    }
    dimension_table_names = {
        "region": "region_performance",
        "state": "state_performance",
        "city": "city_performance",
        "store_id": "store_performance",
        "category": "category_performance",
        "sku_id": "sku_performance",
        "product_name": "product_performance",
    }
    for dimension, table_name in dimension_table_names.items():
        if dimension in data.columns and data[dimension].notna().any():
            tables[table_name] = _aggregate(data, [dimension])

    month_day_source = data.assign(month_number=data["date"].dt.month)
    month_day = (
        month_day_source.groupby("month_number", observed=True)
        .agg(
            records=("units_sold", "size"),
            total_units_sold=("units_sold", "sum"),
            average_units_sold=("units_sold", "mean"),
        )
        .reset_index()
    )
    month_day["average_units_sold"] = month_day["average_units_sold"].round(3)
    tables["month_of_year_demand"] = month_day

    # Descriptive comparisons only: these tables do not establish causal effects.
    for flag_column, table_name in (
        ("promo_flag", "promotion_demand_comparison"),
        ("holiday_flag", "holiday_demand_comparison"),
    ):
        if flag_column in data.columns:
            flag_values = pd.to_numeric(data[flag_column], errors="coerce")
            comparison_source = data.assign(_flag=flag_values)
            comparison = (
                comparison_source.groupby("_flag", dropna=False, observed=True)
                .agg(
                    records=("units_sold", "size"),
                    total_units_sold=("units_sold", "sum"),
                    average_units_sold=("units_sold", "mean"),
                    median_units_sold=("units_sold", "median"),
                    average_sales_value=("analysis_sales_value", "mean"),
                )
                .reset_index()
                .rename(columns={"_flag": flag_column})
            )
            for numeric_column in ("average_units_sold", "median_units_sold", "average_sales_value"):
                comparison[numeric_column] = comparison[numeric_column].round(3)
            tables[table_name] = comparison

    if "closing_stock_units" in data.columns:
        inventory_source = data.assign(
            closing_stock_units=pd.to_numeric(data["closing_stock_units"], errors="coerce")
        )
        inventory_source["closing_stock_zero_flag"] = (
            inventory_source["closing_stock_units"].eq(0).astype("int8")
        )
        inventory_pressure = (
            inventory_source.groupby("closing_stock_zero_flag", observed=True)
            .agg(
                records=("units_sold", "size"),
                total_units_sold=("units_sold", "sum"),
                average_units_sold=("units_sold", "mean"),
            )
            .reset_index()
        )
        inventory_pressure["interpretation"] = inventory_pressure[
            "closing_stock_zero_flag"
        ].map({0: "Closing stock above zero", 1: "Closing stock equals zero"})
        tables["closing_stock_pressure"] = inventory_pressure

    missingness = pd.DataFrame({
        "column": frame.columns,
        "missing_count": [int(frame[column].isna().sum()) for column in frame.columns],
        "missing_pct": [round(float(frame[column].isna().mean() * 100), 4) for column in frame.columns],
        "dtype": [str(frame[column].dtype) for column in frame.columns],
    }).sort_values(["missing_count", "column"], ascending=[False, True])
    tables["missingness"] = missingness

    coverage = _coverage_analysis(data)
    tables["store_sku_coverage"] = coverage
    outliers, outlier_summary = _outlier_analysis(data)
    tables["demand_outliers_iqr"] = outliers

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    correlation = data[numeric_cols].corr(numeric_only=True).round(4) if len(numeric_cols) >= 2 else pd.DataFrame()
    if not correlation.empty:
        correlation.index.name = "feature"
        tables["numeric_correlations"] = correlation.reset_index()

    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    validation = validate_sales_frame(frame)
    dates = data["date"].dt.normalize()
    region_count: int | str = int(data["region"].nunique()) if "region" in data.columns else "not supplied"
    total_sales_value = float(data["analysis_sales_value"].sum(skipna=True))
    summary: dict[str, Any] = {
        "part": "Part 2 — Exploratory Data Analysis",
        "data_source_note": data_source_note,
        "row_count": int(len(data)),
        "column_count": int(frame.shape[1]),
        "date_range": {
            "start": dates.min().strftime("%Y-%m-%d"),
            "end": dates.max().strftime("%Y-%m-%d"),
            "calendar_days": int((dates.max() - dates.min()).days + 1),
        },
        "cardinality": {
            key: int(data[key].nunique(dropna=True)) if key in data.columns else 0
            for key in ("store_id", "sku_id", "region", "state", "city", "category")
        },
        "region_count": region_count,
        "demand": {
            "total_units_sold": float(data["units_sold"].sum()),
            "mean_units_per_record": float(data["units_sold"].mean()),
            "median_units_per_record": float(data["units_sold"].median()),
            "average_daily_units": float(daily["total_units_sold"].mean()),
            "total_sales_value": total_sales_value,
        },
        "sales_value_basis": str(data["sales_value_basis"].iloc[0]),
        "quality": {
            "is_valid_under_part1_contract": validation.is_valid,
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_business_keys": int(validation.duplicate_business_key_count),
            "validation_errors": validation.errors,
            "validation_warnings": validation.warnings,
        },
        "coverage": {
            "series_count": int(len(coverage)),
            "series_with_gaps": int((coverage["missing_calendar_days"] > 0).sum()) if not coverage.empty else 0,
            "mean_coverage_pct": _safe_float(coverage["coverage_pct"].mean()) if not coverage.empty else None,
        },
        "outliers": outlier_summary,
        "inventory_observation": {
            "closing_stock_zero_rows": (
                int(pd.to_numeric(data["closing_stock_units"], errors="coerce").eq(0).sum())
                if "closing_stock_units" in data.columns else None
            ),
            "caution": (
                "Zero closing stock is an inventory-pressure indicator only; it does not reveal "
                "unobserved lost demand or prove a stockout."
            ),
        },
        "highlights": {},
        "generated_tables": [f"{name}.csv" for name in tables],
    }

    for key in ("region_performance", "store_performance", "category_performance", "sku_performance"):
        if key in tables:
            summary["highlights"][key] = tables[key].head(top_n).to_dict(orient="records")

    dashboard_created = _plot_dashboard(tables, output / "eda_dashboard.html")
    summary["interactive_dashboard_created"] = dashboard_created
    summary_path = reports_dir / "part2_eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    findings_path = reports_dir / "part2_eda_findings.md"
    findings_path.write_text(_build_findings(summary), encoding="utf-8")

    return summary
