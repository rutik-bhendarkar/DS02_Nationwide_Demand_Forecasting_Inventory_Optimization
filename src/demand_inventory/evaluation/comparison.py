"""Fair comparison and diagnostic reporting for demand forecasting models.

Model selection is performed on validation WAPE. Test metrics are reported only
as a final holdout estimate and are never used to rank/select the winning model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_inventory.forecasting.data import IDENTIFIER_COLUMNS, TARGET_COLUMN
from demand_inventory.forecasting.metrics import calculate_regression_metrics

KEY_COLUMNS = ["date", "store_id", "sku_id"]
PREDICTION_COLUMNS = [*KEY_COLUMNS, "actual_units_sold", "forecast_units_sold"]
SEGMENT_COLUMNS = ["region", "state", "store_id", "category", "sku_id"]


def _load_prediction_file(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = set(PREDICTION_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing prediction columns: {missing}")
    model_label = None
    if "model" in frame.columns and not frame["model"].dropna().empty:
        model_label = str(frame["model"].dropna().iloc[0])
    frame = frame[PREDICTION_COLUMNS].copy()
    frame.attrs["model_label"] = model_label
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{path.name} contains missing row identifiers")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{path.name} contains duplicate date/store/SKU keys")
    for column in ("actual_units_sold", "forecast_units_sold"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not np.isfinite(frame[column]).all():
            raise ValueError(f"{path.name} contains invalid {column}")
    if (frame["forecast_units_sold"] < 0).any():
        raise ValueError(f"{path.name} contains negative forecasts")
    if split not in path.name:
        raise ValueError(f"Expected a {split} prediction file, got {path.name}")
    result = frame.sort_values(KEY_COLUMNS).reset_index(drop=True)
    result.attrs["model_label"] = model_label
    return result


def _baseline_frame(split_frame: pd.DataFrame) -> pd.DataFrame:
    required = [*KEY_COLUMNS, TARGET_COLUMN, "units_sold_lag_7d"]
    missing = [column for column in required if column not in split_frame]
    if missing:
        raise ValueError(f"Cannot create seasonal-naive baseline; missing columns: {missing}")
    result = split_frame[KEY_COLUMNS].copy()
    result["actual_units_sold"] = split_frame[TARGET_COLUMN].astype(float).to_numpy()
    result["forecast_units_sold"] = np.maximum(
        split_frame["units_sold_lag_7d"].astype(float).to_numpy(), 0.0
    )
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _load_split_predictions(
    root: Path, split: str, split_frame: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    forecast_dir = root / "outputs/forecasts"
    models: dict[str, pd.DataFrame] = {}
    excluded: list[dict[str, str]] = []
    pattern = f"{split}_*_predictions.csv"
    for path in sorted(forecast_dir.glob(pattern)):
        if path.name.startswith(f"{split}_seasonal_naive"):
            continue
        try:
            frame = _load_prediction_file(path, split)
            model_name = frame.attrs.get("model_label") or path.stem.removeprefix(f"{split}_").removesuffix("_predictions").replace("_", " ").title()
            if model_name in models:
                excluded.append({"file": path.name, "reason": f"duplicate model name: {model_name}"})
                continue
            models[model_name] = frame
        except (OSError, ValueError, KeyError) as exc:
            excluded.append({"file": path.name, "reason": str(exc)})

    # A model is compared only when its identifiers exactly match the canonical Part 3 split.
    canonical_keys = split_frame[KEY_COLUMNS].copy().sort_values(KEY_COLUMNS).reset_index(drop=True)
    valid_models: dict[str, pd.DataFrame] = {}
    for model_name, frame in models.items():
        keys = frame[KEY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
        if not keys.equals(canonical_keys):
            excluded.append({
                "file": f"{split}_{model_name}",
                "reason": "prediction identifiers do not exactly match the canonical split",
            })
            continue
        expected_actual = split_frame.sort_values(KEY_COLUMNS).reset_index(drop=True)[TARGET_COLUMN]
        actual_check = np.allclose(
            frame["actual_units_sold"].to_numpy(dtype=float),
            expected_actual.to_numpy(dtype=float),
            rtol=0,
            atol=1e-9,
        )
        if not actual_check:
            excluded.append({
                "file": f"{split}_{model_name}",
                "reason": "actual values do not match the canonical split",
            })
            continue
        valid_models[model_name] = frame

    if not valid_models:
        raise FileNotFoundError(
            f"No compatible trained-model {split} prediction files found in {forecast_dir}. "
            "Run scripts/run_forecasting.py first."
        )
    valid_models["Seasonal naive (7-day lag)"] = _baseline_frame(split_frame)
    return valid_models, excluded


def _metric_row(model_name: str, split: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {"model": model_name, "split": split, **calculate_regression_metrics(
        frame["actual_units_sold"], frame["forecast_units_sold"]
    )}


def _make_segment_metrics(
    split_frame: pd.DataFrame, predictions: dict[str, pd.DataFrame], split: str
) -> pd.DataFrame:
    base_columns = list(dict.fromkeys([*KEY_COLUMNS, *[c for c in SEGMENT_COLUMNS if c in split_frame]]))
    base = split_frame[base_columns].copy()
    base["actual_units_sold"] = split_frame[TARGET_COLUMN].astype(float).to_numpy()
    rows: list[dict[str, Any]] = []
    for model_name, forecast in predictions.items():
        merged = base.merge(forecast[[*KEY_COLUMNS, "forecast_units_sold"]], on=KEY_COLUMNS, how="inner", validate="one_to_one")
        for dimension in SEGMENT_COLUMNS:
            if dimension not in merged.columns:
                continue
            for segment, group in merged.groupby(dimension, dropna=False, observed=False):
                row = {
                    "model": model_name,
                    "split": split,
                    "dimension": dimension,
                    "segment": str(segment),
                }
                row.update(calculate_regression_metrics(group["actual_units_sold"], group["forecast_units_sold"]))
                rows.append(row)
    return pd.DataFrame(rows)


def _make_demand_tier_metrics(
    split_frame: pd.DataFrame, predictions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    base = split_frame[[*KEY_COLUMNS, TARGET_COLUMN]].copy()
    q_low, q_high = base[TARGET_COLUMN].quantile([1 / 3, 2 / 3]).tolist()
    base["demand_tier"] = pd.cut(
        base[TARGET_COLUMN],
        bins=[-np.inf, q_low, q_high, np.inf],
        labels=["Low demand", "Medium demand", "High demand"],
        include_lowest=True,
        duplicates="drop",
    )
    rows: list[dict[str, Any]] = []
    for model_name, forecast in predictions.items():
        merged = base.merge(forecast[[*KEY_COLUMNS, "forecast_units_sold"]], on=KEY_COLUMNS, validate="one_to_one")
        for tier, group in merged.groupby("demand_tier", observed=False):
            row: dict[str, Any] = {"model": model_name, "demand_tier": str(tier)}
            row.update(calculate_regression_metrics(group[TARGET_COLUMN], group["forecast_units_sold"]))
            rows.append(row)
    return pd.DataFrame(rows)


def _daily_metrics(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, frame in predictions.items():
        for date, group in frame.groupby("date", sort=True):
            metrics = calculate_regression_metrics(group["actual_units_sold"], group["forecast_units_sold"])
            rows.append({"model": model_name, "date": pd.Timestamp(date), **metrics})
    return pd.DataFrame(rows)


def _write_html_report(
    root: Path,
    comparison: pd.DataFrame,
    ranking: pd.DataFrame,
    segments: pd.DataFrame,
    daily: pd.DataFrame,
    selected_model: str,
    data_note: str,
) -> str:
    report_path = root / "outputs/reports/part5_evaluation_report.html"
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "not_created_plotly_not_installed"

    validation = comparison[comparison["split"] == "validation"].copy()
    test = comparison[comparison["split"] == "test"].copy()
    fig_wape = px.bar(
        validation.sort_values("wape_pct"), x="model", y="wape_pct",
        title="Validation WAPE — lower is better (used for model selection)",
        text_auto=".2f", color="model",
    )
    fig_wape.update_layout(showlegend=False, yaxis_title="WAPE (%)", xaxis_title="")
    fig_holdout = px.bar(
        test.sort_values("wape_pct"), x="model", y="wape_pct",
        title="Final holdout test WAPE — report only, not used to select the model",
        text_auto=".2f", color="model",
    )
    fig_holdout.update_layout(showlegend=False, yaxis_title="WAPE (%)", xaxis_title="")

    region = segments[segments["dimension"] == "region"] if not segments.empty else pd.DataFrame()
    fig_region = px.bar(
        region, x="segment", y="wape_pct", color="model", barmode="group",
        title="Test WAPE by region — lower is better",
    ) if not region.empty else go.Figure()
    if not daily.empty:
        best_daily = daily[daily["model"] == selected_model]
        fig_daily = px.line(
            best_daily, x="date", y="wape_pct", markers=True,
            title=f"Daily test WAPE — {selected_model}",
        )
    else:
        fig_daily = go.Figure()

    figures = [fig_wape, fig_holdout, fig_region, fig_daily]
    blocks = [fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, fig in enumerate(figures)]
    test_table = test[["model", "n_observations", "mae", "rmse", "wape_pct", "smape_pct", "bias_pct", "r2"]].to_html(index=False, float_format=lambda value: f"{value:,.3f}")
    header = f"""<!doctype html><html><head><meta charset='utf-8'><title>DS-02 Part 5 Evaluation</title>
    <style>body{{font-family:Arial,sans-serif;margin:26px;background:#f8fafc;color:#0f172a}}h1{{margin-bottom:4px}}.note{{color:#475569}}.panel{{background:white;padding:18px;border-radius:12px;margin:18px 0;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #e2e8f0;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head><body>
    <h1>DS-02 — Part 5 Model Evaluation & Comparison</h1>
    <p class='note'>{data_note}</p><div class='panel'><h2>Selected model: {selected_model}</h2>
    <p>Selection rule: lowest validation WAPE among models with compatible predictions. The chronological test split remains a final holdout.</p></div>
    <div class='panel'><h2>Final test metrics</h2>{test_table}</div>"""
    report_path.write_text(header + "\n".join(f"<div class='panel'>{block}</div>" for block in blocks) + "</body></html>", encoding="utf-8")
    return report_path.relative_to(root).as_posix()


def run_model_comparison(
    root: str | Path,
    splits: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    root = Path(root).resolve()
    report_dir = root / "outputs/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: dict[str, dict[str, pd.DataFrame]] = {}
    exclusions: dict[str, list[dict[str, str]]] = {}
    comparison_rows: list[dict[str, Any]] = []

    for split in ("validation", "test"):
        predictions, excluded = _load_split_predictions(root, split, splits[split])
        all_predictions[split] = predictions
        exclusions[split] = excluded
        for model_name, frame in predictions.items():
            comparison_rows.append(_metric_row(model_name, split, frame))

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = report_dir / "part5_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    # Best model is selected on validation only; baseline is not eligible to win model selection.
    validation_rank = comparison[comparison["split"] == "validation"].copy()
    test_model_names = set(comparison.loc[comparison["split"] == "test", "model"])
    validation_rank = validation_rank[
        (validation_rank["model"] != "Seasonal naive (7-day lag)")
        & validation_rank["model"].isin(test_model_names)
    ]
    if validation_rank.empty:
        raise ValueError("No trained model is available for validation-based selection")
    validation_rank = validation_rank.sort_values(["wape_pct", "rmse", "mae"], ascending=True).reset_index(drop=True)
    validation_rank.insert(0, "validation_rank", np.arange(1, len(validation_rank) + 1))
    selected_model = str(validation_rank.iloc[0]["model"])
    validation_rank["selected"] = validation_rank["model"] == selected_model
    baseline_validation = comparison[
        (comparison["split"] == "validation")
        & (comparison["model"] == "Seasonal naive (7-day lag)")
    ]
    baseline_test = comparison[
        (comparison["split"] == "test")
        & (comparison["model"] == "Seasonal naive (7-day lag)")
    ]
    selected_validation = validation_rank.iloc[0]
    selected_test = comparison[
        (comparison["split"] == "test") & (comparison["model"] == selected_model)
    ].iloc[0]
    validation_baseline_wape = float(baseline_validation.iloc[0]["wape_pct"])
    test_baseline_wape = float(baseline_test.iloc[0]["wape_pct"])
    validation_rank["validation_wape_improvement_vs_baseline_pct"] = (
        (validation_baseline_wape - validation_rank["wape_pct"]) / validation_baseline_wape * 100.0
    )
    ranking_path = report_dir / "part5_model_ranking.csv"
    validation_rank.to_csv(ranking_path, index=False)

    test_predictions = all_predictions["test"]
    segments = _make_segment_metrics(splits["test"], test_predictions, "test")
    segments_path = report_dir / "part5_segment_performance.csv"
    segments.to_csv(segments_path, index=False)
    demand_tiers = _make_demand_tier_metrics(splits["test"], test_predictions)
    demand_tiers_path = report_dir / "part5_demand_tier_performance.csv"
    demand_tiers.to_csv(demand_tiers_path, index=False)
    daily = _daily_metrics(test_predictions)
    daily_path = report_dir / "part5_daily_metrics.csv"
    daily.to_csv(daily_path, index=False)

    selected_frame = test_predictions[selected_model].copy()
    selected_frame["signed_error_units"] = selected_frame["forecast_units_sold"] - selected_frame["actual_units_sold"]
    selected_frame["absolute_error_units"] = selected_frame["signed_error_units"].abs()
    id_columns = [column for column in IDENTIFIER_COLUMNS if column in splits["test"].columns]
    selected_frame = selected_frame.merge(
        splits["test"][id_columns], on=KEY_COLUMNS, how="left", validate="one_to_one", suffixes=("", "_metadata")
    )
    worst_errors_path = report_dir / "part5_largest_forecast_errors.csv"
    selected_frame.sort_values("absolute_error_units", ascending=False).head(100).to_csv(worst_errors_path, index=False)

    summary = {
        "part": "Part 5 — Model Evaluation & Comparison",
        "status": "success",
        "data_source_note": metadata.get("data_source_note", "Not specified"),
        "selection_protocol": "Select the model with the lowest validation WAPE; use RMSE and MAE as tie-breakers. Never use the test split to select the model.",
        "forecast_protocol": "Part 4 rolling one-step-ahead predictions on chronological validation/test dates. Seasonal naive uses the 7-day lag feature.",
        "selected_model": selected_model,
        "selected_validation_metrics": selected_validation.to_dict(),
        "selected_test_metrics": selected_test.to_dict(),
        "baseline_validation_wape_pct": validation_baseline_wape,
        "baseline_test_wape_pct": test_baseline_wape,
        "validation_wape_improvement_vs_baseline_pct": float((validation_baseline_wape - float(selected_validation["wape_pct"])) / validation_baseline_wape * 100.0),
        "test_wape_improvement_vs_baseline_pct": float((test_baseline_wape - float(selected_test["wape_pct"])) / test_baseline_wape * 100.0),
        "models_compared": sorted(test_predictions),
        "excluded_prediction_files": exclusions,
        "files": {
            "comparison": comparison_path.relative_to(root).as_posix(),
            "ranking": ranking_path.relative_to(root).as_posix(),
            "segment_performance": segments_path.relative_to(root).as_posix(),
            "demand_tier_performance": demand_tiers_path.relative_to(root).as_posix(),
            "daily_metrics": daily_path.relative_to(root).as_posix(),
            "largest_forecast_errors": worst_errors_path.relative_to(root).as_posix(),
        },
    }
    summary["files"]["html_report"] = _write_html_report(
        root, comparison, validation_rank, segments, daily, selected_model,
        str(summary["data_source_note"]),
    )
    summary_path = report_dir / "part5_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["files"]["summary"] = summary_path.relative_to(root).as_posix()
    return summary
