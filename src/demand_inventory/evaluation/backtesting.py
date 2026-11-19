"""Expanding-origin rolling one-step backtesting for the LightGBM global model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_inventory.forecasting.common import build_prediction_records
from demand_inventory.forecasting.data import TARGET_COLUMN
from demand_inventory.forecasting.metrics import calculate_regression_metrics


def split_dates_into_windows(dates: pd.Series, window_days: int = 7) -> list[list[pd.Timestamp]]:
    """Split sorted unique target dates into non-overlapping evaluation windows."""
    if window_days < 1:
        raise ValueError("window_days must be at least 1")
    unique_dates = sorted(pd.to_datetime(dates).dt.normalize().dropna().unique())
    return [
        [pd.Timestamp(value) for value in unique_dates[index : index + window_days]]
        for index in range(0, len(unique_dates), window_days)
    ]


def _model_matrices(
    train_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train_frame[feature_columns].copy()
    forecast_x = forecast_frame[feature_columns].copy()
    for column in feature_columns:
        if column in categorical_columns:
            categories = sorted(
                set(train_x[column].dropna().tolist())
                | set(forecast_x[column].dropna().tolist())
            )
            train_x[column] = pd.Categorical(train_x[column], categories=categories)
            forecast_x[column] = pd.Categorical(forecast_x[column], categories=categories)
        else:
            train_x[column] = pd.to_numeric(train_x[column], errors="coerce")
            forecast_x[column] = pd.to_numeric(forecast_x[column], errors="coerce")
    if train_x.isna().any().any() or forecast_x.isna().any().any():
        raise ValueError("Rolling-origin feature matrices contain missing values")
    return train_x, forecast_x


def run_lightgbm_rolling_backtest(
    root: str | Path,
    splits: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    *,
    window_days: int = 7,
    n_jobs: int = 4,
) -> dict[str, Any]:
    """Refit at each test window using only observations dated before that window.

    Within each window, this evaluates sequential one-step-ahead forecasts. Part 3
    lag features for each target date contain demand observed before that target
    date. The protocol is not a fixed-origin multi-step forecast.
    """
    try:
        import joblib
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - optional stack
        raise RuntimeError(
            "Rolling-origin backtesting requires LightGBM and joblib. "
            "Install requirements-prophet-lightgbm.txt."
        ) from exc

    root = Path(root).resolve()
    artifact_path = root / "models/trained/lightgbm_demand_regressor.joblib"
    metrics_path = root / "outputs/reports/part4_lightgbm_metrics.json"
    if not artifact_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Run Part 4 LightGBM training before Part 5 rolling-origin backtesting")

    saved_model = joblib.load(artifact_path)
    part4_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata["feature_columns"])
    categorical_columns = list(part4_metrics.get("categorical_code_features", []))
    if not categorical_columns:
        categorical_columns = [name for name in feature_columns if name.endswith("_code")]
    best_iteration = int(part4_metrics.get("best_iteration") or saved_model.get_params().get("n_estimators", 165))
    model_params = saved_model.get_params(deep=False)
    model_params.update({"n_estimators": best_iteration, "n_jobs": n_jobs, "verbosity": -1})
    model_params.pop("callbacks", None)

    history_all = pd.concat([splits["train"], splits["validation"], splits["test"]], ignore_index=True)
    windows = split_dates_into_windows(splits["test"]["date"], window_days=window_days)
    if len(windows) < 2:
        raise ValueError("At least two test-date windows are required for rolling-origin backtesting")

    all_records: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_number, window in enumerate(windows, start=1):
        start_date, end_date = min(window), max(window)
        history = history_all[history_all["date"].dt.normalize() < start_date].copy()
        forecast_frame = splits["test"][splits["test"]["date"].dt.normalize().isin(window)].copy()
        if history.empty or forecast_frame.empty:
            raise ValueError(f"Rolling-origin fold {fold_number} has empty history or forecast rows")
        if history["date"].max().normalize() >= start_date:
            raise AssertionError("Rolling-origin history contains a date at or after the forecast origin")

        x_train, x_forecast = _model_matrices(
            history, forecast_frame, feature_columns, categorical_columns
        )
        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            x_train,
            history[TARGET_COLUMN].astype(float),
            categorical_feature=categorical_columns or "auto",
        )
        prediction = np.maximum(model.predict(x_forecast), 0.0)
        records = build_prediction_records(forecast_frame, prediction, "LightGBM rolling-origin", "test")
        records["fold"] = fold_number
        records["forecast_origin"] = start_date
        records["fold_end_date"] = end_date
        records["step_in_fold"] = records["date"].rank(method="dense").astype(int)
        records["training_rows"] = int(len(history))
        records["training_end_date"] = history["date"].max().normalize()
        baseline = np.maximum(forecast_frame["units_sold_lag_7d"].astype(float).to_numpy(), 0.0)
        baseline_records = build_prediction_records(
            forecast_frame, baseline, "Seasonal naive (7-day lag)", "test"
        )
        model_metrics = calculate_regression_metrics(records["actual_units_sold"], records["forecast_units_sold"])
        baseline_metrics = calculate_regression_metrics(
            baseline_records["actual_units_sold"], baseline_records["forecast_units_sold"]
        )
        fold_rows.append({
            "fold": fold_number,
            "forecast_origin": start_date.date().isoformat(),
            "fold_end_date": end_date.date().isoformat(),
            "n_dates": len(window),
            "n_rows": len(forecast_frame),
            "training_rows": len(history),
            "training_end_date": history["date"].max().date().isoformat(),
            **{f"lightgbm_{key}": value for key, value in model_metrics.items() if key != "n_observations"},
            **{f"baseline_{key}": value for key, value in baseline_metrics.items() if key != "n_observations"},
        })
        all_records.append(records)

    result_frame = pd.concat(all_records, ignore_index=True).sort_values(["date", "store_id", "sku_id"])
    baseline_all = splits["test"].copy()
    baseline_prediction = np.maximum(baseline_all["units_sold_lag_7d"].astype(float).to_numpy(), 0.0)
    baseline_records = build_prediction_records(
        baseline_all, baseline_prediction, "Seasonal naive (7-day lag)", "test"
    )
    lightgbm_overall = calculate_regression_metrics(
        result_frame["actual_units_sold"], result_frame["forecast_units_sold"]
    )
    baseline_overall = calculate_regression_metrics(
        baseline_records["actual_units_sold"], baseline_records["forecast_units_sold"]
    )
    out_dir = root / "outputs/forecasts"
    report_dir = root / "outputs/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "part5_rolling_origin_lightgbm_predictions.csv"
    result_frame.to_csv(prediction_path, index=False)
    fold_path = report_dir / "part5_rolling_origin_fold_metrics.csv"
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    summary = {
        "status": "success",
        "model": "LightGBM",
        "protocol": (
            f"Expanding-window refit every {window_days} test dates. Each target is forecast one-step-ahead; "
            "lag features use previously observed demand. This is not a fixed-origin multi-step forecast."
        ),
        "fold_count": len(windows),
        "window_days": window_days,
        "fold_metrics": fold_rows,
        "overall_lightgbm_metrics": lightgbm_overall,
        "overall_seasonal_naive_metrics": baseline_overall,
        "wape_improvement_vs_baseline_pct": (
            (baseline_overall["wape_pct"] - lightgbm_overall["wape_pct"])
            / baseline_overall["wape_pct"] * 100.0
            if baseline_overall["wape_pct"] not in (None, 0) else None
        ),
        "prediction_path": prediction_path.relative_to(root).as_posix(),
        "fold_metrics_path": fold_path.relative_to(root).as_posix(),
        "data_source_note": metadata.get("data_source_note", "Not specified"),
    }
    summary_path = report_dir / "part5_rolling_origin_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = summary_path.relative_to(root).as_posix()
    return summary
