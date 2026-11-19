"""Shared persistence and evaluation helpers for forecasting adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_inventory.forecasting.data import IDENTIFIER_COLUMNS, TARGET_COLUMN
from demand_inventory.forecasting.metrics import calculate_regression_metrics


def build_prediction_records(
    frame: pd.DataFrame, predictions: np.ndarray, model_name: str, split_name: str
) -> pd.DataFrame:
    values = np.asarray(predictions, dtype=float).reshape(-1)
    if len(values) != len(frame):
        raise ValueError(
            f"{model_name} returned {len(values)} predictions for {len(frame)} {split_name} rows"
        )
    columns = [column for column in IDENTIFIER_COLUMNS if column in frame.columns]
    result = frame[columns].copy()
    result["actual_units_sold"] = frame[TARGET_COLUMN].astype(float).to_numpy()
    result["forecast_units_sold"] = np.maximum(values, 0.0)
    result["error_units"] = result["forecast_units_sold"] - result["actual_units_sold"]
    result["model"] = model_name
    result["split"] = split_name
    return result


def persist_model_evaluation(
    model_name: str,
    predictions_by_split: dict[str, np.ndarray],
    splits: dict[str, pd.DataFrame],
    output_root: str | Path,
    *,
    model_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    forecast_dir = output_root / "outputs/forecasts"
    report_dir = output_root / "outputs/reports"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {"model": model_name, "status": "trained", "splits": {}}
    for split_name, predictions in predictions_by_split.items():
        source = splits[split_name]
        records = build_prediction_records(source, predictions, model_name, split_name)
        path = forecast_dir / f"{split_name}_{_safe_slug(model_name)}_predictions.csv"
        records.to_csv(path, index=False)
        metrics["splits"][split_name] = calculate_regression_metrics(
            records["actual_units_sold"], records["forecast_units_sold"]
        )
        metrics[f"{split_name}_predictions_path"] = path.relative_to(output_root).as_posix()

    if model_path:
        metrics["model_path"] = Path(model_path).resolve().relative_to(output_root).as_posix()
    if extra:
        metrics.update(extra)
    metrics_path = report_dir / f"{_safe_slug(model_name)}_metrics.json"
    metrics["metrics_path"] = metrics_path.relative_to(output_root).as_posix()
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


def _safe_slug(value: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())
